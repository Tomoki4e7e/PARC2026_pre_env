from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import argparse
import json
import shutil
import sys

import numpy as np

from .grasp_metrics import GraspSuccessTracker
from .task_catalog import MIDDLE_DRAWER, TOP_DRAWER, TaskCatalog, TaskSpec
from .base_policy import RemoteBasePolicy


def _flip_libero_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image)[::-1, ::-1].copy()


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q /= np.linalg.norm(q)
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - q[3] ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(q[3]) / denominator).astype(np.float32)


@dataclass
class StageResult:
    name: str
    steps: int
    reached: bool


def _motion(target: np.ndarray, eef: np.ndarray, gripper: float, gain: float = 12.0) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip((target - eef) * gain, -1.0, 1.0)
    action[6] = gripper
    return action


class HandleGraspOracle:
    """Research-only simulator oracle used exclusively to generate training data."""

    def __init__(
        self,
        env: Any,
        base_task: str,
        control_freq: int = 20,
        *,
        base_policy: RemoteBasePolicy | None = None,
        policy_seed: int | None = None,
        handle_only: bool = False,
    ):
        self.env = env
        self.base_task = base_task
        self.tracker = GraspSuccessTracker(env, base_task, control_freq=control_freq)
        self.frames: list[dict[str, Any]] = []
        self.results: list[StageResult] = []
        self.base_policy = base_policy
        self.policy_seed = policy_seed
        self.handle_only = handle_only

    def _record(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        task: str,
        *,
        base_action: np.ndarray | None = None,
    ) -> None:
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        if base_action is None and self.base_policy is not None:
            base_action = np.asarray(self.base_policy.get_action(obs), dtype=np.float32)
        self.frames.append(
            {
                "observation.images.front": _flip_libero_image(obs["agentview_image"]),
                "observation.images.wrist": _flip_libero_image(
                    obs["robot0_eye_in_hand_image"]
                ),
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "base_action": np.asarray(
                    action if base_action is None else base_action, dtype=np.float32
                ),
                "contact": False,
                "closed": False,
                "joint_displacement": 0.0,
                "phase": 0,
                "task": task,
            }
        )

    def _annotate_last(self, snapshot: Any, action: np.ndarray) -> None:
        phase = 0
        if snapshot.grasp_point_error <= 0.08:
            phase = 1
        if snapshot.contact and action[6] > 0.5:
            phase = 2
        if snapshot.joint_displacement >= self.tracker.movement_threshold:
            phase = 3
        if snapshot.success:
            phase = 4
        self.frames[-1].update(
            contact=snapshot.contact,
            closed=snapshot.closed,
            joint_displacement=snapshot.joint_displacement,
            phase=phase,
        )

    def _run_target(
        self,
        name: str,
        obs: dict[str, np.ndarray],
        target_fn,
        *,
        gripper: float,
        max_steps: int,
        tolerance: float,
        task: str,
    ) -> dict[str, np.ndarray]:
        reached = False
        used = 0
        for _ in range(max_steps):
            target = np.asarray(target_fn(), dtype=np.float64)
            action = _motion(target, np.asarray(obs["robot0_eef_pos"]), gripper)
            self._record(obs, action, task)
            obs, _, _, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                len(self.frames),
                gripper_command=float(action[6]),
            )
            self._annotate_last(snapshot, action)
            used += 1
            if np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - target) <= tolerance:
                reached = True
                break
        self.results.append(StageResult(name, used, reached))
        return obs

    def _hold(
        self,
        name: str,
        obs: dict[str, np.ndarray],
        target_fn,
        *,
        gripper: float,
        steps: int,
        task: str,
    ) -> tuple[dict[str, np.ndarray], bool]:
        success = False
        for _ in range(steps):
            action = _motion(
                np.asarray(target_fn()),
                np.asarray(obs["robot0_eef_pos"]),
                gripper,
                gain=8.0,
            )
            self._record(obs, action, task)
            obs, _, _, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                len(self.frames),
                gripper_command=gripper,
            )
            self._annotate_last(snapshot, action)
            success |= snapshot.success
        self.results.append(StageResult(name, steps, success))
        return obs, success

    def _pull_handle(
        self,
        obs: dict[str, np.ndarray],
        start: np.ndarray,
        direction: np.ndarray,
        *,
        task: str,
        distance: float = 0.16,
        steps: int = 70,
    ) -> tuple[dict[str, np.ndarray], bool]:
        success = False
        for index in range(steps):
            fraction = min(1.0, (index + 1) / max(1, steps - 15))
            target = start + distance * fraction * direction
            action = _motion(target, np.asarray(obs["robot0_eef_pos"]), 1.0, gain=8.0)
            self._record(obs, action, task)
            obs, _, _, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                len(self.frames),
                gripper_command=1.0,
            )
            self._annotate_last(snapshot, action)
            success |= snapshot.success
        self.results.append(StageResult("pull_hold", steps, success))
        return obs, success

    def _named_body_position(self, token: str) -> np.ndarray:
        model = self.env.sim.model
        matches = [
            index
            for index in range(model.nbody)
            if token in (model.body_id2name(index) or "")
        ]
        if not matches:
            raise RuntimeError(f"No body contains {token!r}")
        return np.asarray(self.env.sim.data.body_xpos[matches[-1]], dtype=np.float64)

    def collect(
        self, obs: dict[str, np.ndarray], task: str
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        if self.base_policy is not None:
            self.base_policy.reset(task, seed=self.policy_seed)
        handle = self.tracker.handle_center()
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        direction = eef - handle
        direction[2] = 0
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
        pregrasp = handle + 0.05 * direction + np.array([0.0, 0.0, 0.01])
        obs = self._run_target(
            "approach",
            obs,
            lambda: pregrasp,
            gripper=-1.0,
            max_steps=80,
            tolerance=0.02,
            task=task,
        )
        obs = self._run_target(
            "align_handle",
            obs,
            lambda: pregrasp,
            gripper=-1.0,
            max_steps=5,
            tolerance=0.02,
            task=task,
        )
        obs, _ = self._hold(
            "close_handle",
            obs,
            lambda: pregrasp,
            gripper=1.0,
            steps=20,
            task=task,
        )
        initial_handle = self.tracker.handle_center().copy()
        obs, grasp_success = self._pull_handle(
            obs,
            pregrasp,
            direction,
            task=task,
        )
        pull_target = pregrasp + 0.16 * direction
        if self.base_task == TOP_DRAWER and grasp_success and not self.handle_only:
            obs, _ = self._hold(
                "release_handle",
                obs,
                lambda: pull_target,
                gripper=-1.0,
                steps=12,
                task=task,
            )
            bowl = lambda: self._named_body_position("akita_black_bowl_1")
            obs = self._run_target(
                "approach_bowl",
                obs,
                lambda: bowl() + np.array([0.0, 0.0, 0.08]),
                gripper=-1.0,
                max_steps=70,
                tolerance=0.02,
                task=task,
            )
            obs = self._run_target(
                "align_bowl",
                obs,
                lambda: bowl() + np.array([0.0, 0.0, 0.015]),
                gripper=-1.0,
                max_steps=40,
                tolerance=0.012,
                task=task,
            )
            obs, _ = self._hold(
                "close_bowl",
                obs,
                lambda: bowl() + np.array([0.0, 0.0, 0.015]),
                gripper=1.0,
                steps=18,
                task=task,
            )
            obs = self._run_target(
                "lift_bowl",
                obs,
                lambda: bowl() + np.array([0.0, 0.0, 0.12]),
                gripper=1.0,
                max_steps=45,
                tolerance=0.025,
                task=task,
            )
            inside = initial_handle - 0.11 * direction + np.array([0.0, 0.0, 0.06])
            obs = self._run_target(
                "transport_bowl",
                obs,
                lambda: inside,
                gripper=1.0,
                max_steps=70,
                tolerance=0.025,
                task=task,
            )
            obs, _ = self._hold(
                "release_bowl",
                obs,
                lambda: inside,
                gripper=-1.0,
                steps=18,
                task=task,
            )
        task_success = bool(self.env.check_success())
        return self.frames, grasp_success, task_success


class PolicyGraspSupervisor(HandleGraspOracle):
    """Collects DAgger corrections while a frozen VLA supplies task motion."""

    def __init__(
        self,
        env: Any,
        base_task: str,
        policy: RemoteBasePolicy,
        *,
        policy_seed: int | None = None,
    ):
        super().__init__(env, base_task)
        self.policy = policy
        self.policy_seed = policy_seed

    def collect(
        self, obs: dict[str, np.ndarray], task: str
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        self.policy.reset(task, seed=self.policy_seed)
        close_latched = False
        grasp_success = False
        task_success = False
        for step in range(600):
            base_action = np.asarray(self.policy.get_action(obs), dtype=np.float32)
            action = base_action.copy()
            contact, _ = self.tracker._contact_state()
            if contact:
                close_latched = True
            if close_latched:
                action[6] = 1.0
            self._record(obs, action, task, base_action=base_action)
            obs, _, done, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                step,
                gripper_command=float(action[6]),
            )
            grasp_success |= snapshot.success
            phase = 0
            if snapshot.grasp_point_error <= 0.08:
                phase = 1
            if close_latched:
                phase = 2
            if snapshot.joint_displacement >= self.tracker.movement_threshold:
                phase = 3
            if snapshot.success:
                phase = 4
            self.frames[-1].update(
                contact=snapshot.contact,
                closed=snapshot.closed,
                joint_displacement=snapshot.joint_displacement,
                phase=phase,
            )
            if snapshot.joint_displacement >= 0.10 and grasp_success:
                close_latched = False
            task_success |= bool(done) or bool(self.env.check_success())
            if task_success:
                break
        self.results.append(StageResult("policy_supervision", len(self.frames), grasp_success))
        return self.frames, grasp_success, task_success


class ArmCorrectionSupervisor(HandleGraspOracle):
    """Adds bounded oracle translation labels while retaining Base rotations."""

    def __init__(
        self,
        env: Any,
        base_task: str,
        policy: RemoteBasePolicy,
        *,
        policy_seed: int,
        translation_limit: float = 0.05,
    ):
        super().__init__(env, base_task)
        self.policy = policy
        self.policy_seed = policy_seed
        self.translation_limit = translation_limit

    def collect(
        self, obs: dict[str, np.ndarray], task: str
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        self.policy.reset(task, seed=self.policy_seed)
        close_latched = False
        grasp_success = False
        for step in range(600):
            base_action = np.asarray(self.policy.get_action(obs), dtype=np.float32)
            action = base_action.copy()
            contact, _ = self.tracker._contact_state()
            if contact:
                close_latched = True
            if not close_latched:
                handle = self.tracker.handle_center()
                direction = (
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - handle
                )
                direction[2] = 0
                norm = np.linalg.norm(direction)
                direction = (
                    direction / norm
                    if norm > 1e-6
                    else np.array([0.0, -1.0, 0.0])
                )
                pregrasp = (
                    handle
                    + 0.05 * direction
                    + np.array([0.0, 0.0, 0.01])
                )
                expert = _motion(
                    pregrasp,
                    np.asarray(obs["robot0_eef_pos"]),
                    float(base_action[6]),
                )
                correction = np.clip(
                    expert[:3] - base_action[:3],
                    -self.translation_limit,
                    self.translation_limit,
                )
                action[:3] = np.clip(base_action[:3] + correction, -1.0, 1.0)
            if close_latched:
                action[6] = 1.0
            self._record(obs, action, task, base_action=base_action)
            obs, _, _, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                step,
                gripper_command=float(action[6]),
            )
            self._annotate_last(snapshot, action)
            grasp_success |= snapshot.success
            if grasp_success:
                break
        self.results.append(
            StageResult("arm_correction_supervision", len(self.frames), grasp_success)
        )
        return self.frames, grasp_success, bool(self.env.check_success())


def _prepare_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _save_labeled_episode(
    path: Path, frames: list[dict[str, Any]], task: str
) -> None:
    np.savez_compressed(
        path,
        front=np.stack([frame["observation.images.front"] for frame in frames]),
        wrist=np.stack([frame["observation.images.wrist"] for frame in frames]),
        state=np.stack([frame["observation.state"] for frame in frames]),
        action=np.stack([frame["action"] for frame in frames]),
        base_action=np.stack([frame["base_action"] for frame in frames]),
        contact=np.asarray([frame["contact"] for frame in frames]),
        closed=np.asarray([frame["closed"] for frame in frames]),
        joint_displacement=np.asarray(
            [frame["joint_displacement"] for frame in frames], dtype=np.float32
        ),
        phase=np.asarray([frame["phase"] for frame in frames], dtype=np.int8),
        task=np.asarray(task),
    )


def collect_dataset(
    output: Path,
    *,
    episodes_per_task: int,
    max_attempts_per_task: int,
    seed: int,
    lerobot_src: Path | None = None,
    server_url: str | None = None,
    save_failures: int = 0,
    arm_corrections: bool = False,
    task_episode_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    del lerobot_src
    _prepare_output(output)
    failure_output = output.parent / f"{output.name}_failures"
    if save_failures:
        _prepare_output(failure_output)
    catalog = TaskCatalog()
    policy = RemoteBasePolicy(server_url) if server_url else None
    manifest: dict[str, Any] = {
        "seed": seed,
        "tasks": {},
        "gripper_encoding": "libero_signed",
        "collection_mode": "arm_corrections" if arm_corrections else "grasp",
    }
    for task_index, base_task in enumerate((MIDDLE_DRAWER, TOP_DRAWER)):
        target_episodes = (
            task_episode_counts.get(base_task, episodes_per_task)
            if task_episode_counts
            else episodes_per_task
        )
        successes = 0
        failures_saved = 0
        attempts = 0
        rows = []
        while successes < target_episodes and attempts < max_attempts_per_task:
            episode_seed = seed + task_index * 100_000 + attempts
            spec: TaskSpec = catalog.build(
                base_tasks=(base_task,),
                axes=("layout",),
                levels=(0,),
                episodes_per_condition=1,
                seed=episode_seed,
            )[0]
            env = OffScreenRenderEnv(
                bddl_file_name=spec.bddl_file,
                camera_heights=128,
                camera_widths=128,
                control_freq=20,
            )
            env.seed(episode_seed)
            try:
                obs = env.reset()
                for robot in env.robots:
                    robot.controller.use_delta = True
                for _ in range(10):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
                if arm_corrections:
                    if policy is None:
                        raise ValueError("--arm-corrections requires --server-url")
                    oracle = ArmCorrectionSupervisor(
                        env,
                        base_task,
                        policy,
                        policy_seed=episode_seed,
                    )
                else:
                    oracle = (
                        PolicyGraspSupervisor(
                        env, base_task, policy, policy_seed=episode_seed
                        )
                        if policy is not None
                        else HandleGraspOracle(env, base_task)
                    )
                frames, grasp_success, task_success = oracle.collect(obs, spec.language)
                # The strict Phase-1 target is handle GSR. Composite-task TSR is
                # preserved in the sidecar and never relabeled as successful.
                accepted = grasp_success
                if accepted:
                    episode_path = output / f"{base_task}_{successes:03d}.npz"
                    _save_labeled_episode(
                        episode_path,
                        frames,
                        (
                            "open the middle drawer of the cabinet"
                            if base_task == MIDDLE_DRAWER
                            else "open the top drawer and put the bowl inside"
                        ),
                    )
                    successes += 1
                elif failures_saved < save_failures:
                    useful = [
                        frame
                        for index, frame in enumerate(frames)
                        if frame["phase"] >= 1 or index % 8 == 0
                    ]
                    if useful:
                        _save_labeled_episode(
                            failure_output
                            / f"{base_task}_failure_{failures_saved:03d}.npz",
                            useful,
                            (
                                "open the middle drawer of the cabinet"
                                if base_task == MIDDLE_DRAWER
                                else "open the top drawer and put the bowl inside"
                            ),
                        )
                        failures_saved += 1
                rows.append(
                    {
                        "attempt": attempts,
                        "seed": episode_seed,
                        "accepted": accepted,
                        "grasp_success": grasp_success,
                        "task_success": task_success,
                        "frames": len(frames),
                        "stages": [asdict(result) for result in oracle.results],
                    }
                )
            finally:
                env.close()
            attempts += 1
        manifest["tasks"][base_task] = {
            "accepted": successes,
            "attempts": attempts,
            "episodes": rows,
        }
    manifest_path = output.parent / f"{output.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def replay_verify_and_trim(output: Path, manifest_path: Path) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads(manifest_path.read_text())
    catalog = TaskCatalog()
    for base_task, task_data in manifest["tasks"].items():
        accepted_rows = [row for row in task_data["episodes"] if row["accepted"]]
        files = sorted(output.glob(f"{base_task}_*.npz"))
        if len(files) != len(accepted_rows):
            raise RuntimeError(f"Manifest/file mismatch for {base_task}")
        for path, row in zip(files, accepted_rows):
            data = np.load(path)
            spec = catalog.build(
                base_tasks=(base_task,),
                axes=("layout",),
                levels=(0,),
                episodes_per_condition=1,
                seed=row["seed"],
            )[0]
            env = OffScreenRenderEnv(
                bddl_file_name=spec.bddl_file,
                camera_heights=128,
                camera_widths=128,
                control_freq=20,
            )
            env.seed(row["seed"])
            try:
                obs = env.reset()
                for robot in env.robots:
                    robot.controller.use_delta = True
                for _ in range(10):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
                tracker = GraspSuccessTracker(env, base_task, control_freq=20)
                success_step = None
                for index, action in enumerate(data["action"]):
                    obs, _, _, _ = env.step(action)
                    snapshot = tracker.update(
                        obs,
                        index,
                        gripper_command=float(action[6]),
                    )
                    if snapshot.success:
                        success_step = index
                        break
                if success_step is None:
                    raise RuntimeError(f"Replay failed strict GSR: {path}")
                # Strict GSR is the target. Do not teach the release that follows
                # a successful grasp, because it directly conflicts with the
                # required sustained-close behavior.
                stop = min(len(data["action"]), success_step + 1)
                payload = {
                    key: (
                        data[key]
                        if key == "task"
                        else data[key][:stop]
                    )
                    for key in data.files
                }
                np.savez_compressed(path, **payload)
                row["verified_success_step"] = success_step
                row["trimmed_frames"] = stop
            finally:
                env.close()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect strict closed-handle grasp demos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-attempts-per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--server-url")
    parser.add_argument("--replay-trim", action="store_true")
    parser.add_argument("--save-failures", type=int, default=0)
    parser.add_argument("--arm-corrections", action="store_true")
    parser.add_argument("--middle-episodes", type=int)
    parser.add_argument("--top-episodes", type=int)
    args = parser.parse_args()
    if args.replay_trim:
        result = replay_verify_and_trim(
            args.output,
            args.output.parent / f"{args.output.name}_manifest.json",
        )
        print(
            json.dumps(
                {
                    key: value["accepted"]
                    for key, value in result["tasks"].items()
                }
            )
        )
        return
    result = collect_dataset(
        args.output,
        episodes_per_task=args.episodes_per_task,
        max_attempts_per_task=args.max_attempts_per_task,
        seed=args.seed,
        lerobot_src=args.lerobot_src,
        server_url=args.server_url,
        save_failures=args.save_failures,
        arm_corrections=args.arm_corrections,
        task_episode_counts={
            MIDDLE_DRAWER: (
                args.middle_episodes
                if args.middle_episodes is not None
                else args.episodes_per_task
            ),
            TOP_DRAWER: (
                args.top_episodes
                if args.top_episodes is not None
                else args.episodes_per_task
            ),
        },
    )
    print(json.dumps({key: value["accepted"] for key, value in result["tasks"].items()}))


if __name__ == "__main__":
    main()
