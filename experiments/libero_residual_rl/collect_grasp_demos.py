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

    def __init__(self, env: Any, base_task: str, control_freq: int = 20):
        self.env = env
        self.base_task = base_task
        self.tracker = GraspSuccessTracker(env, base_task, control_freq=control_freq)
        self.frames: list[dict[str, Any]] = []
        self.results: list[StageResult] = []

    def _record(self, obs: dict[str, np.ndarray], action: np.ndarray, task: str) -> None:
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        self.frames.append(
            {
                "observation.images.front": _flip_libero_image(obs["agentview_image"]),
                "observation.images.wrist": _flip_libero_image(
                    obs["robot0_eye_in_hand_image"]
                ),
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": task,
            }
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
        if self.base_task == TOP_DRAWER and grasp_success:
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

    def __init__(self, env: Any, base_task: str, policy: RemoteBasePolicy):
        super().__init__(env, base_task)
        self.policy = policy

    def collect(
        self, obs: dict[str, np.ndarray], task: str
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        self.policy.reset(task)
        close_latched = False
        grasp_success = False
        task_success = False
        for step in range(600):
            action = np.asarray(self.policy.get_action(obs), dtype=np.float32)
            contact, _ = self.tracker._contact_state()
            if contact:
                close_latched = True
            if close_latched:
                action[6] = 1.0
            self._record(obs, action, task)
            obs, _, done, _ = self.env.step(action)
            snapshot = self.tracker.update(
                obs,
                step,
                gripper_command=float(action[6]),
            )
            grasp_success |= snapshot.success
            if snapshot.joint_displacement >= 0.10 and grasp_success:
                close_latched = False
            task_success |= bool(done) or bool(self.env.check_success())
            if task_success:
                break
        self.results.append(StageResult("policy_supervision", len(self.frames), grasp_success))
        return self.frames, grasp_success, task_success


def _prepare_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def collect_dataset(
    output: Path,
    *,
    episodes_per_task: int,
    max_attempts_per_task: int,
    seed: int,
    lerobot_src: Path | None = None,
    server_url: str | None = None,
) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    del lerobot_src
    _prepare_output(output)
    catalog = TaskCatalog()
    policy = RemoteBasePolicy(server_url) if server_url else None
    manifest: dict[str, Any] = {"seed": seed, "tasks": {}, "gripper_encoding": "libero_signed"}
    for task_index, base_task in enumerate((MIDDLE_DRAWER, TOP_DRAWER)):
        successes = 0
        attempts = 0
        rows = []
        while successes < episodes_per_task and attempts < max_attempts_per_task:
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
                oracle = (
                    PolicyGraspSupervisor(env, base_task, policy)
                    if policy is not None
                    else HandleGraspOracle(env, base_task)
                )
                frames, grasp_success, task_success = oracle.collect(obs, spec.language)
                # The strict Phase-1 target is handle GSR. Composite-task TSR is
                # preserved in the sidecar and never relabeled as successful.
                accepted = grasp_success
                if accepted:
                    episode_path = output / f"{base_task}_{successes:03d}.npz"
                    np.savez_compressed(
                        episode_path,
                        front=np.stack(
                            [frame["observation.images.front"] for frame in frames]
                        ),
                        wrist=np.stack(
                            [frame["observation.images.wrist"] for frame in frames]
                        ),
                        state=np.stack(
                            [frame["observation.state"] for frame in frames]
                        ),
                        action=np.stack([frame["action"] for frame in frames]),
                        task=np.asarray(spec.language),
                    )
                    successes += 1
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
                stop = min(len(data["action"]), success_step + 21)
                np.savez_compressed(
                    path,
                    front=data["front"][:stop],
                    wrist=data["wrist"][:stop],
                    state=data["state"][:stop],
                    action=data["action"][:stop],
                    task=data["task"],
                )
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
    )
    print(json.dumps({key: value["accepted"] for key, value in result["tasks"].items()}))


if __name__ == "__main__":
    main()
