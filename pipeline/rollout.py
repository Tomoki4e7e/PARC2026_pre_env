import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import EvalConfig, PerturbationConfig
from .environment import EnvironmentManager, TaskInfo
from .total_score import load_scoring_config
from .rl_data import save_rl_episode
from .video import EpisodeVideoRecorder

logger = logging.getLogger(__name__)

# LeRobot LiberoEnv.get_libero_dummy_action() と同一（グリッパ開）
_LIBERO_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)


class PolicyInterface(Protocol):

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        ...


@dataclass
class EpisodeResult:
    task_name: str
    episode_id: int
    success: bool
    total_steps: int
    elapsed_time_sec: float

    joint_positions: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


    collided: bool = False
    video_path: str | None = None

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self.joint_positions


@dataclass
class TaskResult:
    task_info: TaskInfo
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return 0.0
        return sum(e.total_steps for e in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.elapsed_time_sec for e in self.episodes) / len(self.episodes)


class RolloutExecutor:

    def __init__(
        self,
        env_manager: EnvironmentManager,
        eval_config: EvalConfig,
        scoring_config: dict | None = None,
    ):
        self.env_manager = env_manager
        self.config = eval_config


        self.scoring_config = scoring_config or load_scoring_config()
        if self.config.rl_record_dir is not None and not self.config.allow_local_data_recording:
            raise ValueError(
                "rl_record_dir requires allow_local_data_recording=True; "
                "organizer evaluation observations must not be recorded"
            )

    def evaluate_task(
        self,
        policy: PolicyInterface,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
    ) -> TaskResult:
        logger.info(
            "タスク評価開始: %s (%d エピソード)",
            task_info.name, self.config.n_eval_episodes,
        )

        env = self.env_manager.create_env(task_info)


        init_states = self.env_manager.get_perturbed_init_states(
            task_info, perturbation, self.config.n_eval_episodes
        )


        collision_enabled = bool(self.scoring_config.get("collision", {}).get("enabled", True))
        obj_of_interest = (
            self.env_manager.get_obj_of_interest(task_info) if collision_enabled else set()
        )

        episodes: list[EpisodeResult] = []

        try:
            for ep_id in range(self.config.n_eval_episodes):
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    task_info=task_info,
                    init_state=init_states[ep_id],
                    episode_id=ep_id,
                    perturbation=perturbation,
                    obj_of_interest=obj_of_interest,
                )
                episodes.append(result)

                if result.success:
                    logger.debug(
                        "  Episode %d: 成功 (%d steps)", ep_id, result.total_steps
                    )
                else:
                    logger.debug(
                        "  Episode %d: 失敗 (%d steps)", ep_id, result.total_steps
                    )
        finally:
            env.close()

        task_result = TaskResult(task_info=task_info, episodes=episodes)
        logger.info(
            "タスク評価完了: %s — 成功率 %.1f%% (平均 %.1f steps)",
            task_info.name, task_result.success_rate * 100, task_result.avg_steps,
        )
        return task_result

    def _run_episode(
        self,
        env: Any,
        policy: PolicyInterface,
        task_info: TaskInfo,
        init_state: np.ndarray,
        episode_id: int,
        perturbation: PerturbationConfig,
        obj_of_interest: set[str],
    ) -> EpisodeResult:
        start_time = time.time()
        joint_positions: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_orientations: list[np.ndarray] = []
        gripper_qpos_log: list[np.ndarray] = []
        actions_log: list[np.ndarray] = []
        rewards_log: list[float] = []
        record_rl = self.config.rl_record_dir is not None
        observations_log: list[dict[str, np.ndarray]] = []
        next_observations_log: list[dict[str, np.ndarray]] = []
        terminated_log: list[bool] = []
        truncated_log: list[bool] = []
        collision_flags_log: list[bool] = []
        base_actions_log: list[np.ndarray] = []

        cc = self.scoring_config.get("collision", {})
        collision_enabled = bool(cc.get("enabled", True))
        collision_threshold = float(cc.get("threshold_m", 0.001))


        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()


        action_dim = env.robots[0].action_dim
        dummy_action = _LIBERO_DUMMY_ACTION[:action_dim]
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)

        # SmolVLA / LIBERO 学習条件に合わせて相対アクション（OSC delta）を使用
        for robot in env.robots:
            robot.controller.use_delta = True

        video = None
        if self.config.video_dir is not None:
            video = EpisodeVideoRecorder(
                self.config.video_dir, task_info.name, episode_id,
                task_info.language, self.config.video_camera,
                self.config.video_fps, self.config.video_frame_skip,
            )
            video.add(obs, 0, force=True)


        object_init_pos: dict[str, np.ndarray] = {}
        if collision_enabled:
            object_init_pos = {
                k[:-4]: np.asarray(obs[k]).copy()
                for k in obs
                if k.endswith("_pos")
                and not k.startswith("robot0")
                and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in obj_of_interest
            }
        object_max_disp: dict[str, float] = {}
        first_collision: dict[str, float | int | str] | None = None
        collision_diagnostics = os.environ.get(
            "PARC_COLLISION_DIAGNOSTICS", "0"
        ).strip().lower() not in ("", "0", "false", "no")


        episode_seed = self.config.seed + episode_id
        policy.reset(instruction=task_info.language, seed=episode_seed)
        done = False
        total_steps = 0

        for step in range(self.config.max_steps_per_episode):

            obs_for_policy = self.env_manager.apply_observation_noise(
                obs, perturbation
            )


            action = policy.get_action(obs_for_policy)
            if record_rl:
                observations_log.append(
                    {key: np.asarray(value).copy() for key, value in obs_for_policy.items()}
                )
                base_action = getattr(policy, "last_base_action", None)
                if base_action is not None:
                    base_actions_log.append(np.asarray(base_action, dtype=np.float32).copy())


            action = self.env_manager.apply_action_noise(action, perturbation)


            obs, reward, done, info = env.step(action)
            if video is not None:
                video.add(obs, step + 1)


            joint_positions.append(obs.get("robot0_joint_pos", np.zeros(7)).copy())
            ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())
            ee_orientations.append(obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float64)).copy())
            gripper_qpos_log.append(obs.get("robot0_gripper_qpos", np.zeros(2)).copy())
            actions_log.append(action.copy())
            rewards_log.append(float(reward))


            for name, p0 in object_init_pos.items():
                cur = obs.get(name + "_pos")
                if cur is not None:
                    d = float(np.sum(np.abs(np.asarray(cur) - p0)))
                    if d > object_max_disp.get(name, 0.0):
                        object_max_disp[name] = d
                    if first_collision is None and d > collision_threshold:
                        first_collision = {
                            "object": name,
                            "step": step + 1,
                            "displacement": d,
                        }
                        if collision_diagnostics:
                            logger.info(
                                "  [collision-diagnostic] episode=%d object=%s "
                                "step=%d displacement=%.9f",
                                episode_id,
                                name,
                                step + 1,
                                d,
                            )

            if record_rl:
                next_observations_log.append(
                    {key: np.asarray(value).copy() for key, value in obs.items()}
                )
                collision_flags_log.append(
                    any(d > collision_threshold for d in object_max_disp.values())
                )
                terminated_log.append(bool(done))
                truncated_log.append(
                    not bool(done) and step + 1 >= self.config.max_steps_per_episode
                )

            total_steps = step + 1

            if total_steps % 50 == 0:
                logger.info(
                    "  [進捗] %s: %d/%d steps (%.1fs)",
                    task_info.name, total_steps, self.config.max_steps_per_episode,
                    time.time() - start_time,
                )

            if done:
                break

        elapsed = time.time() - start_time


        collided = any(d > collision_threshold for d in object_max_disp.values())
        success = bool(done) and not collided
        video_path = None
        if video is not None:
            saved_video = video.close(total_steps, success, collided)
            if saved_video is not None:
                video_path = str(saved_video.resolve())
                logger.info("  rollout video saved: %s", video_path)

        if record_rl:
            record_path = save_rl_episode(
                self.config.rl_record_dir,
                task_name=task_info.name,
                instruction=task_info.language,
                benchmark=task_info.benchmark_name,
                episode_id=episode_id,
                seed=episode_seed,
                observations=observations_log,
                next_observations=next_observations_log,
                actions=actions_log,
                env_rewards=rewards_log,
                terminated=terminated_log,
                truncated=truncated_log,
                collision_flags=collision_flags_log,
                success=success,
                collision_threshold_m=collision_threshold,
                base_actions=base_actions_log if len(base_actions_log) == len(actions_log) else None,
                collection_context={
                    "init_state_ordinal": int(self.config.init_state_offset + episode_id),
                    "init_state_offset": int(self.config.init_state_offset),
                },
            )
            logger.info("  RL transition saved: %s", record_path)

        return EpisodeResult(
            task_name=task_info.name,
            episode_id=episode_id,
            success=success,
            total_steps=total_steps,
            elapsed_time_sec=elapsed,
            joint_positions=joint_positions,
            ee_positions=ee_positions,
            ee_orientations=ee_orientations,
            gripper_qpos=gripper_qpos_log,
            actions=actions_log,
            rewards=rewards_log,
            collided=collided,
            video_path=video_path,
        )

    def evaluate_tasks(
        self,
        policy: PolicyInterface,
        task_infos: list[TaskInfo],
        perturbation: PerturbationConfig,
    ) -> list[TaskResult]:
        results = []
        for i, task_info in enumerate(task_infos):
            logger.info(
                "=== タスク %d/%d: %s ===",
                i + 1, len(task_infos), task_info.name,
            )
            result = self.evaluate_task(policy, task_info, perturbation)
            results.append(result)
        return results
