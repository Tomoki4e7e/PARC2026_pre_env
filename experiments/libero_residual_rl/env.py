from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import time

from .base_policy import BasePolicy
from .config import ExperimentConfig
from .grasp_metrics import GraspSuccessTracker
from .perturbations import RuntimePerturber
from .task_catalog import TaskSpec


_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)


class ResidualLiberoEnv:
    """Research-only environment; simulator truth never enters policy observations."""

    def __init__(
        self,
        spec: TaskSpec,
        base_policy: BasePolicy,
        config: ExperimentConfig,
        *,
        terminate_on_grasp: bool = False,
    ):
        from libero.libero.envs import OffScreenRenderEnv

        self.spec = spec
        self.base_policy = base_policy
        self.config = config
        self.terminate_on_grasp = terminate_on_grasp
        self.env = OffScreenRenderEnv(
            bddl_file_name=spec.bddl_file,
            camera_heights=config.image_size,
            camera_widths=config.image_size,
            control_freq=config.control_freq,
        )
        self.env.seed(spec.seed)
        self.perturber = RuntimePerturber()
        self.obs: dict[str, np.ndarray] | None = None
        self.tracker: GraspSuccessTracker | None = None
        self.last_residual = np.zeros(7, dtype=np.float32)
        self.previous_distance = 0.0
        self.previous_joint_displacement = 0.0
        self.step_count = 0
        self.task_success = False
        self.applied_perturbation: dict[str, Any] = {}
        self.last_base_latency = 0.0

    def reset(self) -> dict[str, np.ndarray]:
        self.env.reset()
        action = _DUMMY_ACTION[: self.env.robots[0].action_dim]
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(action)
        for robot in self.env.robots:
            robot.controller.use_delta = True
        applied = self.perturber.apply(self.env, self.spec.axis, self.spec.runtime_parameters)
        if not applied.accepted:
            raise RuntimeError(f"Perturbation rejected: {applied.reason}")
        self.applied_perturbation = asdict(applied)
        self.env._update_observables(force=True)
        self.obs = self.env.env._get_observations()
        self.base_policy.reset(self.spec.language, seed=self.spec.seed)
        self.tracker = GraspSuccessTracker(
            self.env,
            self.spec.base_task,
            control_freq=self.config.control_freq,
            hold_seconds=self.config.grasp_hold_seconds,
        )
        self.last_residual.fill(0)
        self.previous_distance = float(
            np.linalg.norm(self.obs["robot0_eef_pos"] - self.tracker.handle_center())
        )
        self.previous_joint_displacement = 0.0
        self.step_count = 0
        self.task_success = False
        return self.obs

    def prepare_step(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.obs is None:
            raise RuntimeError("reset() must be called first")
        started = time.perf_counter()
        base_action = np.asarray(self.base_policy.get_action(self.obs), dtype=np.float32)
        self.last_base_latency = time.perf_counter() - started
        if base_action.shape != (7,):
            raise ValueError(f"Base action must have shape (7,), got {base_action.shape}")
        return base_action, self.obs

    def _safe_residual(self, residual: np.ndarray) -> np.ndarray:
        value = np.asarray(residual, dtype=np.float32).reshape(7)
        limits = np.asarray(self.config.limits.vector(), dtype=np.float32)
        value = np.clip(value, -limits, limits)
        delta = np.clip(
            value - self.last_residual,
            -self.config.limits.rate,
            self.config.limits.rate,
        )
        value = self.last_residual + delta
        if not np.isfinite(value).all():
            value.fill(0)
        return value

    def step(
        self, base_action: np.ndarray, residual: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self.tracker is None:
            raise RuntimeError("reset() must be called first")
        safe_residual = self._safe_residual(residual)
        combined = np.clip(np.asarray(base_action) + safe_residual, -1.0, 1.0)
        self.obs, _, done, info = self.env.step(combined)
        self.step_count += 1
        snapshot = self.tracker.update(
            self.obs,
            self.step_count,
            gripper_command=float(combined[6]),
        )
        distance = snapshot.grasp_point_error
        distance_progress = self.previous_distance - distance
        joint_progress = snapshot.joint_displacement - self.previous_joint_displacement
        reward = (
            10.0 * distance_progress
            + 20.0 * max(0.0, joint_progress)
            + (0.1 if snapshot.contact else 0.0)
            + (10.0 if snapshot.success else 0.0)
            - 0.005 * snapshot.contact_force
            - 0.02 * float(np.square(safe_residual).sum())
            - 0.02 * float(np.square(safe_residual - self.last_residual).sum())
        )
        self.previous_distance = distance
        self.previous_joint_displacement = snapshot.joint_displacement
        self.last_residual = safe_residual
        self.task_success |= bool(done)
        terminated = (
            (self.terminate_on_grasp and snapshot.success)
            or self.step_count >= self.config.max_episode_steps
        )
        return self.obs, float(reward), terminated, {
            **info,
            "grasp": self.tracker.snapshot_dict(snapshot),
            "grasp_success": snapshot.success,
            "task_success": self.task_success,
            "base_action": np.asarray(base_action, dtype=np.float32),
            "residual_action": safe_residual.copy(),
            "combined_action": combined.astype(np.float32),
        }

    def close(self) -> None:
        self.env.close()

