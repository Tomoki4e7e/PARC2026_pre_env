"""Local-simulator RL transition recording and competition-aligned rewards.

This module is intentionally used only when ``EvalConfig.allow_local_data_recording``
is explicitly enabled.  Data from an organizer-hosted evaluation must never be fed
back into training (see the preliminary competition rules).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Dense proxy for the non-public competition score.

    Success and collision dominate.  The smaller terms prefer short, direct and
    smooth trajectories without requiring task-specific privileged state.
    """

    success_bonus: float = 100.0
    collision_penalty: float = 100.0
    env_reward_scale: float = 1.0
    step_penalty: float = 0.01
    cartesian_path_penalty: float = 0.02
    orientation_path_penalty: float = 0.01
    action_delta_penalty: float = 0.01
    cartesian_jerk_penalty: float = 1.0e-5
    control_dt: float = 0.05


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    # LIBERO/robosuite observations use xyzw order.
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = np.sqrt(max(0.0, 1.0 - w * w))
    if denominator < 1.0e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


def observation_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


def shaped_rewards(
    *,
    env_rewards: np.ndarray,
    actions: np.ndarray,
    next_eef_positions: np.ndarray,
    current_eef_positions: np.ndarray | None = None,
    current_eef_quaternions: np.ndarray | None = None,
    next_eef_quaternions: np.ndarray | None = None,
    collision_flags: np.ndarray,
    success: bool,
    weights: RewardWeights,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return per-transition reward and an auditable component summary."""
    count = int(len(actions))
    rewards = np.asarray(env_rewards, dtype=np.float32) * weights.env_reward_scale
    rewards = rewards - weights.step_penalty

    path_cost = np.zeros(count, dtype=np.float32)
    if current_eef_positions is not None:
        path_cost[:] = np.linalg.norm(next_eef_positions - current_eef_positions, axis=1)
    elif count > 1:
        path_cost[1:] = np.linalg.norm(np.diff(next_eef_positions, axis=0), axis=1)
    rewards -= weights.cartesian_path_penalty * path_cost

    orientation_cost = np.zeros(count, dtype=np.float32)
    if current_eef_quaternions is not None and next_eef_quaternions is not None:
        current_q = current_eef_quaternions / np.linalg.norm(
            current_eef_quaternions, axis=1, keepdims=True
        ).clip(1.0e-8)
        next_q = next_eef_quaternions / np.linalg.norm(
            next_eef_quaternions, axis=1, keepdims=True
        ).clip(1.0e-8)
        dots = np.clip(np.abs(np.sum(current_q * next_q, axis=1)), 0.0, 1.0)
        orientation_cost[:] = 2.0 * np.arccos(dots)
        rewards -= weights.orientation_path_penalty * orientation_cost

    action_delta = np.zeros(count, dtype=np.float32)
    if count > 1:
        action_delta[1:] = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
        rewards -= weights.action_delta_penalty * action_delta

    jerk = np.zeros(count, dtype=np.float32)
    if count >= 4:
        velocity = np.diff(next_eef_positions, axis=0) / weights.control_dt
        acceleration = np.diff(velocity, axis=0) / weights.control_dt
        jerk[3:] = np.linalg.norm(
            np.diff(acceleration, axis=0) / weights.control_dt, axis=1
        )
        rewards -= weights.cartesian_jerk_penalty * jerk

    collision_indices = np.flatnonzero(collision_flags)
    if len(collision_indices):
        # A binary episode-level safety gate: charge once at first collision.
        rewards[int(collision_indices[0])] -= weights.collision_penalty
    if success and count:
        rewards[-1] += weights.success_bonus

    components = {
        "env_reward": float(np.sum(env_rewards) * weights.env_reward_scale),
        "step": float(-weights.step_penalty * count),
        "cartesian_path": float(-weights.cartesian_path_penalty * np.sum(path_cost)),
        "orientation_path": float(
            -weights.orientation_path_penalty * np.sum(orientation_cost)
        ),
        "action_delta": float(-weights.action_delta_penalty * np.sum(action_delta)),
        "cartesian_jerk": float(-weights.cartesian_jerk_penalty * np.sum(jerk)),
        "collision": float(-weights.collision_penalty if len(collision_indices) else 0.0),
        "success": float(weights.success_bonus if success else 0.0),
    }
    components["return"] = float(np.sum(rewards))
    return rewards.astype(np.float32), components


def save_rl_episode(
    output_dir: Path,
    *,
    task_name: str,
    instruction: str,
    benchmark: str,
    episode_id: int,
    seed: int,
    observations: list[dict[str, np.ndarray]],
    next_observations: list[dict[str, np.ndarray]],
    actions: list[np.ndarray],
    env_rewards: list[float],
    terminated: list[bool],
    truncated: list[bool],
    collision_flags: list[bool],
    success: bool,
    collision_threshold_m: float,
    reward_weights: RewardWeights | None = None,
    base_actions: list[np.ndarray] | None = None,
    collection_context: dict[str, Any] | None = None,
) -> Path:
    if not actions:
        raise ValueError("cannot save an empty RL episode")
    weights = reward_weights or RewardWeights()
    action_array = np.stack(actions).astype(np.float32)
    next_eef = np.stack(
        [np.asarray(obs["robot0_eef_pos"], dtype=np.float32) for obs in next_observations]
    )
    current_eef = np.stack(
        [np.asarray(obs["robot0_eef_pos"], dtype=np.float32) for obs in observations]
    )
    current_quat = np.stack(
        [np.asarray(obs["robot0_eef_quat"], dtype=np.float32) for obs in observations]
    )
    next_quat = np.stack(
        [np.asarray(obs["robot0_eef_quat"], dtype=np.float32) for obs in next_observations]
    )
    reward_array, components = shaped_rewards(
        env_rewards=np.asarray(env_rewards, dtype=np.float32),
        actions=action_array,
        next_eef_positions=next_eef,
        current_eef_positions=current_eef,
        current_eef_quaternions=current_quat,
        next_eef_quaternions=next_quat,
        collision_flags=np.asarray(collision_flags, dtype=np.bool_),
        success=success,
        weights=weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
    path = output_dir / f"{safe_task}__seed{seed}__ep{episode_id:03d}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite RL episode: {path}")

    payload: dict[str, Any] = {
        "observation.front": np.stack(
            [np.asarray(obs["agentview_image"], dtype=np.uint8) for obs in observations]
        ),
        "observation.wrist": np.stack(
            [np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8) for obs in observations]
        ),
        "observation.state": np.stack([observation_state(obs) for obs in observations]),
        "action": action_array,
        "env_reward": np.asarray(env_rewards, dtype=np.float32),
        "reward": reward_array,
        "next_observation.front": np.stack(
            [np.asarray(obs["agentview_image"], dtype=np.uint8) for obs in next_observations]
        ),
        "next_observation.wrist": np.stack(
            [np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8) for obs in next_observations]
        ),
        "next_observation.state": np.stack(
            [observation_state(obs) for obs in next_observations]
        ),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "collision": np.asarray(collision_flags, dtype=np.bool_),
    }
    if base_actions is not None and len(base_actions) == len(actions):
        payload["base_action"] = np.stack(base_actions).astype(np.float32)

    metadata = {
        "schema_version": 1,
        "source": "local_simulator_rollout",
        "task_name": task_name,
        "instruction": instruction,
        "benchmark": benchmark,
        "episode_id": episode_id,
        "seed": seed,
        "success": bool(success),
        "collided": bool(np.any(collision_flags)),
        "collision_threshold_m": float(collision_threshold_m),
        "reward_weights": asdict(weights),
        "reward_components": components,
    }
    if collection_context is not None:
        metadata["collection_context"] = collection_context
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))

    # A worker may be restarted while another isolated simulator process is
    # winding down.  Per-process temporary names avoid the two writers racing
    # over the same ``.tmp`` inode; the final path remains overwrite-protected.
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
    return path
