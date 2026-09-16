from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


GripperEncoding = Literal["zero_one", "libero_signed"]


@dataclass(frozen=True)
class ActionSchema:
    """Explicitly converts dataset actions into LIBERO controller actions.

    The pinned LeRobot v3 dataset stores 0=close and 1=open. LIBERO's Panda
    controller uses +1=close and -1=open, as verified by a no-motion probe.
    """

    source_gripper: GripperEncoding = "zero_one"
    strict: bool = False
    discrete_gripper: bool = True

    def to_libero(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32).copy()
        if value.shape[-1] != 7:
            raise ValueError(f"Expected (..., 7) action, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("Action contains NaN or infinity")
        if self.source_gripper == "zero_one":
            if self.strict and np.any(
                (value[..., 6] < -1e-4) | (value[..., 6] > 1.0001)
            ):
                raise ValueError("zero_one gripper action is outside [0, 1]")
            value[..., 6] = np.clip(value[..., 6], 0.0, 1.0)
            value[..., 6] = 1.0 - 2.0 * value[..., 6]
        elif self.source_gripper != "libero_signed":
            raise ValueError(f"Unknown gripper encoding: {self.source_gripper}")
        if self.discrete_gripper:
            value[..., 6] = np.where(value[..., 6] >= 0.0, 1.0, -1.0)
        return np.clip(value, -1.0, 1.0)

    def from_libero(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32).copy()
        if value.shape[-1] != 7:
            raise ValueError(f"Expected (..., 7) action, got {value.shape}")
        if self.source_gripper == "zero_one":
            value[..., 6] = (1.0 - value[..., 6]) / 2.0
        return value


def validate_observation_schema(obs: dict[str, np.ndarray]) -> None:
    required = {
        "agentview_image": 3,
        "robot0_eye_in_hand_image": 3,
        "robot0_eef_pos": 1,
        "robot0_eef_quat": 1,
        "robot0_gripper_qpos": 1,
    }
    for key, ndim in required.items():
        if key not in obs:
            raise KeyError(f"Missing observation field: {key}")
        if np.asarray(obs[key]).ndim != ndim:
            raise ValueError(f"{key} must have {ndim} dimensions")
    if np.asarray(obs["robot0_eef_pos"]).shape != (3,):
        raise ValueError("robot0_eef_pos must be (3,)")
    if np.asarray(obs["robot0_eef_quat"]).shape != (4,):
        raise ValueError("robot0_eef_quat must be (4,)")
    if np.asarray(obs["robot0_gripper_qpos"]).shape != (2,):
        raise ValueError("robot0_gripper_qpos must be (2,)")
