from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _axis_quat(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    xyz = np.asarray(axis, dtype=np.float64) * math.sin(angle / 2.0)
    return np.concatenate(([math.cos(angle / 2.0)], xyz))


@dataclass(frozen=True)
class AppliedPerturbation:
    axis: str
    requested: dict[str, float]
    applied: dict[str, float]
    accepted: bool
    reason: str = ""


class RuntimePerturber:
    """Applies only perturbations whose assets are absent from this checkout."""

    def apply(self, env: Any, axis: str, parameters: dict[str, float]) -> AppliedPerturbation:
        if not parameters or axis == "layout":
            return AppliedPerturbation(axis, parameters, {}, True)
        if axis == "camera":
            return self._apply_camera(env, parameters)
        if axis == "robot":
            return self._apply_robot(env, parameters)
        raise ValueError(f"Unknown axis: {axis}")

    def _apply_camera(self, env: Any, parameters: dict[str, float]) -> AppliedPerturbation:
        model = env.sim.model
        try:
            camera_id = model.camera_name2id("agentview")
        except Exception as exc:
            return AppliedPerturbation("camera", parameters, {}, False, str(exc))
        yaw = math.radians(parameters["yaw_deg"])
        pitch = math.radians(parameters["pitch_deg"])
        delta = _quat_mul_wxyz(_axis_quat((0.0, 0.0, 1.0), yaw), _axis_quat((1.0, 0.0, 0.0), pitch))
        original = np.asarray(model.cam_quat[camera_id], dtype=np.float64).copy()
        changed = _quat_mul_wxyz(delta, original)
        model.cam_quat[camera_id] = changed / np.linalg.norm(changed)
        env.sim.forward()
        return AppliedPerturbation(
            "camera",
            parameters,
            {"yaw_deg": parameters["yaw_deg"], "pitch_deg": parameters["pitch_deg"]},
            True,
        )

    def _apply_robot(self, env: Any, parameters: dict[str, float]) -> AppliedPerturbation:
        robot = env.robots[0]
        indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
        data = env.sim.data
        model = env.sim.model
        original = np.asarray(data.qpos[indexes], dtype=np.float64).copy()
        baseline_contacts = int(data.ncon)
        requested_delta = np.zeros(7, dtype=np.float64)
        for i, qpos_index in enumerate(indexes[:7]):
            joint_id = int(model.jnt_qposadr.tolist().index(int(qpos_index)))
            low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
            fraction = float(parameters[f"joint_fraction_{i}"])
            requested_delta[i] = fraction * float(high - low)
        for scale in (1.0, 0.5, 0.25, 0.125):
            changed = original[:7] + scale * requested_delta
            valid = True
            for i, qpos_index in enumerate(indexes[:7]):
                joint_id = int(model.jnt_qposadr.tolist().index(int(qpos_index)))
                low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
                if changed[i] < low or changed[i] > high:
                    valid = False
                    break
            if not valid:
                continue
            data.qpos[indexes[:7]] = changed
            env.sim.forward()
            if int(data.ncon) <= baseline_contacts + 2:
                applied = {f"joint_delta_{i}": float(scale * requested_delta[i]) for i in range(7)}
                applied["backoff_scale"] = scale
                return AppliedPerturbation("robot", parameters, applied, True)
        data.qpos[indexes[:7]] = original[:7]
        env.sim.forward()
        return AppliedPerturbation(
            "robot", parameters, {}, False, "joint-limit or initial-contact rejection after backoff"
        )

