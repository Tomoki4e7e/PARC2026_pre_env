from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import math
import numpy as np

from .task_catalog import MICROWAVE, MIDDLE_DRAWER, TOP_DRAWER


@dataclass(frozen=True)
class HandleTarget:
    body_suffix: str
    joint_suffix: str


TARGETS = {
    MIDDLE_DRAWER: HandleTarget("cabinet_middle", "middle_level"),
    TOP_DRAWER: HandleTarget("cabinet_top", "top_level"),
    MICROWAVE: HandleTarget("microdoorroot", "microjoint"),
}


@dataclass
class GraspSnapshot:
    contact: bool
    closed: bool
    close_command: bool
    tracking_error: float
    joint_displacement: float
    contact_force: float
    grasp_point_error: float
    success: bool


class GraspSuccessTracker:
    """Stateful contact-and-following grasp metric using MuJoCo ground truth."""

    def __init__(
        self,
        env: Any,
        base_task: str,
        *,
        control_freq: int = 20,
        hold_seconds: float = 0.5,
        tracking_threshold: float = 0.02,
        movement_threshold: float = 0.003,
        gripper_closed_threshold: float = 0.055,
    ):
        self.env = env
        self.sim = env.sim
        self.target = TARGETS[base_task]
        self.hold_steps = max(1, math.ceil(control_freq * hold_seconds))
        self.tracking_threshold = tracking_threshold
        self.movement_threshold = movement_threshold
        self.gripper_closed_threshold = gripper_closed_threshold
        self.body_id = self._resolve_id("body", self.target.body_suffix)
        self.joint_id = self._resolve_id("joint", self.target.joint_suffix)
        self.joint_qpos_addr = int(self.sim.model.jnt_qposadr[self.joint_id])
        self.handle_geom_ids = self._resolve_handle_geoms()
        self.finger_geom_ids = self._resolve_finger_geoms()
        self.initial_joint = float(self.sim.data.qpos[self.joint_qpos_addr])
        self.initial_handle = self.handle_center()
        self._last_relative: np.ndarray | None = None
        self._valid_history: deque[bool] = deque(maxlen=self.hold_steps)
        self.max_contact_force = 0.0
        self.min_grasp_point_error = float("inf")
        self.first_success_step: int | None = None
        self.ever_contact = False

    def _names(self, kind: str) -> list[str]:
        count_attribute = {"joint": "njnt"}.get(kind, f"n{kind}")
        count = int(getattr(self.sim.model, count_attribute))
        method = getattr(self.sim.model, f"{kind}_id2name")
        return [method(i) or "" for i in range(count)]

    def _resolve_id(self, kind: str, suffix: str) -> int:
        matches = [i for i, name in enumerate(self._names(kind)) if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {kind} ending in {suffix!r}, got {matches}")
        return matches[0]

    def _resolve_handle_geoms(self) -> set[int]:
        model = self.sim.model
        candidates = [
            i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == self.body_id
        ]
        if not candidates:
            raise RuntimeError(f"No geoms found under handle body {self.body_id}")
        local_y = np.asarray([model.geom_pos[i][1] for i in candidates])
        front = float(local_y.min())
        return {i for i in candidates if float(model.geom_pos[i][1]) <= front + 0.03}

    def _resolve_finger_geoms(self) -> set[int]:
        names = self._names("geom")
        matches = {
            i
            for i, name in enumerate(names)
            if "finger" in name.lower() or "fingerpad" in name.lower()
        }
        if not matches:
            raise RuntimeError("No finger geoms found")
        return matches

    def handle_center(self) -> np.ndarray:
        return np.asarray(
            [self.sim.data.geom_xpos[i] for i in sorted(self.handle_geom_ids)],
            dtype=np.float64,
        ).mean(axis=0)

    def _eef_position(self) -> np.ndarray:
        return np.asarray(
            self.sim.data.site_xpos[self.env.robots[0].eef_site_id],
            dtype=np.float64,
        )

    def _contact_state(self) -> tuple[bool, float]:
        contact_found = False
        force = 0.0
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.handle_geom_ids and pair & self.finger_geom_ids:
                contact_found = True
                address = int(getattr(contact, "efc_address", -1))
                if 0 <= address < len(self.sim.data.efc_force):
                    force = max(force, abs(float(self.sim.data.efc_force[address])))
                else:
                    force = max(force, max(0.0, -float(contact.dist)))
        return contact_found, force

    def update(
        self,
        obs: dict[str, np.ndarray],
        step: int,
        *,
        gripper_command: float | None = None,
    ) -> GraspSnapshot:
        handle = self.handle_center()
        eef = np.asarray(obs.get("robot0_eef_pos", self._eef_position()), dtype=np.float64)
        relative = eef - handle
        tracking_error = (
            0.0 if self._last_relative is None else float(np.linalg.norm(relative - self._last_relative))
        )
        self._last_relative = relative
        contact, force = self._contact_state()
        self.ever_contact |= contact
        gripper = np.asarray(obs.get("robot0_gripper_qpos", [1.0, 1.0]), dtype=np.float64)
        aperture = (
            abs(float(gripper[0] - gripper[1]))
            if gripper.size >= 2
            else abs(float(gripper[0]))
        )
        closed = aperture <= self.gripper_closed_threshold
        close_command = gripper_command is not None and gripper_command >= 0.5
        joint_displacement = abs(
            float(self.sim.data.qpos[self.joint_qpos_addr]) - self.initial_joint
        )
        point_error = float(np.linalg.norm(relative))
        self.min_grasp_point_error = min(self.min_grasp_point_error, point_error)
        self.max_contact_force = max(self.max_contact_force, force)
        valid = (
            contact
            and closed
            and close_command
            and tracking_error <= self.tracking_threshold
            and joint_displacement >= self.movement_threshold
        )
        self._valid_history.append(valid)
        success = len(self._valid_history) == self.hold_steps and all(self._valid_history)
        if success and self.first_success_step is None:
            self.first_success_step = step
        return GraspSnapshot(
            contact=contact,
            closed=closed,
            close_command=close_command,
            tracking_error=tracking_error,
            joint_displacement=joint_displacement,
            contact_force=force,
            grasp_point_error=point_error,
            success=success,
        )

    def failure_mode(self) -> str:
        if self.first_success_step is not None:
            return "success"
        if not self.ever_contact:
            return "target_missed"
        if self.min_grasp_point_error > 0.06:
            return "wrong_approach"
        return "hold_failed"

    @staticmethod
    def snapshot_dict(snapshot: GraspSnapshot) -> dict[str, float | bool]:
        return asdict(snapshot)

