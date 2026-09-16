from types import SimpleNamespace

import numpy as np

from experiments.libero_residual_rl.grasp_metrics import GraspSuccessTracker
from experiments.libero_residual_rl.task_catalog import MIDDLE_DRAWER


class _Model:
    nbody = 2
    njnt = 1
    ngeom = 3
    geom_bodyid = np.array([0, 0, 1])
    geom_pos = np.array([[0, -0.10, 0], [0, -0.08, 0], [0, 0, 0]], dtype=float)
    jnt_qposadr = np.array([0])

    @staticmethod
    def body_id2name(index):
        return ["wooden_cabinet_1_cabinet_middle", "robot_hand"][index]

    @staticmethod
    def joint_id2name(index):
        return ["wooden_cabinet_1_middle_level"][index]

    @staticmethod
    def geom_id2name(index):
        return ["drawer_handle_a", "drawer_handle_b", "gripper0_fingerpad"][index]


def _environment(with_contact=True):
    contact = SimpleNamespace(geom1=0, geom2=2, efc_address=0, dist=-0.001)
    data = SimpleNamespace(
        qpos=np.array([0.01]),
        geom_xpos=np.array([[0.0, 0.0, 0.1], [0.02, 0.0, 0.1], [0.01, 0.0, 0.1]]),
        site_xpos=np.array([[0.01, 0.0, 0.1]]),
        contact=[contact] if with_contact else [],
        ncon=1 if with_contact else 0,
        efc_force=np.array([2.0]),
    )
    sim = SimpleNamespace(model=_Model(), data=data)
    return SimpleNamespace(sim=sim, robots=[SimpleNamespace(eef_site_id=0)])


def test_grasp_requires_sustained_contact_and_following():
    env = _environment()
    tracker = GraspSuccessTracker(env, MIDDLE_DRAWER, control_freq=4, hold_seconds=0.5)
    env.sim.data.qpos[0] = 0.02
    obs = {
        "robot0_eef_pos": np.array([0.01, 0.0, 0.1]),
        "robot0_gripper_qpos": np.zeros(2),
    }

    first = tracker.update(obs, 1, gripper_command=1.0)
    second = tracker.update(obs, 2, gripper_command=1.0)

    assert not first.success
    assert second.success
    assert tracker.first_success_step == 2
    assert tracker.failure_mode() == "success"


def test_no_contact_is_classified_as_target_missed():
    tracker = GraspSuccessTracker(_environment(False), MIDDLE_DRAWER)
    assert tracker.failure_mode() == "target_missed"


def test_open_command_cannot_pass_strict_grasp():
    env = _environment()
    tracker = GraspSuccessTracker(env, MIDDLE_DRAWER, control_freq=1, hold_seconds=0.5)
    env.sim.data.qpos[0] = 0.02
    snapshot = tracker.update(
        {
            "robot0_eef_pos": np.array([0.01, 0.0, 0.1]),
            "robot0_gripper_qpos": np.zeros(2),
        },
        1,
        gripper_command=-1.0,
    )
    assert snapshot.contact and snapshot.closed
    assert not snapshot.close_command
    assert not snapshot.success

