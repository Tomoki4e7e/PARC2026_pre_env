import json

import numpy as np
import torch

from experiments.libero_residual_rl.arm_residual_head import (
    ArmHeadConfig,
    LearnedArmResidualPolicy,
    LearnedFrontGraspPolicy,
    TemporalArmResidualHead,
)
from experiments.libero_residual_rl.gripper_head import (
    GripperHeadConfig,
    TemporalGripperHead,
)


def test_arm_head_shape_and_limits():
    config = ArmHeadConfig()
    model = TemporalArmResidualHead(config)
    residual, approach, contact = model(
        torch.zeros(2, 4, 3, 64, 64), torch.zeros(2, 4, 15)
    )
    assert residual.shape == (2, 6)
    assert approach.shape == contact.shape == (2,)
    assert torch.all(residual.abs() <= model.limits + 1e-6)


def test_two_camera_arm_head_shape():
    config = ArmHeadConfig(mode="two_camera", task_conditioned=True)
    model = TemporalArmResidualHead(config)
    residual, approach, contact = model(
        torch.zeros(2, 4, 6, 64, 64), torch.zeros(2, 4, 16)
    )
    assert residual.shape == (2, 6)
    assert approach.shape == contact.shape == (2,)


def test_combined_policy_uses_front_only_and_separates_coordinates(tmp_path):
    arm_dir, gripper_dir = tmp_path / "arm", tmp_path / "gripper"
    arm_dir.mkdir()
    gripper_dir.mkdir()
    arm_config = ArmHeadConfig()
    arm = TemporalArmResidualHead(arm_config)
    for parameter in arm.parameters():
        parameter.data.zero_()
    arm.residual.bias.data[0] = 10
    arm.approach.bias.data.fill_(10)
    torch.save(arm.state_dict(), arm_dir / "model.pt")
    (arm_dir / "config.json").write_text(json.dumps(arm_config.__dict__))
    gripper_config = GripperHeadConfig(mode="front", threshold=0.1)
    gripper = TemporalGripperHead(gripper_config)
    for parameter in gripper.parameters():
        parameter.data.zero_()
    gripper.close.bias.data.fill_(10)
    torch.save(gripper.state_dict(), gripper_dir / "model.pt")
    (gripper_dir / "config.json").write_text(json.dumps(gripper_config.__dict__))

    class Base:
        def reset(self, instruction="", seed=None):
            pass

        def get_action(self, obs):
            return np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, -1], np.float32)

    policy = LearnedFrontGraspPolicy(Base(), arm_dir, gripper_dir)
    obs = {
        "agentview_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.zeros(2, np.float32),
    }
    action = policy.get_action(obs)
    assert 0 < action[0] <= arm_config.rate_limit
    assert np.array_equal(action[1:6], Base().get_action(obs)[1:6])
    assert action[6] == 1


def test_arm_only_wrapper_preserves_gripper_coordinate(tmp_path):
    config = ArmHeadConfig()
    model = TemporalArmResidualHead(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.residual.bias.data[0] = 10
    model.approach.bias.data.fill_(10)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))

    class Base:
        def reset(self, instruction="", seed=None):
            pass

        def get_action(self, obs):
            return np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7], np.float32)

    policy = LearnedArmResidualPolicy(Base(), tmp_path)
    obs = {
        "agentview_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.zeros(2, np.float32),
    }
    action = policy.get_action(obs)
    assert 0 < action[0] <= config.rate_limit
    assert action[6] == -0.7

