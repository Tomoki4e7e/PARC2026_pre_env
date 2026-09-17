import json
import numpy as np
import torch

from experiments.libero_residual_rl.gripper_head import (
    GripperHeadConfig,
    LearnedGripperResidualPolicy,
    TemporalGripperHead,
)


def test_gripper_head_shapes_for_all_modes():
    for mode, channels in (
        ("vector", 0),
        ("front", 3),
        ("wrist", 3),
        ("two_camera", 6),
    ):
        config = GripperHeadConfig(mode=mode)
        model = TemporalGripperHead(config)
        images = torch.zeros(2, config.history, channels, config.image_size, config.image_size)
        vectors = torch.zeros(2, config.history, 15)
        close, contact, phase = model(images, vectors)
        assert close.shape == (2,)
        assert contact.shape == (2,)
        assert phase.shape == (2, 5)


def test_learned_head_never_changes_arm_coordinates(tmp_path):
    config = GripperHeadConfig(mode="vector", threshold=0.1)
    model = TemporalGripperHead(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.close.bias.data.fill_(10)
    model.contact.bias.data.fill_(10)
    model.phase.bias.data[1] = 10
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))

    class Base:
        def __init__(self):
            self.horizons = []

        def reset(self, instruction="", seed=None):
            pass

        def set_execution_horizon(self, horizon):
            self.horizons.append(horizon)

        def get_action(self, obs):
            return np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1], np.float32)

    base = Base()
    policy = LearnedGripperResidualPolicy(
        base, tmp_path, post_close_execution_horizon=50, trigger_steps=1
    )
    obs = {
        "agentview_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.zeros(2, np.float32),
    }
    action = policy.get_action(obs)
    assert np.array_equal(action[:6], Base().get_action(obs)[:6])
    assert action[6] == 1
    assert base.horizons == [50]
    policy.get_action(obs)
    assert base.horizons == [50]


def test_gripper_latch_switches_to_post_latch_scale_when_closed(tmp_path):
    config = GripperHeadConfig(mode="vector", threshold=0.5)
    model = TemporalGripperHead(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.close.bias.data.fill_(10)
    model.contact.bias.data.fill_(10)
    model.phase.bias.data[2] = 10
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))

    class Base:
        def reset(self, instruction="", seed=None):
            pass

        def get_action(self, obs):
            return np.array([0.2, -0.2, 0.4, 0.1, -0.1, 0.3, -1], np.float32)

    policy = LearnedGripperResidualPolicy(
        Base(),
        tmp_path,
        post_close_arm_scale=0.5,
        post_latch_arm_scale=0.3,
        post_latch_translation_scale=0.7,
        post_latch_vertical_scale=0.1,
        trigger_steps=2,
        closure_damping_steps=10,
    )
    obs = {
        "agentview_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], np.float32),
    }
    assert policy.get_action(obs)[6] == -1
    closing = policy.get_action(obs)
    assert np.allclose(closing[:6], Base().get_action(obs)[:6] * 0.5)
    assert closing[6] == 1
    assert policy.close_latched

    obs["robot0_gripper_qpos"] = np.array([0.02, -0.02], np.float32)
    closed = policy.get_action(obs)
    expected = Base().get_action(obs)
    assert np.allclose(closed[:2], expected[:2] * 0.7)
    assert np.isclose(closed[2], expected[2] * 0.1)
    assert np.allclose(closed[3:6], expected[3:6] * 0.3)
    assert closed[6] == 1

    policy.reset()
    assert not policy.close_latched
    assert policy.trigger_count == 0
    assert policy.get_action(obs)[6] == -1


def test_gripper_trigger_requires_learned_contact(tmp_path):
    config = GripperHeadConfig(mode="vector", threshold=0.5)
    model = TemporalGripperHead(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.close.bias.data.fill_(10)
    model.contact.bias.data.fill_(-10)
    model.phase.bias.data[2] = 10
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))

    class Base:
        def reset(self, instruction="", seed=None):
            pass

        def get_action(self, obs):
            return np.array([0, 0, 0, 0, 0, 0, -1], np.float32)

    policy = LearnedGripperResidualPolicy(Base(), tmp_path, trigger_steps=1)
    obs = {
        "agentview_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], np.float32),
    }
    assert policy.get_action(obs)[6] == -1
    assert not policy.close_latched

