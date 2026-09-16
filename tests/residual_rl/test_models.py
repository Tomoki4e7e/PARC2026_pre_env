import numpy as np
import torch

from experiments.libero_residual_rl.config import ExperimentConfig
from experiments.libero_residual_rl.env import ResidualLiberoEnv
from experiments.libero_residual_rl.models import (
    ConditionalFlowResidualPolicy,
    TwinQCritic,
)


def test_flow_shapes_bounds_and_gradients():
    policy = ConditionalFlowResidualPolicy(horizon=4, flow_steps=2)
    images = torch.rand(2, 3, 2, 3, 32, 32)
    vectors = torch.rand(2, 3, 15)
    residual = policy(images, vectors)

    assert residual.shape == (2, 4, 7)
    limits = policy.limits.reshape(4, 7)
    assert torch.all(residual.abs() <= limits + 1e-6)
    loss = residual.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_flow_matching_and_critic_are_finite():
    policy = ConditionalFlowResidualPolicy(horizon=4, flow_steps=2)
    images = torch.rand(3, 2, 2, 3, 32, 32)
    vectors = torch.rand(3, 2, 15)
    condition = policy.condition(images, vectors)
    target = torch.zeros(3, 4, 7)
    cfm = policy.flow_matching_loss(condition, target)
    critic = TwinQCritic()
    q1, q2 = critic(condition, target)

    assert torch.isfinite(cfm)
    assert q1.shape == q2.shape == (3, 1)


def test_residual_safety_limits_magnitude_and_rate():
    env = ResidualLiberoEnv.__new__(ResidualLiberoEnv)
    env.config = ExperimentConfig()
    env.last_residual = np.zeros(7, dtype=np.float32)
    value = env._safe_residual(np.full(7, 100, dtype=np.float32))

    assert np.all(value <= env.config.limits.rate + 1e-6)
    assert np.isfinite(value).all()

