from types import SimpleNamespace

import pytest

from experiments.libero_residual_rl.base_policy import configure_execution_horizon


class _Policy:
    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=50, chunk_size=50)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_configure_execution_horizon_uses_checkpoint_default():
    policy = _Policy()
    assert configure_execution_horizon(policy, None) == 50
    assert policy.config.n_action_steps == 50
    assert policy.reset_calls == 1


def test_configure_execution_horizon_overrides_queue_length():
    policy = _Policy()
    assert configure_execution_horizon(policy, 5) == 5
    assert policy.config.n_action_steps == 5
    assert policy.reset_calls == 1


@pytest.mark.parametrize("value", [0, 51])
def test_configure_execution_horizon_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="execution_horizon"):
        configure_execution_horizon(_Policy(), value)
