import numpy as np
import pytest

from experiments.libero_residual_rl.action_schema import ActionSchema


def test_zero_one_gripper_maps_to_libero_signed():
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = [0.0, 1.0]
    converted = ActionSchema("zero_one").to_libero(actions)
    assert converted[:, 6].tolist() == [1.0, -1.0]


def test_action_conversion_round_trip():
    action = np.array([0.1, -0.2, 0.3, 0, 0, 0, 0.25], dtype=np.float32)
    schema = ActionSchema("zero_one", discrete_gripper=False)
    assert np.allclose(schema.from_libero(schema.to_libero(action)), action)


def test_zero_one_encoding_rejects_out_of_range_values():
    action = np.zeros(7, dtype=np.float32)
    action[6] = -1
    with pytest.raises(ValueError):
        ActionSchema("zero_one", strict=True).to_libero(action)

