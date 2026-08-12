from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.rl_data import RewardWeights, save_rl_episode, shaped_rewards


def _obs(value: int) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 5, 3), value + 1, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([value * 0.01, 0.0, 0.2], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02], dtype=np.float32),
    }


def test_collision_penalty_is_charged_once():
    reward, components = shaped_rewards(
        env_rewards=np.zeros(4, dtype=np.float32),
        actions=np.zeros((4, 7), dtype=np.float32),
        next_eef_positions=np.zeros((4, 3), dtype=np.float32),
        collision_flags=np.asarray([False, True, True, True]),
        success=False,
        weights=RewardWeights(step_penalty=0.0),
    )
    assert reward.tolist() == pytest.approx([0.0, -100.0, 0.0, 0.0])
    assert components["collision"] == -100.0


def test_save_complete_transition_schema(tmp_path):
    observations = [_obs(0), _obs(1)]
    next_observations = [_obs(1), _obs(2)]
    path = save_rl_episode(
        tmp_path,
        task_name="task a",
        instruction="do task a",
        benchmark="local_test",
        episode_id=0,
        seed=42,
        observations=observations,
        next_observations=next_observations,
        actions=[np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
        env_rewards=[0.0, 1.0],
        terminated=[False, True],
        truncated=[False, False],
        collision_flags=[False, False],
        success=True,
        collision_threshold_m=0.001,
        base_actions=[np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)],
    )
    with np.load(path, allow_pickle=False) as episode:
        assert episode["observation.front"].shape == (2, 4, 5, 3)
        assert episode["observation.state"].shape == (2, 8)
        assert episode["next_observation.state"].shape == (2, 8)
        assert episode["base_action"].shape == (2, 7)
        assert episode["terminated"].tolist() == [False, True]
        metadata = json.loads(str(episode["metadata_json"].item()))
    assert metadata["source"] == "local_simulator_rollout"
    assert metadata["success"] is True
    assert metadata["reward_components"]["success"] == 100.0


def test_refuses_to_overwrite_episode(tmp_path):
    kwargs = dict(
        task_name="same", instruction="same", benchmark="test", episode_id=0, seed=1,
        observations=[_obs(0)], next_observations=[_obs(1)],
        actions=[np.zeros(7, dtype=np.float32)], env_rewards=[0.0],
        terminated=[False], truncated=[True], collision_flags=[False],
        success=False, collision_threshold_m=0.001,
    )
    save_rl_episode(tmp_path, **kwargs)
    with pytest.raises(FileExistsError):
        save_rl_episode(tmp_path, **kwargs)
