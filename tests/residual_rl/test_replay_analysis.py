import json

import numpy as np

from experiments.libero_residual_rl.analyze import compare_paired, summarize
from experiments.libero_residual_rl.replay import ReplayBuffer


def test_replay_round_trip():
    replay = ReplayBuffer(10, seed=1)
    for index in range(5):
        replay.add(
            np.zeros((2, 2, 3, 8, 8)),
            np.zeros((2, 15)),
            np.zeros((4, 7)),
            float(index),
            np.ones((2, 2, 3, 8, 8)),
            np.ones((2, 15)),
            index == 4,
        )
    batch = replay.sample(3)
    assert batch.images.shape == (3, 2, 2, 3, 8, 8)
    assert batch.residuals.shape == (3, 4, 7)


def _record(condition, seed, success, level, norm=0.0):
    return {
        "condition": condition,
        "base_task": "drawer",
        "task_name": "variant",
        "axis": "camera",
        "level": level,
        "seed": seed,
        "grasp_success": success,
        "task_success": success,
        "max_contact_force": 1.0,
        "grasp_point_error": 0.01,
        "residual_norm_mean": norm,
    }


def test_paired_statistics_and_summary_are_serializable():
    base = [_record("base", 1, False, 1), _record("base", 2, True, 2)]
    residual = [
        _record("residual", 1, True, 1, 0.1),
        _record("residual", 2, True, 2, 0.2),
    ]
    comparison = compare_paired(base, residual)
    grouped = summarize(residual)

    assert comparison["gsr_delta"] == 0.5
    assert comparison["residual_only_successes"] == 1
    assert grouped["drawer/camera/1"]["gsr"] == 1.0
    json.dumps({"comparison": comparison, "summary": grouped})

