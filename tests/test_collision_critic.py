from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "submission_template"))

from collision_critic import (CollisionCriticConfig, CollisionCriticNetwork,
                              stove_geometry_features)


def test_collision_critic_network_has_one_logit_per_candidate():
    model = CollisionCriticNetwork(CollisionCriticConfig())
    assert tuple(model(torch.randn(4, 23)).shape) == (4,)


def test_collision_critic_config_round_trip(tmp_path):
    config = CollisionCriticConfig(threshold=0.8, risk_margin=0.1, horizon=15)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    restored = CollisionCriticConfig.from_json(path)
    assert restored.threshold == 0.8
    assert restored.risk_margin == 0.1
    assert restored.horizon == 15
    assert len(restored.input_mean) == 23


def test_invalid_image_returns_missing_visual_features():
    features = stove_geometry_features(np.zeros((10, 10), dtype=np.uint8))
    assert features.shape == (8,)
    assert np.array_equal(features, np.zeros(8, dtype=np.float32))
