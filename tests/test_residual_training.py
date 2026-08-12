from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _ROOT / "examples" / "smolvla_libero_spatial_lora_local"
sys.path.insert(0, str(_EXAMPLES))

from train_residual_policy import load_episodes


def _episode(path: Path, reward: float) -> None:
    np.savez_compressed(
        path,
        **{
            "observation.state": np.zeros((2, 8), dtype=np.float32),
            "base_action": np.zeros((2, 7), dtype=np.float32),
            "action": np.full((2, 7), 0.01, dtype=np.float32),
            "reward": np.asarray([0.0, reward], dtype=np.float32),
        },
    )


def test_reward_weights_favor_high_return_episodes(tmp_path):
    low = tmp_path / "low.npz"
    high = tmp_path / "high.npz"
    _episode(low, -100.0)
    _episode(high, 100.0)
    state, base, target, weights = load_episodes([low, high], temperature=10.0)
    assert state.shape == (4, 8)
    assert base.shape == (4, 7)
    assert target.shape == (4, 6)
    assert weights[2:].mean() > weights[:2].mean()
