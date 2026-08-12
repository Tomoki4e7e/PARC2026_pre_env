import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "submission_template"))

from residual_policy import ResidualActionNetwork, ResidualPolicyConfig


def test_residual_network_has_six_bounded_outputs():
    config = ResidualPolicyConfig()
    model = ResidualActionNetwork(config)
    output = model(torch.randn(3, 15))
    assert tuple(output.shape) == (3, 6)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)


def test_config_round_trip(tmp_path):
    config = ResidualPolicyConfig(
        translation_limit=0.02,
        rotation_limit=0.03,
        output_scale=0.25,
        state_mean=tuple(float(value) for value in np.arange(8)),
    )
    path = tmp_path / "config.json"
    import json

    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    restored = ResidualPolicyConfig.from_json(path)
    assert restored.translation_limit == 0.02
    assert restored.rotation_limit == 0.03
    assert restored.output_scale == 0.25
    assert restored.state_mean == config.state_mean
