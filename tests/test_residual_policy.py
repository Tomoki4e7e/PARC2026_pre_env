import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "submission_template"))

from residual_policy import (
    GatedResidualPolicyConfig,
    ResidualActionNetwork,
    ResidualPolicyConfig,
    VisualLanguageGatedResidualNetwork,
    language_features,
    visual_features,
)


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


def test_visual_language_gated_network_and_features():
    config = GatedResidualPolicyConfig()
    model = VisualLanguageGatedResidualNetwork(config)
    inputs = torch.randn(3, 8 + 7 + config.visual_dim + config.language_dim)
    correction, gate = model(inputs)
    assert tuple(correction.shape) == (3, 6)
    assert tuple(gate.shape) == (3, 1)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    assert visual_features(image, image).shape == (36,)
    assert language_features("put the bowl on the stove").shape == (64,)
    assert np.array_equal(
        language_features("put the bowl on the stove"),
        language_features("put the bowl on the stove"),
    )


def test_gated_config_round_trip_preserves_instruction_allowlist(tmp_path):
    config = GatedResidualPolicyConfig(allowed_instructions=("put the bowl on the stove",))
    path = tmp_path / "config.json"
    import json
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert GatedResidualPolicyConfig.from_json(path).allowed_instructions == config.allowed_instructions
