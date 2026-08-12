#!/usr/bin/env python3
"""Offline sanity gate for a trained residual-policy candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "submission_template"))

from residual_policy import ResidualActionNetwork, ResidualPolicyConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    states, bases, actions = [], [], []
    success_masks, collision_masks = [], []
    paths = sorted(path for root in args.input_dir for path in root.rglob("*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            state = np.asarray(episode["observation.state"], dtype=np.float32)
            base = np.asarray(episode["base_action"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            metadata = json.loads(str(episode["metadata_json"].item()))
        states.append(state)
        bases.append(base)
        actions.append(action)
        success_masks.append(np.full(len(action), bool(metadata["success"])))
        collision_masks.append(np.full(len(action), bool(metadata["collided"])))
    state = np.concatenate(states)
    base = np.concatenate(bases)
    action = np.concatenate(actions)
    success = np.concatenate(success_masks)
    collision = np.concatenate(collision_masks)

    config = ResidualPolicyConfig.from_json(args.model_dir / "config.json")
    model = ResidualActionNetwork(config)
    model.load_state_dict(load_file(str(args.model_dir / "model.safetensors")))
    model.eval()
    state_mean = np.asarray(config.state_mean, dtype=np.float32)
    state_std = np.asarray(config.state_std, dtype=np.float32)
    action_mean = np.asarray(config.base_action_mean, dtype=np.float32)
    action_std = np.asarray(config.base_action_std, dtype=np.float32)
    inputs = np.concatenate(
        [(state - state_mean) / state_std, (base - action_mean) / action_std], axis=1
    )
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 4096):
            predictions.append(model(torch.from_numpy(inputs[start : start + 4096])).numpy())
    limits = np.asarray(
        [config.translation_limit] * 3 + [config.rotation_limit] * 3,
        dtype=np.float32,
    )
    correction = np.concatenate(predictions) * limits * config.output_scale
    residual_action = np.clip(base[:, :6] + correction, -1.0, 1.0)
    base_error = np.square(base[:, :6] - action[:, :6]).mean(axis=1)
    residual_error = np.square(residual_action - action[:, :6]).mean(axis=1)

    def subgroup(mask: np.ndarray) -> dict[str, float | int | None]:
        if not np.any(mask):
            return {"transitions": 0, "base_mse": None, "residual_mse": None}
        return {
            "transitions": int(mask.sum()),
            "base_mse": float(base_error[mask].mean()),
            "residual_mse": float(residual_error[mask].mean()),
        }

    absolute = np.abs(correction)
    report = {
        "episodes": len(paths),
        "transitions": len(state),
        "base_mse": float(base_error.mean()),
        "residual_mse": float(residual_error.mean()),
        "relative_mse_change": float(residual_error.mean() / base_error.mean() - 1.0),
        "correction_abs_percentiles": {
            str(q): np.percentile(absolute, q, axis=0).tolist()
            for q in (50, 90, 99, 100)
        },
        "limit_saturation_rate": (absolute >= limits * 0.99).mean(axis=0).tolist(),
        "within_configured_limits": bool(np.all(absolute <= limits + 1.0e-7)),
        "subgroups": {
            "success": subgroup(success),
            "failure": subgroup(~success),
            "collision": subgroup(collision),
            "non_collision": subgroup(~collision),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
