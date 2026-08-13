#!/usr/bin/env python3
"""Measure fixed-noise chosen/rejected flow-matching loss gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import config
from smolvla_mypolicy import _load_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=config.RAMEKIN_PREFERENCE_DATASET_ROOT)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    meta = LeRobotDatasetMetadata(config.RAMEKIN_PREFERENCE_DATASET_REPO,
                                  root=args.dataset_root)
    delta = resolve_delta_timestamps(SmolVLAConfig(), meta)
    dataset = LeRobotDataset(config.RAMEKIN_PREFERENCE_DATASET_REPO,
                             root=args.dataset_root, delta_timestamps=delta,
                             return_uint8=True)
    pair_ranges = [
        (int(episode["dataset_from_index"]), int(episode["dataset_to_index"]))
        for episode in (dataset.meta.episodes[index] for index in range(24))
    ]
    candidates = [start + (end - start) // 2 for start, end in pair_ranges]
    indices = candidates[:args.max_samples]
    results = {}
    for model_path in args.models:
        device = "cuda"
        policy = SmolVLAPolicy.from_pretrained(str(model_path)).to(device).eval()
        preprocessor, _ = _load_pre_post_processors(str(model_path), device)
        chosen_total = rejected_total = 0.0
        for index in indices:
            raw = dataset[index]
            rejected = raw.pop("preference.rejected_action")
            raw.pop("preference.weight")
            batch = {}
            for key, value in raw.items():
                if isinstance(value, torch.Tensor):
                    if key.startswith("observation.images") and value.dtype == torch.uint8:
                        value = value.float() / 255.0
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = [value]
            chosen_batch = preprocessor(dict(batch))
            rejected_raw = dict(batch)
            rejected_raw["action"] = rejected.unsqueeze(0)
            rejected_batch = preprocessor(rejected_raw)
            torch.manual_seed(2026 + index)
            torch.cuda.manual_seed_all(2026 + index)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, _ = policy.forward(chosen_batch, reduction="none")
            torch.manual_seed(2026 + index)
            torch.cuda.manual_seed_all(2026 + index)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rejected_loss, _ = policy.forward(rejected_batch, reduction="none")
            chosen_total += float(chosen.item())
            rejected_total += float(rejected_loss.item())
        count = len(indices)
        results[model_path.name] = {
            "model": str(model_path.resolve()), "samples": count,
            "chosen_loss": chosen_total / count,
            "rejected_loss": rejected_total / count,
            "preference_gap": (rejected_total - chosen_total) / count,
        }
        del policy
        torch.cuda.empty_cache()
        print(model_path.name, results[model_path.name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
