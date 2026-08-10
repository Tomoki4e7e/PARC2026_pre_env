#!/usr/bin/env python3
"""Two compatible safetensors models are linearly interpolated on CPU."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROCESSOR_PATTERNS = (
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
)


def load_tensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for key in source.keys():
            tensors[key] = source.get_tensor(key)
    return tensors, metadata


def interpolate(base_dir: Path, candidate_dir: Path, output_dir: Path, alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base, metadata = load_tensors(base_dir / "model.safetensors")
    candidate, candidate_metadata = load_tensors(candidate_dir / "model.safetensors")
    if base.keys() != candidate.keys():
        missing = sorted(base.keys() - candidate.keys())[:5]
        extra = sorted(candidate.keys() - base.keys())[:5]
        raise RuntimeError(f"model keys differ: missing={missing}, extra={extra}")
    if metadata != candidate_metadata:
        raise RuntimeError("safetensors metadata differs")

    blended: dict[str, torch.Tensor] = {}
    for key, base_tensor in base.items():
        candidate_tensor = candidate[key]
        if base_tensor.shape != candidate_tensor.shape or base_tensor.dtype != candidate_tensor.dtype:
            raise RuntimeError(f"incompatible tensor {key}")
        if base_tensor.is_floating_point():
            blended[key] = torch.lerp(base_tensor.float(), candidate_tensor.float(), alpha).to(base_tensor.dtype)
        else:
            if not torch.equal(base_tensor, candidate_tensor):
                raise RuntimeError(f"non-floating tensor differs: {key}")
            blended[key] = base_tensor

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(blended, output_dir / "model.safetensors", metadata=metadata)
    for pattern in PROCESSOR_PATTERNS:
        for source in base_dir.glob(pattern):
            shutil.copy2(source, output_dir / source.name)
    manifest = {
        "base": str(base_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "alpha": alpha,
    }
    (output_dir / "interpolation.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    interpolate(args.base, args.candidate, args.output_dir, args.alpha)
    print(f"Interpolated model ready: {args.output_dir} (alpha={args.alpha})")


if __name__ == "__main__":
    main()
