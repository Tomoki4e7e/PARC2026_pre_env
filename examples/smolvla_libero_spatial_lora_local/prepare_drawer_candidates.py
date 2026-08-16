#!/usr/bin/env python3
"""Merge drawer LoRA checkpoints and create stage2 delta interpolations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="500,1000,1500,2000")
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--output-root", type=Path,
                        default=config.WORK_DIR / "drawer_checkpoint_models_v3_033443")
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for step in steps:
        merged = args.output_root / f"ckpt_{step}"
        subprocess.run([
            sys.executable, str(SCRIPT_DIR / "merge_lora.py"), "--drawer-targeted",
            "--steps", str(step), "--output-dir", str(merged),
        ], check=True)
        for alpha in alphas:
            tag = f"{alpha:.2f}".replace(".", "")
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "interpolate_models.py"),
                "--base", str(config.DRAWER_BASE_MODEL_DIR),
                "--candidate", str(merged),
                "--output-dir", str(args.output_root / f"ckpt_{step}_a{tag}"),
                "--alpha", str(alpha),
            ], check=True)


if __name__ == "__main__":
    main()
