#!/usr/bin/env python3
"""Compare augmentation checkpoints and stage2-interpolated candidates."""

from __future__ import annotations

from pathlib import Path

import run_phase4_unified_select as evaluator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORK = SCRIPT_DIR / "work"
EXPERIMENT = WORK / "augmentation_checkpoint_models"

evaluator.SELECTION_DIR = REPO_ROOT / "results" / "augmentation_checkpoint_select"
evaluator.SELECTION_JSON = evaluator.SELECTION_DIR / "selection.json"
evaluator.SELECTED_MERGED = WORK / "smolvla_libero_aug_checkpoint_selected_merged"
evaluator.CANDIDATES = {
    "stage2": WORK / "smolvla_libero_plus_track1_stage2_merged",
    "aug_1000": EXPERIMENT / "aug_1000",
    "aug_2000": EXPERIMENT / "aug_2000",
    "aug_3000": EXPERIMENT / "aug_3000",
    "aug_4000": WORK / "smolvla_libero_plus_image_aug_merged",
    "blend_4000_a025": EXPERIMENT / "blend_4000_a025",
    "blend_4000_a050": EXPERIMENT / "blend_4000_a050",
    "blend_4000_a075": EXPERIMENT / "blend_4000_a075",
}
evaluator.TRAINED_STAGE_HINT = {name: "track1_stage2" for name in evaluator.CANDIDATES}


if __name__ == "__main__":
    evaluator.main()
