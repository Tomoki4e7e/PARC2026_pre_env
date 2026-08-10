#!/usr/bin/env python3
"""stage2 と画像 Augmentation 追加学習モデルを同一ゲートで比較する。"""

from __future__ import annotations

from pathlib import Path

import run_phase4_unified_select as evaluator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORK = SCRIPT_DIR / "work"

evaluator.SELECTION_DIR = REPO_ROOT / "results" / "image_augmentation_select"
evaluator.SELECTION_JSON = evaluator.SELECTION_DIR / "selection.json"
evaluator.SELECTED_MERGED = WORK / "smolvla_libero_image_aug_selected_merged"
evaluator.CANDIDATES = {
    "stage2": WORK / "smolvla_libero_plus_track1_stage2_merged",
    "image_augmentation": WORK / "smolvla_libero_plus_image_aug_merged",
}
evaluator.TRAINED_STAGE_HINT = {
    "stage2": "track1_stage2",
    "image_augmentation": "track1_stage2",
}


if __name__ == "__main__":
    evaluator.main()
