#!/usr/bin/env python3
"""学習用データセットを事前ダウンロードする（multi-suite 対応）。

train.py 実行前に単独で実行してもよい。ネットワークエラー時は自動リトライする。

使い方:
    source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
    conda activate ~/.local/smolvla-conda-env
    python examples/smolvla_libero_spatial_lora_local/download_dataset.py
    python examples/smolvla_libero_spatial_lora_local/download_dataset.py --spatial-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from utils import (
    ensure_work_dirs,
    prefetch_dataset,
    print_episode_selection_summary,
    select_training_episodes,
    set_hf_env,
)


def resolve_train_suites(args: argparse.Namespace) -> list[config.TrainSuiteSpec]:
    if args.spatial_only:
        return [
            config.TrainSuiteSpec(
                "spatial",
                config.SPATIAL_TASK_NAMES,
                config.TRAIN_EPISODES_PER_TASK,
            )
        ]
    enabled = {name.strip() for name in args.suites.split(",") if name.strip()}
    suites = [suite for suite in config.TRAIN_SUITES if suite.name in enabled]
    if not suites:
        available = ", ".join(suite.name for suite in config.TRAIN_SUITES)
        raise ValueError(f"No suites matched {args.suites!r}. Available: {available}")
    return list(suites)


def main() -> None:
    parser = argparse.ArgumentParser(description="multi-suite 学習データの事前取得")
    parser.add_argument(
        "--suites",
        default="spatial,object,goal",
        help="取得 suite（カンマ区切り）",
    )
    parser.add_argument(
        "--spatial-only",
        action="store_true",
        help="spatial のみ取得",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Phase 3 hold-out object/goal 弱点タスクのみ取得",
    )
    args = parser.parse_args()

    if args.phase3 and args.spatial_only:
        parser.error("--phase3 と --spatial-only は同時に指定できません")

    ensure_work_dirs(config.WORK_DIR, config.HF_HOME, config.LEROBOT_CACHE)
    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)

    lerobot_src = config.LEROBOT_DIR / "src"
    if not lerobot_src.is_dir():
        raise RuntimeError(
            "LeRobot が未セットアップです。setup_lerobot.sh を実行してください。"
        )

    if args.phase3:
        config.apply_phase3_config()
        suites = list(config.PHASE3_TRAIN_SUITES)
        task_episode_overrides = config.PHASE3_TASK_EPISODE_OVERRIDES
    else:
        suites = resolve_train_suites(args)
        task_episode_overrides = config.TASK_EPISODE_OVERRIDES

    episode_indices, selected_by_task = select_training_episodes(
        config.DATASET_REPO,
        config.DATASET_REVISION,
        suites,
        task_episode_overrides=task_episode_overrides,
    )
    print(f"Suites: {[suite.name for suite in suites]}")
    print_episode_selection_summary(selected_by_task, suites)
    print("HF_HUB_DISABLE_XET=1 (通常 HTTP ダウンロードを使用)")
    prefetch_dataset(
        config.DATASET_REPO,
        config.DATASET_REVISION,
        episode_indices,
        lerobot_src,
    )
    print("Done.")


if __name__ == "__main__":
    main()
