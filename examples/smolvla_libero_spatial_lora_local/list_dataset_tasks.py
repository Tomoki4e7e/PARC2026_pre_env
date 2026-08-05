#!/usr/bin/env python3
"""libero_plus データセット上の task 名と config の突合確認。"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from utils import (
    ensure_work_dirs,
    normalize_task_name,
    run_hf_with_retry,
    set_hf_env,
    task_name_from_cell,
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
    parser = argparse.ArgumentParser(description="libero_plus task 名の突合")
    parser.add_argument(
        "--suites",
        default="spatial,object,goal",
        help="確認 suite（カンマ区切り）",
    )
    parser.add_argument(
        "--spatial-only",
        action="store_true",
        help="spatial のみ確認",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Phase 3 hold-out object/goal 弱点タスクのみ確認",
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
    sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = run_hf_with_retry(
        lambda: LeRobotDatasetMetadata(
            config.DATASET_REPO,
            revision=config.DATASET_REVISION,
        )
    )

    counts: dict[str, int] = defaultdict(int)
    for task_cell in meta.episodes["tasks"]:
        counts[task_name_from_cell(task_cell)] += 1

    available_norm = {normalize_task_name(name): name for name in counts}
    if args.phase3:
        config.apply_phase3_config()
        suites = list(config.PHASE3_TRAIN_SUITES)
        overrides = config.PHASE3_TASK_EPISODE_OVERRIDES
    else:
        suites = resolve_train_suites(args)
        overrides = config.TASK_EPISODE_OVERRIDES

    missing = 0
    for suite in suites:
        print(f"\n=== {suite.name} ===")
        for task_name in suite.task_names:
            norm = normalize_task_name(task_name)
            if norm in available_norm:
                actual = available_norm[norm]
                print(f"  OK  {task_name!r} -> {counts[actual]} episodes")
            else:
                missing += 1
                print(f"  NG  {task_name!r} NOT FOUND")

    if overrides:
        print("\n=== overrides ===")
        for task_name, episode_count in overrides.items():
            norm = normalize_task_name(task_name)
            if norm in available_norm:
                actual = available_norm[norm]
                print(
                    f"  OK  {task_name!r} -> {episode_count} requested "
                    f"({counts[actual]} available)"
                )
            else:
                missing += 1
                print(f"  NG  {task_name!r} NOT FOUND")

    if missing:
        raise SystemExit(f"{missing} task(s) not found in dataset metadata.")
    print("\nAll configured tasks are present in the dataset.")


if __name__ == "__main__":
    main()
