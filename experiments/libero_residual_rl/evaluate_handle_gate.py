from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .base_policy import RemoteBasePolicy
from .config import ExperimentConfig
from .evaluate import evaluate_policy
from .task_catalog import MIDDLE_DRAWER, TOP_DRAWER, TaskCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict GSR-only two-drawer gate")
    parser.add_argument("--server-url", default="http://127.0.0.1:8010")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--task",
        choices=(MIDDLE_DRAWER, TOP_DRAWER),
        action="append",
        help="limit evaluation to one or more drawer tasks",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        action="append",
        help="evaluate exact environment seeds; requires exactly one --task",
    )
    args = parser.parse_args()
    catalog = TaskCatalog()
    specs = []
    tasks = tuple(args.task or (MIDDLE_DRAWER, TOP_DRAWER))
    if args.episode_seed:
        if len(tasks) != 1:
            parser.error("--episode-seed requires exactly one --task")
        template = catalog.build(
            base_tasks=tasks,
            axes=("layout",),
            levels=(0,),
            episodes_per_condition=1,
            seed=args.seed,
        )[0]
        specs = [replace(template, seed=seed) for seed in args.episode_seed]
    else:
        for task in tasks:
            specs.extend(
                catalog.build(
                    base_tasks=(task,),
                    axes=("layout",),
                    levels=(0,),
                    episodes_per_condition=args.episodes,
                    seed=args.seed,
                )
            )
    rows = evaluate_policy(
        specs,
        RemoteBasePolicy(args.server_url),
        ExperimentConfig(image_size=128, max_episode_steps=600),
        output=args.output,
        condition_name=args.condition,
        terminate_on_grasp=True,
        report_task_success=False,
    )
    summary = {}
    for task in tasks:
        task_rows = [row for row in rows if row["base_task"] == task]
        summary[task] = {
            "gsr_count": sum(row["grasp_success"] for row in task_rows),
            "episodes": len(task_rows),
            "failure_modes": [row["failure_mode"] for row in task_rows],
            "max_inference_sec": max(
                row["base_inference_sec_max"] for row in task_rows
            ),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
