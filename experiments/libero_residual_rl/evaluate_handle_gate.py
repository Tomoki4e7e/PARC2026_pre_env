from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    catalog = TaskCatalog()
    specs = []
    for task in (MIDDLE_DRAWER, TOP_DRAWER):
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
    for task in (MIDDLE_DRAWER, TOP_DRAWER):
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
