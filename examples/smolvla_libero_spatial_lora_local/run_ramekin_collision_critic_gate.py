#!/usr/bin/env python3
"""A/B gate Base versus the ramekin collision critic on identical episodes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
from pathlib import Path

import eval_expanded_generalization as expanded_eval
from run_collision_safe_checkpoint_gate import metrics, run_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TARGETS = {
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_light_41",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path, default=(
        REPO_ROOT / "results/curriculum_checkpoint_select/confirm_seed123_n5/003000/tasks.csv"))
    parser.add_argument("--model-dir", type=Path, default=(
        SCRIPT_DIR / "work/curriculum_checkpoint_models/003000"))
    parser.add_argument("--critic-dir", type=Path, default=(
        SCRIPT_DIR / "work/ramekin_collision_critic"))
    parser.add_argument("--result-root", type=Path, default=(
        REPO_ROOT / "results/ramekin_collision_critic_gate"))
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.tasks_csv.open(encoding="utf-8")))
    task_ids = [row["task_id"] for row in rows if row["task_id"] in TARGETS]
    if set(task_ids) != TARGETS:
        raise RuntimeError(f"target task mismatch: {task_ids}")
    results = {}
    for name, enabled in (("base", False), ("collision_critic", True)):
        output = args.result_root / name
        output.mkdir(parents=True, exist_ok=True)
        proc = expanded_eval.start_server(
            args.port, args.model_dir, None, output / "server.log",
            residual_enabled=False, critic_dir=args.critic_dir, critic_enabled=enabled,
        )
        try:
            result = run_pipeline(
                port=args.port, tasks_csv=args.tasks_csv, task_ids=task_ids,
                output=output / "raw", n_episodes=args.n_episodes, seed=args.seed,
            )
            results[name] = metrics(result)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
        row = results[name]
        print(f"[critic-gate] {name}: success={row['success_rate']:.3f} "
              f"collision={row['collision_rate']:.3f}", flush=True)

    base, critic = results["base"], results["collision_critic"]
    passed = (critic["collision_rate"] < base["collision_rate"]
              and critic["success_rate"] >= base["success_rate"])
    summary = {
        "protocol": {"tasks": task_ids, "n_episodes": args.n_episodes,
                     "seed": args.seed, "same_episode_set": True,
                     "gate": "collision < base and success >= base"},
        "model": str(args.model_dir.resolve()), "critic": str(args.critic_dir.resolve()),
        "results": results, "passed": passed, "selected": "collision_critic" if passed else "base",
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[critic-gate] passed={passed}; selected={summary['selected']}", flush=True)


if __name__ == "__main__":
    main()
