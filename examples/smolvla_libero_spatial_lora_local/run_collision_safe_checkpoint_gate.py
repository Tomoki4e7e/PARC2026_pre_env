#!/usr/bin/env python3
"""Evaluate Base and 500/1000/1500 safety checkpoints on identical episodes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
from pathlib import Path

import eval_expanded_generalization as expanded_eval

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORK = SCRIPT_DIR / "work"
DEFAULT_TASKS_CSV = (
    REPO_ROOT / "results/curriculum_checkpoint_select/confirm_seed123_n5/003000/tasks.csv"
)
RESULT_ROOT = REPO_ROOT / "results/curriculum_collision_safe_checkpoint_gate"
TARGETS = {
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_light_41",
    "open_the_top_drawer_and_put_the_bowl_inside_light_41",
}


def run_pipeline(*, port: int, tasks_csv: Path, task_ids: list[str], output: Path,
                 n_episodes: int, seed: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PARC_ENABLE_PROXY_SUITE": "1",
        "PARC_PROXY_TASKS_CSV": str(tasks_csv.resolve()),
        "PYTHONPATH": ":".join([
            str(REPO_ROOT / "LIBERO-plus"), str(REPO_ROOT), str(REPO_ROOT / "compe"),
            env.get("PYTHONPATH", ""),
        ]),
        "LIBERO_ROOT": str(REPO_ROOT / "LIBERO-plus"),
        "LD_LIBRARY_PATH": ":".join(filter(None, [
            "/home/cm-pc/miniconda3/envs/parc2026/lib", env.get("LD_LIBRARY_PATH", "")
        ])),
    })
    command = [
        "/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python", "-m", "pipeline",
        "--server-url", f"http://127.0.0.1:{port}", "--track", "track1",
        "--benchmark", "libero_proxy", "--n-episodes", str(n_episodes),
        "--max-steps", "600", "--seed", str(seed), "--output-dir", str(output),
        "--tasks", *task_ids,
    ]
    with (output / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return sorted(output.glob("server_*.json"))[-1]


def metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = [episode for task in payload["tracks"][0]["tasks"]
                for episode in task["episodes"]]
    return {
        "episodes": len(episodes),
        "successes": sum(bool(x["success"]) for x in episodes),
        "collisions": sum(bool(x.get("collided")) for x in episodes),
        "success_rate": sum(bool(x["success"]) for x in episodes) / len(episodes),
        "collision_rate": sum(bool(x.get("collided")) for x in episodes) / len(episodes),
        "result": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--model-root", type=Path,
                        default=WORK / "curriculum_collision_safe_checkpoint_models")
    parser.add_argument("--base-model", type=Path,
                        default=WORK / "curriculum_checkpoint_models/003000")
    parser.add_argument(
        "--target-ids", default=",".join(sorted(TARGETS)),
        help="comma-separated task IDs from tasks.csv",
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--port", type=int, default=8071)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.tasks_csv.open(encoding="utf-8")))
    targets = {value.strip() for value in args.target_ids.split(",") if value.strip()}
    task_ids = [row["task_id"] for row in rows if row["task_id"] in targets]
    if set(task_ids) != targets:
        raise RuntimeError(f"target task mismatch: found={task_ids}")
    models = {
        "base_003000": args.base_model,
        **{f"step_{step:06d}": args.model_root / f"{step:06d}"
           for step in (500, 1000, 1500)},
    }
    results = {}
    for name, model in models.items():
        (args.result_root / name).mkdir(parents=True, exist_ok=True)
        proc = expanded_eval.start_server(
            args.port, model, None, args.result_root / name / "server.log",
            residual_enabled=False,
        )
        try:
            result_path = run_pipeline(
                port=args.port, tasks_csv=args.tasks_csv, task_ids=task_ids,
                output=args.result_root / name / "raw", n_episodes=args.n_episodes,
                seed=args.seed,
            )
            results[name] = {"model": str(model.resolve()), **metrics(result_path)}
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
        row = results[name]
        print(f"[collision-safe-gate] {name}: success={row['success_rate']:.3f} "
              f"collision={row['collision_rate']:.3f}", flush=True)

    base = results["base_003000"]
    passed = [name for name, row in results.items() if name != "base_003000"
              and row["success_rate"] >= base["success_rate"]
              and row["collision_rate"] < base["collision_rate"]]
    selected = min(
        passed,
        key=lambda name: (
            results[name]["collision_rate"], -results[name]["success_rate"],
            int(name.split("_")[-1]),
        ),
        default="base_003000",
    )
    summary = {
        "protocol": {"tasks": task_ids, "n_episodes": args.n_episodes, "seed": args.seed,
                     "success_gate": ">= base", "collision_gate": "< base"},
        "results": results, "passed": passed, "selected": selected,
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[collision-safe-gate] passed={passed}; selected={selected}")


if __name__ == "__main__":
    main()
