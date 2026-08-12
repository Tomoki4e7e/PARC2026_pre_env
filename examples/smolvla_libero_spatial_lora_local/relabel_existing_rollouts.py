#!/usr/bin/env python3
"""Replay existing local NPZ actions and emit complete RL transition records.

This is for independently generated simulator trajectories only.  Do not pass
organizer evaluation observations or results to this script.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REPLAY_TOOL = SCRIPT_DIR / "validate_perturbation_replay.py"
DEFAULT_TASKS = REPO_ROOT / "results/track1_distribution_audit/tasks.csv"
_ORDINAL = re.compile(r"__ep(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, action="append", required=True,
        help="既存NPZディレクトリ（複数回指定可）",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--python", default="/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument(
        "--visual-domain-mode", choices=["none", "bright", "dark", "mixed"],
        default="none",
    )
    parser.add_argument("--visual-seed", type=int, default=2026)
    parser.add_argument(
        "--confirm-local-generated-data",
        action="store_true",
        help="入力が運営評価観測ではなくローカル生成データであることを確認する",
    )
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required by competition data rules")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_new_jobs is not None and args.max_new_jobs < 1:
        parser.error("--max-new-jobs must be >= 1")

    recordings = sorted(
        (root, path)
        for root in args.input_dir
        for path in root.glob("*.npz")
    )
    if not recordings:
        raise RuntimeError(f"no NPZ trajectories found: {args.input_dir}")
    with args.tasks_csv.open(newline="", encoding="utf-8") as handle:
        known_tasks = {row["task_id"] for row in csv.DictReader(handle)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT / 'compe'}"
    )
    jobs = []
    for index, (source_root, recording) in enumerate(recordings):
        match = _ORDINAL.search(recording.stem)
        if match is None:
            print(f"skip (no __epNNN ordinal): {recording.name}", flush=True)
            continue
        with np.load(recording, allow_pickle=False) as episode:
            task_id = str(episode["task_id"].item())
            actions = np.asarray(episode["action"], dtype=np.float32).tolist()
        if task_id not in known_tasks:
            print(f"skip (task not present in tasks CSV): {recording.name}", flush=True)
            continue
        ordinal = int(match.group(1))
        item_dir = args.output_dir / source_root.name / recording.stem
        payload = {
            "tasks_csv": str(args.tasks_csv.resolve()),
            "task_id": task_id,
            "init_ordinal": ordinal,
            "actions": actions,
            "record_path": None,
            "rl_record_dir": str(item_dir.resolve()),
            "visual_domain": (
                {"mode": args.visual_domain_mode, "seed": args.visual_seed + index}
                if args.visual_domain_mode != "none" else None
            ),
        }
        payload_path = args.output_dir / f"payload_{index:04d}.json"
        output_path = args.output_dir / f"result_{index:04d}.json"
        import json

        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        jobs.append((recording.name, payload_path, output_path))

    def run_job(job):
        name, payload_path, output_path = job
        if not output_path.is_file():
            subprocess.run(
                [args.python, str(REPLAY_TOOL), "--worker-payload", str(payload_path),
                 "--worker-output", str(output_path)],
                cwd=REPO_ROOT,
                env=worker_env,
                check=True,
            )
        return name

    pending_jobs = []
    for job in jobs:
        _, payload_path, output_path = job
        from validate_perturbation_replay import _recover_worker_output
        if not _recover_worker_output(payload_path, output_path):
            pending_jobs.append(job)
    if args.max_new_jobs is not None:
        pending_jobs = pending_jobs[:args.max_new_jobs]

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(run_job, job) for job in pending_jobs]
        for future in as_completed(pending):
            name = future.result()
            completed += 1
            print(f"relabeled {completed}/{len(pending_jobs)}: {name}", flush=True)
    total_completed = sum(output_path.is_file() for _, _, output_path in jobs)
    print(f"progress: {total_completed}/{len(jobs)} episodes -> {args.output_dir}")


if __name__ == "__main__":
    main()
