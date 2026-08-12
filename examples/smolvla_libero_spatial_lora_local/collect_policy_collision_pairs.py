#!/usr/bin/env python3
"""Find safe replay counterparts for collisions produced by the deployed policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKER = SCRIPT_DIR / "validate_perturbation_replay.py"
PYTHON = Path("/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python")


def _candidate_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[-60:]


def _run(payload: Path, output: Path, env: dict[str, str]) -> dict:
    if not output.is_file():
        subprocess.run(
            [str(PYTHON), str(WORKER), "--worker-payload", str(payload),
             "--worker-output", str(output)],
            cwd=REPO_ROOT, env=env, check=True,
        )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rollout-dir", type=Path, required=True)
    parser.add_argument("--safe-actions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-safe-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks-csv", type=Path,
        default=REPO_ROOT / "results/track1_distribution_audit/tasks.csv",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required")

    collisions = []
    for path in sorted(args.policy_rollout_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
            if bool(metadata["collided"]):
                collisions.append((path, metadata))
    candidates = sorted(args.safe_actions_dir.glob("*.npz"))
    if not collisions or not candidates:
        raise RuntimeError("collision rollouts and safe action candidates are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for collision_index, (collision_path, metadata) in enumerate(collisions):
        ordinal = int(metadata["collection_context"]["init_state_ordinal"])
        for candidate_index, source in enumerate(candidates):
            with np.load(source, allow_pickle=False) as episode:
                actions = np.asarray(episode["action"], dtype=np.float32).tolist()
            index = len(jobs)
            record_dir = args.output_dir / "records" / (
                f"init{ordinal:03d}__candidate{candidate_index:02d}_{_candidate_id(source)}"
            )
            payload = {
                "tasks_csv": str(args.tasks_csv.resolve()),
                "task_id": metadata["task_name"],
                "init_ordinal": ordinal,
                "actions": actions,
                "record_path": None,
                "rl_record_dir": str(record_dir),
                "visual_domain": None,
                "camera_size": 256,
            }
            payload_path = args.output_dir / f"worker_payload_{index:04d}.json"
            output_path = args.output_dir / f"worker_output_{index:04d}.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            jobs.append({
                "index": index, "ordinal": ordinal, "collision_path": collision_path,
                "source": source, "payload": payload_path, "output": output_path,
            })

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT / 'compe'}"
    )
    outcomes: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run, job["payload"], job["output"], env): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            outcomes[job["index"]] = future.result()
            print(
                f"[safe-search] {completed}/{len(jobs)} init={job['ordinal']} "
                f"source={job['source'].name} accepted={outcomes[job['index']]['accepted']} "
                f"collision={outcomes[job['index']]['collided']}", flush=True,
            )

    args.selected_safe_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for collision_path, metadata in collisions:
        ordinal = int(metadata["collection_context"]["init_state_ordinal"])
        matching = [job for job in jobs if job["ordinal"] == ordinal]
        valid = [
            job for job in matching
            if outcomes[job["index"]]["accepted"] and not outcomes[job["index"]]["collided"]
        ]
        if not valid:
            pairs.append({"init_state_ordinal": ordinal, "collision": str(collision_path),
                          "safe": None, "valid": False})
            continue
        chosen = valid[0]
        safe_record = Path(outcomes[chosen["index"]]["rl_record_path"])
        destination = args.selected_safe_dir / f"init{ordinal:03d}__safe.npz"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite selected safe record: {destination}")
        shutil.copy2(safe_record, destination)
        pairs.append({
            "init_state_ordinal": ordinal, "collision": str(collision_path),
            "safe": str(destination), "safe_action_source": str(chosen["source"]),
            "valid": True, "n_valid_candidates": len(valid),
        })
    summary = {
        "protocol": "deployed_policy_collision_same_init_safe_counterpart",
        "fixed_seed42_eval_used_for_training": False,
        "n_collisions": len(collisions),
        "n_valid_pairs": sum(item["valid"] for item in pairs),
        "pairs": pairs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
