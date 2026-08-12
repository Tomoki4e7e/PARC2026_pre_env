#!/usr/bin/env python3
"""Replay safe trajectories at the exact init and visual domain of collisions."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKER = SCRIPT_DIR / "validate_perturbation_replay.py"
PYTHON = Path("/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python")


def _ordinal(path: Path) -> int:
    match = re.search(r"__ep(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot determine init ordinal: {path}")
    return int(match.group(1))


def _safe_sources(directory: Path) -> dict[int, Path]:
    grouped: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*.npz")):
        grouped.setdefault(_ordinal(path), []).append(path)
    # Prefer validated donor trajectories over exploratory xy variants.
    return {
        ordinal: sorted(paths, key=lambda path: ("__donor_" not in path.name, path.name))[0]
        for ordinal, paths in grouped.items()
    }


def _run(payload_path: Path, output_path: Path, env: dict[str, str]) -> dict:
    if not output_path.is_file():
        subprocess.run(
            [str(PYTHON), str(WORKER), "--worker-payload", str(payload_path),
             "--worker-output", str(output_path)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unsafe-control-dir", type=Path,
        default=REPO_ROOT / "results/rl_additional/collision_unsafe_visual",
    )
    parser.add_argument(
        "--safe-actions-dir", type=Path,
        default=SCRIPT_DIR / "work/stove_corrective_records",
    )
    parser.add_argument(
        "--rl-record-dir", type=Path,
        default=SCRIPT_DIR / "work/rl_collision_paired_safe_visual",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results/rl_collision_pairs",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("recording requires --confirm-local-generated-data")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    sources = _safe_sources(args.safe_actions_dir)
    unsafe_payloads = sorted(args.unsafe_control_dir.glob("worker_payload_*.json"))
    if not unsafe_payloads:
        raise FileNotFoundError(f"unsafe control payloads not found: {args.unsafe_control_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, unsafe_path in enumerate(unsafe_payloads):
        unsafe = json.loads(unsafe_path.read_text(encoding="utf-8"))
        ordinal = int(unsafe["init_ordinal"])
        if ordinal not in sources:
            raise ValueError(f"safe action source missing for init ordinal {ordinal}")
        source = sources[ordinal]
        with np.load(source, allow_pickle=False) as episode:
            actions = np.asarray(episode["action"], dtype=np.float32).tolist()
        visual = unsafe.get("visual_domain") or {}
        visual_seed = int(visual.get("seed", -1))
        record_dir = args.rl_record_dir / (
            f"{unsafe['task_id']}__ep{ordinal:03d}__paired_vis{index:02d}_seed{visual_seed}"
        )
        payload = {
            "tasks_csv": unsafe["tasks_csv"],
            "task_id": unsafe["task_id"],
            "init_ordinal": ordinal,
            "actions": actions,
            "record_path": None,
            "rl_record_dir": str(record_dir),
            "visual_domain": unsafe.get("visual_domain"),
            "camera_size": int(unsafe.get("camera_size", 128)),
        }
        payload_path = args.output_dir / f"worker_payload_{index:04d}.json"
        output_path = args.output_dir / f"worker_output_{index:04d}.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        jobs.append((index, unsafe_path, payload_path, output_path, source))

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT / 'compe'}"
    )
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run, payload, output, env): (index, unsafe, source)
            for index, unsafe, payload, output, source in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, unsafe_path, source = futures[future]
            safe = future.result()
            unsafe_output = args.unsafe_control_dir / unsafe_path.name.replace(
                "worker_payload_", "worker_output_"
            )
            unsafe = json.loads(unsafe_output.read_text(encoding="utf-8"))
            results[index] = {
                "pair_index": index,
                "init_ordinal": json.loads(unsafe_path.read_text(encoding="utf-8"))["init_ordinal"],
                "safe_action_source": str(source),
                "unsafe": unsafe,
                "safe": safe,
            }
            print(
                f"[pair] {completed}/{len(jobs)} init={results[index]['init_ordinal']} "
                f"unsafe_collision={unsafe['collided']} safe_collision={safe['collided']} "
                f"safe_success={safe['accepted']}",
                flush=True,
            )

    ordered = [results[index] for index in sorted(results)]
    valid = [
        item for item in ordered
        if item["unsafe"]["collided"] and not item["safe"]["collided"]
        and item["safe"]["accepted"]
    ]
    summary = {
        "protocol": "same_init_same_visual_collision_safe_pairs",
        "n_pairs": len(ordered),
        "n_valid_pairs": len(valid),
        "valid_pair_rate": len(valid) / len(ordered),
        "pairs": ordered,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {summary_path} ({len(valid)}/{len(ordered)} valid pairs)")


if __name__ == "__main__":
    main()
