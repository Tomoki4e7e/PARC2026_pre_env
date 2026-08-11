#!/usr/bin/env python3
"""衝突するstove初期状態へ、把持後の上方回避waypointを探索して記録する。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import validate_perturbation_replay as replay_tool

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TASK_ID = "put_the_bowl_on_the_stove_light_11"
DEFAULT_ORDINALS = "1,8,13,15,18,19"

# cream cheeseはボウルの+x/+y側。上空で反対側へ迂回し、降下前に戻す。
WAYPOINTS = [
    (-0.2, 0.0, 2), (-0.4, 0.0, 2),
    (0.0, -0.2, 2), (0.0, -0.4, 2),
    (-0.2, -0.2, 2), (-0.4, -0.4, 2),
]


def corrected(actions: list[list[float]], dx: float, dy: float, repeats: int) -> list[list[float]]:
    array = np.asarray(actions, dtype=np.float32)
    close_indices = np.flatnonzero(array[:, 6] > 0.0)
    if not len(close_indices):
        raise RuntimeError("gripper close transition not found")
    close_at = int(close_indices[0])
    outbound_at = max(5, close_at - 24)
    inbound_at = max(outbound_at + 1, close_at - 10)
    outbound = np.asarray([dx, dy, 0, 0, 0, 0, -1], dtype=np.float32)
    inbound = np.asarray([-dx, -dy, 0, 0, 0, 0, -1], dtype=np.float32)
    result = np.concatenate([
        array[:outbound_at], np.repeat(outbound[None], repeats, axis=0),
        array[outbound_at:inbound_at], np.repeat(inbound[None], repeats, axis=0),
        array[inbound_at:],
    ])
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path,
                        default=REPO_ROOT / "results/track1_distribution_audit/tasks.csv")
    parser.add_argument("--ordinals", default=DEFAULT_ORDINALS)
    parser.add_argument("--record-dir", type=Path,
                        default=SCRIPT_DIR / "work/stove_corrective_records")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results/stove_corrective_replay/summary.json")
    args = parser.parse_args()

    with args.tasks_csv.open(newline="", encoding="utf-8") as handle:
        row = next(item for item in csv.DictReader(handle) if item["task_id"] == TASK_ID)
    tasks, episodes, frames = replay_tool.load_dataset_tables()
    task_index = {
        name.replace(" ", "_"): int(item.task_index) for name, item in tasks.iterrows()
    }
    source_index = task_index[row["base_task"]]
    episode_ids = [
        int(ep.episode_index) for _, ep in episodes.iterrows()
        if source_index in [int(v) for v in frames.loc[
            frames.episode_index == int(ep.episode_index), "task_index"
        ].unique()]
    ]
    ordinals = [int(value) for value in args.ordinals.split(",")]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    donor_paths = sorted((SCRIPT_DIR / "work/stove_collision_safe_records").glob("*.npz"))
    if not donor_paths:
        raise RuntimeError("safe stove donor recordings not found")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT / 'compe'}"

    for ordinal in ordinals:
        episode_id = episode_ids[ordinal]
        episode_frames = frames[frames.episode_index == episode_id].sort_values("frame_index")
        original = [np.asarray(action).tolist() for action in episode_frames.action]
        attempts = []
        accepted = None
        candidates = []
        for donor_path in donor_paths:
            with np.load(donor_path, allow_pickle=False) as donor:
                candidates.append((f"donor_{donor_path.stem.split('__ep')[-1]}",
                                   donor["action"].tolist(), {"donor": donor_path.name}))
        for dx, dy, repeats in WAYPOINTS:
            candidates.append((f"xy{dx:+.2f}_{dy:+.2f}_x{repeats}",
                               corrected(original, dx, dy, repeats),
                               {"dx": dx, "dy": dy, "repeats": repeats}))
        for tag, candidate_actions, candidate_meta in candidates:
            payload = args.output.parent / f"payload_ep{ordinal:03d}_{tag}.json"
            worker_output = args.output.parent / f"worker_ep{ordinal:03d}_{tag}.json"
            record_path = args.record_dir / f"{TASK_ID}__ep{ordinal:03d}__{tag}.npz"
            payload.write_text(json.dumps({
                "tasks_csv": str(args.tasks_csv.resolve()), "task_id": TASK_ID,
                "init_ordinal": ordinal, "actions": candidate_actions,
                "record_path": str(record_path),
            }))
            subprocess.run([
                "/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python",
                str(Path(replay_tool.__file__).resolve()),
                "--worker-payload", str(payload), "--worker-output", str(worker_output),
            ], cwd=REPO_ROOT, env=env, check=True)
            outcome = json.loads(worker_output.read_text())
            attempt = {**candidate_meta, **outcome}
            attempts.append(attempt)
            print(f"ordinal={ordinal} {tag}: {outcome}", flush=True)
            if outcome["accepted"]:
                accepted = attempt
                break
        results.append({"ordinal": ordinal, "source_episode": episode_id,
                        "accepted": accepted, "attempts": attempts})

    report = {
        "task_id": TASK_ID, "ordinals": ordinals,
        "n_accepted": sum(item["accepted"] is not None for item in results),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"accepted {report['n_accepted']}/{len(ordinals)}; saved: {args.output}")


if __name__ == "__main__":
    main()
