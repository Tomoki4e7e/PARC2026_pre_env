#!/usr/bin/env python3
"""Build paired collision-safety correction data with representative replay.

Only successful, collision-free trajectories that carry an auditable
``corrective_pair.same_initial_state`` marker are eligible for the corrective
bucket.  Unsafe trajectories are retained outside this BC dataset as the
paired audit source; training teaches the safe action for the exact state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from build_curriculum_datasets import (
    append_episode,
    create_output,
    episode_map,
    stratified_replay,
)


def unsafe_index(root: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
        ordinal = int(metadata.get("collection_context", {}).get(
            "init_state_ordinal", metadata.get("episode_id", -1)))
        if metadata.get("collided"):
            result[(str(metadata["task_name"]), ordinal)] = path
    return result


def load_safe_paths(root: Path, unsafe: dict[tuple[str, int], Path]) -> list[tuple[Path, dict]]:
    accepted: list[tuple[Path, dict]] = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
        pair = metadata.get("corrective_pair", {})
        key = (str(pair.get("base_task_id")), int(pair.get("base_episode_id", -1)))
        if (metadata.get("success") and not metadata.get("collided")
                and pair.get("same_initial_state")
                and pair.get("base_collided") and key in unsafe):
            metadata["unsafe_pair_source"] = str(unsafe[key].resolve())
            accepted.append((path, metadata))
    return accepted


def balanced_cycle(items: list[tuple[Path, dict]], count: int) -> list[tuple[Path, dict]]:
    by_task: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for item in items:
        by_task[str(item[1]["instruction"])].append(item)
    ordered: list[tuple[Path, dict]] = []
    depth = 0
    tasks = sorted(by_task)
    while len(ordered) < count:
        for task in tasks:
            pool = by_task[task]
            ordered.append(pool[depth % len(pool)])
            if len(ordered) == count:
                break
        depth += 1
    return ordered


def append_npz(output, path: Path, metadata: dict) -> int:
    with np.load(path, allow_pickle=False) as episode:
        action = episode["action"]
        for index in range(len(action)):
            output.add_frame({
                "observation.images.front": np.asarray(
                    episode["observation.front"][index], dtype=np.uint8),
                "observation.images.wrist": np.asarray(
                    episode["observation.wrist"][index], dtype=np.uint8),
                "observation.state": np.asarray(
                    episode["observation.state"][index], dtype=np.float32),
                "action": np.asarray(action[index], dtype=np.float32),
                "task": str(metadata["instruction"]),
            })
    output.save_episode(parallel_encoding=False)
    return len(action)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--unsafe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=config.COLLISION_SAFE_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=config.COLLISION_SAFE_MANIFEST_PATH)
    parser.add_argument("--repo-id", default=config.COLLISION_SAFE_DATASET_REPO)
    parser.add_argument("--expanded-root", type=Path, default=config.EXPANDED_DATASET_ROOT)
    parser.add_argument("--safe-episodes", type=int, default=20)
    parser.add_argument("--replay-ratio", type=float, default=0.40)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required")
    if not 20 <= args.safe_episodes <= 40:
        parser.error("--safe-episodes must be in [20, 40]")
    if not 0.30 <= args.replay_ratio <= 0.50:
        parser.error("--replay-ratio must be in [0.30, 0.50]")

    unsafe = unsafe_index(args.unsafe_root)
    safe_unique = load_safe_paths(args.safe_root, unsafe)
    if not safe_unique:
        raise RuntimeError(f"no verified same-state safe pair found under {args.safe_root}")
    safe_entries = balanced_cycle(safe_unique, args.safe_episodes)
    replay_count = round(args.safe_episodes * args.replay_ratio / (1.0 - args.replay_ratio))

    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    expanded = LeRobotDataset(
        config.EXPANDED_DATASET_REPO, root=args.expanded_root,
        video_backend=config.VIDEO_BACKEND,
    )
    target_tasks = {str(meta["instruction"]).lower() for _, meta in safe_unique}
    replay = stratified_replay(episode_map(expanded), target_tasks, replay_count)
    output = create_output(args.output_root, args.repo_id)
    manifest_entries: list[dict] = []

    for index, (path, metadata) in enumerate(safe_entries, start=1):
        pair = metadata["corrective_pair"]
        frames = append_npz(output, path, metadata)
        manifest_entries.append({
            "bucket": "same_state_safe_corrective",
            "source": str(path.resolve()),
            "unsafe_pair_source": metadata["unsafe_pair_source"],
            "instruction": metadata["instruction"],
            "base_task_id": pair["base_task_id"],
            "base_episode_id": pair["base_episode_id"],
            "donor_ordinal": pair["expert_donor_ordinal"],
            "frames": frames,
        })
        print(f"[collision-safe] corrective {index}/{len(safe_entries)}", flush=True)

    for index, (task, episode_index) in enumerate(replay, start=1):
        frames = append_episode(output, expanded, episode_index)
        manifest_entries.append({
            "bucket": "expanded_representative_replay",
            "source": "expanded",
            "task": task,
            "episode": episode_index,
            "frames": frames,
        })
        print(f"[collision-safe] replay {index}/{len(replay)}: {task}", flush=True)
    output.finalize()

    actual_ratio = replay_count / (args.safe_episodes + replay_count)
    manifest = {
        "protocol": "same_init_collision_vs_safe_bc_with_expanded_replay",
        "repo_id": args.repo_id,
        "root": str(args.output_root.resolve()),
        "unique_verified_safe_pairs": len(safe_unique),
        "safe_output_episodes": args.safe_episodes,
        "safe_oversampling_factor": args.safe_episodes / len(safe_unique),
        "representative_replay_episodes": replay_count,
        "representative_replay_ratio": actual_ratio,
        "total_episodes": len(manifest_entries),
        "total_frames": sum(int(entry["frames"]) for entry in manifest_entries),
        "target_tasks": sorted({str(meta["instruction"]) for _, meta in safe_unique}),
        "entries": manifest_entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
