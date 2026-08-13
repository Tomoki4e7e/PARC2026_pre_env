#!/usr/bin/env python3
"""Build the 580-episode expanded multi-suite/multi-domain training dataset.

The split is deliberately explicit and reproducible:
  * LIBERO-Spatial: 10 tasks x 16 episodes = 160
  * LIBERO-Object:  10 tasks x 16 episodes = 160
  * LIBERO-Goal:    10 tasks x 10 episodes = 100
  * domain replay: 119 disjoint LIBERO-Plus episodes + 41 local table/light
    successful replays = 160

Selected source episode indices are written to the manifest so evaluation can
exclude the exact training samples and select held-out perturbation task IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = (
    config.LEROBOT_CACHE
    / "hub/datasets--lerobot--libero_plus/snapshots"
    / config.LIBERO_PLUS_DATASET_REVISION
)
LOCAL_REPLAY_ROOT = config.WORK_DIR / "datasets/libero_track1_perturbation_replay_v4"


def task_text(cell) -> str:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return str(cell[0])
    return str(cell)


def evenly(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"need {count} episodes, only {len(values)} available")
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def to_uint8_hwc(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--local-replay-root", type=Path, default=LOCAL_REPLAY_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.EXPANDED_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=config.EXPANDED_MANIFEST_PATH)
    parser.add_argument("--corrective-root", type=Path)
    parser.add_argument("--repo-id", default=config.EXPANDED_DATASET_REPO)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(
        config.LIBERO_PLUS_DATASET_REPO,
        root=args.source_root,
        video_backend=config.VIDEO_BACKEND,
    )
    replay = LeRobotDataset(
        "local/libero_track1_perturbation_replay",
        root=args.local_replay_root,
        video_backend=config.VIDEO_BACKEND,
    )

    by_task: dict[str, list[int]] = defaultdict(list)
    for episode in source.meta.episodes:
        by_task[task_text(episode["tasks"]).lower()].append(int(episode["episode_index"]))

    category_specs = [
        ("spatial", config.SPATIAL_TASK_NAMES, 16),
        ("object", config.OBJECT_TASK_NAMES, 16),
        ("goal", config.GOAL_TASK_NAMES, 10),
    ]
    selected: set[int] = set()
    entries: list[dict] = []
    for category, tasks, count in category_specs:
        for task in tasks:
            pool = by_task[task.lower()]
            picked = evenly(pool, count)
            selected.update(picked)
            entries.extend(
                {"source": "libero_plus", "bucket": category, "episode": ep, "task": task}
                for ep in picked
            )

    # Complete the requested 160-domain-replay bucket. The 41 locally rendered
    # table/light successful trajectories are all retained; 119 additional,
    # disjoint LIBERO-Plus episodes are spread uniformly over the same 30 tasks.
    extra_needed = 160 - replay.num_episodes
    all_tasks = config.SPATIAL_TASK_NAMES + config.OBJECT_TASK_NAMES + config.GOAL_TASK_NAMES
    quotient, remainder = divmod(extra_needed, len(all_tasks))
    for task_index, task in enumerate(all_tasks):
        available = [ep for ep in by_task[task.lower()] if ep not in selected]
        picked = evenly(available, quotient + int(task_index < remainder))
        selected.update(picked)
        entries.extend(
            {"source": "libero_plus", "bucket": "domain_replay", "episode": ep, "task": task}
            for ep in picked
        )

    features = {
        "observation.images.front": {"dtype": "image", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "image", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,),
                              "names": [f"state_{i}" for i in range(8)]},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": [f"action_{i}" for i in range(7)]},
    }
    output = LeRobotDataset.create(
        args.repo_id,
        fps=20,
        features=features,
        root=args.output_root,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=8,
    )

    def append_episode(dataset, episode_index: int) -> int:
        episode = dataset.meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        task = task_text(episode["tasks"])
        for frame_index in range(start, end):
            frame = dataset[frame_index]
            output.add_frame({
                "observation.images.front": to_uint8_hwc(frame["observation.images.front"]),
                "observation.images.wrist": to_uint8_hwc(frame["observation.images.wrist"]),
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "action": np.asarray(frame["action"], dtype=np.float32),
                "task": task,
            })
        output.save_episode(parallel_encoding=False)
        return end - start

    for index, entry in enumerate(entries):
        entry["frames"] = append_episode(source, int(entry["episode"]))
        print(f"[expanded] {index + 1}/580 {entry['bucket']}: {entry['task']}", flush=True)
    for episode_index in range(replay.num_episodes):
        episode = replay.meta.episodes[episode_index]
        entry = {
            "source": "local_table_light_replay",
            "bucket": "domain_replay",
            "episode": episode_index,
            "task": task_text(episode["tasks"]),
        }
        entry["frames"] = append_episode(replay, episode_index)
        entries.append(entry)
        print(f"[expanded] {len(entries)}/580 domain_replay(local): {entry['task']}", flush=True)
    if args.corrective_root is not None:
        corrective_paths = sorted(args.corrective_root.glob("*.npz"))
        if not corrective_paths:
            raise RuntimeError(f"corrective records not found: {args.corrective_root}")
        for path in corrective_paths:
            with np.load(path, allow_pickle=False) as episode:
                metadata = json.loads(str(episode["metadata_json"].item()))
                if not metadata.get("success") or metadata.get("collided"):
                    raise ValueError(f"corrective must be successful and collision-free: {path}")
                length = len(episode["action"])
                for frame_index in range(length):
                    output.add_frame({
                        "observation.images.front": np.asarray(episode["observation.front"][frame_index], dtype=np.uint8),
                        "observation.images.wrist": np.asarray(episode["observation.wrist"][frame_index], dtype=np.uint8),
                        "observation.state": np.asarray(episode["observation.state"][frame_index], dtype=np.float32),
                        "action": np.asarray(episode["action"][frame_index], dtype=np.float32),
                        "task": str(metadata["instruction"]),
                    })
            output.save_episode(parallel_encoding=False)
            entries.append({"source": "eval_failure_corrective", "bucket": "corrective",
                            "episode": path.name, "task": metadata["instruction"], "frames": length})
            print(f"[expanded] {len(entries)} corrective: {metadata['instruction']}", flush=True)
    output.finalize()

    bucket_counts = {bucket: sum(e["bucket"] == bucket for e in entries)
                     for bucket in ("spatial", "object", "goal", "domain_replay")}
    manifest = {
        "repo_id": args.repo_id,
        "root": str(args.output_root.resolve()),
        "source_revision": config.LIBERO_PLUS_DATASET_REVISION,
        "total_episodes": len(entries),
        "total_frames": sum(int(e["frames"]) for e in entries),
        "bucket_counts": bucket_counts,
        "entries": entries,
    }
    expected_total = 580 + (len(corrective_paths) if args.corrective_root is not None else 0)
    if len(entries) != expected_total or bucket_counts != {
        "spatial": 160, "object": 160, "goal": 100, "domain_replay": 160
    }:
        raise RuntimeError(f"unexpected split: {bucket_counts}, total={len(entries)}")
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
