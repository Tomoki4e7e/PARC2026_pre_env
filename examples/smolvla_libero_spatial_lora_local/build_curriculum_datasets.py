#!/usr/bin/env python3
"""Build datasets for the expanded-base-first curriculum.

Two outputs are intentionally derived from immutable inputs:
  * spatial: only Spatial base tasks that failed the expanded-base evaluation;
  * track1: the four weak tasks plus a stratified 20% replay from the expanded base.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

SOURCE_ROOT = (
    config.LEROBOT_CACHE / "hub/datasets--lerobot--libero_plus/snapshots"
    / config.LIBERO_PLUS_DATASET_REVISION
)


def task_text(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0])
    return str(value)


def evenly(values: list[int], count: int) -> list[int]:
    if count < 0 or len(values) < count:
        raise ValueError(f"cannot select {count} episodes from {len(values)}")
    if count == 0:
        return []
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def failed_spatial_tasks(summary_path: Path, threshold: float) -> list[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = {
        str(row["base_task"]).lower()
        for row in summary.get("tasks", [])
        if row.get("group") == "spatial_clean"
        and float(row.get("success_rate", 0.0)) < threshold
    }
    known = {task.lower(): task for task in config.SPATIAL_TASK_NAMES}
    unknown = sorted(failed - set(known))
    if unknown:
        raise ValueError(f"unknown Spatial base tasks in evaluation: {unknown}")
    return [known[name] for name in sorted(failed)]


def episode_map(dataset) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for episode in dataset.meta.episodes:
        result[task_text(episode["tasks"]).lower()].append(int(episode["episode_index"]))
    return result


def to_uint8_hwc(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def create_output(root: Path, repo_id: str):
    if root.exists():
        raise FileExistsError(f"refusing to overwrite: {root}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    return LeRobotDataset.create(
        repo_id, fps=20, features=features, root=root, robot_type="panda",
        use_videos=False, image_writer_threads=8,
    )


def append_episode(output, dataset, episode_index: int) -> int:
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


def stratified_replay(by_task: dict[str, list[int]], excluded: set[str], count: int) -> list[tuple[str, int]]:
    """Round-robin over tasks so replay represents the broad base, not large tasks."""
    pools = {task: evenly(indices, len(indices)) for task, indices in sorted(by_task.items())
             if task not in excluded}
    picked: list[tuple[str, int]] = []
    depth = 0
    while len(picked) < count:
        added = False
        for task, indices in pools.items():
            if depth < len(indices):
                picked.append((task, indices[depth]))
                added = True
                if len(picked) == count:
                    break
        if not added:
            raise ValueError(f"only {len(picked)} representative replay episodes available")
        depth += 1
    return picked


def write_dataset(output, entries: list[dict], datasets: dict[str, object], manifest_path: Path,
                  metadata: dict) -> None:
    for index, entry in enumerate(entries, start=1):
        entry["frames"] = append_episode(
            output, datasets[str(entry["source"])], int(entry["episode"])
        )
        print(f"[curriculum-data] {index}/{len(entries)} {entry['bucket']}: {entry['task']}", flush=True)
    output.finalize()
    manifest = {
        **metadata,
        "total_episodes": len(entries),
        "total_frames": sum(int(entry["frames"]) for entry in entries),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


def build_spatial(args, expanded) -> None:
    failed = failed_spatial_tasks(args.eval_summary, args.failure_threshold)
    if not failed:
        manifest = {
            "mode": "spatial_failure_only", "failure_threshold": args.failure_threshold,
            "failed_tasks": [], "total_episodes": 0, "spatial_corrective_required": False,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return
    by_task = episode_map(expanded)
    entries = [
        {"source": "expanded", "bucket": "spatial_failure", "episode": episode,
         "task": task}
        for task in failed for episode in by_task[task.lower()]
    ]
    output = create_output(args.output_root, args.repo_id)
    write_dataset(output, entries, {"expanded": expanded}, args.manifest, {
        "mode": "spatial_failure_only", "failure_threshold": args.failure_threshold,
        "failed_tasks": failed, "spatial_corrective_required": True,
    })


def build_track1(args, expanded, source) -> None:
    target_counts = config.TRACK1_STAGE2_TASK_EPISODE_OVERRIDES
    source_by_task = episode_map(source)
    expanded_by_task = episode_map(expanded)
    target_entries: list[dict] = []
    for task, count in target_counts.items():
        for episode in evenly(source_by_task[task.lower()], count):
            target_entries.append({
                "source": "source", "bucket": "track1_weak", "episode": episode,
                "task": task,
            })
    target_names = {task.lower() for task in target_counts}
    replay = stratified_replay(expanded_by_task, target_names, args.replay_episodes)
    replay_entries = [
        {"source": "expanded", "bucket": "expanded_replay", "episode": episode,
         "task": task}
        for task, episode in replay
    ]
    entries = target_entries + replay_entries
    output = create_output(args.output_root, args.repo_id)
    ratio = len(replay_entries) / len(entries)
    write_dataset(output, entries, {"expanded": expanded, "source": source}, args.manifest, {
        "mode": "track1_weak_with_expanded_replay",
        "target_episodes": len(target_entries), "replay_episodes": len(replay_entries),
        "replay_ratio": ratio, "target_task_counts": target_counts,
        "representative_replay_tasks": len({entry["task"] for entry in replay_entries}),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    spatial = sub.add_parser("spatial")
    spatial.add_argument("--eval-summary", type=Path, required=True)
    spatial.add_argument("--failure-threshold", type=float, default=0.75)
    spatial.add_argument("--output-root", type=Path, default=config.SPATIAL_CORRECTIVE_DATASET_ROOT)
    spatial.add_argument("--manifest", type=Path, default=config.SPATIAL_CORRECTIVE_MANIFEST_PATH)
    spatial.add_argument("--repo-id", default=config.SPATIAL_CORRECTIVE_DATASET_REPO)
    track1 = sub.add_parser("track1")
    track1.add_argument("--replay-episodes", type=int, default=config.TRACK1_REPLAY_EPISODES)
    track1.add_argument("--output-root", type=Path, default=config.TRACK1_REPLAY_DATASET_ROOT)
    track1.add_argument("--manifest", type=Path, default=config.TRACK1_REPLAY_MANIFEST_PATH)
    track1.add_argument("--repo-id", default=config.TRACK1_REPLAY_DATASET_REPO)
    parser.add_argument("--expanded-root", type=Path, default=config.EXPANDED_DATASET_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()

    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    expanded = LeRobotDataset(config.EXPANDED_DATASET_REPO, root=args.expanded_root,
                              video_backend=config.VIDEO_BACKEND)
    if args.mode == "spatial":
        build_spatial(args, expanded)
    else:
        source = LeRobotDataset(config.LIBERO_PLUS_DATASET_REPO, root=args.source_root,
                                video_backend=config.VIDEO_BACKEND)
        build_track1(args, expanded, source)


if __name__ == "__main__":
    main()
