#!/usr/bin/env python3
"""既存Track1混合データへ非衝突stove replayを重点追加する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORK = SCRIPT_DIR / "work"
LEROBOT_SRC = WORK / "lerobot/src"
BASE_ROOT = WORK / "datasets/libero_track1_perturbation_mixed_v1"
SAFE_ROOT = WORK / "datasets/stove_collision_safe_replay_v1"
OUTPUT_ROOT = WORK / "datasets/stove_collision_safe_mixed_v1"
MANIFEST = WORK / "stove_collision_training_manifest.json"


def _task_text(cell) -> str:
    return str(cell[0] if isinstance(cell, (list, tuple, np.ndarray)) else cell)


def _uint8_hwc(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--safe-root", type=Path, default=SAFE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--safe-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base = LeRobotDataset(
        "local/libero_track1_perturbation_mixed", root=args.base_root, video_backend="pyav"
    )
    safe = LeRobotDataset(
        "local/libero_track1_perturbation_replay", root=args.safe_root, video_backend="pyav"
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
        "local/stove_collision_safe_mixed", fps=20, features=features,
        root=args.output_root, robot_type="panda", use_videos=False,
        image_writer_threads=8,
    )

    entries = []
    def append(dataset, episode_index: int, source: str) -> None:
        episode = dataset.meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        task = _task_text(episode["tasks"])
        for frame_index in range(start, end):
            frame = dataset[frame_index]
            output.add_frame({
                "observation.images.front": _uint8_hwc(frame["observation.images.front"]),
                "observation.images.wrist": _uint8_hwc(frame["observation.images.wrist"]),
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "action": np.asarray(frame["action"], dtype=np.float32),
                "task": task,
            })
        output.save_episode(parallel_encoding=False)
        entries.append({"source": source, "episode": episode_index,
                        "task": task, "frames": end - start})
        print(f"[safe-mix] {source}: {task}: {end - start} frames", flush=True)

    for episode_index in range(base.num_episodes):
        append(base, episode_index, "track1_mixed")
    for repeat in range(args.safe_repeats):
        for episode_index in range(safe.num_episodes):
            append(safe, episode_index, f"safe_stove_x{repeat + 1}")
    output.finalize()
    manifest = {
        "repo_id": "local/stove_collision_safe_mixed",
        "root": str(args.output_root.resolve()),
        "base_episodes": base.num_episodes,
        "safe_episodes": safe.num_episodes * args.safe_repeats,
        "total_episodes": len(entries),
        "total_frames": sum(item["frames"] for item in entries),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
