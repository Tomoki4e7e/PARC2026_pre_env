#!/usr/bin/env python3
"""通常Track1デモと摂動replayを50:50で混合した学習datasetを作る。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORK = SCRIPT_DIR / "work"
LEROBOT_SRC = WORK / "lerobot/src"
BASE_ROOT = (WORK / "lerobot_cache/hub/datasets--lerobot--libero_plus/snapshots/"
             "f3f49f426d75030177b18778374005bc12ccd588")
PERTURBED_ROOT = WORK / "datasets/libero_track1_perturbation_replay_v4"
OUTPUT_ROOT = WORK / "datasets/libero_track1_perturbation_mixed_v1"
MANIFEST = WORK / "perturbation_training_manifest.json"

TRACK1_TASKS = {
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the tomato sauce and place it in the basket",
    "pick up the milk and place it in the basket",
    "put the bowl on the stove",
}


def task_text(cell) -> str:
    return str(cell[0] if isinstance(cell, (list, tuple, np.ndarray)) else cell)


def to_uint8_hwc(value) -> np.ndarray:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def evenly(values: list[int], count: int) -> list[int]:
    if count >= len(values):
        return values
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--perturbed-root", type=Path, default=PERTURBED_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    base = LeRobotDataset("lerobot/libero_plus", root=args.base_root, video_backend="pyav")
    perturbed = LeRobotDataset(
        "local/libero_track1_perturbation_replay", root=args.perturbed_root,
        video_backend="pyav",
    )
    target_base_count = perturbed.num_episodes
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode in base.meta.episodes:
        task = task_text(episode["tasks"]).lower()
        if task in TRACK1_TASKS:
            by_task[task].append(int(episode["episode_index"]))
    quotient, remainder = divmod(target_base_count, len(TRACK1_TASKS))
    selected_base = []
    for index, task in enumerate(sorted(TRACK1_TASKS)):
        selected_base.extend(evenly(by_task[task], quotient + (index < remainder)))
    selected_base = sorted(selected_base)

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
        "local/libero_track1_perturbation_mixed", fps=20, features=features,
        root=args.output_root, robot_type="panda", use_videos=False,
        image_writer_threads=8,
    )

    def append_episode(dataset, episode_index: int) -> tuple[str, int]:
        episode = dataset.meta.episodes[episode_index]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
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
        print(f"[mix] {task}: {end - start} frames", flush=True)
        return task, end - start

    entries = []
    for episode_index in selected_base:
        task, frames = append_episode(base, episode_index)
        entries.append({"source": "base", "episode": episode_index, "task": task, "frames": frames})
    for episode_index in range(perturbed.num_episodes):
        task, frames = append_episode(perturbed, episode_index)
        entries.append({"source": "perturbed", "episode": episode_index, "task": task, "frames": frames})
    output.finalize()

    manifest = {
        "repo_id": "local/libero_track1_perturbation_mixed",
        "root": str(args.output_root.resolve()),
        "base_episodes": len(selected_base), "perturbed_episodes": perturbed.num_episodes,
        "perturbed_fraction": perturbed.num_episodes / len(entries),
        "total_episodes": len(entries), "total_frames": sum(e["frames"] for e in entries),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
