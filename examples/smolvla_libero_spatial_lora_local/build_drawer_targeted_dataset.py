#!/usr/bin/env python3
"""Build drawer-targeted BC data while retaining 70--90% stage2 replay.

Drawer examples come from successful table/light replay episodes.  The replay
bucket is sampled evenly from the other three Track-1 tasks, so an 80% default
ratio explicitly protects the six currently strong table/light series.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from build_curriculum_datasets import append_episode, create_output

DRAWER_TASK = (
    "pick up the black bowl in the top drawer of the wooden cabinet "
    "and place it on the plate"
)
SOURCE_ROOT = config.WORK_DIR / "datasets/libero_track1_perturbation_replay_v4"
CORRECTIVE_ROOT = config.WORK_DIR / "eval_failure_corrective_pairs"
COLLECTED_ROOT = config.WORK_DIR / "drawer_collection_records_v2"
BASE_ROOT = (
    config.WORK_DIR / "lerobot_cache/hub/datasets--lerobot--libero_plus/snapshots"
    / config.LIBERO_PLUS_DATASET_REVISION
)


def task_text(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    text = str(value).strip().lower()
    # Rendered Track-1 BDDL uses paraphrases while the HF base uses canonical text.
    text = text.replace("akita black bowl", "black bowl")
    text = text.replace("top layer of the wooden cabinet", "top drawer of the wooden cabinet")
    if text.startswith("pick the "):
        text = "pick up the " + text[len("pick the "):]
    return text


def episode_groups(dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in dataset.meta.episodes:
        groups[task_text(row["tasks"])].append(int(row["episode_index"]))
    return groups


def cycle_evenly(groups: dict[str, list[int]], tasks: list[str], count: int):
    missing = [task for task in tasks if not groups.get(task)]
    if missing:
        raise RuntimeError(f"replay tasks missing from base dataset: {missing}")
    chosen = []
    depth = 0
    while len(chosen) < count:
        for task in tasks:
            pool = groups[task]
            chosen.append((task, pool[depth % len(pool)]))
            if len(chosen) == count:
                break
        depth += 1
    return chosen


def successful_corrective_paths(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
        if (metadata.get("success") and not metadata.get("collided")
                and "pick_up_the_black_bowl_in_the_top_drawer" in metadata.get("task_name", "")):
            paths.append(path)
    return paths


def append_npz(output, path: Path) -> int:
    with np.load(path, allow_pickle=False) as episode:
        metadata = json.loads(str(episode["metadata_json"].item()))
        actions = np.asarray(episode["action"], dtype=np.float32)
        for index, action in enumerate(actions):
            output.add_frame({
                "observation.images.front": np.asarray(episode["observation.front"][index], dtype=np.uint8),
                "observation.images.wrist": np.asarray(episode["observation.wrist"][index], dtype=np.uint8),
                "observation.state": np.asarray(episode["observation.state"][index], dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "task": str(metadata["instruction"]),
            })
    output.save_episode(parallel_encoding=False)
    return len(actions)


def append_record_npz(output, path: Path) -> int:
    with np.load(path, allow_pickle=False) as episode:
        actions = np.asarray(episode["action"], dtype=np.float32)
        task = str(episode["task"].item())
        for index, action in enumerate(actions):
            output.add_frame({
                "observation.images.front": np.asarray(episode["front"][index], dtype=np.uint8),
                "observation.images.wrist": np.asarray(episode["wrist"][index], dtype=np.uint8),
                "observation.state": np.asarray(episode["state"][index], dtype=np.float32),
                "action": action, "task": task,
            })
    output.save_episode(parallel_encoding=False)
    return len(actions)


def evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise RuntimeError(f"need {count} unique paths, found {len(paths)}")
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[round(i * (len(paths) - 1) / (count - 1))] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawer-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--corrective-root", type=Path, default=CORRECTIVE_ROOT)
    parser.add_argument("--collected-root", type=Path, default=COLLECTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.DRAWER_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=config.DRAWER_MANIFEST_PATH)
    parser.add_argument("--drawer-episodes", type=int, default=30)
    parser.add_argument("--min-unique-drawer", type=int, default=20)
    parser.add_argument("--replay-ratio", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.70 <= args.replay_ratio <= 0.90:
        parser.error("--replay-ratio must be in [0.70, 0.90]")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")

    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    drawer = LeRobotDataset(
        "local/libero_track1_perturbation_replay", root=args.drawer_root,
        video_backend=config.VIDEO_BACKEND,
    )
    base = LeRobotDataset(
        config.LIBERO_PLUS_DATASET_REPO, root=args.base_root,
        video_backend=config.VIDEO_BACKEND,
    )
    collected = sorted(args.collected_root.glob("*.npz"))
    table = [path for path in collected if "_table_" in path.name]
    light = [path for path in collected if "_light_" in path.name]
    unique_count = len({path.stem for path in collected})
    if unique_count < args.min_unique_drawer:
        raise RuntimeError(
            f"unique drawer gate failed: {unique_count} < {args.min_unique_drawer}"
        )
    table_count = args.drawer_episodes // 2
    light_count = args.drawer_episodes - table_count
    drawer_selected = [("collected_table", path) for path in evenly(table, table_count)]
    drawer_selected += [("collected_light", path) for path in evenly(light, light_count)]
    replay_count = round(
        args.drawer_episodes * args.replay_ratio / (1.0 - args.replay_ratio)
    )
    protected = sorted(
        set(config.TRACK1_STAGE2_OBJECT_TASK_NAMES + config.TRACK1_STAGE2_GOAL_TASK_NAMES)
    )
    replay_selected = cycle_evenly(episode_groups(base), protected, replay_count)
    output = create_output(args.output_root, config.DRAWER_DATASET_REPO)
    entries = []
    for source, value in drawer_selected:
        frames = append_record_npz(output, value)
        provenance = {"path": str(value.resolve())}
        entries.append({"bucket": f"drawer_table_light_{source}_success",
                        **provenance, "task": DRAWER_TASK, "frames": frames})
    for task, episode_index in replay_selected:
        frames = append_episode(output, base, episode_index)
        entries.append({"bucket": "stage2_protection_replay", "episode": episode_index,
                        "task": task, "frames": frames})
    output.finalize()
    manifest = {
        "protocol": "drawer_table_light_success_with_strong_stage2_replay",
        "repo_id": config.DRAWER_DATASET_REPO,
        "root": str(args.output_root.resolve()),
        "drawer_unique_episodes_available": unique_count,
        "drawer_unique_episodes_selected": len(drawer_selected),
        "selected_table_episodes": table_count,
        "selected_light_episodes": light_count,
        "minimum_unique_gate": args.min_unique_drawer,
        "drawer_output_episodes": len(drawer_selected),
        "replay_episodes": len(replay_selected),
        "replay_ratio": len(replay_selected) / len(entries),
        "protected_tasks": protected,
        "total_episodes": len(entries),
        "total_frames": sum(entry["frames"] for entry in entries),
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
