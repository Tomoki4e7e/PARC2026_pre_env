#!/usr/bin/env python3
"""Build same-observation chosen/rejected action data for ramekin safety."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
from build_curriculum_datasets import episode_map, stratified_replay, to_uint8_hwc


def grasp_index(actions: np.ndarray) -> int:
    values = np.flatnonzero(actions[:, -1] > 0.25)
    return int(values[0]) if len(values) else min(35, len(actions) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/libero_ramekin_pairwise_preference_v1")
    parser.add_argument("--expanded-root", type=Path, default=config.EXPANDED_DATASET_ROOT)
    parser.add_argument("--pair-repeats", type=int, default=3)
    parser.add_argument("--replay-ratio", type=float, default=0.40)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    if not 0.30 <= args.replay_ratio <= 0.50:
        parser.error("--replay-ratio must be in [0.30, 0.50]")

    safe_paths = []
    for path in sorted(args.safe_root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
        pair = metadata.get("preference_pair", {})
        if (metadata.get("success") and not metadata.get("collided")
                and pair.get("same_initial_state")):
            safe_paths.append(path)
    if not safe_paths:
        raise RuntimeError("no verified closed-loop safe trajectories found")

    sys.path.insert(0, str(config.LEROBOT_DIR / "src"))
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
        "preference.rejected_action": {"dtype": "float32", "shape": (7,),
                                       "names": [f"rejected_{i}" for i in range(7)]},
        "preference.weight": {"dtype": "float32", "shape": (1,),
                              "names": ["preference_weight"]},
    }
    output = LeRobotDataset.create(
        args.repo_id, fps=20, features=features, root=args.output_root,
        robot_type="panda", use_videos=False, image_writer_threads=8,
    )
    entries = []
    for repeat in range(args.pair_repeats):
        for safe_path in safe_paths:
            with np.load(safe_path, allow_pickle=False) as safe:
                metadata = json.loads(str(safe["metadata_json"].item()))
                pair = metadata["preference_pair"]
                chosen = np.asarray(safe["action"], dtype=np.float32)
                chosen_grasp = grasp_index(chosen)
                length = min(len(chosen), chosen_grasp + 15)
                with np.load(pair["unsafe"], allow_pickle=False) as unsafe:
                    rejected_source = np.asarray(unsafe["action"], dtype=np.float32)
                    collision = np.asarray(unsafe["collision"], dtype=np.bool_)
                collision_step = int(np.flatnonzero(collision)[0])
                rejected_indices = []
                for index in range(length):
                    if index <= chosen_grasp:
                        mapped = round(index * collision_step / max(chosen_grasp, 1))
                    else:
                        mapped = collision_step + index - chosen_grasp
                    rejected_indices.append(min(mapped, len(rejected_source) - 1))
                for index, rejected_index in enumerate(rejected_indices):
                    output.add_frame({
                        "observation.images.front": np.asarray(
                            safe["observation.front"][index], dtype=np.uint8),
                        "observation.images.wrist": np.asarray(
                            safe["observation.wrist"][index], dtype=np.uint8),
                        "observation.state": np.asarray(
                            safe["observation.state"][index], dtype=np.float32),
                        "action": chosen[index],
                        "preference.rejected_action": rejected_source[rejected_index],
                        "preference.weight": np.asarray([1.0], dtype=np.float32),
                        "task": metadata["instruction"],
                    })
            output.save_episode(parallel_encoding=False)
            entries.append({
                "bucket": "same_observation_pairwise", "repeat": repeat,
                "safe": str(safe_path.resolve()), "unsafe": pair["unsafe"],
                "task": metadata["task_name"], "init_state_ordinal": pair["init_state_ordinal"],
                "frames": length, "collision_step": collision_step,
            })
            print(f"[preference-data] pair {len(entries)}: {metadata['task_name']}", flush=True)

    pair_episodes = len(entries)
    replay_count = round(pair_episodes * args.replay_ratio / (1.0 - args.replay_ratio))
    expanded = LeRobotDataset(
        config.EXPANDED_DATASET_REPO, root=args.expanded_root,
        video_backend=config.VIDEO_BACKEND,
    )
    target_tasks = {entry["task"].replace("_", " ") for entry in entries}
    replay = stratified_replay(episode_map(expanded), target_tasks, replay_count)
    for task, episode_index in replay:
        episode = expanded.meta.episodes[episode_index]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        for frame_index in range(start, end):
            frame = expanded[frame_index]
            action = np.asarray(frame["action"], dtype=np.float32)
            output.add_frame({
                "observation.images.front": to_uint8_hwc(frame["observation.images.front"]),
                "observation.images.wrist": to_uint8_hwc(frame["observation.images.wrist"]),
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "action": action, "preference.rejected_action": action,
                "preference.weight": np.asarray([0.0], dtype=np.float32), "task": task,
            })
        output.save_episode(parallel_encoding=False)
        entries.append({"bucket": "expanded_representative_replay", "task": task,
                        "episode": episode_index, "frames": end - start})
        print(f"[preference-data] replay {len(entries) - pair_episodes}/{replay_count}: {task}",
              flush=True)
    output.finalize()

    manifest = {
        "protocol": "same_observation_flow_matching_pairwise_preference",
        "unique_closed_loop_safe_trajectories": len(safe_paths),
        "pair_repeats": args.pair_repeats, "pair_episodes": pair_episodes,
        "replay_episodes": replay_count,
        "replay_ratio": replay_count / (pair_episodes + replay_count),
        "total_episodes": len(entries),
        "total_frames": sum(int(entry["frames"]) for entry in entries),
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
