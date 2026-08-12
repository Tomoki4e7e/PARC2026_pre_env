#!/usr/bin/env python3
"""Build a LeRobot dataset with episode-level reward-weighted resampling.

The source must be complete local-simulator RL NPZ records produced by
``pipeline.rl_data``.  Successful, collision-free, efficient trajectories are
repeated more often; failures remain once so the manifest retains data coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/libero_reward_weighted_bc")
    parser.add_argument("--max-repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="失敗軌道もBC対象に含める（既定はRL用に保持するだけで模倣しない）",
    )
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required by competition data rules")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {args.output_root}")

    paths = sorted(args.input_dir.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"no RL records found: {args.input_dir}")
    source_count = len(paths)
    returns = []
    metadata = []
    selected_paths = []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            item_metadata = json.loads(str(episode["metadata_json"].item()))
            if not args.include_failures and (
                not item_metadata["success"] or item_metadata["collided"]
            ):
                continue
            returns.append(float(np.asarray(episode["reward"]).sum()))
            metadata.append(item_metadata)
            selected_paths.append(path)
    paths = selected_paths
    if not paths:
        raise RuntimeError("no successful collision-free trajectories selected for BC")
    returns_array = np.asarray(returns, dtype=np.float64)
    centered = returns_array - np.median(returns_array)
    raw_weight = np.exp(np.clip(centered / args.temperature, -5.0, 5.0))
    raw_weight /= max(float(raw_weight.min()), 1.0e-8)
    repeats = np.clip(np.rint(raw_weight), 1, args.max_repeats).astype(int)

    lerobot_src = SCRIPT_DIR / "work/lerobot/src"
    sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with np.load(paths[0], allow_pickle=False) as first:
        height, width, channels = first["observation.front"].shape[1:]
        state_dim = first["observation.state"].shape[1]
        action_dim = first["action"].shape[1]
    features = {
        "observation.images.front": {
            "dtype": "image", "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image", "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,),
            "names": [f"state_{index}" for index in range(state_dim)],
        },
        "action": {
            "dtype": "float32", "shape": (action_dim,),
            "names": [f"action_{index}" for index in range(action_dim)],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=20,
        features=features,
        root=args.output_root,
        robot_type="panda",
        use_videos=False,
        image_writer_threads=4,
    )
    entries = []
    for path, meta, episode_return, repeat in zip(paths, metadata, returns, repeats):
        with np.load(path, allow_pickle=False) as episode:
            front = episode["observation.front"]
            wrist = episode["observation.wrist"]
            state = episode["observation.state"]
            action = episode["action"]
            for copy_index in range(int(repeat)):
                for frame_index in range(len(action)):
                    dataset.add_frame(
                        {
                            "observation.images.front": front[frame_index],
                            "observation.images.wrist": wrist[frame_index],
                            "observation.state": state[frame_index],
                            "action": action[frame_index],
                            "task": meta["instruction"],
                        }
                    )
                dataset.save_episode(parallel_encoding=False)
                entries.append(
                    {
                        "source": str(path),
                        "copy": copy_index,
                        "return": episode_return,
                        "success": meta["success"],
                        "collided": meta["collided"],
                    }
                )
    dataset.finalize()
    manifest = {
        "method": "episode_reward_weighted_resampling",
        "source_episodes": source_count,
        "selected_source_episodes": len(paths),
        "output_episodes": len(entries),
        "temperature": args.temperature,
        "max_repeats": args.max_repeats,
        "return_min": float(returns_array.min()),
        "return_median": float(np.median(returns_array)),
        "return_max": float(returns_array.max()),
        "entries": entries,
    }
    (args.output_root / "reward_weighting_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"built {len(entries)} episodes from {len(paths)} sources: {args.output_root}")


if __name__ == "__main__":
    main()
