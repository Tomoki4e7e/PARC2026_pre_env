#!/usr/bin/env python3
"""Run the frozen SmolVLA over RL observations and attach ``base_action``.

This bridges relabeled expert/replay trajectories to residual training.  Temporal
ensemble stays enabled, so annotations match the deployed base-policy interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUBMISSION = REPO_ROOT / "submission_template"
sys.path.insert(0, str(SUBMISSION))
os.environ["PARC_RESIDUAL_POLICY"] = "0"


def axis_angle_to_quat(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1.0e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = value / angle
    return np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    parser.add_argument("--max-new-episodes", type=int)
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required by competition data rules")
    if args.max_new_episodes is not None and args.max_new_episodes < 1:
        parser.error("--max-new-episodes must be >= 1")
    paths = sorted(
        (root, path) for root in args.input_dir for path in root.rglob("*.npz")
    )
    if not paths:
        raise RuntimeError(f"no RL records found: {args.input_dir}")

    pending = []
    for root, path in paths:
        output = args.output_dir / root.name / path.relative_to(root)
        if output.is_file():
            with np.load(output, allow_pickle=False) as episode:
                if "base_action" not in episode.files:
                    raise ValueError(f"existing annotation lacks base_action: {output}")
            continue
        pending.append((root, path, output))
    if args.max_new_episodes is not None:
        pending = pending[:args.max_new_episodes]
    if not pending:
        print(f"complete: {len(paths)}/{len(paths)} annotations already exist")
        return

    from smolvla_mypolicy import SmolVLAMyPolicy

    policy = SmolVLAMyPolicy(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (root, path, output) in enumerate(pending):
        with np.load(path, allow_pickle=False) as episode:
            payload = {key: np.asarray(episode[key]) for key in episode.files}
        metadata = json.loads(str(payload["metadata_json"].item()))
        state = payload["observation.state"]
        front = payload["observation.front"]
        wrist = payload["observation.wrist"]
        policy.reset(metadata["instruction"], seed=int(metadata.get("seed", 0)))
        base_actions = []
        for frame_index in range(len(state)):
            obs = {
                "agentview_image": front[frame_index],
                "robot0_eye_in_hand_image": wrist[frame_index],
                "robot0_eef_pos": state[frame_index, :3],
                "robot0_eef_quat": axis_angle_to_quat(state[frame_index, 3:6]),
                "robot0_gripper_qpos": state[frame_index, 6:8],
            }
            policy.get_action(obs)
            base_actions.append(np.asarray(policy.last_base_action, dtype=np.float32))
        payload["base_action"] = np.stack(base_actions)
        relative = Path(root.name) / path.relative_to(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".npz.tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, output)
        print(f"annotated {index + 1}/{len(pending)}: {relative}", flush=True)


if __name__ == "__main__":
    main()
