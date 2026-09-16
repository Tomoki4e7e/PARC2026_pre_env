from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq


def _segments(close: np.ndarray) -> list[tuple[bool, int, int]]:
    if not len(close):
        return []
    result: list[tuple[bool, int, int]] = []
    start = 0
    for index in range(1, len(close)):
        if bool(close[index]) != bool(close[start]):
            result.append((bool(close[start]), start, index))
            start = index
    result.append((bool(close[start]), start, len(close)))
    return result


def audit_dataset(
    dataset_root: str | Path,
    selected_by_task: dict[str, list[int]],
    *,
    lerobot_src: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_root)
    sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata("local/experiment_a_audit", root=root)
    selected = {episode for episodes in selected_by_task.values() for episode in episodes}
    file_paths = {metadata.get_data_file_path(episode) for episode in selected}
    rows: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for relative_path in file_paths:
        table = pq.read_table(
            root / relative_path,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        ).to_pydict()
        for episode, frame, action, state in zip(
            table["episode_index"],
            table["frame_index"],
            table["action"],
            table["observation.state"],
        ):
            if int(episode) in selected:
                rows[int(episode)].append(
                    (int(frame), np.asarray(action), np.asarray(state))
                )

    task_reports: dict[str, Any] = {}
    for task, episodes in selected_by_task.items():
        episode_reports = []
        for episode in episodes:
            frames = sorted(rows[episode], key=lambda item: item[0])
            actions = np.stack([item[1] for item in frames])
            states = np.stack([item[2] for item in frames])
            close = actions[:, 6] <= 0.5
            segments = _segments(close)
            close_segments = [(start, stop) for active, start, stop in segments if active]
            first_close = close_segments[0] if close_segments else None
            second_close = close_segments[1] if len(close_segments) > 1 else None
            first_motion = (
                float(np.linalg.norm(states[first_close[1] - 1, :3] - states[first_close[0], :3]))
                if first_close
                else 0.0
            )
            episode_reports.append(
                {
                    "episode": episode,
                    "frames": len(frames),
                    "duration_sec": len(frames) / metadata.fps,
                    "close_segments": close_segments,
                    "approach_frames": first_close[0] if first_close else len(frames),
                    "handle_close_frames": (
                        first_close[1] - first_close[0] if first_close else 0
                    ),
                    "handle_close_motion_m": first_motion,
                    "has_second_close": second_close is not None,
                    "bowl_close_frames": (
                        second_close[1] - second_close[0] if second_close else 0
                    ),
                    "post_release_frames": (
                        len(frames) - close_segments[-1][1] if close_segments else 0
                    ),
                }
            )
        task_reports[task] = {
            "episodes": len(episodes),
            "frames": sum(row["frames"] for row in episode_reports),
            "with_handle_close": sum(row["handle_close_frames"] > 0 for row in episode_reports),
            "with_second_close": sum(row["has_second_close"] for row in episode_reports),
            "episode_reports": episode_reports,
        }

    return {
        "dataset_root": str(root),
        "fps": metadata.fps,
        "features": {
            key: {
                "shape": list(feature["shape"] if isinstance(feature, dict) else feature.shape),
                "type": str(
                    feature["dtype"] if isinstance(feature, dict) else feature.type
                ),
            }
            for key, feature in metadata.features.items()
            if key in {"action", "observation.state"} or "images" in key
        },
        "gripper_encoding": "zero_one",
        "tasks": task_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Experiment A demonstration stages")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = audit_dataset(
        manifest["dataset_root"],
        manifest["selected_by_task"],
        lerobot_src=args.lerobot_src,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({task: {k: v for k, v in data.items() if k != "episode_reports"} for task, data in report["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
