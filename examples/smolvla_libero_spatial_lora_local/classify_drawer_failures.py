#!/usr/bin/env python3
"""Create auditable four-way drawer failure labels from recorded rollouts.

The automatic label is a triage hint, not ground truth.  It uses collision,
gripper closure and post-closure lift signals and writes a CSV with an empty
``reviewed_label`` column for video/manual correction before data selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS = (
    "drawer_not_opened", "grasp_failure", "drawer_interference",
    "plate_transport_failure",
)


def classify(path: Path) -> dict[str, object] | None:
    with np.load(path, allow_pickle=False) as episode:
        metadata = json.loads(str(episode["metadata_json"].item()))
        if metadata.get("success") or "top_drawer" not in metadata.get("task_name", ""):
            return None
        state = np.asarray(episode["observation.state"], dtype=np.float32)
        collision = bool(metadata.get("collided") or np.any(episode["collision"]))
    gripper = state[:, 6:8].mean(axis=1)
    baseline = float(np.median(gripper[: min(10, len(gripper))]))
    closure = np.abs(gripper - baseline)
    close_steps = np.flatnonzero(closure > 0.012)
    first_close = int(close_steps[0]) if len(close_steps) else -1
    lift = 0.0
    if first_close >= 0:
        lift = float(np.max(state[first_close:, 2]) - state[first_close, 2])
    if collision:
        label, confidence = "drawer_interference", "high"
    elif first_close < 0:
        label, confidence = "drawer_not_opened", "medium"
    elif lift < 0.035:
        label, confidence = "grasp_failure", "medium"
    else:
        label, confidence = "plate_transport_failure", "medium"
    return {
        "path": str(path.resolve()), "task_id": metadata.get("task_name"),
        "episode_id": metadata.get("episode_id"), "automatic_label": label,
        "confidence": confidence, "reviewed_label": "", "collision": collision,
        "first_gripper_change_step": first_close, "post_gripper_lift_m": round(lift, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for root in args.roots for path in sorted(root.rglob("*.npz"))
            if (row := classify(path)) is not None]
    if not rows:
        raise RuntimeError("no failed top-drawer rollout found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = {label: sum(row["automatic_label"] == label for row in rows) for label in LABELS}
    print(json.dumps({"episodes": len(rows), "automatic_counts": counts,
                      "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
