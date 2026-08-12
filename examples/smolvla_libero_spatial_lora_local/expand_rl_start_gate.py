#!/usr/bin/env python3
"""Build a restartable RL start-gate set from annotated local trajectories.

Each successful local rollout is retained once and relabeled at several strict
prefixes.  A strict prefix is a real observed transition sequence, but is an
unsuccessful truncated episode; it never inherits the terminal success bonus.
Organizer evaluation data must not be passed to this utility.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as episode:
            for key in episode.files:
                np.asarray(episode[key])
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefixes-per-episode", type=int, default=8)
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required by competition data rules")
    if args.prefixes_per_episode < 1:
        parser.error("--prefixes-per-episode must be >= 1")

    paths = sorted(args.input_dir.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"no annotated RL records found: {args.input_dir}")
    ratios = np.linspace(0.15, 0.85, args.prefixes_per_episode)
    written = 0
    for source_index, source in enumerate(paths):
        relative = source.relative_to(args.input_dir)
        full_output = args.output_dir / "full" / relative
        if not full_output.exists():
            full_output.parent.mkdir(parents=True, exist_ok=True)
            # The source is immutable local training data.  Prefer a hard link
            # so the gate view does not duplicate its large image arrays.
            try:
                os.link(source, full_output)
            except OSError:
                shutil.copy2(source, full_output)
        with np.load(source, allow_pickle=False) as episode:
            payload = {key: np.asarray(episode[key]) for key in episode.files}
        if "base_action" not in payload:
            raise ValueError(f"base_action is required before expansion: {source}")
        length = len(payload["action"])
        if length < 2:
            raise ValueError(f"episode is too short to truncate: {source}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        for prefix_index, ratio in enumerate(ratios):
            cutoff = min(length - 1, max(1, int(round(length * float(ratio)))))
            prefix_payload: dict[str, np.ndarray] = {}
            for key, value in payload.items():
                if key == "metadata_json":
                    continue
                prefix_payload[key] = value[:cutoff].copy()
            prefix_payload["terminated"][:] = False
            prefix_payload["truncated"][:] = False
            prefix_payload["truncated"][-1] = True
            prefix_metadata = dict(metadata)
            prefix_metadata["success"] = False
            prefix_metadata["collided"] = bool(np.any(prefix_payload["collision"]))
            prefix_metadata["reward_components"] = {
                "return": float(np.sum(prefix_payload["reward"])),
                "success": 0.0,
                "prefix_steps": cutoff,
            }
            context = dict(prefix_metadata.get("collection_context") or {})
            context["prefix_relabel"] = {
                "source": str(source),
                "source_steps": length,
                "prefix_steps": cutoff,
                "ratio": float(ratio),
            }
            prefix_metadata["collection_context"] = context
            prefix_payload["metadata_json"] = np.asarray(
                json.dumps(prefix_metadata, ensure_ascii=False)
            )
            output = (
                args.output_dir / "failure_prefixes" / relative.parent
                / f"prefix_{prefix_index:02d}__{relative.name}"
            )
            _write_npz(output, prefix_payload)
            written += 1
        print(
            f"expanded {source_index + 1}/{len(paths)}: {relative}", flush=True
        )
    print(
        f"complete: {len(paths)} full + {written} failure prefixes = "
        f"{len(paths) + written} episodes"
    )


if __name__ == "__main__":
    main()
