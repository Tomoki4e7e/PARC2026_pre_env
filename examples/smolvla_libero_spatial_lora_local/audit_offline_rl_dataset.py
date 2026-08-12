#!/usr/bin/env python3
"""Audit whether local RL records are ready for residual IQL/CQL training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-episodes", type=int, default=500)
    parser.add_argument("--min-failures", type=int, default=100)
    parser.add_argument("--min-tasks", type=int, default=20)
    args = parser.parse_args()
    paths = sorted(path for root in args.input_dir for path in root.rglob("*.npz"))
    tasks = Counter()
    returns, action_std_inputs = [], []
    transitions = successes = collisions = with_base = visual_domain_episodes = 0
    independent_initial_states: set[tuple[str, int]] = set()
    invalid = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as episode:
                # Access every member so ZIP/NPY CRC damage in large image arrays
                # cannot hide behind an audit that only reads scalar fields.
                for key in episode.files:
                    np.asarray(episode[key])
                required = {
                    "observation.state", "action", "reward", "next_observation.state",
                    "terminated", "truncated", "collision", "metadata_json",
                }
                missing = sorted(required - set(episode.files))
                if missing:
                    raise ValueError(f"missing {missing}")
                metadata = json.loads(str(episode["metadata_json"].item()))
                length = len(episode["action"])
                if any(len(episode[key]) != length for key in required if key != "metadata_json"):
                    raise ValueError("transition length mismatch")
                transitions += length
                returns.append(float(episode["reward"].sum()))
                action_std_inputs.append(np.asarray(episode["action"], dtype=np.float32))
                successes += int(metadata["success"])
                collisions += int(metadata["collided"])
                with_base += int("base_action" in episode.files)
                tasks[metadata["task_name"]] += 1
                independent_initial_states.add(
                    (metadata["task_name"], int(metadata.get("seed", -1)))
                )
                context = metadata.get("collection_context") or {}
                visual_domain_episodes += int(context.get("visual_domain") is not None)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    failures = len(paths) - successes
    action_std = (
        np.concatenate(action_std_inputs).std(axis=0).tolist() if action_std_inputs else []
    )
    gates = {
        "enough_episodes": len(paths) >= args.min_episodes,
        "enough_failures": failures >= args.min_failures,
        "enough_tasks": len(tasks) >= args.min_tasks,
        "all_valid": not invalid,
        "all_have_base_action": with_base == len(paths) and bool(paths),
        "behavior_action_diversity": bool(action_std) and min(action_std[:6]) > 1.0e-3,
    }
    report = {
        "episodes": len(paths),
        "transitions": transitions,
        "tasks": len(tasks),
        "episodes_by_task": tasks,
        "successes": successes,
        "failures": failures,
        "collisions": collisions,
        "episodes_with_base_action": with_base,
        "episodes_with_visual_domain": visual_domain_episodes,
        "independent_initial_states": len(independent_initial_states),
        "return_percentiles": (
            {str(q): float(np.percentile(returns, q)) for q in (0, 25, 50, 75, 100)}
            if returns else {}
        ),
        "action_std": action_std,
        "invalid": invalid,
        "gates": gates,
        "ready_for_iql_cql": all(gates.values()),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
