#!/usr/bin/env python3
"""Apply promotion gates to expanded-first and legacy-order evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_map(summary: dict) -> dict[str, dict]:
    return {str(row["task_id"]): row for row in summary["tasks"]}


def assert_same_protocol(summaries: list[dict]) -> None:
    reference = set(task_map(summaries[0]))
    keys = ("n_episodes", "max_steps", "seed")
    expected_protocol = {key: summaries[0]["protocol"].get(key) for key in keys}
    for summary in summaries[1:]:
        if set(task_map(summary)) != reference:
            raise ValueError("evaluation task manifests differ; refusing an invalid A/B")
        protocol = {key: summary["protocol"].get(key) for key in keys}
        if protocol != expected_protocol:
            raise ValueError(f"evaluation protocols differ: {expected_protocol} != {protocol}")


def aggregate(summary: dict, *, group: str | None = None) -> dict:
    rows = [row for row in summary["tasks"] if group is None or row["group"] == group]
    if not rows:
        raise ValueError(f"evaluation has no rows for group={group!r}")
    return {
        "tasks": len(rows),
        "success_rate": sum(float(row["success_rate"]) for row in rows) / len(rows),
        "collision_rate": sum(float(row.get("collision_rate", 0.0)) for row in rows) / len(rows),
    }


def failed_spatial(summary: dict, threshold: float) -> list[str]:
    return sorted({
        str(row["base_task"])
        for row in summary["tasks"]
        if row["group"] == "spatial_clean" and float(row["success_rate"]) < threshold
    })


def delta(candidate: dict, baseline: dict) -> dict:
    return {
        "success_rate": candidate["success_rate"] - baseline["success_rate"],
        "collision_rate": candidate["collision_rate"] - baseline["collision_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--expanded-base", type=Path, required=True)
    parser.add_argument("--spatial-corrective", type=Path)
    parser.add_argument("--new-final", type=Path)
    parser.add_argument("--failure-threshold", type=float, default=0.75)
    parser.add_argument("--max-regression", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy = load(args.legacy)
    expanded = load(args.expanded_base)
    named = {"legacy": legacy, "expanded_base": expanded}
    if args.spatial_corrective:
        named["spatial_corrective"] = load(args.spatial_corrective)
    if args.new_final:
        named["new_final"] = load(args.new_final)
    assert_same_protocol(list(named.values()))

    failures = failed_spatial(expanded, args.failure_threshold)
    selected_pre_track1 = "expanded_base"
    spatial_gate = {"required": bool(failures), "failed_tasks": failures, "promoted": False}
    if failures and "spatial_corrective" in named:
        base_spatial = aggregate(expanded, group="spatial_clean")
        candidate_spatial = aggregate(named["spatial_corrective"], group="spatial_clean")
        base_overall = aggregate(expanded)
        candidate_overall = aggregate(named["spatial_corrective"])
        spatial_gate.update({
            "base_spatial": base_spatial,
            "candidate_spatial": candidate_spatial,
            "spatial_delta": delta(candidate_spatial, base_spatial),
            "overall_delta": delta(candidate_overall, base_overall),
        })
        promoted = (
            candidate_spatial["success_rate"] > base_spatial["success_rate"]
            and candidate_overall["success_rate"] + args.max_regression >= base_overall["success_rate"]
            and candidate_overall["collision_rate"] <= base_overall["collision_rate"]
        )
        spatial_gate["promoted"] = promoted
        if promoted:
            selected_pre_track1 = "spatial_corrective"

    final_comparison = None
    improved = None
    if "new_final" in named:
        legacy_all = aggregate(legacy)
        final_all = aggregate(named["new_final"])
        final_comparison = {
            "legacy": legacy_all, "new_order": final_all,
            "delta": delta(final_all, legacy_all),
        }
        improved = (
            final_all["success_rate"] > legacy_all["success_rate"]
            and final_all["collision_rate"] <= legacy_all["collision_rate"]
        )
        final_comparison["improved"] = improved

    report = {
        "same_task_manifest": True,
        "evaluation_protocol": {
            key: expanded["protocol"].get(key) for key in ("n_episodes", "max_steps", "seed")
        },
        "spatial_failure_threshold": args.failure_threshold,
        "spatial_gate": spatial_gate,
        "selected_pre_track1": selected_pre_track1,
        "selected_pre_track1_model_dir": named[selected_pre_track1]["protocol"]["model_dir"],
        "final_comparison": final_comparison,
        "improved_over_legacy": improved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
