#!/usr/bin/env python3
"""Summarize like-for-like weak-four and perturbation proxy evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    labels = ("score_033443", "legacy_order_expanded", "expanded_first_current")
    rows = []
    for label in labels:
        weak_path = next((args.root / label / "weak4").glob("server_*.json"))
        weak = json.loads(weak_path.read_text(encoding="utf-8"))["tracks"][0]
        proxy_path = args.root.parent / f"proxy_{args.root.name}" / label / "proxy" / "summary.json"
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        rows.append({
            "model": label,
            "weak4_score": weak["overall_score"],
            "weak4_collision_rate": weak["overall_metrics"].get("mean_collision_rate"),
            "weak4_tasks": {
                item["task_name"]: item["success_rate"] for item in weak["tasks"]
            },
            "proxy_category_mean": proxy["category_mean"],
            "proxy_categories": {
                name: value["score"] for name, value in proxy["categories"].items()
            },
        })
    payload = {"protocol": {"seed": 42}, "models": rows}
    output = args.root / "comparison.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
