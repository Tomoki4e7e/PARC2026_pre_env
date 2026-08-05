#!/usr/bin/env python3
"""results/proxy_*/summary.json を並べてカテゴリ別に比較する。

使い方:
    python examples/smolvla_libero_spatial_lora_local/compare_proxy_results.py
    python examples/smolvla_libero_spatial_lora_local/compare_proxy_results.py --tags multisuite_legacy base_generic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def load_summaries(tags: list[str] | None) -> dict[str, dict]:
    out = {}
    for path in sorted(RESULTS_DIR.glob("proxy_*/summary.json")):
        tag = path.parent.name.removeprefix("proxy_")
        if tags and tag not in tags:
            continue
        out[tag] = json.loads(path.read_text())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="プロキシ評価結果の比較")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument(
        "--per-task", action="store_true", help="タスク別成功率も表示する"
    )
    args = parser.parse_args()

    summaries = load_summaries(args.tags)
    if not summaries:
        raise SystemExit("summary.json が見つかりません")

    categories: list[str] = []
    for payload in summaries.values():
        for category in payload["categories"]:
            if category not in categories:
                categories.append(category)

    width = max(len(tag) for tag in summaries) + 2
    header = "config".ljust(width) + "".join(c.rjust(10) for c in categories)
    print(header + "平均".rjust(10))
    print("-" * len(header + "平均".rjust(10)))
    for tag, payload in summaries.items():
        row = tag.ljust(width)
        scores = []
        for category in categories:
            value = payload["categories"].get(category)
            if value is None:
                row += "-".rjust(10)
                continue
            scores.append(value["score"])
            row += f"{value['score']:.3f}".rjust(10)
        mean = sum(scores) / len(scores) if scores else 0.0
        print(row + f"{mean:.3f}".rjust(10))

    if args.per_task:
        for tag, payload in summaries.items():
            print(f"\n=== {tag} タスク別 ===")
            for category, value in payload["categories"].items():
                print(f"  [{category}]")
                for task, rate in sorted(value["tasks"].items()):
                    print(f"    {rate:.2f}  {task}")


if __name__ == "__main__":
    main()
