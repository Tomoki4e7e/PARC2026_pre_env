#!/usr/bin/env python3
"""Generate a proxy CSV containing every top-drawer table/light variant."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import run_track1_distribution_audit as audit

DRAWER = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.output.with_suffix(".all.csv")
    rows = audit.build_csv(temporary, 10_000, ["table", "light"])
    rows = [row for row in rows if row["base_task"] == DRAWER]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.unlink()
    counts = {category: sum(row["category"] == category for row in rows)
              for category in ("table", "light")}
    print(f"saved {len(rows)} variants {counts}: {args.output}")


if __name__ == "__main__":
    main()
