#!/usr/bin/env python3
"""摂動カテゴリ層別のプロキシタスク CSV を生成する。

本番（非公開タスク）は LIBERO-plus の摂動バリアントを CSV 登録して評価される。
compe/t1 と同じ形式でローカル用の層別セットを作り、instruction は bddl の
`(:language ...)`（= 本番 CSV と同じ言い回し）から取る。

Hold-out: example 4 タスクに対応するベースタスクは除外する。

使い方:
    source env.sh
    python -m compe.proxy.build_proxy_tasks              # 既定 (6 タスク × 7 カテゴリ)
    python -m compe.proxy.build_proxy_tasks --per-category 2
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
OUTPUT_CSV = Path(__file__).resolve().parent / "PROXY_TASKS.csv"

# example 4（compe/t1/T1_TASKS.csv）のベースタスク。汎化計測のため proxy から除く。
EXCLUDED_BASE_STEMS = {
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "put_the_bowl_on_the_stove",
}

# 過去の hold-out 9 タスクと比較できるよう、同じベースタスクを使う。
PROXY_BASE_STEMS: dict[str, list[str]] = {
    "libero_spatial": [
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_plate",
    ],
}

# 摂動カテゴリ → バリアント名の判定正規表現。
# bddl ファイルが実在するものだけを対象にする。採点パイプラインは
# obj_of_interest を bddl から読むため、ファイルの無い variant（_view_/_initstate_）は
# 本番でも評価できない。
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "table": re.compile(r"_table_\d+$"),
    "tb": re.compile(r"_tb_\d+$"),
    "light": re.compile(r"_light_\d+$"),
    "add": re.compile(r"_add_\d+$"),
    "level": re.compile(r"_level\d+_sample\d+$"),
    "language": re.compile(r"_language_\d+$"),
}
LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")

# カテゴリごとに選ぶバリアント数。add / level は init state が 1 本しか無いため
# エピソードを増やす代わりにバリアント数で分散させる。
DEFAULT_PER_CATEGORY: dict[str, int] = {
    "table": 1,
    "tb": 1,
    "light": 1,
    "language": 2,
    "add": 4,
    "level": 4,
}


def variants_for(suite: str, base_stem: str, category: str) -> list[str]:
    pattern = CATEGORY_PATTERNS[category]
    out = []
    for path in sorted((BDDL_ROOT / suite).glob(f"{base_stem}*.bddl")):
        stem = path.stem
        if " copy" in stem or not pattern.search(stem):
            continue
        out.append(stem)
    return out


def choose_evenly(names: list[str], count: int) -> list[str]:
    if count >= len(names):
        return names
    if count == 1:
        return [names[len(names) // 2]]
    step = (len(names) - 1) / (count - 1)
    return [names[round(index * step)] for index in range(count)]


def instruction_for(suite: str, stem: str) -> str:
    match = LANGUAGE_RE.search((BDDL_ROOT / suite / f"{stem}.bddl").read_text())
    if not match:
        raise RuntimeError(f"(:language ...) が見つかりません: {suite}/{stem}")
    return " ".join(match.group(1).split())


def main() -> None:
    parser = argparse.ArgumentParser(description="層別プロキシタスク CSV の生成")
    parser.add_argument(
        "--per-category",
        type=int,
        default=None,
        help="ベースタスク × カテゴリごとに選ぶバリアント数（既定はカテゴリ別）",
    )
    parser.add_argument(
        "--categories",
        default=",".join(CATEGORY_PATTERNS),
        help="対象カテゴリ（カンマ区切り）",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = set(categories) - set(CATEGORY_PATTERNS)
    if unknown:
        parser.error(f"unknown categories: {sorted(unknown)}")

    rows = []
    task_num = 0
    for suite, base_stems in PROXY_BASE_STEMS.items():
        for base_stem in base_stems:
            if base_stem in EXCLUDED_BASE_STEMS:
                raise RuntimeError(f"example 4 のベースタスクは使用不可: {base_stem}")
            for category in categories:
                names = variants_for(suite, base_stem, category)
                if not names:
                    print(f"  警告: 該当なし {suite}/{base_stem} [{category}]")
                    continue
                count = args.per_category or DEFAULT_PER_CATEGORY.get(category, 1)
                for stem in choose_evenly(names, count):
                    task_num += 1
                    rows.append(
                        {
                            "task_num": task_num,
                            "task_id": stem,
                            "instruction": instruction_for(suite, stem),
                            "suite": suite,
                            "category": category,
                            "base_task": base_stem,
                        }
                    )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_num",
                "task_id",
                "instruction",
                "suite",
                "category",
                "base_task",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} タスクを書き出しました: {args.output}")
    for row in rows[:6]:
        print(f"  [{row['category']:9s}] {row['task_id']}")
        print(f"              instruction='{row['instruction']}'")


if __name__ == "__main__":
    main()
