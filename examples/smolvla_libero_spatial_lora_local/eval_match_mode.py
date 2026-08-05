#!/usr/bin/env python3
"""submission_template の instruction 正規化を language 変種全件で採点する。

`_language_N.bddl` の `(:language ...)` は評価時に policy へ渡る言い換えで、正解は
ファイル名（変種サフィックス除去）が指すベースタスクの正規 task 文字列。実際に
提出物が使う `_normalize_instruction` をそのまま呼ぶので、モード切替の効果を
シミュレーション無しで比較できる。

使い方:
    python examples/smolvla_libero_spatial_lora_local/eval_match_mode.py
    PARC_INSTRUCTION_MODE=generic python .../eval_match_mode.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_instruction_aliases import (  # noqa: E402
    LANGUAGE_RE,
    SUITES,
    dataset_task_strings,
    resolve_target,
)

# examples/ 側にも同名の古い写しがあるため、提出物のファイルを明示的に読み込む。
sys.path.insert(0, str(REPO_ROOT / "submission_template"))
_spec = importlib.util.spec_from_file_location(
    "submission_smolvla_mypolicy",
    REPO_ROOT / "submission_template" / "smolvla_mypolicy.py",
)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)

LANGUAGE_SUFFIX_RE = re.compile(r"_language_\d+$")


def collect_cases() -> list[tuple[str, str, str]]:
    """(言い換え instruction, 正解 task 文字列, 変種ファイル名) を集める。"""
    allowed = dataset_task_strings()
    cases: list[tuple[str, str, str]] = []
    for suite in SUITES:
        suite_dir = BDDL_ROOT / suite
        if not suite_dir.is_dir():
            continue
        for path in sorted(suite_dir.glob("*_language_*.bddl")):
            base_stem = LANGUAGE_SUFFIX_RE.sub("", path.stem)
            target = resolve_target(base_stem.replace("_", " "), allowed)
            if target is None:
                continue
            match = LANGUAGE_RE.search(path.read_text())
            if not match:
                continue
            cases.append((" ".join(match.group(1).split()), target, path.stem))
    return cases


def show_failures(needle: str) -> None:
    """正解が needle を含む task の失敗例を原文付きで出す（同義語規則の設計用）。"""
    for text, target, _stem in collect_cases():
        if needle not in target:
            continue
        predicted = policy._normalize_instruction(text)
        if predicted != target:
            print(f"  IN : {text}")
            print(f"  GOT: {predicted}\n")


def main() -> None:
    if len(sys.argv) > 1:
        show_failures(sys.argv[1])
        return
    mode = policy.INSTRUCTION_MODE
    print(
        f"policy: mode={mode} aliases={len(policy.INSTRUCTION_ALIASES)} "
        f"canonical={len(policy.CANONICAL_TASKS)}"
    )
    cases = collect_cases()
    global CANONICAL
    CANONICAL = set(policy.CANONICAL_TASKS)
    hits = 0
    per_task: Counter[str] = Counter()
    per_task_total: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()

    for text, target, _stem in cases:
        predicted = policy._normalize_instruction(text)
        per_task_total[target] += 1
        if predicted == target:
            hits += 1
            per_task[target] += 1
        else:
            confusions[(target, predicted)] += 1

    total = len(cases)
    print(f"mode={mode}  {hits}/{total} = {hits / max(total, 1):.1%}")
    if getattr(policy, "_PARAPHRASE_CENTROIDS", None):
        print(
            "注意: match モードの分類器はこの 2000 件で学習済みのため、この数値は"
            "学習データ上の再現率。汎化性能は build_paraphrase_model.py の CV を見る。\n"
        )
    else:
        print()

    print("タスク別正解率（低い順）:")
    for target, count in sorted(
        per_task_total.items(), key=lambda kv: per_task[kv[0]] / kv[1]
    ):
        rate = per_task[target] / count
        print(f"  {rate:6.1%}  ({per_task[target]:3d}/{count:3d})  {target}")

    print("\n頻出誤りトップ 40:")
    for (target, predicted), count in confusions.most_common(40):
        print(f"  {count:4d}  {target}\n        -> {predicted}")

    print("\n誤写像先の集計（どの task が吸い込んでいるか）:")
    sinks: Counter[str] = Counter()
    for (_target, predicted), count in confusions.items():
        sinks[predicted if predicted in CANONICAL else "(写像せず=原文のまま)"] += count
    for predicted, count in sinks.most_common(12):
        print(f"  {count:4d}  {predicted}")


if __name__ == "__main__":
    main()
