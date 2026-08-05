#!/usr/bin/env python3
"""評価 instruction → 学習 task 文字列の対応表を生成する。

LIBERO-plus の bddl `(:language ...)` は評価側が使う言い回し（`Pick the akita
black bowl ...`）で、`lerobot/libero_plus` データセットの task 文字列（bddl の
ファイル名由来、`pick up the black bowl ...`）とは全タスクで異なる。SmolVLA は
言語条件付きのため、この差を吸収しないと未知タスクで性能が出ない。

出力: submission_template/instruction_aliases.py（生成物、手で編集しない）

使い方:
    python examples/smolvla_libero_spatial_lora_local/gen_instruction_aliases.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
OUTPUT = REPO_ROOT / "submission_template" / "instruction_aliases.py"

# libero_90 の task 文字列は lerobot/libero_plus に含まれないため対象外
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

DATASET_TASKS_PARQUET = (
    Path(__file__).resolve().parent
    / "work"
    / "lerobot_cache"
    / "hub"
    / "datasets--lerobot--libero_plus"
    / "snapshots"
)

SCENE_PREFIX_RE = re.compile(r"^[A-Z][A-Z ]*SCENE\d+\s+")

# 変種サフィックス（table_1 / light_5 / add_12 / level1_sample1 / initstate / noise ...）
VARIANT_RE = re.compile(
    r"_(?:table|tb|light|add|language)_\d+$"
    r"|_level\d+_sample\d+$"
    r"|_view_[\d_]+_initstate_\d+(?:_noise_\d+)?$"
    r"|_initstate_\d+(?:_noise_\d+)?$"
)
LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")


def base_task_stems(suite_dir: Path) -> list[Path]:
    """変種でもコピーでもないベースタスクの bddl を返す。"""
    out = []
    for path in sorted(suite_dir.glob("*.bddl")):
        stem = path.stem
        if VARIANT_RE.search(stem) or re.search(r"_copy(_\d+)?$", stem):
            continue
        out.append(path)
    return out


def dataset_task_strings() -> set[str]:
    """lerobot/libero_plus が持つ task 文字列（学習時に policy が見た言い回し）。"""
    candidates = sorted(DATASET_TASKS_PARQUET.glob("*/meta/tasks.parquet"))
    if not candidates:
        return set()
    import pandas as pd

    df = pd.read_parquet(candidates[-1])
    column = df.index if df.index.name == "task" else df.iloc[:, 0]
    return {str(value) for value in column}


def resolve_target(train_text: str, allowed: set[str]) -> str | None:
    """bddl ファイル名由来の文字列を、データセット実在の task 文字列に解決する。"""
    if not allowed:
        return train_text
    for candidate in (train_text, SCENE_PREFIX_RE.sub("", train_text)):
        if candidate in allowed:
            return candidate
    # libero_10 等はファイル名とデータセット表記が僅かに異なるため語集合で照合する
    stripped = SCENE_PREFIX_RE.sub("", train_text)
    tokens = set(stripped.split())
    best, best_score = None, 0.0
    for candidate in allowed:
        candidate_tokens = set(candidate.split())
        score = len(tokens & candidate_tokens) / max(
            len(tokens | candidate_tokens), 1
        )
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= 0.8 else None


def build_mapping() -> tuple[dict[str, str], list[str]]:
    allowed = dataset_task_strings()
    mapping: dict[str, str] = {}
    dropped: list[str] = []
    ambiguous: set[str] = set()

    def add_alias(eval_text: str, target: str, source: str) -> None:
        eval_text = " ".join(eval_text.split()).lower().rstrip(".")
        if eval_text in ambiguous:
            return
        if eval_text == target:
            return
        existing = mapping.get(eval_text)
        if existing is not None and existing != target:
            # 同じ instruction が複数 task を指す場合、強制写像すると危険なので除外する。
            mapping.pop(eval_text, None)
            ambiguous.add(eval_text)
            dropped.append(
                f"{source}: ambiguous eval={eval_text!r}: "
                f"{existing!r} vs {target!r}"
            )
            return
        mapping[eval_text] = target

    for suite in SUITES:
        suite_dir = BDDL_ROOT / suite
        if not suite_dir.is_dir():
            continue
        for path in base_task_stems(suite_dir):
            match = LANGUAGE_RE.search(path.read_text())
            if not match:
                continue
            eval_text = " ".join(match.group(1).split()).lower()
            target = resolve_target(path.stem.replace("_", " "), allowed)
            if target is None:
                dropped.append(f"{suite}/{path.stem}: eval={eval_text!r}")
                continue
            add_alias(eval_text, target, f"{suite}/{path.name}")

            # language 変種は既知タスクの人手言い換え。完全一致辞書に入れることで、
            # 未知タスクを最近傍40タスクへ誤写像せずに language 摂動だけを救う。
            for variant in sorted(suite_dir.glob(f"{path.stem}_language_*.bddl")):
                variant_match = LANGUAGE_RE.search(variant.read_text())
                if variant_match:
                    add_alias(
                        variant_match.group(1),
                        target,
                        f"{suite}/{variant.name}",
                    )
    return mapping, dropped


HEADER = '''"""評価 instruction → 学習 task 文字列の対応表（自動生成）。

生成元: examples/smolvla_libero_spatial_lora_local/gen_instruction_aliases.py
LIBERO-plus の bddl `(:language ...)` と `lerobot/libero_plus` データセットの
task 文字列の対応。手で編集せず、生成スクリプトを更新して再生成する。
"""

from __future__ import annotations

INSTRUCTION_ALIASES: dict[str, str] = {
'''

FOOTER = "}\n"

CANONICAL_HEADER = '''
# lerobot/libero_plus が持つ task 文字列（policy が学習時に見た全 40 タスク）。
# 未知の言い換えを最も近い正規表記へ写すときの候補集合。
CANONICAL_TASKS: tuple[str, ...] = (
'''


def main() -> None:
    mapping, dropped = build_mapping()
    lines = [HEADER]
    for eval_text in sorted(mapping):
        train_text = mapping[eval_text]
        lines.append(f"    {eval_text!r}: (\n        {train_text!r}\n    ),\n")
    lines.append(FOOTER)

    canonical = sorted(dataset_task_strings())
    if canonical:
        lines.append(CANONICAL_HEADER)
        for task in canonical:
            lines.append(f"    {task!r},\n")
        lines.append(")\n")

    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"{len(mapping)} 件 + 正規タスク {len(canonical)} 件を書き出しました: {OUTPUT}")
    for eval_text in sorted(mapping):
        print(f"  {eval_text!r}\n    -> {mapping[eval_text]!r}")
    if dropped:
        print(f"\nデータセットに対応 task が無く除外 ({len(dropped)}):")
        for item in dropped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
