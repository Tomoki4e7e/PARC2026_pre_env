#!/usr/bin/env python3
"""言い換え instruction → 正規 task 文字列 のマッチング精度をオフライン計測する。

LIBERO-plus の `*_language_N.bddl` は同じタスクの言い換え（例: "Grasp the
darkcolored vessel situated on the cooking appliance ..."）。SmolVLA は学習時の
task 文字列しか見ていないため、言い換えを正規表記へ写せれば成功率が戻る可能性がある。
シミュレーション不要で、写像の正解率だけを先に測る。

使い方:
    python examples/smolvla_libero_spatial_lora_local/eval_paraphrase_matching.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
SUITES = ("libero_spatial", "libero_object", "libero_goal")

LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")
LANGUAGE_VARIANT_RE = re.compile(r"^(?P<base>.+)_language_\d+$")

STOPWORDS = {
    "the", "a", "an", "and", "to", "on", "in", "it", "of", "then", "please",
    "that", "this", "with", "into", "at", "onto", "up", "your", "you", "is",
    "its", "for", "from", "there", "be", "by", "as", "after", "before",
}


def canonical_tasks() -> dict[str, list[str]]:
    """正規 task 文字列 -> その suite 内での base stem 一覧。"""
    out: dict[str, list[str]] = defaultdict(list)
    for suite in SUITES:
        for path in sorted((BDDL_ROOT / suite).glob("*.bddl")):
            stem = path.stem
            if " copy" in stem or re.search(
                r"_(?:table|tb|light|add|language)_\d+$|_level\d+_sample\d+$", stem
            ):
                continue
            out[stem.replace("_", " ")].append(f"{suite}/{stem}")
    return dict(out)


def paraphrase_samples() -> list[tuple[str, str]]:
    """(言い換え instruction, 正解の正規 task 文字列) のリスト。"""
    samples = []
    for suite in SUITES:
        for path in sorted((BDDL_ROOT / suite).glob("*_language_*.bddl")):
            match = LANGUAGE_VARIANT_RE.match(path.stem)
            if not match or " copy" in path.stem:
                continue
            language = LANGUAGE_RE.search(path.read_text())
            if not language:
                continue
            text = " ".join(language.group(1).split()).lower()
            samples.append((text, match.group("base").replace("_", " ")))
    return samples


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def build_idf(documents: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for document in documents:
        counts.update(set(tokens(document)))
    total = len(documents)
    return {
        word: math.log((total + 1) / (count + 0.5)) for word, count in counts.items()
    }


def char_ngrams(text: str, n: int = 4) -> Counter[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return Counter(cleaned[i : i + n] for i in range(max(len(cleaned) - n + 1, 0)))


def cosine(a: Counter[str], b: Counter[str], weights: dict[str, float] | None = None) -> float:
    if weights is None:
        weights = {}
    keys = set(a) | set(b)
    dot = sum(
        a.get(k, 0) * b.get(k, 0) * (weights.get(k, 1.0) ** 2) for k in keys
    )
    norm_a = math.sqrt(sum((a.get(k, 0) * weights.get(k, 1.0)) ** 2 for k in keys))
    norm_b = math.sqrt(sum((b.get(k, 0) * weights.get(k, 1.0)) ** 2 for k in keys))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# 言い換えで使われる語 → LIBERO の正規語。長いフレーズから順に適用する。
SYNONYM_RULES: tuple[tuple[str, str], ...] = (
    # 物体（黒いボウル）
    ("dark hued", "darkhued"),
    ("dark colored", "darkcolored"),
    ("darkhued curved container", "black bowl"),
    ("darkhued concave container", "black bowl"),
    ("darkhued mixing vessel", "black bowl"),
    ("darkcolored spherical vessel", "black bowl"),
    ("darkcolored rounded container", "black bowl"),
    ("black ceramic bowl", "black bowl"),
    ("akita black bowl", "black bowl"),
    ("darkhued", "black"),
    ("darkcolored", "black"),
    ("dark", "black"),
    ("obsidian", "black"),
    ("ebony", "black"),
    # 容器類
    ("small ceramic baking dish", "ramekin"),
    ("small ceramic dish", "ramekin"),
    ("small baking dish", "ramekin"),
    ("flat dining ware", "plate"),
    ("flat dish holder", "plate"),
    ("flat serving dish", "plate"),
    ("flat dish", "plate"),
    ("dinner plate", "plate"),
    ("serving dish", "plate"),
    ("cookies box", "cookie box"),
    ("biscuit box", "cookie box"),
    ("wicker container", "basket"),
    ("woven container", "basket"),
    ("woven basket", "basket"),
    # 家具・設備
    ("cooking appliance", "stove"),
    ("cooking range", "stove"),
    ("stove surface", "stove"),
    ("burner", "stove"),
    ("wooden cupboard", "wooden cabinet"),
    ("wooden storage unit", "wooden cabinet"),
    ("cupboard", "cabinet"),
    ("drawer of the cabinet", "drawer of the cabinet"),
    # 場所
    ("middle of the flat surface for meals", "table center"),
    ("middle of the dining surface", "table center"),
    ("middle of the eating surface", "table center"),
    ("heart of the eating surface", "table center"),
    ("center of the table", "table center"),
    ("middle of the table", "table center"),
    ("table centre", "table center"),
    # 動作
    ("relocate", "place"),
    ("reposition", "place"),
    ("position", "place"),
    ("set it atop", "place it on"),
    ("set it upon", "place it on"),
    ("set it down onto", "place it on"),
    ("set it on", "place it on"),
    ("place it atop", "place it on"),
    ("atop", "on"),
    ("upon", "on"),
    ("grasp", "pick up"),
    ("retrieve", "pick up"),
    ("collect", "pick up"),
    ("lift", "pick up"),
    ("take", "pick up"),
    ("amid", "between"),
    ("midway between", "between"),
    ("in the narrow space separating", "between"),
    ("adjacent to", "next to"),
    ("beside", "next to"),
    ("alongside", "next to"),
)


def apply_synonyms(text: str) -> str:
    out = " ".join(text.lower().split())
    for before, after in SYNONYM_RULES:
        if before in out:
            out = out.replace(before, after)
    return out


def match_ensemble(text: str, candidates: list[str], idf: dict[str, float]) -> str:
    normalized = apply_synonyms(text)
    query_words = Counter(tokens(normalized))
    query_chars = char_ngrams(normalized)
    best, best_score = candidates[0], -1.0
    for candidate in candidates:
        score = cosine(query_words, Counter(tokens(candidate)), idf) + cosine(
            query_chars, char_ngrams(candidate)
        )
        if score > best_score:
            best, best_score = candidate, score
    return best


def match_word_idf(text: str, candidates: list[str], idf: dict[str, float]) -> str:
    query = Counter(tokens(text))
    best, best_score = candidates[0], -1.0
    for candidate in candidates:
        score = cosine(query, Counter(tokens(candidate)), idf)
        if score > best_score:
            best, best_score = candidate, score
    return best


def match_char_ngram(text: str, candidates: list[str]) -> str:
    query = char_ngrams(text)
    best, best_score = candidates[0], -1.0
    for candidate in candidates:
        score = cosine(query, char_ngrams(candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best


def main() -> None:
    canon = canonical_tasks()
    candidates = sorted(canon)
    samples = paraphrase_samples()
    idf = build_idf(candidates)

    print(f"正規 task 候補: {len(candidates)} 件 / 言い換えサンプル: {len(samples)} 件\n")

    for name, matcher in (
        ("word+idf ", lambda t: match_word_idf(t, candidates, idf)),
        ("char4gram", lambda t: match_char_ngram(t, candidates)),
        ("syn+char ", lambda t: match_char_ngram(apply_synonyms(t), candidates)),
        ("ensemble ", lambda t: match_ensemble(t, candidates, idf)),
    ):
        correct = 0
        mistakes: list[tuple[str, str, str]] = []
        for text, truth in samples:
            prediction = matcher(text)
            if prediction == truth:
                correct += 1
            elif len(mistakes) < 5:
                mistakes.append((text, truth, prediction))
        print(f"{name}: 正解率 {correct / len(samples):.1%} ({correct}/{len(samples)})")
        for text, truth, prediction in mistakes:
            print(f"    入力: {text}")
            print(f"    正解: {truth}")
            print(f"    予測: {prediction}")
        print()


if __name__ == "__main__":
    main()
