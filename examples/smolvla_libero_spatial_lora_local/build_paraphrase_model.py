#!/usr/bin/env python3
"""言い換え instruction → 正規 task 文字列の分類器を学習して重みを書き出す。

LIBERO-plus の `_language_N.bddl` は各ベースタスク 50 通りの人手言い換えを持ち、
ファイル名から正解タスクが分かる。この 2000 件を教師データとして TF-IDF セントロイド
（Rocchio）を学習すると、手書き同義語より広い語彙をカバーできる。

汎化性能は「タスクごとに言い換えを分割する」k-fold で測る。学習に使った言い換えを
テストに使わないので、未知の言い換えに対する精度の正直な推定になる。

出力: submission_template/paraphrase_model.py（生成物、手で編集しない）

使い方:
    python examples/smolvla_libero_spatial_lora_local/build_paraphrase_model.py
    python .../build_paraphrase_model.py --folds 5   # CV のみ
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
OUTPUT = REPO_ROOT / "submission_template" / "paraphrase_model.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_instruction_aliases import (  # noqa: E402
    LANGUAGE_RE,
    SUITES,
    dataset_task_strings,
    resolve_target,
)

LANGUAGE_SUFFIX_RE = re.compile(r"_language_\d+$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# 内容語をほぼ持たない機能語。IDF でも下がるが、明示的に落として次元を減らす。
STOPWORDS = frozenset(
    """a an the of to in on at it its this that these those and or for with from
    into onto over under by as is are be been being do does did can could would
    should will shall may might must please kindly you your i me my we our then
    now just also very really quite so such if while when where which who whom
    there here what how why not no nor but than too own same s t don now
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def features(text: str) -> Counter[str]:
    """単語 unigram / bigram と文字 4-gram を混ぜた素性。"""
    tokens = tokenize(text)
    feats: Counter[str] = Counter()
    for token in tokens:
        feats[f"w:{token}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        feats[f"b:{left}_{right}"] += 1
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    for index in range(max(len(cleaned) - 3, 0)):
        feats[f"c:{cleaned[index : index + 4]}"] += 1
    return feats


def collect_cases() -> list[tuple[str, str]]:
    """(言い換え, 正解 task 文字列) を集める。"""
    allowed = dataset_task_strings()
    cases: list[tuple[str, str]] = []
    for suite in SUITES:
        suite_dir = BDDL_ROOT / suite
        if not suite_dir.is_dir():
            continue
        for path in sorted(suite_dir.glob("*_language_*.bddl")):
            target = resolve_target(
                LANGUAGE_SUFFIX_RE.sub("", path.stem).replace("_", " "), allowed
            )
            if target is None:
                continue
            match = LANGUAGE_RE.search(path.read_text())
            if match:
                cases.append((" ".join(match.group(1).split()), target))
    return cases


def compute_idf(documents: list[Counter[str]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for doc in documents:
        counts.update(doc.keys())
    total = len(documents)
    return {
        feat: math.log((total + 1.0) / (count + 1.0)) + 1.0
        for feat, count in counts.items()
    }


def to_vector(feats: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    vector = {
        feat: (1.0 + math.log(count)) * idf[feat]
        for feat, count in feats.items()
        if feat in idf
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {feat: value / norm for feat, value in vector.items()}


def train(pairs: list[tuple[str, str]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """クラスごとの TF-IDF セントロイドを返す。"""
    docs = [features(text) for text, _ in pairs]
    idf = compute_idf(docs)
    sums: dict[str, dict[str, float]] = defaultdict(dict)
    for doc, (_text, label) in zip(docs, pairs):
        vector = to_vector(doc, idf)
        centroid = sums[label]
        for feat, value in vector.items():
            centroid[feat] = centroid.get(feat, 0.0) + value
    centroids: dict[str, dict[str, float]] = {}
    for label, centroid in sums.items():
        norm = math.sqrt(sum(v * v for v in centroid.values()))
        if norm > 0.0:
            centroids[label] = {f: v / norm for f, v in centroid.items()}
    return centroids, idf


def predict(
    text: str, centroids: dict[str, dict[str, float]], idf: dict[str, float]
) -> tuple[str | None, float]:
    vector = to_vector(features(text), idf)
    if not vector:
        return None, 0.0
    best, best_score = None, 0.0
    for label, centroid in centroids.items():
        score = sum(value * centroid.get(feat, 0.0) for feat, value in vector.items())
        if score > best_score:
            best, best_score = label, score
    return best, best_score


def cross_validate(pairs: list[tuple[str, str]], folds: int) -> None:
    """タスクごとに言い換えを folds 分割し、未知の言い換えに対する精度を測る。"""
    by_label: dict[str, list[str]] = defaultdict(list)
    for text, label in pairs:
        by_label[label].append(text)

    hits = total = 0
    per_label_hits: Counter[str] = Counter()
    per_label_total: Counter[str] = Counter()
    for fold in range(folds):
        train_pairs: list[tuple[str, str]] = []
        test_pairs: list[tuple[str, str]] = []
        for label, texts in by_label.items():
            for index, text in enumerate(texts):
                bucket = test_pairs if index % folds == fold else train_pairs
                bucket.append((text, label))
            # 正規表記そのものも学習文書に入れる（完全一致の指示に効く）
            train_pairs.append((label, label))
        centroids, idf = train(train_pairs)
        for text, label in test_pairs:
            predicted, _score = predict(text, centroids, idf)
            total += 1
            per_label_total[label] += 1
            if predicted == label:
                hits += 1
                per_label_hits[label] += 1

    print(f"{folds}-fold CV: {hits}/{total} = {hits / max(total, 1):.1%}\n")
    print("タスク別（低い順）:")
    for label, count in sorted(
        per_label_total.items(), key=lambda kv: per_label_hits[kv[0]] / kv[1]
    )[:12]:
        print(f"  {per_label_hits[label] / count:6.1%}  ({per_label_hits[label]:3d}/{count:3d})  {label}")


HEADER = '''"""言い換え instruction 分類器の重み（自動生成）。

生成元: examples/smolvla_libero_spatial_lora_local/build_paraphrase_model.py
LIBERO-plus の language 変種を教師データにした TF-IDF セントロイド。
手で編集せず、生成スクリプトを更新して再生成する。
"""

from __future__ import annotations

'''


def write_model(
    centroids: dict[str, dict[str, float]], idf: dict[str, float], min_weight: float
) -> None:
    """小さい重みを捨ててファイルサイズを抑えつつ書き出す。"""
    kept_feats: set[str] = set()
    trimmed: dict[str, dict[str, float]] = {}
    for label, centroid in centroids.items():
        pruned = {f: round(v, 5) for f, v in centroid.items() if v >= min_weight}
        trimmed[label] = pruned
        kept_feats.update(pruned)

    lines = [HEADER, "IDF: dict[str, float] = {\n"]
    for feat in sorted(kept_feats):
        lines.append(f"    {feat!r}: {round(idf[feat], 5)},\n")
    lines.append("}\n\nCENTROIDS: dict[str, dict[str, float]] = {\n")
    for label in sorted(trimmed):
        lines.append(f"    {label!r}: {{\n")
        for feat in sorted(trimmed[label]):
            lines.append(f"        {feat!r}: {trimmed[label][feat]},\n")
        lines.append("    },\n")
    lines.append("}\n")
    OUTPUT.write_text("".join(lines), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(
        f"\n書き出し: {OUTPUT} ({size_kb:.0f} KB, "
        f"素性 {len(kept_feats)}, クラス {len(trimmed)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="言い換え分類器の学習と CV")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="この値未満のセントロイド重みは書き出さない",
    )
    parser.add_argument(
        "--cv-only", action="store_true", help="CV だけ実行しモデルを書き出さない"
    )
    args = parser.parse_args()

    pairs = collect_cases()
    print(f"教師データ: {len(pairs)} 件 / {len(set(l for _, l in pairs))} タスク\n")
    cross_validate(pairs, args.folds)

    if args.cv_only:
        return
    full = pairs + [(label, label) for label in {l for _, l in pairs}]
    centroids, idf = train(full)
    write_model(centroids, idf, args.min_weight)


if __name__ == "__main__":
    main()
