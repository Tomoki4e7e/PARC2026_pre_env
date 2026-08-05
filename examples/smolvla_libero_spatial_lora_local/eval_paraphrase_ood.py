#!/usr/bin/env python3
"""言い換え分類器の ID/OOD 分離性能を測る。

ID は 40 既知タスクの language 変種を task ごとに 5-fold 分割する。
OOD は分類器の候補外である LIBERO-90 の 90 task instruction。
top-1 score と top-1/top-2 margin の閾値を組み合わせ、未知タスクを既知タスクへ
誤写像しない保守的な設定を選ぶ。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from build_paraphrase_model import (
    BDDL_ROOT,
    LANGUAGE_RE,
    collect_cases,
    features,
    to_vector,
    train,
)


def ranked_scores(text: str, centroids, idf) -> tuple[str | None, float, float]:
    vector = to_vector(features(text), idf)
    scores = sorted(
        (
            (sum(value * centroid.get(feat, 0.0) for feat, value in vector.items()), label)
            for label, centroid in centroids.items()
        ),
        reverse=True,
    )
    if not scores:
        return None, 0.0, 0.0
    top_score, label = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    return label, top_score, top_score - second_score


def ood_instructions() -> list[str]:
    out = []
    for path in sorted((BDDL_ROOT / "libero_90").glob("*.bddl")):
        match = LANGUAGE_RE.search(path.read_text())
        if match:
            out.append(" ".join(match.group(1).split()))
        else:
            stem = re.sub(r"^[A-Z][A-Z0-9_]*_SCENE\d+_", "", path.stem)
            out.append(stem.replace("_", " "))
    return out


def main() -> None:
    by_label: dict[str, list[str]] = defaultdict(list)
    for text, label in collect_cases():
        by_label[label].append(text)

    id_results: list[tuple[bool, float, float]] = []
    ood = ood_instructions()
    # OOD は各 fold のモデルで採点し平均せず全件を数える（90 * 5）。
    ood_results: list[tuple[float, float]] = []
    for fold in range(5):
        train_pairs = []
        test_pairs = []
        for label, texts in by_label.items():
            for index, text in enumerate(texts):
                (test_pairs if index % 5 == fold else train_pairs).append((text, label))
            train_pairs.append((label, label))
        centroids, idf = train(train_pairs)
        for text, target in test_pairs:
            predicted, score, margin = ranked_scores(text, centroids, idf)
            id_results.append((predicted == target, score, margin))
        for text in ood:
            _predicted, score, margin = ranked_scores(text, centroids, idf)
            ood_results.append((score, margin))

    print(f"ID={len(id_results)} OOD={len(ood_results)}")
    print("score margin | ID accepted  ID correct  OOD false-accept")
    for score_min in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for margin_min in (0.00, 0.03, 0.05, 0.08, 0.10, 0.15):
            id_accept = [
                correct
                for correct, score, margin in id_results
                if score >= score_min and margin >= margin_min
            ]
            ood_accept = sum(
                score >= score_min and margin >= margin_min
                for score, margin in ood_results
            )
            accepted_rate = len(id_accept) / len(id_results)
            correct_rate = sum(id_accept) / len(id_results)
            false_rate = ood_accept / len(ood_results)
            if false_rate <= 0.02 or (
                score_min in (0.30, 0.40) and margin_min in (0.05, 0.10)
            ):
                print(
                    f"{score_min:4.2f}  {margin_min:4.2f} |"
                    f"   {accepted_rate:7.1%}    {correct_rate:7.1%}"
                    f"       {false_rate:7.1%}"
                )


if __name__ == "__main__":
    main()
