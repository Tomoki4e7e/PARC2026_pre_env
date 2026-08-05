#!/usr/bin/env python3
"""言い換え instruction の写像精度を、SmolVLM の言語モデル埋め込みで測る。

eval_paraphrase_matching.py（語彙ベース）は約 60% だった。SmolVLA が内部に持つ
言語モデルの埋め込みで意味的に照合すればどこまで上がるかを確認する。
提出物に追加の重みは不要（policy 内の VLM を使えるため）。

使い方（CPU 実行、GPU は評価サーバーが使用中でも可）:
    python examples/smolvla_libero_spatial_lora_local/eval_paraphrase_matching_embed.py
    python examples/smolvla_libero_spatial_lora_local/eval_paraphrase_matching_embed.py --limit 300
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval_paraphrase_matching import canonical_tasks, paraphrase_samples  # noqa: E402


def embed(texts: list[str], model, tokenizer, device: str, batch_size: int):
    import torch

    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=48, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outputs.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="埋め込みによる言い換え写像精度")
    parser.add_argument("--limit", type=int, default=None, help="サンプル数上限")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    import config

    repo = config.VLM_REPO
    os.environ.setdefault("HF_HOME", str(config.HF_HOME))

    tokenizer = AutoTokenizer.from_pretrained(repo)
    full = AutoModel.from_pretrained(repo, dtype=torch.float32)
    text_model = getattr(getattr(full, "model", full), "text_model", None)
    if text_model is None:
        text_model = getattr(full, "text_model", full)
    text_model = text_model.to(args.device).eval()

    candidates = sorted(canonical_tasks())
    samples = paraphrase_samples()
    if args.limit:
        step = max(len(samples) // args.limit, 1)
        samples = samples[::step][: args.limit]

    print(f"候補 {len(candidates)} 件 / サンプル {len(samples)} 件, device={args.device}")

    candidate_emb = embed(candidates, text_model, tokenizer, args.device, args.batch_size)
    sample_emb = embed(
        [text for text, _ in samples], text_model, tokenizer, args.device, args.batch_size
    )

    similarity = sample_emb @ candidate_emb.T
    best = similarity.argmax(dim=1)
    top_scores = similarity.gather(1, best.unsqueeze(1)).squeeze(1)

    correct = 0
    mistakes = []
    for index, (text, truth) in enumerate(samples):
        prediction = candidates[int(best[index])]
        if prediction == truth:
            correct += 1
        elif len(mistakes) < 5:
            mistakes.append((text, truth, prediction, float(top_scores[index])))

    print(f"埋め込み: 正解率 {correct / len(samples):.1%} ({correct}/{len(samples)})")
    for text, truth, prediction, score in mistakes:
        print(f"    入力: {text}")
        print(f"    正解: {truth}")
        print(f"    予測: {prediction}  (cos={score:.3f})")

    # 閾値を上げた場合の精度と適用率（誤写像を避けたい場合の指標）
    for threshold in (0.8, 0.85, 0.9, 0.95):
        mask = top_scores >= threshold
        applied = int(mask.sum())
        if applied == 0:
            print(f"    閾値 {threshold}: 適用 0 件")
            continue
        hits = sum(
            1
            for index, (_, truth) in enumerate(samples)
            if mask[index] and candidates[int(best[index])] == truth
        )
        print(
            f"    閾値 {threshold}: 適用率 {applied / len(samples):.1%}, "
            f"適用分の正解率 {hits / applied:.1%}"
        )


if __name__ == "__main__":
    main()
