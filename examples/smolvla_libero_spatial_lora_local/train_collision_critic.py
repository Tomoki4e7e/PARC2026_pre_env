#!/usr/bin/env python3
"""Train/evaluate an action-conditioned critic from paired local rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "submission_template"))

from collision_critic import (CollisionCriticConfig, CollisionCriticNetwork,
                              stove_geometry_features)


def _load(paths: list[Path], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata_json"].item()))
            state = np.asarray(episode["observation.state"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            image = np.asarray(episode["observation.front"])
            collision = np.asarray(episode["collision"], dtype=np.bool_)
        context = metadata.get("collection_context", {})
        ordinal = int(context.get("init_state_ordinal", metadata["seed"]))
        collision_indices = np.flatnonzero(collision)
        first = int(collision_indices[0]) if len(collision_indices) else None
        # Post-contact observations cannot help prevention and would inflate metrics.
        keep = first if first is not None else len(state)
        for index in range(keep):
            visual = stove_geometry_features(image[index])
            features.append(np.concatenate([state[index], action[index], visual]))
            labels.append(float(first is not None and 0 < first - index <= horizon))
            groups.append(ordinal)
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32),
            np.asarray(groups, dtype=np.int64))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    return float(((positives[:, None] > negatives).mean()
                  + 0.5 * (positives[:, None] == negatives).mean()))


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    prediction = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    return {
        "samples": int(len(labels)), "positives": int(positive.sum()),
        "auc": _auc(labels, scores), "threshold": float(threshold),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _select_threshold(labels: np.ndarray, scores: np.ndarray, min_recall: float) -> float:
    candidates = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    feasible = []
    for threshold in candidates:
        metrics = _metrics(labels, scores, float(threshold))
        if metrics["recall"] >= min_recall:
            feasible.append((metrics["balanced_accuracy"], metrics["specificity"], threshold))
    return float(max(feasible)[2]) if feasible else 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsafe-dir", type=Path, action="append", required=True)
    parser.add_argument("--safe-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-ordinals", default="18,19")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--risk-margin", type=float, default=0.05)
    parser.add_argument(
        "--min-recall", type=float, default=0.50,
        help="minimum holdout danger recall; balanced accuracy selects within this gate",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(
        path for root in args.unsafe_dir for path in root.rglob("*.npz")
    ) + sorted(path for root in args.safe_dir for path in root.rglob("*.npz"))
    if not paths:
        raise RuntimeError("no paired RL episodes found")
    features, labels, groups = _load(paths, args.horizon)
    holdout = {int(value) for value in args.holdout_ordinals.split(",") if value.strip()}
    train_mask = ~np.isin(groups, sorted(holdout))
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("both train and holdout init groups are required")
    mean = features[train_mask].mean(0)
    std = features[train_mask].std(0).clip(1e-6)
    config = CollisionCriticConfig(
        input_mean=tuple(float(v) for v in mean), input_std=tuple(float(v) for v in std),
        risk_margin=args.risk_margin, horizon=args.horizon,
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CollisionCriticNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    x = torch.from_numpy((features - mean) / std)
    y = torch.from_numpy(labels)
    train_indices = np.flatnonzero(train_mask)
    positives = float(labels[train_mask].sum())
    negatives = float(train_mask.sum() - positives)
    positive_weight = torch.tensor(negatives / max(1.0, positives), device=device)
    generator = torch.Generator().manual_seed(args.seed)
    model.train()
    for step in range(args.steps):
        selected = train_indices[
            torch.randint(len(train_indices), (args.batch_size,), generator=generator).numpy()
        ]
        logits = model(x[selected].to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y[selected].to(device), pos_weight=positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)

    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(x.to(device))).cpu().numpy()
    threshold = _select_threshold(labels[validation_mask], scores[validation_mask], args.min_recall)
    config = CollisionCriticConfig(
        input_mean=config.input_mean, input_std=config.input_std,
        threshold=threshold, risk_margin=args.risk_margin, horizon=args.horizon,
    )
    report = {
        "split": {
            "train_ordinals": sorted(set(groups[train_mask].tolist())),
            "holdout_ordinals": sorted(set(groups[validation_mask].tolist())),
        },
        "train": _metrics(labels[train_mask], scores[train_mask], threshold),
        "holdout": _metrics(labels[validation_mask], scores[validation_mask], threshold),
        "visual_detection_rate": float(features[:, -8].mean()),
        "horizon": args.horizon, "risk_margin": args.risk_margin,
        "episodes": len(paths), "seed": args.seed,
        "input_dirs": [
            *[str(path) for path in args.unsafe_dir],
            *[str(path) for path in args.safe_dir],
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()},
              str(args.output_dir / "model.safetensors"))
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
