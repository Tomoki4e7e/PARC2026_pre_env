#!/usr/bin/env python3
"""Train a ramekin critic from same-observation safe/collision action pairs."""

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

from collision_critic import CollisionCriticConfig, CollisionCriticNetwork


def grasp_index(actions: np.ndarray) -> int:
    indices = np.flatnonzero(actions[:, -1] > 0.25)
    return int(indices[0]) if len(indices) else min(35, len(actions) - 1)


def load_pairs(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    records: list[dict] = []
    group_ids: dict[tuple[str, int], int] = {}
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as safe:
            metadata = json.loads(str(safe["metadata_json"].item()))
            pair = metadata.get("preference_pair", {})
            if not (metadata.get("success") and not metadata.get("collided")
                    and pair.get("same_initial_state")):
                continue
            task = str(metadata["task_name"])
            ordinal = int(pair["init_state_ordinal"])
            unsafe_path = str(pair["unsafe"])
            # Multiple candidate searches can rediscover the exact same trajectory.
            fingerprint = (task, ordinal, str(np.asarray(safe["action"]).tobytes()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            state = np.asarray(safe["observation.state"], dtype=np.float32)
            chosen = np.asarray(safe["action"], dtype=np.float32)
            chosen_grasp = grasp_index(chosen)
            with np.load(unsafe_path, allow_pickle=False) as unsafe:
                rejected = np.asarray(unsafe["action"], dtype=np.float32)
                collision = np.asarray(unsafe["collision"], dtype=np.bool_)
            collision_indices = np.flatnonzero(collision)
            if not len(collision_indices):
                raise ValueError(f"paired unsafe rollout has no collision: {unsafe_path}")
            collision_step = int(collision_indices[0])
            group = group_ids.setdefault((task, ordinal), len(group_ids))
            deltas = []
            for index in range(chosen_grasp + 1):
                rejected_index = min(
                    round(index * collision_step / max(chosen_grasp, 1)), len(rejected) - 1
                )
                safe_action = chosen[index]
                unsafe_action = rejected[rejected_index]
                # Same observation on both sides prevents state-only shortcuts.
                for alpha, label in ((0.0, 0.0), (0.33, 0.33), (0.67, 0.67), (1.0, 1.0)):
                    action = safe_action + alpha * (unsafe_action - safe_action)
                    features.append(np.concatenate([state[index], action]))
                    labels.append(label)
                    groups.append(group)
                deltas.append((index, chosen_grasp, safe_action - unsafe_action))
            records.append({
                "safe": str(path.resolve()), "unsafe": str(Path(unsafe_path).resolve()),
                "task": task, "ordinal": ordinal, "chosen_grasp": chosen_grasp,
                "collision_step": collision_step, "deltas": deltas,
            })
    if not records:
        raise RuntimeError(f"no verified preference pairs under {root}")
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32),
            np.asarray(groups, dtype=np.int64), records)


def pair_metrics(model: CollisionCriticNetwork, x: torch.Tensor, records: list[dict],
                 mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict:
    chosen_scores, rejected_scores = [], []
    with torch.inference_mode():
        for record in records:
            with np.load(record["safe"], allow_pickle=False) as safe, np.load(
                    record["unsafe"], allow_pickle=False) as unsafe:
                states = np.asarray(safe["observation.state"], dtype=np.float32)
                chosen = np.asarray(safe["action"], dtype=np.float32)
                rejected = np.asarray(unsafe["action"], dtype=np.float32)
                for index in range(record["chosen_grasp"] + 1):
                    rejected_index = min(round(index * record["collision_step"] /
                                               max(record["chosen_grasp"], 1)), len(rejected) - 1)
                    pair = np.stack([
                        np.concatenate([states[index], chosen[index]]),
                        np.concatenate([states[index], rejected[rejected_index]]),
                    ])
                    inputs = torch.as_tensor((pair - mean) / std, device=device)
                    scores = torch.sigmoid(model(inputs)).cpu().numpy()
                    chosen_scores.append(float(scores[0]))
                    rejected_scores.append(float(scores[1]))
    chosen_array, rejected_array = np.asarray(chosen_scores), np.asarray(rejected_scores)
    return {
        "pairs": len(chosen_array),
        "ranking_accuracy": float(np.mean(rejected_array > chosen_array)),
        "chosen_risk_mean": float(chosen_array.mean()),
        "rejected_risk_mean": float(rejected_array.mean()),
        "risk_gap_mean": float((rejected_array - chosen_array).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-margin", type=float, default=0.05)
    args = parser.parse_args()

    features, labels, groups, records = load_pairs(args.pair_root)
    mean = features.mean(0)
    std = features.std(0).clip(1e-4)
    # Empirical safe-minus-unsafe corrections, aligned to the online episode phases.
    phase_deltas = []
    for lower, upper in ((0.0, 0.4), (0.4, 0.7), (0.7, 1.01)):
        values = [delta for record in records for index, grasp, delta in record["deltas"]
                  if lower <= index / max(grasp, 1) < upper]
        phase_deltas.append(np.mean(values, axis=0).astype(np.float32))

    config = CollisionCriticConfig(
        visual_dim=0, input_mean=tuple(float(v) for v in mean),
        input_std=tuple(float(v) for v in std), feature_mode="state_action",
        task_gate="pick up the black bowl on the ramekin and place it on the plate",
        threshold=0.50, risk_margin=args.min_margin, shield_start=8, shield_end=42,
        phase_boundaries=(15, 27),
        candidate_deltas=tuple(tuple(float(v) for v in row) for row in phase_deltas),
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CollisionCriticNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    x = torch.from_numpy((features - mean) / std)
    y = torch.from_numpy(labels)
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        selected = torch.randint(len(x), (args.batch_size,), generator=generator)
        logits = model(x[selected].to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)

    model.eval()
    metrics = pair_metrics(model, x, records, mean, std, device)
    # Threshold is the midpoint of mean paired risks; the margin still gates action edits.
    config = CollisionCriticConfig(**{
        **config.__dict__,
        "threshold": 0.5 * (metrics["chosen_risk_mean"] + metrics["rejected_risk_mean"]),
    })
    report = {
        "protocol": "same_observation_ramekin_action_critic",
        "device": str(device), "seed": args.seed, "steps": args.steps,
        "unique_safe_trajectories": len(records), "init_groups": int(len(set(groups))),
        "samples": int(len(features)), "metrics": metrics,
        "phase_boundaries": list(config.phase_boundaries),
        "candidate_deltas": [list(row) for row in config.candidate_deltas],
        "threshold": config.threshold, "risk_margin": config.risk_margin,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()},
              str(args.output_dir / "model.safetensors"))
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
