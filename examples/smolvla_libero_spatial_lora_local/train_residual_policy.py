#!/usr/bin/env python3
"""Train the common residual initializer from locally generated RL NPZ files.

Each input episode must contain ``state``, ``base_action`` and executed ``action``.
The frozen base policy is never loaded or updated here.  Reward-to-go weighting
provides a conservative reward-weighted BC initialization before Offline RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "submission_template"))

from residual_policy import ResidualActionNetwork, ResidualPolicyConfig


def _array(data: np.lib.npyio.NpzFile, preferred: str, fallback: str | None = None) -> np.ndarray:
    if preferred in data:
        return np.asarray(data[preferred])
    if fallback is not None and fallback in data:
        return np.asarray(data[fallback])
    raise KeyError(preferred)


def load_episodes(paths: list[Path], temperature: float) -> tuple[np.ndarray, ...]:
    states, bases, targets, returns_to_go = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            state = _array(episode, "observation.state", "state").astype(np.float32)
            base = _array(episode, "base_action").astype(np.float32)
            action = _array(episode, "action").astype(np.float32)
            reward = _array(episode, "reward").astype(np.float32)
        if not (len(state) == len(base) == len(action) == len(reward)):
            raise ValueError(f"transition length mismatch: {path}")
        returns = np.cumsum(reward[::-1])[::-1].copy()
        states.append(state)
        bases.append(base)
        targets.append(action[:, :6] - base[:, :6])
        returns_to_go.append(returns)
    concatenated = [np.concatenate(items, axis=0) for items in (states, bases, targets)]
    returns_array = np.concatenate(returns_to_go, axis=0)
    centered = returns_array - np.median(returns_array)
    sample_weight = np.exp(np.clip(centered / temperature, -5.0, 5.0)).astype(np.float32)
    return (*concatenated, sample_weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-dir", type=Path, required=True,
        help="Frozen SmolVLA used to produce base_action annotations",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(path for root in args.input_dir for path in root.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"no RL episodes found: {args.input_dir}")
    state, base, target, sample_weight = load_episodes(paths, args.temperature)
    state_mean, state_std = state.mean(0), state.std(0).clip(1.0e-6)
    action_mean, action_std = base.mean(0), base.std(0).clip(1.0e-6)
    config = ResidualPolicyConfig(
        state_mean=tuple(float(v) for v in state_mean),
        state_std=tuple(float(v) for v in state_std),
        base_action_mean=tuple(float(v) for v in action_mean),
        base_action_std=tuple(float(v) for v in action_std),
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualActionNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    state_t = torch.from_numpy((state - state_mean) / state_std)
    base_t = torch.from_numpy((base - action_mean) / action_std)
    limits = np.asarray([config.translation_limit] * 3 + [config.rotation_limit] * 3)
    target_t = torch.from_numpy(np.clip(target / limits, -1.0, 1.0).astype(np.float32))
    weight_t = torch.from_numpy(sample_weight / sample_weight.mean())
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        indices = torch.randint(len(state_t), (args.batch_size,), generator=generator)
        inputs = torch.cat([state_t[indices], base_t[indices]], dim=1).to(device)
        expected = target_t[indices].to(device)
        batch_weight = weight_t[indices].to(device)
        prediction = model(inputs)
        loss = ((prediction - expected).square().mean(1) * batch_weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()},
              str(args.output_dir / "model.safetensors"))
    manifest = {
        "method": "reward_weighted_behavior_cloning_residual_initializer",
        "base_policy_frozen": True,
        "episodes": len(paths),
        "transitions": int(len(state)),
        "steps": args.steps,
        "temperature": args.temperature,
        "seed": args.seed,
        "input_dirs": [str(path) for path in args.input_dir],
        "base_model_dir": str(args.base_model_dir),
        "base_model_sha256": hashlib.sha256(
            (args.base_model_dir / "model.safetensors").read_bytes()
        ).hexdigest(),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
