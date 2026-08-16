#!/usr/bin/env python3
"""Train a conservative visual-language gated residual on successful records only."""

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

from residual_policy import (
    GatedResidualPolicyConfig,
    VisualLanguageGatedResidualNetwork,
    language_features,
    visual_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gate-penalty", type=float, default=0.02)
    parser.add_argument(
        "--failure-negative-gate",
        action="store_true",
        help="Use failed transitions as explicit gate=0 negatives with balanced batches",
    )
    parser.add_argument("--gate-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    excluded = 0
    for root in args.input_dir:
        for path in sorted(root.rglob("*.npz")):
            with np.load(path, allow_pickle=False) as episode:
                metadata = json.loads(str(episode["metadata_json"].item()))
            if not args.failure_negative_gate and not bool(metadata.get("success")):
                excluded += 1
                continue
            records.append((path, metadata))
    if not records:
        raise RuntimeError("no successful trajectories found")

    states, bases, targets, visuals, languages, gate_targets, success_masks = [], [], [], [], [], [], []
    corrective = 0
    for path, metadata in records:
        corrective += int(bool(metadata.get("success")) and "corrective" in str(path).lower())
        with np.load(path, allow_pickle=False) as episode:
            state = np.asarray(episode["observation.state"], dtype=np.float32)
            base = np.asarray(episode["base_action"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            front = np.asarray(episode["observation.front"])
            wrist = np.asarray(episode["observation.wrist"])
        states.append(state)
        bases.append(base)
        successful = bool(metadata.get("success"))
        targets.append(action[:, :6] - base[:, :6] if successful else np.zeros((len(state), 6), dtype=np.float32))
        visuals.append(np.stack([visual_features(f, w) for f, w in zip(front, wrist)]))
        lang = language_features(metadata.get("instruction", ""))
        languages.append(np.repeat(lang[None], len(state), axis=0))
        gate_targets.append(np.full((len(state), 1), float(successful), dtype=np.float32))
        success_masks.append(np.full(len(state), successful, dtype=bool))

    state = np.concatenate(states); base = np.concatenate(bases)
    target = np.concatenate(targets); visual = np.concatenate(visuals)
    language = np.concatenate(languages)
    gate_target = np.concatenate(gate_targets)
    success_mask = np.concatenate(success_masks)
    state_mean, state_std = state.mean(0), state.std(0).clip(1e-6)
    action_mean, action_std = base.mean(0), base.std(0).clip(1e-6)
    config = GatedResidualPolicyConfig(
        state_mean=tuple(float(x) for x in state_mean),
        state_std=tuple(float(x) for x in state_std),
        base_action_mean=tuple(float(x) for x in action_mean),
        base_action_std=tuple(float(x) for x in action_std),
    )
    inputs = np.concatenate([
        (state-state_mean)/state_std, (base-action_mean)/action_std, visual, language
    ], axis=1).astype(np.float32)
    limits = np.asarray([config.translation_limit]*3+[config.rotation_limit]*3, dtype=np.float32)
    expected = np.clip(target/limits, -1, 1).astype(np.float32)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualLanguageGatedResidualNetwork(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    input_t=torch.from_numpy(inputs); expected_t=torch.from_numpy(expected)
    gate_target_t=torch.from_numpy(gate_target); success_mask_t=torch.from_numpy(success_mask)
    generator=torch.Generator().manual_seed(args.seed)
    positive_indices=torch.where(success_mask_t)[0]
    negative_indices=torch.where(~success_mask_t)[0]
    if args.failure_negative_gate and (len(positive_indices) == 0 or len(negative_indices) == 0):
        raise RuntimeError("--failure-negative-gate requires both successful and failed transitions")
    for step in range(args.steps):
        if args.failure_negative_gate:
            positive_count=args.batch_size//2
            index=torch.cat([
                positive_indices[torch.randint(len(positive_indices),(positive_count,),generator=generator)],
                negative_indices[torch.randint(len(negative_indices),(args.batch_size-positive_count,),generator=generator)],
            ])
        else:
            index=torch.randint(len(input_t),(args.batch_size,),generator=generator)
        direction,gate=model(input_t[index].to(device))
        prediction=direction*gate
        if args.failure_negative_gate:
            positive_batch=success_mask_t[index].to(device)
            correction_loss=(prediction[positive_batch]-expected_t[index].to(device)[positive_batch]).square().mean()
            gate_loss=torch.nn.functional.binary_cross_entropy(gate,gate_target_t[index].to(device))
            loss=correction_loss+args.gate_loss_weight*gate_loss
        else:
            correction_loss=(prediction-expected_t[index].to(device)).square().mean()
            gate_loss=args.gate_penalty*gate.mean()
            loss=correction_loss+gate_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%500==0 or step+1==args.steps:
            if args.failure_negative_gate:
                print(f"step={step+1} loss={loss.item():.6f} correction={correction_loss.item():.6f} gate_loss={gate_loss.item():.6f} pos_gate={gate[positive_batch].mean().item():.4f} neg_gate={gate[~positive_batch].mean().item():.4f}",flush=True)
            else:
                print(f"step={step+1} loss={loss.item():.6f} gate={gate.mean().item():.4f}",flush=True)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"config.json").write_text(json.dumps(config.to_dict(),indent=2)+"\n")
    save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(args.output_dir/"model.safetensors"))
    manifest={
        "method":"failure_negative_visual_language_gated_residual" if args.failure_negative_gate else "successful_corrective_visual_language_gated_residual",
        "episodes":len(records),"successful_episodes":sum(bool(metadata.get("success")) for _, metadata in records),
        "failed_negative_episodes":sum(not bool(metadata.get("success")) for _, metadata in records),
        "excluded_failure_episodes":excluded,
        "corrective_success_episodes":corrective,"transitions":int(len(state)),
        "successful_transitions":int(success_mask.sum()),"failed_negative_transitions":int((~success_mask).sum()),
        "steps":args.steps,"gate_penalty":args.gate_penalty,"gate_loss_weight":args.gate_loss_weight,"seed":args.seed,
        "input_dirs":[str(x) for x in args.input_dir],
        "base_model_dir":str(args.base_model_dir),
        "base_model_sha256":hashlib.sha256((args.base_model_dir/"model.safetensors").read_bytes()).hexdigest(),
    }
    (args.output_dir/"training_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
