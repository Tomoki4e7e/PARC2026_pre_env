#!/usr/bin/env python3
"""LoRA アダプタをベース重みへマージし、推論用モデル一式を保存する。

Colab ノートブック §9 相当。train.py 完了後、または単独で実行できる。

使い方:
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py --stage1
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py --track1-stage2
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py --phase3
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py --stage1 --steps 20000
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from utils import (
    cached_or_downloaded_snapshot,
    copy_processors_from_model,
    ensure_work_dirs,
    set_hf_env,
)


def _canonicalize_visual_shapes(model_dir: Path) -> None:
    """LeRobot dataset metadata が HWC でも読み込みは CHW のため、推論設定を CHW に揃える。"""
    pre_path = model_dir / "policy_preprocessor.json"
    if pre_path.is_file():
        pre = json.loads(pre_path.read_text())
        for step in pre.get("steps", []):
            if step.get("registry_name") != "normalizer_processor":
                continue
            for key, feat in step["config"]["features"].items():
                if not key.startswith("observation.images."):
                    continue
                shape = feat["shape"]
                if len(shape) == 3 and shape[-1] in (1, 3, 4) and shape[0] != 3:
                    feat["shape"] = [shape[2], shape[0], shape[1]]
        pre_path.write_text(json.dumps(pre, indent=2) + "\n")

    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        for key, feat in cfg.get("input_features", {}).items():
            if not key.startswith("observation.images."):
                continue
            shape = feat.get("shape")
            if shape and len(shape) == 3 and shape[-1] in (1, 3, 4) and shape[0] != 3:
                feat["shape"] = [shape[2], shape[0], shape[1]]
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def resolve_base_model_local() -> Path:
    if config.BASE_MODEL_LOCAL is not None:
        base_model_local = Path(config.BASE_MODEL_LOCAL)
        if not (base_model_local / "model.safetensors").is_file():
            raise FileNotFoundError(f"Local base model not found: {base_model_local}")
        return base_model_local

    return cached_or_downloaded_snapshot(
        config.BASE_MODEL_REPO,
        config.BASE_MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "train_config.json",
            "policy_preprocessor.json",
            "policy_preprocessor*.safetensors",
            "policy_postprocessor.json",
            "policy_postprocessor*.safetensors",
        ],
        ignore_patterns=["README.md", "eval/**"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA アダプタをベース重みへマージ")
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="第1段階 multi-suite LoRA 出力をマージする",
    )
    parser.add_argument(
        "--track1-stage2",
        action="store_true",
        help="第2段階 Track 1 LoRA 出力をマージする",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="第3段階 hold-out object/goal LoRA 出力をマージする",
    )
    parser.add_argument(
        "--phase4",
        action="store_true",
        help="第4段階 LIBERO-90 多様タスク LoRA 出力をマージする",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="マージする checkpoint step（未指定時は config.STEPS）",
    )
    args = parser.parse_args()

    stage_flags = sum(
        int(x) for x in (args.stage1, args.track1_stage2, args.phase3, args.phase4)
    )
    if stage_flags > 1:
        parser.error(
            "--stage1 / --track1-stage2 / --phase3 / --phase4 は同時に指定できません"
        )

    if args.phase4:
        config.apply_phase4_config()
    elif args.phase3:
        config.apply_phase3_config()
    elif args.track1_stage2:
        config.apply_track1_stage2_config()
    elif args.stage1:
        config.apply_stage1_config()

    merge_steps = args.steps if args.steps is not None else config.STEPS

    ensure_work_dirs(config.WORK_DIR, config.HF_HOME, config.LEROBOT_CACHE)
    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)

    lerobot_src = config.LEROBOT_DIR / "src"
    if not lerobot_src.is_dir():
        raise RuntimeError("LeRobot が未セットアップです。setup_lerobot.sh を実行してください。")
    sys.path.insert(0, str(lerobot_src))

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint_dir = (
        config.OUTPUT_DIR
        / "checkpoints"
        / f"{merge_steps:06d}"
        / "pretrained_model"
    )
    if not (checkpoint_dir / "adapter_model.safetensors").is_file():
        checkpoints_root = config.OUTPUT_DIR / "checkpoints"
        if checkpoints_root.is_dir():
            candidates = sorted(
                int(path.name)
                for path in checkpoints_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
            if candidates:
                latest = candidates[-1]
                fallback = (
                    checkpoints_root / f"{latest:06d}" / "pretrained_model"
                )
                if (fallback / "adapter_model.safetensors").is_file():
                    print(
                        f"Checkpoint {merge_steps:06d} not found; "
                        f"using latest {latest:06d}"
                    )
                    checkpoint_dir = fallback
    if not (checkpoint_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Final adapter not found: {checkpoint_dir}")

    base_model_local = resolve_base_model_local()

    gc.collect()
    torch.cuda.empty_cache()

    merge_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    merge_config.device = "cpu"
    merge_config.pretrained_path = base_model_local
    merge_config.use_peft = False

    quiet_output = io.StringIO()
    with (
        contextlib.redirect_stdout(quiet_output),
        contextlib.redirect_stderr(quiet_output),
    ):
        base_policy = SmolVLAPolicy.from_pretrained(
            base_model_local,
            config=merge_config,
            strict=False,
        )
        peft_policy = PeftModel.from_pretrained(
            base_policy,
            checkpoint_dir,
            is_trainable=False,
            torch_device="cpu",
        )
        merged_policy = peft_policy.merge_and_unload(safe_merge=True)

    shutil.rmtree(config.MERGED_MODEL_DIR, ignore_errors=True)
    config.MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    merged_policy.config.use_peft = False
    merged_policy.config.pretrained_path = None
    merged_policy.config.push_to_hub = False
    merged_policy.config.repo_id = None
    merged_policy.config.device = None
    merged_policy.config.load_vlm_weights = False
    merged_policy.config.vlm_model_name = config.VLM_REPO

    merged_policy.save_pretrained(config.MERGED_MODEL_DIR)

    for pattern in [
        "policy_preprocessor.json",
        "policy_preprocessor*.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor*.safetensors",
    ]:
        for source_path in checkpoint_dir.glob(pattern):
            shutil.copy2(source_path, config.MERGED_MODEL_DIR / source_path.name)

    # Phase4 v2: 学習中に dataset stats を注入しても、最終は base normalizer を正とする。
    if (
        getattr(config, "PHASE4_PRESERVE_BASE_NORMALIZER", False)
        and args.phase4
        and (base_model_local / "policy_preprocessor.json").is_file()
    ):
        copy_processors_from_model(base_model_local, config.MERGED_MODEL_DIR)
        print(f"Preserved base normalizer from: {base_model_local}")

    _canonicalize_visual_shapes(config.MERGED_MODEL_DIR)

    merged_weights_path = config.MERGED_MODEL_DIR / "model.safetensors"
    with safe_open(merged_weights_path, framework="pt", device="cpu") as weights:
        if any("lora_" in key.lower() for key in weights.keys()):
            raise RuntimeError("LoRA parameters remain after merge.")

    del peft_policy, base_policy, merged_policy
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Merged model ready: {config.MERGED_MODEL_DIR}")
    print("提出用にコピーする例:")
    print(
        f"  cp -r {config.MERGED_MODEL_DIR} "
        "submission_template/model_weights/smolvla_spatial_lora"
    )


if __name__ == "__main__":
    main()
