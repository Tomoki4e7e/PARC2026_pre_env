#!/usr/bin/env python3
"""SmolVLA × LIBERO-plus multi-suite LoRA 追加学習（ローカル GPU 版）。

使い方:
    # 1) 環境構築（初回のみ）
    bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh

    # Phase 2: 第1段階 multi-suite 学習（spatial + object + goal, 740 ep）
    python examples/smolvla_libero_spatial_lora_local/train.py --stage1

    # spatial のみ（従来互換）
    python examples/smolvla_libero_spatial_lora_local/train.py --spatial-only

    # Track 1 弱点 4 タスク特化（第2段階 LoRA）
    python examples/smolvla_libero_spatial_lora_local/train.py --track1-stage2

    # Phase 3: hold-out object/goal 弱点 LoRA（stage2 merged ベース）
    python examples/smolvla_libero_spatial_lora_local/train.py --phase3

    # 3) マージのみ再実行
    python examples/smolvla_libero_spatial_lora_local/merge_lora.py
"""

from __future__ import annotations

import os

# huggingface_hub import 前に設定すること（XET はネットワーク不安定時に失敗しやすい）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from utils import (
    cached_or_downloaded_snapshot,
    ensure_work_dirs,
    prefetch_dataset,
    print_episode_selection_summary,
    run_quiet,
    select_training_episodes,
    set_hf_env,
)


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        version = torch.__version__
        hint = ""
        if "+cu130" in version or "+cu126" in version:
            hint = (
                f"\n\ntorch {version} は現在の NVIDIA ドライバと非互換の可能性があります。"
                "\nLeRobot インストール後に torch が自動更新された場合:"
                "\n  bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh"
            )
        raise RuntimeError(
            "CUDA GPU が利用できません。"
            f" (torch={version})"
            f"{hint}"
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})")


def _require_lerobot() -> Path:
    lerobot_dir = config.LEROBOT_DIR
    lerobot_src = lerobot_dir / "src"
    if not (lerobot_src / "lerobot" / "scripts" / "lerobot_train.py").is_file():
        raise RuntimeError(
            "LeRobot が未セットアップです。先に setup_lerobot.sh を実行してください。"
        )
    return lerobot_src


def resolve_base_model() -> Path:
    if config.BASE_MODEL_LOCAL is not None:
        base_model_local = Path(config.BASE_MODEL_LOCAL)
        if not (base_model_local / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Local base model not found: {base_model_local}\n"
                "第2段階 LoRA には旧 merged モデルが必要です。"
                "先に spatial-only 学習 + merge を完了してください。"
            )
        print(f"Base model (local): {base_model_local}")
        return base_model_local

    return download_base_model()


def download_base_model() -> Path:
    base_model_local = cached_or_downloaded_snapshot(
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
    if not (base_model_local / "model.safetensors").is_file():
        raise FileNotFoundError(f"Base model not found: {base_model_local}")
    print(f"Base model: {base_model_local}")
    return base_model_local


def resolve_train_suites(args: argparse.Namespace) -> list[config.TrainSuiteSpec]:
    if args.spatial_only:
        return [
            config.TrainSuiteSpec(
                "spatial",
                config.SPATIAL_TASK_NAMES,
                config.TRAIN_EPISODES_PER_TASK,
            )
        ]
    enabled = {name.strip() for name in args.suites.split(",") if name.strip()}
    suites = [suite for suite in config.TRAIN_SUITES if suite.name in enabled]
    if not suites:
        available = ", ".join(suite.name for suite in config.TRAIN_SUITES)
        raise ValueError(f"No suites matched {args.suites!r}. Available: {available}")
    return list(suites)


def build_train_command(
    base_model_local: Path,
    episode_indices: list[int],
    *,
    job_name: str,
) -> list[str]:
    episodes_json = "[" + ",".join(map(str, episode_indices)) + "]"
    save_freq = max(config.LOG_FREQ, config.STEPS // 4)
    return [
        "lerobot-train",
        f"--policy.path={base_model_local}",
        f"--policy.vlm_model_name={config.VLM_REPO}",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        f"--policy.freeze_vision_encoder={str(config.FREEZE_VISION_ENCODER).lower()}",
        f"--policy.train_expert_only={str(config.TRAIN_EXPERT_ONLY).lower()}",
        f"--policy.optimizer_lr={config.LEARNING_RATE}",
        f"--policy.scheduler_decay_lr={config.FINAL_LEARNING_RATE}",
        f"--policy.scheduler_warmup_steps={config.WARMUP_STEPS}",
        f"--policy.scheduler_decay_steps={config.STEPS}",
        f"--dataset.repo_id={config.DATASET_REPO}",
        f"--dataset.revision={config.DATASET_REVISION}",
        f"--dataset.episodes={episodes_json}",
        "--dataset.use_imagenet_stats=false",
        f"--dataset.video_backend={config.VIDEO_BACKEND}",
        f"--output_dir={config.OUTPUT_DIR}",
        f"--job_name={job_name}",
        f"--steps={config.STEPS}",
        f"--batch_size={config.BATCH_SIZE}",
        "--num_workers=0",
        "--persistent_workers=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--seed={config.SEED}",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={config.LOG_FREQ}",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        f"--peft.r={config.LORA_R}",
        f"--peft.lora_alpha={config.LORA_ALPHA}",
    ]


def run_training(
    base_model_local: Path,
    episode_indices: list[int],
    lerobot_src: Path,
    *,
    job_name: str,
) -> None:
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    training_env = os.environ.copy()
    training_env["PYTHONPATH"] = (
        str(lerobot_src) + os.pathsep + training_env.get("PYTHONPATH", "")
    )
    training_env["ACCELERATE_MIXED_PRECISION"] = mixed_precision
    training_env["PYTHONUNBUFFERED"] = "1"
    training_env["TQDM_DISABLE"] = "1"
    training_env["PYTHONWARNINGS"] = "ignore"
    training_env["HF_HUB_DISABLE_XET"] = "1"
    training_env["HF_HOME"] = os.environ.get("HF_HOME", "")
    training_env["HF_LEROBOT_HOME"] = os.environ.get("HF_LEROBOT_HOME", "")

    shutil.rmtree(config.OUTPUT_DIR, ignore_errors=True)

    command = build_train_command(base_model_local, episode_indices, job_name=job_name)
    print("Preparing data and starting training...")
    print(f"  output: {config.OUTPUT_DIR}")
    print(f"  mixed precision: {mixed_precision}")
    print(f"  suites: freeze_vision_encoder={config.FREEZE_VISION_ENCODER}")

    process = subprocess.Popen(
        command,
        cwd=config.LEROBOT_DIR,
        env=training_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    recent_lines: deque[str] = deque(maxlen=80)
    report_step = config.LOG_FREQ
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.replace("\r", "").strip()
        if not line:
            continue
        recent_lines.append(line)
        if "step:" in line and "loss:" in line:
            loss_match = re.search(r"loss:([0-9.eE+-]+)", line)
            lr_match = re.search(r"lr:([0-9.eE+-]+)", line)
            loss = loss_match.group(1) if loss_match else "n/a"
            lr = lr_match.group(1) if lr_match else "n/a"
            print(f"step {report_step:4d}/{config.STEPS}  loss={loss}  lr={lr}")
            report_step += config.LOG_FREQ

    return_code = process.wait()
    if return_code != 0:
        print("\n".join(recent_lines))
        raise RuntimeError(f"Training failed: {return_code}")
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA multi-suite LoRA 学習")
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="学習後の LoRA マージをスキップする",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="ログ出力先（既定: work/train.log）",
    )
    parser.add_argument(
        "--suites",
        default="spatial,object,goal",
        help="学習 suite（カンマ区切り）: spatial,object,goal",
    )
    parser.add_argument(
        "--spatial-only",
        action="store_true",
        help="spatial のみ学習（従来互換）",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="第1段階 multi-suite LoRA（spatial+object+goal, 既定と同じ）",
    )
    parser.add_argument(
        "--track1-stage2",
        action="store_true",
        help="Track 1 弱点 4 タスク特化 LoRA（旧 merged をベース）",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="hold-out object/goal 弱点 LoRA（stage2 merged をベース）",
    )
    args = parser.parse_args()

    stage_flags = sum(
        int(x)
        for x in (args.stage1, args.track1_stage2, args.phase3, args.spatial_only)
    )
    if stage_flags > 1:
        parser.error(
            "--stage1 / --track1-stage2 / --phase3 / --spatial-only は同時に指定できません"
        )
    if args.stage1 and args.track1_stage2:
        parser.error("--stage1 と --track1-stage2 は同時に指定できません")
    if args.track1_stage2 and args.spatial_only:
        parser.error("--track1-stage2 と --spatial-only は同時に指定できません")
    if args.stage1 and args.spatial_only:
        parser.error("--stage1 と --spatial-only は同時に指定できません")

    log_path = args.log_file or (config.WORK_DIR / "train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data: str) -> None:
            for stream in self._streams:
                stream.write(data)
                stream.flush()

        def flush(self) -> None:
            for stream in self._streams:
                stream.flush()

    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"\n=== train.py started (log: {log_path}) ===")

    ensure_work_dirs(config.WORK_DIR, config.HF_HOME, config.LEROBOT_CACHE)
    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)
    _require_gpu()
    lerobot_src = _require_lerobot()

    sys.path.insert(0, str(lerobot_src))

    if args.phase3:
        config.apply_phase3_config()
        suites = list(config.PHASE3_TRAIN_SUITES)
        task_episode_overrides = config.PHASE3_TASK_EPISODE_OVERRIDES
        job_name = "smolvla_libero_plus_phase3_holdout_lora"
    elif args.track1_stage2:
        config.apply_track1_stage2_config()
        suites = list(config.TRACK1_STAGE2_TRAIN_SUITES)
        task_episode_overrides = config.TRACK1_STAGE2_TASK_EPISODE_OVERRIDES
        job_name = "smolvla_libero_plus_track1_stage2_lora"
    else:
        if args.stage1 or not args.spatial_only:
            config.apply_stage1_config()
        suites = resolve_train_suites(args)
        task_episode_overrides = config.TASK_EPISODE_OVERRIDES
        if args.spatial_only:
            config.OUTPUT_DIR = config.LEGACY_OUTPUT_DIR
            config.MERGED_MODEL_DIR = config.LEGACY_MERGED_MODEL_DIR
            job_name = "smolvla_libero_plus_spatial_lora"
        else:
            job_name = "smolvla_libero_plus_multisuite_lora"

    episode_indices, selected_by_task = select_training_episodes(
        config.DATASET_REPO,
        config.DATASET_REVISION,
        suites,
        task_episode_overrides=task_episode_overrides,
    )
    print(f"Training suites: {[suite.name for suite in suites]}")
    print_episode_selection_summary(selected_by_task, suites)

    base_model_local = resolve_base_model()
    print(f"Prefetching dataset ({len(episode_indices)} episodes)...")
    prefetch_dataset(
        config.DATASET_REPO,
        config.DATASET_REVISION,
        episode_indices,
        lerobot_src,
    )
    run_training(base_model_local, episode_indices, lerobot_src, job_name=job_name)

    if not args.skip_merge:
        merge_script = SCRIPT_DIR / "merge_lora.py"
        merge_cmd = [sys.executable, str(merge_script)]
        if args.phase3:
            merge_cmd.append("--phase3")
        elif args.track1_stage2:
            merge_cmd.append("--track1-stage2")
        elif args.stage1 or not args.spatial_only:
            merge_cmd.append("--stage1")
        run_quiet(merge_cmd)


if __name__ == "__main__":
    main()
