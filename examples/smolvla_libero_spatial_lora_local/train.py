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

    # Phase 4: LIBERO-90 多様タスク + replay LoRA（OOD early stopping 用）
    python examples/smolvla_libero_spatial_lora_local/train.py --phase4 --skip-merge

    # Phase 5: 品質フィルタ（低 jerk / 良 SPARC / 低回転）デモでの追加 LoRA
    python examples/smolvla_libero_spatial_lora_local/score_episode_quality.py --phase5
    python examples/smolvla_libero_spatial_lora_local/train.py --phase5 --skip-merge
    python examples/smolvla_libero_spatial_lora_local/run_phase4_proxy_select.py --phase5

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
    inject_model_stats_into_dataset,
    prefetch_dataset,
    print_episode_selection_summary,
    restore_dataset_stats_backup,
    run_quiet,
    select_phase4_mixture_episodes,
    select_training_episodes,
    set_hf_env,
)


def load_phase4_episode_indices() -> list[int]:
    import json

    manifest_path = config.PHASE4_MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Phase4 manifest がありません: {manifest_path}\n"
            "先に prepare_diverse_retrain_data.py --prepare を実行してください。"
        )
    if config.PHASE4_DATASET_ROOT is None or not (
        config.PHASE4_DATASET_ROOT / "meta" / "info.json"
    ).is_file():
        raise FileNotFoundError(
            f"Phase4 dataset がありません: {config.PHASE4_DATASET_ROOT}\n"
            "先に prepare_diverse_retrain_data.py --prepare を実行してください。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes, summary = select_phase4_mixture_episodes(
        manifest,
        target_replay_fraction=config.PHASE4_TARGET_REPLAY_FRACTION,
    )
    if not episodes:
        raise RuntimeError("Phase4 manifest に学習 episode がありません。")
    print(
        "Phase4 mixture (replay-up): "
        f"diverse {summary['diverse_kept']}/{summary['diverse_available']}, "
        f"replay {summary['replay_kept']}, total {summary['total']}, "
        f"replay_frac={summary['replay_fraction']:.3f} "
        f"(target={summary['target_replay_fraction']:.3f})"
    )
    return episodes


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
    save_freq = (
        config.SAVE_FREQ
        if config.SAVE_FREQ is not None
        else max(config.LOG_FREQ, config.STEPS // 4)
    )
    command = [
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
    if getattr(config, "IMAGE_AUGMENTATION_ENABLED", False):
        command.extend(
            [
                "--dataset.image_transforms.enable=true",
                "--dataset.image_transforms.max_num_transforms=4",
                "--dataset.image_transforms.random_order=true",
            ]
        )
    if config.DATASET_ROOT is not None:
        command.append(f"--dataset.root={config.DATASET_ROOT}")
    elif config.DATASET_REVISION is not None:
        command.append(f"--dataset.revision={config.DATASET_REVISION}")
    return command


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
    # conda activate なしで python 直実行しても lerobot-train を見つけられるようにする
    conda_bin = Path(sys.executable).resolve().parent
    path_parts = training_env.get("PATH", "").split(os.pathsep)
    if str(conda_bin) not in path_parts:
        training_env["PATH"] = str(conda_bin) + os.pathsep + training_env.get("PATH", "")

    shutil.rmtree(config.OUTPUT_DIR, ignore_errors=True)

    # 絶対パスで起動（PATH 依存をさらに減らす）
    lerobot_train = conda_bin / "lerobot-train"
    command = build_train_command(base_model_local, episode_indices, job_name=job_name)
    if lerobot_train.is_file():
        command[0] = str(lerobot_train)
    print("Preparing data and starting training...")
    print(f"  output: {config.OUTPUT_DIR}")
    print(f"  mixed precision: {mixed_precision}")
    print(f"  suites: freeze_vision_encoder={config.FREEZE_VISION_ENCODER}")

    preserve_normalizer = (
        getattr(config, "PHASE4_PRESERVE_BASE_NORMALIZER", False)
        and config.DATASET_ROOT is not None
        and (base_model_local / "policy_preprocessor.json").is_file()
    )
    if preserve_normalizer:
        backup = inject_model_stats_into_dataset(
            config.DATASET_ROOT, base_model_local
        )
        print(f"  normalizer: injected base stats into dataset (backup={backup.name})")

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

    try:
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
    finally:
        if preserve_normalizer:
            restore_dataset_stats_backup(config.DATASET_ROOT)
            print("  normalizer: restored dataset stats backup")


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
        "--augmentation",
        action="store_true",
        help="stage2 ベースの color/contrast/gamma/texture 複合 Augmentation 追加学習",
    )
    parser.add_argument(
        "--perturbation-replay",
        action="store_true",
        help="table/light 実レンダリング50:50 replayの低LR LoRA",
    )
    parser.add_argument(
        "--reward-weighted-bc",
        action="store_true",
        help="ローカルRLログから構築した報酬重み付きBCデータで低LR LoRA",
    )
    parser.add_argument(
        "--stove-collision",
        action="store_true",
        help="非衝突stove replay重点配分の低LR LoRA",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="hold-out object/goal 弱点 LoRA（stage2 merged をベース）",
    )
    parser.add_argument(
        "--phase4",
        action="store_true",
        help="LIBERO-90 多様タスク + replay LoRA（ローカル集約データセット）",
    )
    parser.add_argument(
        "--phase5",
        action="store_true",
        help="品質フィルタ済みデモでの追加 LoRA（jerk/SPARC/回転総量）",
    )
    parser.add_argument(
        "--quality-cache",
        type=Path,
        default=None,
        help="episode_quality JSON（指定時は品質上位選定を有効化）",
    )
    args = parser.parse_args()

    stage_flags = sum(
        int(x)
        for x in (
            args.stage1,
            args.track1_stage2,
            args.augmentation,
            args.perturbation_replay,
            args.reward_weighted_bc,
            args.stove_collision,
            args.phase3,
            args.phase4,
            args.phase5,
            args.spatial_only,
        )
    )
    if stage_flags > 1:
        parser.error(
            "--stage1 / --track1-stage2 / --augmentation / --perturbation-replay / --reward-weighted-bc / --stove-collision / "
            "--phase3 / --phase4 / --phase5 / "
            "--spatial-only は同時に指定できません"
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

    # ignored checkout にも再現可能な custom gamma/texture transform を反映する。
    from apply_lerobot_patches import patch_image_augmentations

    patch_image_augmentations(config.LEROBOT_DIR)

    sys.path.insert(0, str(lerobot_src))

    if args.perturbation_replay or args.reward_weighted_bc or args.stove_collision:
        if args.stove_collision:
            config.apply_stove_collision_config()
            job_name = "smolvla_stove_collision_safe_lora"
        elif args.reward_weighted_bc:
            config.apply_reward_weighted_bc_config()
            job_name = "smolvla_reward_weighted_bc_lora"
        else:
            config.apply_perturbation_replay_config()
            job_name = "smolvla_track1_perturbation_replay_lora"
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(
            config.DATASET_REPO,
            root=config.DATASET_ROOT,
        )
        episode_indices = list(range(metadata.total_episodes))
        selected_by_task = {"perturbation_mixed": episode_indices}
        print(
            f"Training local replay: episodes={len(episode_indices)} "
            f"root={config.DATASET_ROOT}"
        )
    elif args.phase4:
        config.apply_phase4_config()
        episode_indices = load_phase4_episode_indices()
        selected_by_task = {
            "phase4_mixture": episode_indices,
        }
        job_name = "smolvla_libero_diverse_phase4_lora"
        print(
            f"Training phase4: episodes={len(episode_indices)} "
            f"root={config.DATASET_ROOT}"
        )
        print_episode_selection_summary(
            selected_by_task,
            [
                config.TrainSuiteSpec(
                    "phase4_mixture",
                    ["phase4_mixture"],
                    len(episode_indices),
                )
            ],
        )
    elif args.phase5:
        config.apply_phase5_config()
        suites = list(config.PHASE5_TRAIN_SUITES)
        task_episode_overrides = config.PHASE5_TASK_EPISODE_OVERRIDES
        job_name = "smolvla_libero_plus_phase5_quality_lora"
    elif args.phase3:
        config.apply_phase3_config()
        suites = list(config.PHASE3_TRAIN_SUITES)
        task_episode_overrides = config.PHASE3_TASK_EPISODE_OVERRIDES
        job_name = "smolvla_libero_plus_phase3_holdout_lora"
    elif args.augmentation:
        config.apply_augmentation_config()
        suites = list(config.TRACK1_STAGE2_TRAIN_SUITES)
        task_episode_overrides = config.TRACK1_STAGE2_TASK_EPISODE_OVERRIDES
        job_name = "smolvla_libero_plus_image_aug_lora"
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

    if args.quality_cache is not None:
        config.QUALITY_CACHE_PATH = args.quality_cache

    if not args.phase4 and not args.perturbation_replay and not args.reward_weighted_bc and not args.stove_collision:
        episode_indices, selected_by_task = select_training_episodes(
            config.DATASET_REPO,
            config.DATASET_REVISION,
            suites,
            task_episode_overrides=task_episode_overrides,
            quality_cache_path=config.QUALITY_CACHE_PATH,
            min_quality=config.QUALITY_MIN_QUALITY,
            oversample_top_frac=config.QUALITY_OVERSAMPLE_TOP_FRAC,
        )
        print(f"Training suites: {[suite.name for suite in suites]}")
        print_episode_selection_summary(selected_by_task, suites)

    base_model_local = resolve_base_model()
    if config.DATASET_ROOT is None:
        print(f"Prefetching dataset ({len(episode_indices)} episodes)...")
        prefetch_dataset(
            config.DATASET_REPO,
            config.DATASET_REVISION,
            episode_indices,
            lerobot_src,
        )
    else:
        print(
            f"Using local dataset ({len(episode_indices)} episodes): "
            f"{config.DATASET_ROOT}"
        )
    run_training(base_model_local, episode_indices, lerobot_src, job_name=job_name)

    if not args.skip_merge:
        merge_script = SCRIPT_DIR / "merge_lora.py"
        merge_cmd = [sys.executable, str(merge_script)]
        if args.stove_collision:
            merge_cmd.append("--stove-collision")
        elif args.reward_weighted_bc:
            merge_cmd.append("--reward-weighted-bc")
        elif args.perturbation_replay:
            merge_cmd.append("--perturbation-replay")
        elif args.phase5:
            merge_cmd.append("--phase5")
        elif args.augmentation:
            merge_cmd.append("--augmentation")
        elif args.phase4:
            merge_cmd.append("--phase4")
        elif args.phase3:
            merge_cmd.append("--phase3")
        elif args.track1_stage2:
            merge_cmd.append("--track1-stage2")
        elif args.stage1 or not args.spatial_only:
            merge_cmd.append("--stage1")
        run_quiet(merge_cmd)


if __name__ == "__main__":
    main()
