"""SmolVLA multi-suite LoRA 学習の共通設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# リポジトリルート（PARC2026_pre）
REPO_ROOT = Path(__file__).resolve().parents[2]

# ローカル作業ディレクトリ（キャッシュ・出力）
WORK_DIR = Path(__file__).resolve().parent / "work"
LEROBOT_DIR = WORK_DIR / "lerobot"
HF_HOME = WORK_DIR / "hf_cache"
LEROBOT_CACHE = WORK_DIR / "lerobot_cache"

# Hugging Face リビジョン（ノートブックと同一）
BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
LIBERO_PLUS_DATASET_REPO = "lerobot/libero_plus"
LIBERO_PLUS_DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
DATASET_REPO = LIBERO_PLUS_DATASET_REPO
DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION

SPATIAL_TASK_NAMES = [
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
]

OBJECT_TASK_NAMES = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
]

GOAL_TASK_NAMES = [
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "push the plate to the front of the stove",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "turn on the stove",
]

TRAIN_EPISODES_PER_TASK = 20
OBJECT_EPISODES_PER_TASK = 30
GOAL_EPISODES_PER_TASK = 20

# Track 1 example の弱点タスク（第1段階 multi-suite 用 overrides）
TASK_EPISODE_OVERRIDES: dict[str, int] = {
    "pick up the tomato sauce and place it in the basket": 40,
    "pick up the milk and place it in the basket": 30,
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 40,
    "put the bowl on the stove": 30,
}

# ── 第1段階: multi-suite 学習（740 ep） ──────────────────────────────────────
STAGE1_STEPS = 40_000
STAGE1_LEARNING_RATE = 3e-4
STAGE1_FINAL_LEARNING_RATE = 3e-5
STAGE1_LORA_R = 16
STAGE1_LORA_ALPHA = 16
STAGE1_FREEZE_VISION_ENCODER = False

STAGE1_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_multisuite_lora"
STAGE1_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_multisuite_merged"

# 旧 spatial-only 出力（第1段階 spatial-only / 第2段階ベース）
LEGACY_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_spatial_lora"
LEGACY_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_spatial_lora_merged"

# ── 第2段階: Track 1 弱点 4 タスク特化 LoRA ─────────────────────────────────
# compe/t1/T1_TASKS.csv に対応する lerobot/libero_plus task 名のみ学習する。
TRACK1_STAGE2_SPATIAL_TASK_NAMES = [
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
]
TRACK1_STAGE2_OBJECT_TASK_NAMES = [
    "pick up the tomato sauce and place it in the basket",
    "pick up the milk and place it in the basket",
]
TRACK1_STAGE2_GOAL_TASK_NAMES = [
    "put the bowl on the stove",
]

# 弱点タスクは ep 数を増やす（全 170 ep 想定）
TRACK1_STAGE2_TASK_EPISODE_OVERRIDES: dict[str, int] = {
    "pick up the tomato sauce and place it in the basket": 50,
    "pick up the milk and place it in the basket": 40,
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 50,
    "put the bowl on the stove": 30,
}

STAGE2_STEPS = 10_000
STAGE2_LEARNING_RATE = 1e-4
STAGE2_FINAL_LEARNING_RATE = 1e-5
STAGE2_LORA_R = 32
STAGE2_LORA_ALPHA = 32
STAGE2_FREEZE_VISION_ENCODER = True

STAGE2_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_track1_stage2_lora"
STAGE2_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_track1_stage2_merged"
STAGE2_BASE_MODEL_DIR = LEGACY_MERGED_MODEL_DIR

# ── 画像 Augmentation 追加学習（stage2 を壊さない独立候補） ────────────────
# Track1 stage2 と同じ 4 タスクを replay し、色・コントラスト・gamma・texture
# の複合摂動に対する頑健性だけを低 LR で追加する。
AUGMENTATION_STEPS = 4_000
AUGMENTATION_SAVE_FREQ = 1_000
AUGMENTATION_LEARNING_RATE = 5e-5
AUGMENTATION_FINAL_LEARNING_RATE = 5e-6
AUGMENTATION_LORA_R = 16
AUGMENTATION_LORA_ALPHA = 16
AUGMENTATION_FREEZE_VISION_ENCODER = True
AUGMENTATION_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_image_aug_lora"
AUGMENTATION_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_image_aug_merged"
AUGMENTATION_BASE_MODEL_DIR = STAGE2_MERGED_MODEL_DIR

# ── 報酬重み付きBC（ローカルRLログをepisode returnで再標本化） ──────────────
REWARD_WEIGHTED_BC_DATASET_REPO = "local/libero_reward_weighted_bc"
REWARD_WEIGHTED_BC_DATASET_ROOT = WORK_DIR / "datasets" / "libero_reward_weighted_bc"
REWARD_WEIGHTED_BC_STEPS = 3_000
REWARD_WEIGHTED_BC_SAVE_FREQ = 500
REWARD_WEIGHTED_BC_LEARNING_RATE = 2e-5
REWARD_WEIGHTED_BC_FINAL_LEARNING_RATE = 2e-6
REWARD_WEIGHTED_BC_LORA_R = 16
REWARD_WEIGHTED_BC_LORA_ALPHA = 16
REWARD_WEIGHTED_BC_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_reward_weighted_bc_lora"
REWARD_WEIGHTED_BC_MERGED_MODEL_DIR = WORK_DIR / "smolvla_reward_weighted_bc_merged"

# ── Track1 table/light 実レンダリング replay LoRA ──────────────────────────
PERTURBATION_REPLAY_DATASET_REPO = "local/libero_track1_perturbation_mixed"
PERTURBATION_REPLAY_DATASET_ROOT = WORK_DIR / "datasets" / "libero_track1_perturbation_mixed_v1"
PERTURBATION_REPLAY_STEPS = 3_000
PERTURBATION_REPLAY_SAVE_FREQ = 500
PERTURBATION_REPLAY_LEARNING_RATE = 2e-5
PERTURBATION_REPLAY_FINAL_LEARNING_RATE = 2e-6
PERTURBATION_REPLAY_LORA_R = 16
PERTURBATION_REPLAY_LORA_ALPHA = 16
PERTURBATION_REPLAY_FREEZE_VISION_ENCODER = True
PERTURBATION_REPLAY_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_track1_perturbation_replay_lora"
PERTURBATION_REPLAY_MERGED_MODEL_DIR = WORK_DIR / "smolvla_track1_perturbation_replay_merged"
PERTURBATION_REPLAY_BASE_MODEL_DIR = STAGE2_MERGED_MODEL_DIR

# ── 引き出し弱点専用 LoRA（成功table/light 20% + stage2 replay 80%） ──────
DRAWER_DATASET_REPO = "local/libero_drawer_targeted_v3"
DRAWER_DATASET_ROOT = WORK_DIR / "datasets" / "libero_drawer_targeted_v3"
DRAWER_MANIFEST_PATH = WORK_DIR / "drawer_targeted_v3_manifest.json"
DRAWER_STEPS = 2_000
DRAWER_SAVE_FREQ = 500
DRAWER_LEARNING_RATE = 1e-5
DRAWER_FINAL_LEARNING_RATE = 1e-6
DRAWER_LORA_R = 16
DRAWER_LORA_ALPHA = 16
DRAWER_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_drawer_targeted_v3_033443_lora"
DRAWER_MERGED_MODEL_DIR = WORK_DIR / "smolvla_drawer_targeted_v3_033443_merged"
# 0.33443 submission の実base。旧名 stage2 は現在 80aad1... を指す一方、
# residual manifest と提出zipは perturbation replay の 8c8a26... を基準にする。
DRAWER_BASE_MODEL_DIR = PERTURBATION_REPLAY_MERGED_MODEL_DIR

# ── 現行モデル起点の multi-suite / multi-domain 拡張 LoRA ────────────────
# Spatial 160 + Object 160 + Goal 100 + texture/light replay 160 = 580 ep。
EXPANDED_DATASET_REPO = "local/libero_expanded_multisuite_v1"
EXPANDED_DATASET_ROOT = WORK_DIR / "datasets" / "libero_expanded_multisuite_v1"
EXPANDED_MANIFEST_PATH = WORK_DIR / "expanded_multisuite_manifest.json"
EXPANDED_STEPS = 6_000
EXPANDED_SAVE_FREQ = 1_500
EXPANDED_LEARNING_RATE = 2e-5
EXPANDED_FINAL_LEARNING_RATE = 2e-6
EXPANDED_LORA_R = 16
EXPANDED_LORA_ALPHA = 16
EXPANDED_FREEZE_VISION_ENCODER = False
EXPANDED_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_expanded_multisuite_lora"
EXPANDED_MERGED_MODEL_DIR = WORK_DIR / "smolvla_expanded_multisuite_merged"
EXPANDED_BASE_MODEL_DIR = PERTURBATION_REPLAY_MERGED_MODEL_DIR

# ── 拡張Base先行カリキュラム（旧順序とのA/B用に出力を分離） ─────────────
# 1) 初期SmolVLAから580epを学習、2) 評価失敗Spatialのみ補正、
# 3) 弱点4タスク170epへ拡張Base代表43epを混ぜる（43 / 213 = 20.2%）。
CURRICULUM_EXPANDED_OUTPUT_DIR = WORK_DIR / "outputs/smolvla_curriculum_expanded_base_lora"
CURRICULUM_EXPANDED_MERGED_MODEL_DIR = WORK_DIR / "smolvla_curriculum_expanded_base_merged"
CURRICULUM_EXPANDED_LEARNING_RATE = 5e-5
CURRICULUM_EXPANDED_FINAL_LEARNING_RATE = 5e-6

SPATIAL_CORRECTIVE_DATASET_REPO = "local/libero_spatial_failure_corrective_v1"
SPATIAL_CORRECTIVE_DATASET_ROOT = WORK_DIR / "datasets/libero_spatial_failure_corrective_v1"
SPATIAL_CORRECTIVE_MANIFEST_PATH = WORK_DIR / "spatial_failure_corrective_manifest.json"
SPATIAL_CORRECTIVE_STEPS = 2_000
SPATIAL_CORRECTIVE_SAVE_FREQ = 500
SPATIAL_CORRECTIVE_LEARNING_RATE = 1e-5
SPATIAL_CORRECTIVE_FINAL_LEARNING_RATE = 1e-6
SPATIAL_CORRECTIVE_OUTPUT_DIR = WORK_DIR / "outputs/smolvla_spatial_failure_corrective_lora"
SPATIAL_CORRECTIVE_MERGED_MODEL_DIR = WORK_DIR / "smolvla_spatial_failure_corrective_merged"

TRACK1_REPLAY_DATASET_REPO = "local/libero_track1_with_expanded_replay_v1"
TRACK1_REPLAY_DATASET_ROOT = WORK_DIR / "datasets/libero_track1_with_expanded_replay_v1"
TRACK1_REPLAY_MANIFEST_PATH = WORK_DIR / "track1_with_expanded_replay_manifest.json"
TRACK1_REPLAY_EPISODES = 43
TRACK1_REPLAY_STEPS = 6_000
TRACK1_REPLAY_SAVE_FREQ = 1_500
TRACK1_REPLAY_LEARNING_RATE = 2e-5
TRACK1_REPLAY_FINAL_LEARNING_RATE = 2e-6
TRACK1_REPLAY_OUTPUT_DIR = WORK_DIR / "outputs/smolvla_track1_with_expanded_replay_lora"
TRACK1_REPLAY_MERGED_MODEL_DIR = WORK_DIR / "smolvla_track1_with_expanded_replay_merged"

# ── 拡張Base中間checkpointへの衝突安全補正 ───────────────────────────────
# 同一初期状態の「衝突Base軌道 / 非衝突成功軌道」ペアから、安全側のみを
# BCし、拡張Base代表episodeを40% replayする。500/1000/1500 stepを評価する。
COLLISION_SAFE_DATASET_REPO = "local/libero_curriculum_collision_safe_v1"
COLLISION_SAFE_DATASET_ROOT = WORK_DIR / "datasets/libero_curriculum_collision_safe_v1"
COLLISION_SAFE_MANIFEST_PATH = WORK_DIR / "curriculum_collision_safe_manifest.json"
COLLISION_SAFE_STEPS = 1_500
COLLISION_SAFE_SAVE_FREQ = 500
COLLISION_SAFE_LEARNING_RATE = 5e-6
COLLISION_SAFE_FINAL_LEARNING_RATE = 2e-6
COLLISION_SAFE_OUTPUT_DIR = WORK_DIR / "outputs/smolvla_curriculum_collision_safe_lora"
COLLISION_SAFE_MERGED_MODEL_DIR = WORK_DIR / "smolvla_curriculum_collision_safe_merged"
COLLISION_SAFE_BASE_MODEL_DIR = WORK_DIR / "curriculum_checkpoint_models/003000"

RAMEKIN_PREFERENCE_DATASET_REPO = "local/libero_ramekin_pairwise_preference_v1"
RAMEKIN_PREFERENCE_DATASET_ROOT = WORK_DIR / "datasets/libero_ramekin_pairwise_preference_v1"
RAMEKIN_PREFERENCE_STEPS = 1_500
RAMEKIN_PREFERENCE_SAVE_FREQ = 500
RAMEKIN_PREFERENCE_LEARNING_RATE = 2e-6
RAMEKIN_PREFERENCE_FINAL_LEARNING_RATE = 1e-6
RAMEKIN_PREFERENCE_OUTPUT_DIR = WORK_DIR / "outputs/smolvla_ramekin_pairwise_preference_lora"
RAMEKIN_PREFERENCE_MERGED_MODEL_DIR = WORK_DIR / "smolvla_ramekin_pairwise_preference_merged"

STOVE_COLLISION_DATASET_REPO = "local/stove_collision_safe_mixed"
STOVE_COLLISION_DATASET_ROOT = WORK_DIR / "datasets" / "stove_corrective_mixed_v1"
STOVE_COLLISION_STEPS = 2_500
STOVE_COLLISION_SAVE_FREQ = 500
STOVE_COLLISION_LEARNING_RATE = 2e-5
STOVE_COLLISION_FINAL_LEARNING_RATE = 2e-6
STOVE_COLLISION_LORA_R = 16
STOVE_COLLISION_LORA_ALPHA = 16
STOVE_COLLISION_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_stove_corrective_lora"
STOVE_COLLISION_MERGED_MODEL_DIR = WORK_DIR / "smolvla_stove_corrective_merged"
STOVE_COLLISION_BASE_MODEL_DIR = PERTURBATION_REPLAY_MERGED_MODEL_DIR

# ── 第3段階: hold-out proxy 弱点（object / goal）LoRA ─────────────────────────
# hold-out n=20 で弱かったベースタスクのみ追加学習。ベースは stage2 merged。
PHASE3_OBJECT_TASK_NAMES = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the ketchup and place it in the basket",
]
PHASE3_GOAL_TASK_NAMES = [
    "open the middle drawer of the cabinet",
    "turn on the stove",
]

PHASE3_TASK_EPISODE_OVERRIDES: dict[str, int] = {
    "pick up the alphabet soup and place it in the basket": 50,
    "pick up the butter and place it in the basket": 40,
    "pick up the ketchup and place it in the basket": 40,
    "open the middle drawer of the cabinet": 50,
    "turn on the stove": 50,
}

PHASE3_STEPS = 10_000
PHASE3_LEARNING_RATE = 1e-4
PHASE3_FINAL_LEARNING_RATE = 1e-5
PHASE3_LORA_R = 32
PHASE3_LORA_ALPHA = 32
PHASE3_FREEZE_VISION_ENCODER = True

PHASE3_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_phase3_holdout_lora"
PHASE3_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_phase3_holdout_merged"
PHASE3_BASE_MODEL_DIR = STAGE2_MERGED_MODEL_DIR

# ── 第4段階: LIBERO-90 多様タスク + 既知40 replay LoRA ───────────────────────
# prepare_diverse_retrain_data.py が作るローカル集約データセットを使う。
# v2: replay 比率↑・steps 短縮・base normalizer 固定・checkpoint は proxy 選定。
PHASE4_DATASET_REPO = "local/libero_diverse_retrain"
PHASE4_DATASET_ROOT = WORK_DIR / "datasets" / "libero_diverse_retrain"
PHASE4_MANIFEST_PATH = WORK_DIR / "diverse_retrain_manifest.json"
PHASE4_STEPS = 5_000
PHASE4_SAVE_FREQ = 1_000
PHASE4_LEARNING_RATE = 5e-5
PHASE4_FINAL_LEARNING_RATE = 5e-6
PHASE4_LORA_R = 16
PHASE4_LORA_ALPHA = 16
PHASE4_FREEZE_VISION_ENCODER = True
# 学習 episode のうち replay（既知 suite）の目標比率。diverse 側を間引いて達成する。
PHASE4_TARGET_REPLAY_FRACTION = 0.55
# True なら学習・merge 後とも stage2 base の normalizer stats を使う。
PHASE4_PRESERVE_BASE_NORMALIZER = True
PHASE4_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_diverse_phase4_v2_lora"
PHASE4_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_diverse_phase4_v2_merged"
# stage2 merged をベースにしないと Track1 / hold-out 能力が HF 素モデルへ巻き戻る。
PHASE4_BASE_MODEL_DIR: Path | None = WORK_DIR / "smolvla_libero_plus_track1_stage2_merged"

# ── 第5段階: 品質フィルタ済みデモでの追加 LoRA（滑らかさ重視） ──────────────
# Track1 + Phase3 弱点タスクのプールから jerk/SPARC/回転総量が良い demo を優先選定。
# ベースは Phase4 v2 merged（無ければ stage2）。checkpoint は複合 proxy で選定。
PHASE5_QUALITY_CACHE = WORK_DIR / "episode_quality_phase5.json"
PHASE5_MIN_QUALITY: float | None = None  # None なら上位 count のみ（閾値なし）
PHASE5_OVERSAMPLE_TOP_FRAC = 0.0  # lerobot episodes は集合扱いのため通常は 0
PHASE5_STEPS = 6_000
PHASE5_SAVE_FREQ = 1_000
PHASE5_LEARNING_RATE = 5e-5
PHASE5_FINAL_LEARNING_RATE = 5e-6
PHASE5_LORA_R = 16
PHASE5_LORA_ALPHA = 16
PHASE5_FREEZE_VISION_ENCODER = True
PHASE5_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_libero_plus_phase5_quality_lora"
PHASE5_MERGED_MODEL_DIR = WORK_DIR / "smolvla_libero_plus_phase5_quality_merged"
PHASE5_BASE_MODEL_DIR: Path | None = WORK_DIR / "smolvla_libero_diverse_phase4_v2_merged"
# Phase5 の ep 数（品質上位）。弱点タスクは多め。
PHASE5_TASK_EPISODE_OVERRIDES: dict[str, int] = {
    "pick up the tomato sauce and place it in the basket": 50,
    "pick up the milk and place it in the basket": 40,
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 50,
    "put the bowl on the stove": 30,
    "pick up the alphabet soup and place it in the basket": 40,
    "pick up the butter and place it in the basket": 35,
    "pick up the ketchup and place it in the basket": 35,
    "open the middle drawer of the cabinet": 40,
    "turn on the stove": 40,
}

# ── train.py / merge_lora.py が参照するアクティブ設定（第1段階が既定） ───────
STEPS = STAGE1_STEPS
LOG_FREQ = 100
BATCH_SIZE = 1
LEARNING_RATE = STAGE1_LEARNING_RATE
FINAL_LEARNING_RATE = STAGE1_FINAL_LEARNING_RATE
WARMUP_STEPS = 100
LORA_R = STAGE1_LORA_R
LORA_ALPHA = STAGE1_LORA_ALPHA
SEED = 42
SAVE_FREQ: int | None = None

FREEZE_VISION_ENCODER = STAGE1_FREEZE_VISION_ENCODER
TRAIN_EXPERT_ONLY = not FREEZE_VISION_ENCODER

OUTPUT_DIR = STAGE1_OUTPUT_DIR
MERGED_MODEL_DIR = STAGE1_MERGED_MODEL_DIR

# 品質フィルタ（None なら従来の均等間隔選定）
QUALITY_CACHE_PATH: Path | None = None
QUALITY_MIN_QUALITY: float | None = None
QUALITY_OVERSAMPLE_TOP_FRAC = 0.0
IMAGE_AUGMENTATION_ENABLED = False

# ローカル merged モデルをベースにする場合（第2段階 LoRA）。None なら HF から取得。
BASE_MODEL_LOCAL: Path | None = None
# ローカル LeRobot データセット root（phase4）。None なら HF repo のみ。
DATASET_ROOT: Path | None = None

# Colab は torchcodec だが、ローカルでは libffi / FFmpeg 競合を避け pyav を使う
VIDEO_BACKEND = "pyav"

LEROBOT_TAG = "v0.6.0"


@dataclass(frozen=True)
class TrainSuiteSpec:
    name: str
    task_names: list[str]
    episodes_per_task: int


TRAIN_SUITES: tuple[TrainSuiteSpec, ...] = (
    TrainSuiteSpec("spatial", SPATIAL_TASK_NAMES, TRAIN_EPISODES_PER_TASK),
    TrainSuiteSpec("object", OBJECT_TASK_NAMES, OBJECT_EPISODES_PER_TASK),
    TrainSuiteSpec("goal", GOAL_TASK_NAMES, GOAL_EPISODES_PER_TASK),
)

TRACK1_STAGE2_TRAIN_SUITES: tuple[TrainSuiteSpec, ...] = (
    TrainSuiteSpec(
        "spatial",
        TRACK1_STAGE2_SPATIAL_TASK_NAMES,
        TRAIN_EPISODES_PER_TASK,
    ),
    TrainSuiteSpec(
        "object",
        TRACK1_STAGE2_OBJECT_TASK_NAMES,
        OBJECT_EPISODES_PER_TASK,
    ),
    TrainSuiteSpec(
        "goal",
        TRACK1_STAGE2_GOAL_TASK_NAMES,
        GOAL_EPISODES_PER_TASK,
    ),
)

PHASE3_TRAIN_SUITES: tuple[TrainSuiteSpec, ...] = (
    TrainSuiteSpec(
        "object",
        PHASE3_OBJECT_TASK_NAMES,
        OBJECT_EPISODES_PER_TASK,
    ),
    TrainSuiteSpec(
        "goal",
        PHASE3_GOAL_TASK_NAMES,
        GOAL_EPISODES_PER_TASK,
    ),
)

PHASE5_TRAIN_SUITES: tuple[TrainSuiteSpec, ...] = (
    TrainSuiteSpec(
        "spatial",
        TRACK1_STAGE2_SPATIAL_TASK_NAMES,
        TRAIN_EPISODES_PER_TASK,
    ),
    TrainSuiteSpec(
        "object",
        list(
            dict.fromkeys(
                TRACK1_STAGE2_OBJECT_TASK_NAMES + PHASE3_OBJECT_TASK_NAMES
            )
        ),
        OBJECT_EPISODES_PER_TASK,
    ),
    TrainSuiteSpec(
        "goal",
        list(
            dict.fromkeys(TRACK1_STAGE2_GOAL_TASK_NAMES + PHASE3_GOAL_TASK_NAMES)
        ),
        GOAL_EPISODES_PER_TASK,
    ),
)


def apply_stage1_config() -> None:
    """第1段階 multi-suite LoRA 用に train/merge が参照する設定を復元する。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC

    STEPS = STAGE1_STEPS
    LEARNING_RATE = STAGE1_LEARNING_RATE
    FINAL_LEARNING_RATE = STAGE1_FINAL_LEARNING_RATE
    LORA_R = STAGE1_LORA_R
    LORA_ALPHA = STAGE1_LORA_ALPHA
    FREEZE_VISION_ENCODER = STAGE1_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not STAGE1_FREEZE_VISION_ENCODER
    OUTPUT_DIR = STAGE1_OUTPUT_DIR
    MERGED_MODEL_DIR = STAGE1_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = None
    DATASET_ROOT = None
    SAVE_FREQ = None
    DATASET_REPO = LIBERO_PLUS_DATASET_REPO
    DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION
    QUALITY_CACHE_PATH = None
    QUALITY_MIN_QUALITY = None
    QUALITY_OVERSAMPLE_TOP_FRAC = 0.0


def apply_track1_stage2_config() -> None:
    """第2段階 Track 1 特化 LoRA 用に train/merge が参照する設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC

    STEPS = STAGE2_STEPS
    LEARNING_RATE = STAGE2_LEARNING_RATE
    FINAL_LEARNING_RATE = STAGE2_FINAL_LEARNING_RATE
    LORA_R = STAGE2_LORA_R
    LORA_ALPHA = STAGE2_LORA_ALPHA
    FREEZE_VISION_ENCODER = STAGE2_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not STAGE2_FREEZE_VISION_ENCODER
    OUTPUT_DIR = STAGE2_OUTPUT_DIR
    MERGED_MODEL_DIR = STAGE2_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = STAGE2_BASE_MODEL_DIR
    DATASET_ROOT = None
    SAVE_FREQ = None
    DATASET_REPO = LIBERO_PLUS_DATASET_REPO
    DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION
    QUALITY_CACHE_PATH = None
    QUALITY_MIN_QUALITY = None
    QUALITY_OVERSAMPLE_TOP_FRAC = 0.0


def apply_augmentation_config() -> None:
    """stage2 ベースの画像 Augmentation 追加学習設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC
    global IMAGE_AUGMENTATION_ENABLED

    STEPS = AUGMENTATION_STEPS
    LEARNING_RATE = AUGMENTATION_LEARNING_RATE
    FINAL_LEARNING_RATE = AUGMENTATION_FINAL_LEARNING_RATE
    LORA_R = AUGMENTATION_LORA_R
    LORA_ALPHA = AUGMENTATION_LORA_ALPHA
    FREEZE_VISION_ENCODER = AUGMENTATION_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not AUGMENTATION_FREEZE_VISION_ENCODER
    OUTPUT_DIR = AUGMENTATION_OUTPUT_DIR
    MERGED_MODEL_DIR = AUGMENTATION_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = AUGMENTATION_BASE_MODEL_DIR
    DATASET_ROOT = None
    SAVE_FREQ = AUGMENTATION_SAVE_FREQ
    DATASET_REPO = LIBERO_PLUS_DATASET_REPO
    DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION
    QUALITY_CACHE_PATH = None
    QUALITY_MIN_QUALITY = None
    QUALITY_OVERSAMPLE_TOP_FRAC = 0.0
    IMAGE_AUGMENTATION_ENABLED = True


def apply_perturbation_replay_config() -> None:
    """実レンダリングtable/light replayの低LR LoRA設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = PERTURBATION_REPLAY_STEPS
    LEARNING_RATE = PERTURBATION_REPLAY_LEARNING_RATE
    FINAL_LEARNING_RATE = PERTURBATION_REPLAY_FINAL_LEARNING_RATE
    LORA_R = PERTURBATION_REPLAY_LORA_R
    LORA_ALPHA = PERTURBATION_REPLAY_LORA_ALPHA
    FREEZE_VISION_ENCODER = PERTURBATION_REPLAY_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not FREEZE_VISION_ENCODER
    OUTPUT_DIR = PERTURBATION_REPLAY_OUTPUT_DIR
    MERGED_MODEL_DIR = PERTURBATION_REPLAY_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = PERTURBATION_REPLAY_BASE_MODEL_DIR
    DATASET_ROOT = PERTURBATION_REPLAY_DATASET_ROOT
    SAVE_FREQ = PERTURBATION_REPLAY_SAVE_FREQ
    DATASET_REPO = PERTURBATION_REPLAY_DATASET_REPO
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_drawer_config() -> None:
    """引き出し成功軌道へ80%のstage2 replayを混ぜる低LR LoRA設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = DRAWER_STEPS
    LEARNING_RATE = DRAWER_LEARNING_RATE
    FINAL_LEARNING_RATE = DRAWER_FINAL_LEARNING_RATE
    LORA_R = DRAWER_LORA_R
    LORA_ALPHA = DRAWER_LORA_ALPHA
    FREEZE_VISION_ENCODER = True
    TRAIN_EXPERT_ONLY = False
    OUTPUT_DIR = DRAWER_OUTPUT_DIR
    MERGED_MODEL_DIR = DRAWER_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = DRAWER_BASE_MODEL_DIR
    DATASET_ROOT = DRAWER_DATASET_ROOT
    SAVE_FREQ = DRAWER_SAVE_FREQ
    DATASET_REPO = DRAWER_DATASET_REPO
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_expanded_config() -> None:
    """現行摂動Replayモデルを起点にしたmulti-suite拡張LoRA設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = EXPANDED_STEPS
    LEARNING_RATE = EXPANDED_LEARNING_RATE
    FINAL_LEARNING_RATE = EXPANDED_FINAL_LEARNING_RATE
    LORA_R = EXPANDED_LORA_R
    LORA_ALPHA = EXPANDED_LORA_ALPHA
    FREEZE_VISION_ENCODER = EXPANDED_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not FREEZE_VISION_ENCODER
    OUTPUT_DIR = Path(os.environ.get("PARC_EXPANDED_OUTPUT_DIR", str(EXPANDED_OUTPUT_DIR)))
    MERGED_MODEL_DIR = Path(os.environ.get("PARC_EXPANDED_MERGED_MODEL_DIR", str(EXPANDED_MERGED_MODEL_DIR)))
    BASE_MODEL_LOCAL = EXPANDED_BASE_MODEL_DIR
    DATASET_ROOT = Path(os.environ.get("PARC_EXPANDED_DATASET_ROOT", str(EXPANDED_DATASET_ROOT)))
    SAVE_FREQ = EXPANDED_SAVE_FREQ
    DATASET_REPO = os.environ.get("PARC_EXPANDED_DATASET_REPO", EXPANDED_DATASET_REPO)
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_curriculum_expanded_config() -> None:
    """初期SmolVLAを起点にした拡張Base先行学習。"""
    apply_expanded_config()
    global LEARNING_RATE, FINAL_LEARNING_RATE, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL

    LEARNING_RATE = CURRICULUM_EXPANDED_LEARNING_RATE
    FINAL_LEARNING_RATE = CURRICULUM_EXPANDED_FINAL_LEARNING_RATE
    OUTPUT_DIR = CURRICULUM_EXPANDED_OUTPUT_DIR
    MERGED_MODEL_DIR = CURRICULUM_EXPANDED_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = None


def _apply_curriculum_local_dataset(
    *, dataset_repo: str, dataset_root: Path, steps: int, save_freq: int,
    learning_rate: float, final_learning_rate: float, output_dir: Path,
    merged_model_dir: Path, default_base_model_dir: Path,
) -> None:
    """Spatial補正と弱点+Replay段に共通する低LR設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = steps
    LEARNING_RATE = learning_rate
    FINAL_LEARNING_RATE = final_learning_rate
    LORA_R = 16
    LORA_ALPHA = 16
    FREEZE_VISION_ENCODER = True
    TRAIN_EXPERT_ONLY = False
    OUTPUT_DIR = output_dir
    MERGED_MODEL_DIR = merged_model_dir
    BASE_MODEL_LOCAL = Path(os.environ.get(
        "PARC_CURRICULUM_BASE_MODEL_DIR", str(default_base_model_dir)
    ))
    DATASET_ROOT = dataset_root
    SAVE_FREQ = save_freq
    DATASET_REPO = dataset_repo
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_spatial_corrective_config() -> None:
    """拡張Base評価で失敗したSpatial配置だけを補正する。"""
    _apply_curriculum_local_dataset(
        dataset_repo=SPATIAL_CORRECTIVE_DATASET_REPO,
        dataset_root=SPATIAL_CORRECTIVE_DATASET_ROOT,
        steps=SPATIAL_CORRECTIVE_STEPS,
        save_freq=SPATIAL_CORRECTIVE_SAVE_FREQ,
        learning_rate=SPATIAL_CORRECTIVE_LEARNING_RATE,
        final_learning_rate=SPATIAL_CORRECTIVE_FINAL_LEARNING_RATE,
        output_dir=SPATIAL_CORRECTIVE_OUTPUT_DIR,
        merged_model_dir=SPATIAL_CORRECTIVE_MERGED_MODEL_DIR,
        default_base_model_dir=CURRICULUM_EXPANDED_MERGED_MODEL_DIR,
    )


def apply_track1_replay_config() -> None:
    """弱点4タスクに拡張Base代表episodeを20%混ぜる追加学習。"""
    _apply_curriculum_local_dataset(
        dataset_repo=TRACK1_REPLAY_DATASET_REPO,
        dataset_root=TRACK1_REPLAY_DATASET_ROOT,
        steps=TRACK1_REPLAY_STEPS,
        save_freq=TRACK1_REPLAY_SAVE_FREQ,
        learning_rate=TRACK1_REPLAY_LEARNING_RATE,
        final_learning_rate=TRACK1_REPLAY_FINAL_LEARNING_RATE,
        output_dir=TRACK1_REPLAY_OUTPUT_DIR,
        merged_model_dir=TRACK1_REPLAY_MERGED_MODEL_DIR,
        default_base_model_dir=CURRICULUM_EXPANDED_MERGED_MODEL_DIR,
    )


def apply_collision_safe_config() -> None:
    """選定した拡張Base中間checkpointへ低LRの衝突安全補正を行う。"""
    _apply_curriculum_local_dataset(
        dataset_repo=COLLISION_SAFE_DATASET_REPO,
        dataset_root=COLLISION_SAFE_DATASET_ROOT,
        steps=COLLISION_SAFE_STEPS,
        save_freq=COLLISION_SAFE_SAVE_FREQ,
        learning_rate=COLLISION_SAFE_LEARNING_RATE,
        final_learning_rate=COLLISION_SAFE_FINAL_LEARNING_RATE,
        output_dir=COLLISION_SAFE_OUTPUT_DIR,
        merged_model_dir=COLLISION_SAFE_MERGED_MODEL_DIR,
        default_base_model_dir=COLLISION_SAFE_BASE_MODEL_DIR,
    )


def apply_ramekin_preference_config() -> None:
    """Flow-matching pairwise preferenceによるramekin安全補正。"""
    _apply_curriculum_local_dataset(
        dataset_repo=RAMEKIN_PREFERENCE_DATASET_REPO,
        dataset_root=RAMEKIN_PREFERENCE_DATASET_ROOT,
        steps=RAMEKIN_PREFERENCE_STEPS,
        save_freq=RAMEKIN_PREFERENCE_SAVE_FREQ,
        learning_rate=RAMEKIN_PREFERENCE_LEARNING_RATE,
        final_learning_rate=RAMEKIN_PREFERENCE_FINAL_LEARNING_RATE,
        output_dir=RAMEKIN_PREFERENCE_OUTPUT_DIR,
        merged_model_dir=RAMEKIN_PREFERENCE_MERGED_MODEL_DIR,
        default_base_model_dir=COLLISION_SAFE_BASE_MODEL_DIR,
    )


def apply_reward_weighted_bc_config() -> None:
    """Local RLログから作った報酬重み付きBCデータの低LR LoRA設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = REWARD_WEIGHTED_BC_STEPS
    LEARNING_RATE = REWARD_WEIGHTED_BC_LEARNING_RATE
    FINAL_LEARNING_RATE = REWARD_WEIGHTED_BC_FINAL_LEARNING_RATE
    LORA_R = REWARD_WEIGHTED_BC_LORA_R
    LORA_ALPHA = REWARD_WEIGHTED_BC_LORA_ALPHA
    FREEZE_VISION_ENCODER = True
    TRAIN_EXPERT_ONLY = False
    OUTPUT_DIR = REWARD_WEIGHTED_BC_OUTPUT_DIR
    MERGED_MODEL_DIR = REWARD_WEIGHTED_BC_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = STAGE2_MERGED_MODEL_DIR
    DATASET_ROOT = REWARD_WEIGHTED_BC_DATASET_ROOT
    SAVE_FREQ = REWARD_WEIGHTED_BC_SAVE_FREQ
    DATASET_REPO = REWARD_WEIGHTED_BC_DATASET_REPO
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_stove_collision_config() -> None:
    """非衝突stove replay重点配分による低LR追加LoRA設定。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION, IMAGE_AUGMENTATION_ENABLED

    STEPS = STOVE_COLLISION_STEPS
    LEARNING_RATE = STOVE_COLLISION_LEARNING_RATE
    FINAL_LEARNING_RATE = STOVE_COLLISION_FINAL_LEARNING_RATE
    LORA_R = STOVE_COLLISION_LORA_R
    LORA_ALPHA = STOVE_COLLISION_LORA_ALPHA
    FREEZE_VISION_ENCODER = True
    TRAIN_EXPERT_ONLY = False
    OUTPUT_DIR = STOVE_COLLISION_OUTPUT_DIR
    MERGED_MODEL_DIR = STOVE_COLLISION_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = STOVE_COLLISION_BASE_MODEL_DIR
    DATASET_ROOT = STOVE_COLLISION_DATASET_ROOT
    SAVE_FREQ = STOVE_COLLISION_SAVE_FREQ
    DATASET_REPO = STOVE_COLLISION_DATASET_REPO
    DATASET_REVISION = None
    IMAGE_AUGMENTATION_ENABLED = False


def apply_phase3_config() -> None:
    """第3段階 hold-out object/goal 弱点 LoRA 用に train/merge が参照する設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC

    STEPS = PHASE3_STEPS
    LEARNING_RATE = PHASE3_LEARNING_RATE
    FINAL_LEARNING_RATE = PHASE3_FINAL_LEARNING_RATE
    LORA_R = PHASE3_LORA_R
    LORA_ALPHA = PHASE3_LORA_ALPHA
    FREEZE_VISION_ENCODER = PHASE3_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not PHASE3_FREEZE_VISION_ENCODER
    OUTPUT_DIR = PHASE3_OUTPUT_DIR
    MERGED_MODEL_DIR = PHASE3_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = PHASE3_BASE_MODEL_DIR
    DATASET_ROOT = None
    SAVE_FREQ = None
    DATASET_REPO = LIBERO_PLUS_DATASET_REPO
    DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION
    QUALITY_CACHE_PATH = None
    QUALITY_MIN_QUALITY = None
    QUALITY_OVERSAMPLE_TOP_FRAC = 0.0


def apply_phase4_config() -> None:
    """第4段階 LIBERO-90 多様タスク + replay LoRA 用に train/merge 設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC

    STEPS = PHASE4_STEPS
    LEARNING_RATE = PHASE4_LEARNING_RATE
    FINAL_LEARNING_RATE = PHASE4_FINAL_LEARNING_RATE
    LORA_R = PHASE4_LORA_R
    LORA_ALPHA = PHASE4_LORA_ALPHA
    FREEZE_VISION_ENCODER = PHASE4_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not PHASE4_FREEZE_VISION_ENCODER
    OUTPUT_DIR = PHASE4_OUTPUT_DIR
    MERGED_MODEL_DIR = PHASE4_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = PHASE4_BASE_MODEL_DIR
    DATASET_ROOT = PHASE4_DATASET_ROOT
    SAVE_FREQ = PHASE4_SAVE_FREQ
    DATASET_REPO = PHASE4_DATASET_REPO
    DATASET_REVISION = None
    QUALITY_CACHE_PATH = None
    QUALITY_MIN_QUALITY = None
    QUALITY_OVERSAMPLE_TOP_FRAC = 0.0


def apply_phase5_config() -> None:
    """第5段階: 品質フィルタ済みデモでの追加 LoRA。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION
    global QUALITY_CACHE_PATH, QUALITY_MIN_QUALITY, QUALITY_OVERSAMPLE_TOP_FRAC

    STEPS = PHASE5_STEPS
    LEARNING_RATE = PHASE5_LEARNING_RATE
    FINAL_LEARNING_RATE = PHASE5_FINAL_LEARNING_RATE
    LORA_R = PHASE5_LORA_R
    LORA_ALPHA = PHASE5_LORA_ALPHA
    FREEZE_VISION_ENCODER = PHASE5_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not PHASE5_FREEZE_VISION_ENCODER
    OUTPUT_DIR = PHASE5_OUTPUT_DIR
    MERGED_MODEL_DIR = PHASE5_MERGED_MODEL_DIR
    base = PHASE5_BASE_MODEL_DIR
    if base is None or not (base / "model.safetensors").is_file():
        base = STAGE2_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = base
    DATASET_ROOT = None
    SAVE_FREQ = PHASE5_SAVE_FREQ
    DATASET_REPO = LIBERO_PLUS_DATASET_REPO
    DATASET_REVISION = LIBERO_PLUS_DATASET_REVISION
    QUALITY_CACHE_PATH = PHASE5_QUALITY_CACHE
    QUALITY_MIN_QUALITY = PHASE5_MIN_QUALITY
    QUALITY_OVERSAMPLE_TOP_FRAC = PHASE5_OVERSAMPLE_TOP_FRAC
