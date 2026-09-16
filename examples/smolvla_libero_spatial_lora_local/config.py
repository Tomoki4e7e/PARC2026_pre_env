"""SmolVLA multi-suite LoRA 学習の共通設定。"""

from __future__ import annotations

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

# ── Experiment A: handle-grasp Base Policy ──────────────────────────────────
# The primary perturbation study uses the two goal-suite drawer tasks. Microwave
# remains an L0-only shape-generalization probe and is intentionally held out.
EXPERIMENT_A_DATASET_REPO = "nvidia/LIBERO_LeRobot_v3/libero_goal"
EXPERIMENT_A_DATASET_REVISION = "e5907374380b8f96511957e6ba5582be52a1e179"
EXPERIMENT_A_STEPS = 10_000
EXPERIMENT_A_DATASET_ROOT = WORK_DIR / "datasets" / "experiment_a_source" / "libero_goal"
EXPERIMENT_A_PRESERVE_BASE_NORMALIZER = True
EXPERIMENT_A_OUTPUT_DIR = WORK_DIR / "outputs" / "smolvla_experiment_a_base_lora"
EXPERIMENT_A_MERGED_MODEL_DIR = WORK_DIR / "smolvla_experiment_a_base_merged"
EXPERIMENT_A_ORACLE_ROOT = WORK_DIR / "datasets" / "experiment_a_oracle"
EXPERIMENT_A_DAGGER_ROOT = WORK_DIR / "datasets" / "experiment_a_dagger"
EXPERIMENT_A_RECOVERY_DATASET_REPO = "local/experiment_a_recovery"
EXPERIMENT_A_RECOVERY_DATASET_ROOT = WORK_DIR / "datasets" / "experiment_a_recovery"
EXPERIMENT_A_RECOVERY_MANIFEST = (
    EXPERIMENT_A_RECOVERY_DATASET_ROOT.parent / "experiment_a_recovery_manifest.json"
)
EXPERIMENT_A_RECOVERY_OUTPUT_DIR = (
    WORK_DIR / "outputs" / "smolvla_experiment_a_recovery_v4_lora"
)
EXPERIMENT_A_RECOVERY_MERGED_MODEL_DIR = (
    WORK_DIR / "smolvla_experiment_a_recovery_v4_merged"
)

# ── Experiment A v5: v4継続 arm-correction LoRA ─────────────────────────────
EXPERIMENT_A_V5_DATASET_REPO = "local/experiment_a_arm_lora_v5"
EXPERIMENT_A_V5_DATASET_ROOT = WORK_DIR / "datasets" / "experiment_a_arm_lora_v5"
EXPERIMENT_A_V5_MANIFEST = (
    EXPERIMENT_A_V5_DATASET_ROOT.parent / "experiment_a_arm_lora_v5_manifest.json"
)
EXPERIMENT_A_V5_BASE_MODEL_DIR = EXPERIMENT_A_RECOVERY_MERGED_MODEL_DIR
EXPERIMENT_A_V5_OUTPUT_DIR = (
    WORK_DIR / "outputs" / "smolvla_experiment_a_arm_lora_v5"
)
EXPERIMENT_A_V5_MERGED_MODEL_DIR = WORK_DIR / "smolvla_experiment_a_arm_lora_v5_merged"
EXPERIMENT_A_V5_STEPS = 12_000
EXPERIMENT_A_V5_SAVE_FREQ = 2_000
EXPERIMENT_A_V5_LEARNING_RATE = 5e-5
EXPERIMENT_A_V5_FINAL_LEARNING_RATE = 5e-6
EXPERIMENT_A_V5_LORA_R = 16
EXPERIMENT_A_V5_LORA_ALPHA = 16
EXPERIMENT_A_V5_FREEZE_VISION_ENCODER = True
EXPERIMENT_A_V5_PRESERVE_BASE_NORMALIZER = True

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


EXPERIMENT_A_TRAIN_SUITES = (
    TrainSuiteSpec(
        "libero_goal",
        [
            "open the middle drawer of the cabinet",
            "open the top drawer and put the bowl inside",
        ],
        50,
    ),
)


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


def apply_stage1_config() -> None:
    """第1段階 multi-suite LoRA 用に train/merge が参照する設定を復元する。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

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


def apply_track1_stage2_config() -> None:
    """第2段階 Track 1 特化 LoRA 用に train/merge が参照する設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

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


def apply_phase3_config() -> None:
    """第3段階 hold-out object/goal 弱点 LoRA 用に train/merge が参照する設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

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


def apply_phase4_config() -> None:
    """第4段階 LIBERO-90 多様タスク + replay LoRA 用に train/merge 設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

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


def apply_experiment_a_config() -> None:
    """Configure the two-drawer, no-domain-randomization Base Policy."""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

    STEPS = EXPERIMENT_A_STEPS
    LEARNING_RATE = 1e-4
    FINAL_LEARNING_RATE = 1e-5
    LORA_R = 16
    LORA_ALPHA = 16
    FREEZE_VISION_ENCODER = False
    TRAIN_EXPERT_ONLY = False
    OUTPUT_DIR = EXPERIMENT_A_OUTPUT_DIR
    MERGED_MODEL_DIR = EXPERIMENT_A_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = None
    DATASET_ROOT = EXPERIMENT_A_DATASET_ROOT
    SAVE_FREQ = 2_500
    DATASET_REPO = EXPERIMENT_A_DATASET_REPO
    DATASET_REVISION = EXPERIMENT_A_DATASET_REVISION


def apply_experiment_a_recovery_config() -> None:
    """Configure schema-corrected public + strict simulated grasp training."""
    apply_experiment_a_config()
    global OUTPUT_DIR, MERGED_MODEL_DIR, DATASET_ROOT, DATASET_REPO, DATASET_REVISION
    global SAVE_FREQ
    OUTPUT_DIR = EXPERIMENT_A_RECOVERY_OUTPUT_DIR
    MERGED_MODEL_DIR = EXPERIMENT_A_RECOVERY_MERGED_MODEL_DIR
    DATASET_ROOT = EXPERIMENT_A_RECOVERY_DATASET_ROOT
    DATASET_REPO = EXPERIMENT_A_RECOVERY_DATASET_REPO
    DATASET_REVISION = None
    SAVE_FREQ = 500


def apply_experiment_a_v5_config() -> None:
    """Configure conservative LoRA continuation from the merged v4 policy."""
    apply_experiment_a_config()
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL, DATASET_ROOT, SAVE_FREQ
    global DATASET_REPO, DATASET_REVISION

    STEPS = EXPERIMENT_A_V5_STEPS
    LEARNING_RATE = EXPERIMENT_A_V5_LEARNING_RATE
    FINAL_LEARNING_RATE = EXPERIMENT_A_V5_FINAL_LEARNING_RATE
    LORA_R = EXPERIMENT_A_V5_LORA_R
    LORA_ALPHA = EXPERIMENT_A_V5_LORA_ALPHA
    FREEZE_VISION_ENCODER = EXPERIMENT_A_V5_FREEZE_VISION_ENCODER
    TRAIN_EXPERT_ONLY = not FREEZE_VISION_ENCODER
    OUTPUT_DIR = EXPERIMENT_A_V5_OUTPUT_DIR
    MERGED_MODEL_DIR = EXPERIMENT_A_V5_MERGED_MODEL_DIR
    BASE_MODEL_LOCAL = EXPERIMENT_A_V5_BASE_MODEL_DIR
    DATASET_ROOT = EXPERIMENT_A_V5_DATASET_ROOT
    SAVE_FREQ = EXPERIMENT_A_V5_SAVE_FREQ
    DATASET_REPO = EXPERIMENT_A_V5_DATASET_REPO
    DATASET_REVISION = None
