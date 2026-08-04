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
DATASET_REPO = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

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

FREEZE_VISION_ENCODER = STAGE1_FREEZE_VISION_ENCODER
TRAIN_EXPERT_ONLY = not FREEZE_VISION_ENCODER

OUTPUT_DIR = STAGE1_OUTPUT_DIR
MERGED_MODEL_DIR = STAGE1_MERGED_MODEL_DIR

# ローカル merged モデルをベースにする場合（第2段階 LoRA）。None なら HF から取得。
BASE_MODEL_LOCAL: Path | None = None

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


def apply_track1_stage2_config() -> None:
    """第2段階 Track 1 特化 LoRA 用に train/merge が参照する設定を上書き。"""
    global STEPS, LEARNING_RATE, FINAL_LEARNING_RATE, LORA_R, LORA_ALPHA
    global FREEZE_VISION_ENCODER, TRAIN_EXPERT_ONLY, OUTPUT_DIR, MERGED_MODEL_DIR
    global BASE_MODEL_LOCAL

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
