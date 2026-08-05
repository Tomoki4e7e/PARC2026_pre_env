"""SmolVLA マージ済みモデルを MyPolicy に組み込む参考実装。

submission_template/policy_server.py の MyPolicy クラスを、この内容で置き換えて使用する。
モデル重みは model_weights/smolvla_spatial_lora/ に配置すること。

注意:
  - 学習環境（LeRobot v0.6.0 + CUDA torch）と評価環境（setup.sh の venv）は別。
    提出 zip 内の requirements.txt に推論に必要な依存を記載する。
  - 推論 1 リクエストは 10 秒以内（action chunking により多くの step はキャッシュ返却）。
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "PyTorch の CUDA ライブラリが壊れています。\n"
        "  SmolVLA 環境で pip install -r requirements.txt を実行した場合、"
        "torch が 2.11.0 等に上がりこのエラー/クラッシュが起きます。\n"
        "  修復: bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh\n"
        "  推奨: bash examples/smolvla_libero_spatial_lora_local/run_policy_server.sh"
    ) from exc

# policy_server.py から BasePolicy を import する想定。
try:
    from policy_server import BasePolicy
except ImportError:

    class BasePolicy:  # type: ignore[no-redef]
        pass


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """robosuite 形式 (x, y, z, w) のクォータニオンを axis-angle へ変換。"""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def _flip_libero_image(img: np.ndarray) -> np.ndarray:
    """LIBERO カメラ画像を 180° 反転（LeRobot LiberoProcessorStep と同一）。"""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[-1] < arr.shape[0]:
        return arr[::-1, ::-1].copy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return arr[:, ::-1, ::-1].copy()
    return arr


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """PARC の HWC uint8 画像を LeRobot 前処理向け CHW float32 [0, 1] へ変換。"""
    arr = _flip_libero_image(img)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[-1] < arr.shape[0]:
        arr = arr.transpose(2, 0, 1)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


# PARC 評価 instruction → lerobot/libero_plus 学習 task 文字列。
# LIBERO-plus の bddl (:language) は評価側の言い回し（"Pick the akita black bowl ..."）で、
# 学習データの task 文字列（"pick up the black bowl ..."）とは全タスクで異なる。
# 表は gen_instruction_aliases.py が全ベースタスク分を生成する。
try:
    from instruction_aliases import INSTRUCTION_ALIASES
except ImportError:  # 生成物が無い場合は example 4 タスクのみ最低限カバーする
    INSTRUCTION_ALIASES: dict[str, str] = {
        "pick the akita black bowl in the top layer of the wooden cabinet "
        "and place it on the plate": (
            "pick up the black bowl in the top drawer of the wooden cabinet "
            "and place it on the plate"
        ),
        "pick the tomato sauce and place it in the basket": (
            "pick up the tomato sauce and place it in the basket"
        ),
        "pick the milk and place it in the basket": (
            "pick up the milk and place it in the basket"
        ),
    }

# 表に無い言い回し向けの規則。評価側の語彙を学習側へ寄せる。
INSTRUCTION_REWRITE_RULES: tuple[tuple[str, str], ...] = (
    ("pick the akita black bowl", "pick up the black bowl"),
    ("pick the", "pick up the"),
    ("the akita black bowl", "the black bowl"),
    ("the cookies box", "the cookie box"),
    ("the top layer of the wooden cabinet", "the top drawer of the wooden cabinet"),
    ("the middle layer of the drawer", "the middle drawer of the cabinet"),
    ("the top layer of the drawer", "the top drawer"),
    ("on the top of the drawer", "on top of the cabinet"),
    ("put the cream cheese on the bowl", "put the cream cheese in the bowl"),
)

# ローカル proxy 評価では task 名由来のサフィックス（"... table 1" 等）が
# instruction に混入する。本番 CSV には無いので落として同条件にする。
_VARIANT_SUFFIX_RE = re.compile(
    r"\s+(?:table|tb|light|add|language)\s+\d+$"
    r"|\s+level\d+\s+sample\d+$"
    r"|\s+view(?:\s+\d+)+\s+initstate\s+\d+(?:\s+noise\s+\d+)?$"
    r"|\s+initstate\s+\d+(?:\s+noise\s+\d+)?$"
)

# 旧実装が持っていた example 4 タスク分のみの対応表（A/B 比較の対照）
LEGACY_INSTRUCTION_ALIASES: dict[str, str] = {
    "pick the akita black bowl in the top layer of the wooden cabinet "
    "and place it on the plate": (
        "pick up the black bowl in the top drawer of the wooden cabinet "
        "and place it on the plate"
    ),
    "pick the tomato sauce and place it in the basket": (
        "pick up the tomato sauce and place it in the basket"
    ),
    "pick the milk and place it in the basket": (
        "pick up the milk and place it in the basket"
    ),
    "put the bowl on the stove": "put the bowl on the stove",
}

# instruction 正規化モード（既定 generic）。A/B 比較用に環境変数で切替できる。
#   generic: 全タスク対応表 + 規則書き換え + 変種サフィックス除去
#   alias:   全タスク対応表のみ
#   legacy:  example 4 タスクのみの対応表（提出済みバージョンの挙動）
#   raw:     小文字化のみ
INSTRUCTION_MODE = os.environ.get("PARC_INSTRUCTION_MODE", "generic").strip().lower()

# 新 chunk 先頭 step のスケーリングを使うか（0 で無効化）
ENABLE_CHUNK_SCALE = os.environ.get("PARC_CHUNK_SCALE", "1").strip() not in ("0", "false")

# Phase 1: 新 chunk 先頭 step の並進・回転スケール（gripper は変更しない）
# collision_rate が高いタスク向け。未指定は 1.0（スケールなし）。
CHUNK_FIRST_STEP_ACTION_SCALE: dict[str, float] = {
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 0.8,
    "pick up the tomato sauce and place it in the basket": 0.75,
    "put the bowl on the stove": 0.75,
    "open the middle drawer of the cabinet": 0.75,
    "turn on the stove": 0.7,
    "pick up the alphabet soup and place it in the basket": 0.8,
    "pick up the ketchup and place it in the basket": 0.8,
}


def _normalize_instruction(instruction: str) -> str:
    """評価パイプラインの instruction を学習データの task 文字列へ変換。"""
    key = " ".join(instruction.split()).lower().rstrip(".")
    if INSTRUCTION_MODE == "raw":
        return key
    if INSTRUCTION_MODE == "legacy":
        return LEGACY_INSTRUCTION_ALIASES.get(key, key)
    if INSTRUCTION_MODE != "alias":
        key = _VARIANT_SUFFIX_RE.sub("", key)
    aliased = INSTRUCTION_ALIASES.get(key)
    if aliased is not None:
        return aliased
    if INSTRUCTION_MODE == "alias":
        return key
    for before, after in INSTRUCTION_REWRITE_RULES:
        if before in key:
            key = key.replace(before, after)
    return INSTRUCTION_ALIASES.get(key, key)


def _scale_action_motion(action: np.ndarray, scale: float) -> np.ndarray:
    """並進・回転成分のみスケール（gripper=index 6 はそのまま）。"""
    if scale == 1.0:
        return action
    scaled = action.copy()
    scaled[:6] *= scale
    return scaled


def _build_lerobot_observation(obs: dict[str, np.ndarray], instruction: str) -> dict:
    """PARC 評価パイプラインの観測を LeRobot / SmolVLA 形式へ変換。"""
    state = np.concatenate(
        [
            obs["robot0_eef_pos"].astype(np.float32),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"].astype(np.float32),
        ]
    )
    return {
        "observation.images.front": _prepare_image(obs["agentview_image"]),
        "observation.images.wrist": _prepare_image(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": instruction,
    }


def _default_model_dir() -> Path:
    here = Path(__file__).resolve().parent
    # Phase 2 汎化モデル（multi-suite）を優先。submission の model_weights は deploy_model.sh で上書き。
    candidates = [
        here / "model_weights" / "smolvla_spatial_lora",
        here.parent
        / "smolvla_libero_spatial_lora_local"
        / "work"
        / "smolvla_libero_plus_multisuite_merged",
        here.parent
        / "smolvla_libero_spatial_lora_local"
        / "work"
        / "smolvla_libero_plus_spatial_lora_merged",
        here.parent
        / "smolvla_libero_spatial_lora_local"
        / "work"
        / "smolvla_libero_plus_track1_stage2_merged",
    ]
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    return candidates[0]


def _load_pre_post_processors(pretrained_path: str, device: str):
    """保存済みの pre/post processor を読み込む。

    lerobot.policies.factory の make_pre_post_processors と同じ処理だが、
    factory は lerobot.envs（gymnasium 必須）や全 policy の config を eager import する。
    採点 venv にはそれらが無いため、推論に必要な部分だけをここで組み立てる。
    """
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # relative→absolute の参照はシリアライズされないため、読み込み後に張り直す。
    relative_step = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None
    )
    if relative_step is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                step.relative_step = relative_step

    return preprocessor, postprocessor


def _to_preprocessor_frame(frame: dict) -> dict:
    """LeRobot preprocessor が期待する batched torch.Tensor 形式へ変換。"""
    prepared = dict(frame)
    for key, value in prepared.items():
        if not key.startswith("observation."):
            continue
        if not isinstance(value, np.ndarray):
            continue
        tensor = torch.from_numpy(value)
        if tensor.dim() in (1, 3):
            tensor = tensor.unsqueeze(0)
        prepared[key] = tensor
    return prepared


class SmolVLAMyPolicy(BasePolicy):
    """SmolVLA マージ済みチェックポイント用 MyPolicy 実装。"""

    def __init__(self, model_dir: str | Path | None = None):
        from lerobot_bootstrap import setup_vendored_lerobot

        setup_vendored_lerobot()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        model_path = Path(model_dir) if model_dir else _default_model_dir()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}\n"
                "train.py + merge_lora.py 実行後、"
                "submission_template/model_weights/smolvla_spatial_lora/ へコピーしてください。"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(str(model_path)).to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = _load_pre_post_processors(
            str(model_path), str(self.device)
        )

        self.instruction = ""

    def reset(self, instruction: str = "") -> None:
        self.instruction = _normalize_instruction(instruction)
        self.policy.reset()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.utils.constants import ACTION

        # 新 chunk 生成直前かどうか（queue 空 = 次の select_action で推論が走る）
        new_chunk = len(self.policy._queues[ACTION]) == 0

        frame = _to_preprocessor_frame(_build_lerobot_observation(obs, self.instruction))
        with torch.inference_mode():
            batch = self.preprocess(frame)
            action = self.policy.select_action(batch)
            action = self.postprocess(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        if new_chunk and ENABLE_CHUNK_SCALE:
            scale = CHUNK_FIRST_STEP_ACTION_SCALE.get(self.instruction, 1.0)
            action = _scale_action_motion(action, scale)

        return action


# policy_server.py への組み込み例:
#
# class MyPolicy(BasePolicy):
#     def __init__(self):
#         from smolvla_mypolicy import SmolVLAMyPolicy
#         self._impl = SmolVLAMyPolicy("model_weights/smolvla_spatial_lora")
#
#     def get_action(self, obs):
#         return self._impl.get_action(obs)
#
#     def reset(self, instruction=""):
#         self._impl.reset(instruction=instruction)
