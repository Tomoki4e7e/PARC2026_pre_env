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


# PARC 評価 instruction → lerobot/libero_plus 学習 task 文字列（compe/t1/T1_TASKS.csv）
INSTRUCTION_ALIASES: dict[str, str] = {
    "pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate": (
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    ),
    "pick the tomato sauce and place it in the basket": (
        "pick up the tomato sauce and place it in the basket"
    ),
    "pick the milk and place it in the basket": (
        "pick up the milk and place it in the basket"
    ),
}


def _normalize_instruction(instruction: str) -> str:
    """評価パイプラインの instruction を学習データの task 文字列へ変換。"""
    key = instruction.strip().lower()
    return INSTRUCTION_ALIASES.get(key, instruction.strip())


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
    ]
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    return candidates[0]


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
        from lerobot.policies.factory import make_pre_post_processors
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

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(model_path),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
            },
        )

        self.instruction = ""

    def reset(self, instruction: str = "") -> None:
        self.instruction = _normalize_instruction(instruction)
        self.policy.reset()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
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
