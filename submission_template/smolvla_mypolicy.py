"""SmolVLA マージ済みモデルを MyPolicy に組み込む参考実装。

submission_template/policy_server.py の MyPolicy クラスを、この内容で置き換えて使用する。
モデル重みは model_weights/smolvla_spatial_lora/ に配置すること。

注意:
  - 学習環境（LeRobot v0.6.0 + CUDA torch）と評価環境（setup.sh の venv）は別。
    提出 zip 内の requirements.txt に推論に必要な依存を記載する。
  - 推論 1 リクエストは 10 秒以内（action chunking により多くの step はキャッシュ返却）。
"""

from __future__ import annotations

import json
import hashlib
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


def _visual_layout_from_preprocessor(pretrained_path: str | Path) -> str:
    """policy_preprocessor.json の visual shape から CHW / HWC を判定する。"""
    path = Path(pretrained_path) / "policy_preprocessor.json"
    if not path.is_file():
        return "chw"
    pre = json.loads(path.read_text())
    for step in pre.get("steps", []):
        if step.get("registry_name") != "normalizer_processor":
            continue
        shape = step["config"]["features"]["observation.images.front"]["shape"]
        if len(shape) == 3:
            if shape[0] == 3:
                return "chw"
            if shape[-1] in (1, 3, 4):
                return "hwc"
    return "chw"


def _prepare_image(img: np.ndarray, *, layout: str = "chw") -> np.ndarray:
    """PARC の HWC uint8 画像をモデル preprocessor が期待する layout へ変換。"""
    arr = _flip_libero_image(img)
    is_hwc = (
        arr.ndim == 3
        and arr.shape[-1] in (1, 3, 4)
        and arr.shape[-1] < arr.shape[0]
    )
    if layout == "chw" and is_hwc:
        arr = arr.transpose(2, 0, 1)
    elif layout == "hwc" and not is_hwc and arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


# PARC 評価 instruction → lerobot/libero_plus 学習 task 文字列。
# LIBERO-plus の bddl (:language) は評価側の言い回し（"Pick the akita black bowl ..."）で、
# 学習データの task 文字列（"pick up the black bowl ..."）とは全タスクで異なる。
# 表は gen_instruction_aliases.py が全ベースタスク分を生成する。
try:
    from instruction_aliases import CANONICAL_TASKS, INSTRUCTION_ALIASES
except ImportError:  # 生成物が無い場合は example 4 タスクのみ最低限カバーする
    CANONICAL_TASKS: tuple[str, ...] = ()
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

# 言い換え instruction（LIBERO-plus の language 摂動）で使われる語 → LIBERO の正規語。
# 学習データの task 文字列に寄せてから類似度照合する。長いフレーズを先に適用する。
PARAPHRASE_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("dark hued", "darkhued"),
    ("dark colored", "darkcolored"),
    ("darkhued curved container", "black bowl"),
    ("darkhued concave container", "black bowl"),
    ("darkhued mixing vessel", "black bowl"),
    ("darkcolored spherical vessel", "black bowl"),
    ("darkcolored rounded container", "black bowl"),
    ("black ceramic bowl", "black bowl"),
    ("akita black bowl", "black bowl"),
    ("darkhued", "black"),
    ("darkcolored", "black"),
    ("obsidian", "black"),
    ("ebony", "black"),
    ("small ceramic baking dish", "ramekin"),
    ("small ceramic dish", "ramekin"),
    ("small baking dish", "ramekin"),
    ("flat dining ware", "plate"),
    ("flat dish holder", "plate"),
    ("flat serving dish", "plate"),
    ("flat dish", "plate"),
    ("dinner plate", "plate"),
    ("serving dish", "plate"),
    ("cookies box", "cookie box"),
    ("biscuit box", "cookie box"),
    ("wicker container", "basket"),
    ("woven container", "basket"),
    ("cooking appliance", "stove"),
    ("cooking range", "stove"),
    ("stove surface", "stove"),
    ("burner", "stove"),
    ("wooden cupboard", "wooden cabinet"),
    ("wooden storage unit", "wooden cabinet"),
    ("cupboard", "cabinet"),
    ("middle of the flat surface for meals", "table center"),
    ("middle of the dining surface", "table center"),
    ("middle of the eating surface", "table center"),
    ("heart of the eating surface", "table center"),
    ("center of the table", "table center"),
    ("middle of the table", "table center"),
    ("table centre", "table center"),
    ("relocate", "place"),
    ("reposition", "place"),
    ("position", "place"),
    ("set it atop", "place it on"),
    ("set it upon", "place it on"),
    ("set it down onto", "place it on"),
    ("set it on", "place it on"),
    ("place it atop", "place it on"),
    ("atop", "on"),
    ("upon", "on"),
    ("grasp", "pick up"),
    ("retrieve", "pick up"),
    ("collect", "pick up"),
    ("lift", "pick up"),
    ("midway between", "between"),
    ("in the narrow space separating", "between"),
    ("amid", "between"),
    ("adjacent to", "next to"),
    ("beside", "next to"),
    ("alongside", "next to"),
)

# 文字 n-gram 類似度がこの値を下回る場合は写像せず、元の instruction を使う
PARAPHRASE_MATCH_MIN_SCORE = 0.25

# LIBERO-plus の language 変種を教師データにした TF-IDF セントロイド分類器。
# build_paraphrase_model.py が生成する。5-fold CV で未知の言い換えに 93% 前後。
try:
    from paraphrase_model import CENTROIDS as _PARAPHRASE_CENTROIDS
    from paraphrase_model import IDF as _PARAPHRASE_IDF
except ImportError:  # 生成物が無い場合は文字 n-gram 照合のみで動く
    _PARAPHRASE_CENTROIDS: dict[str, dict[str, float]] = {}
    _PARAPHRASE_IDF: dict[str, float] = {}

# 分類器スコアがこの値を下回る場合は文字 n-gram 照合へ落とす
PARAPHRASE_MODEL_MIN_SCORE = 0.10

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# build_paraphrase_model.py の STOPWORDS と一致させること
_STOPWORDS = frozenset(
    """a an the of to in on at it its this that these those and or for with from
    into onto over under by as is are be been being do does did can could would
    should will shall may might must please kindly you your i me my we our then
    now just also very really quite so such if while when where which who whom
    there here what how why not no nor but than too own same s t don now
    """.split()
)

# instruction 正規化モード（既定 generic）。A/B 比較用に環境変数で切替できる。
#   match:   generic + 未知 instruction も最も近い学習 task 文字列へ写す（実験専用）
#   generic: 全タスク対応表 + 規則書き換え + 変種サフィックス除去（推奨・既定）
#   alias:   既知の完全一致対応表のみ。未知 task は原文を保持（サフィックス非除去）
#   legacy:  example 4 タスクのみの対応表（提出済みバージョンの挙動）
#   raw:     小文字化のみ
# 2026-08-09 A/B (stage2): holdout_spatial alias 0.400 → generic/match 0.867。
# inset/language は 3 モード同点。match は OOD 誤写像リスクのみ増えるため不採用。
INSTRUCTION_MODE = os.environ.get("PARC_INSTRUCTION_MODE", "generic").strip().lower()

# 新 chunk 先頭の並進・回転スケール（gripper は変更しない）。
# PARC_CHUNK_SCALE=0 で無効化。
#   off    : 無効（推奨・既定。2026-08-09 A/B で最良）
#   ramp   : 全タスク共通で先頭 N step にランプ（提案③。攻撃的設定は悪化）
#   legacy : タスク辞書に載った instruction の先頭 1 step のみ（旧挙動。inset 悪化）
ENABLE_CHUNK_SCALE = os.environ.get("PARC_CHUNK_SCALE", "1").strip() not in ("0", "false")
CHUNK_SCALE_MODE = os.environ.get("PARC_CHUNK_SCALE_MODE", "off").strip().lower()
if not ENABLE_CHUNK_SCALE:
    CHUNK_SCALE_MODE = "off"

# Temporal ensemble（ACT Algorithm 2）。推論時のみ。追加学習は不要。
#   PARC_TEMPORAL_ENSEMBLE=0|1（既定 1。2026-08-09 A/B で stage2 を全面上回り採用）
#   PARC_TE_COEFF: 指数重み係数（ACT 既定 0.01。正→古い予測を重視）
#   PARC_TE_STRIDE: 何 step ごとに再推論するか（採用値 5。SmolVLA ~150ms/推論）
#   PARC_TE_CHUNK: 予測 chunk の先頭何 step を ensemble に使うか（既定=モデル chunk_size）
ENABLE_TEMPORAL_ENSEMBLE = os.environ.get("PARC_TEMPORAL_ENSEMBLE", "1").strip().lower() not in (
    "0",
    "false",
    "off",
)
TEMPORAL_ENSEMBLE_COEFF = float(os.environ.get("PARC_TE_COEFF", "0.01"))
TEMPORAL_ENSEMBLE_STRIDE = max(1, int(os.environ.get("PARC_TE_STRIDE", "5")))
_TE_CHUNK_RAW = os.environ.get("PARC_TE_CHUNK", "").strip()
TEMPORAL_ENSEMBLE_CHUNK: int | None = int(_TE_CHUNK_RAW) if _TE_CHUNK_RAW else None

# bowl-on-stove 専用の画像ベース衝突回避シールド（既定は無効、A/B 用）。
# cream cheese が画像上でボウルの右側にある接近区間だけ、並進 x を負側へ補正する。
ENABLE_COLLISION_SHIELD = os.environ.get("PARC_COLLISION_SHIELD", "1").strip().lower() not in (
    "0", "false", "off",
)
COLLISION_SHIELD_GAIN = float(os.environ.get("PARC_SHIELD_GAIN", "0.10"))
COLLISION_SHIELD_START = int(os.environ.get("PARC_SHIELD_START", "18"))
COLLISION_SHIELD_END = int(os.environ.get("PARC_SHIELD_END", "40"))
COLLISION_SHIELD_DEBUG = os.environ.get("PARC_SHIELD_DEBUG", "0").strip().lower() not in (
    "0", "false", "off",
)


def _parse_chunk_ramp(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("PARC_CHUNK_SCALE_RAMP が空です")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"PARC_CHUNK_SCALE_RAMP は正の値である必要があります: {values}")
    return values


# 新 chunk 先頭 N step のスケール列。例: "0.6,0.8,1.0" → 3 step かけて復帰。
CHUNK_RAMP_SCALES: tuple[float, ...] = _parse_chunk_ramp(
    os.environ.get("PARC_CHUNK_SCALE_RAMP", "0.6,0.8,1.0")
)

# legacy モード専用: instruction ごとの先頭 1 step スケール。
CHUNK_FIRST_STEP_ACTION_SCALE: dict[str, float] = {
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 0.8,
    "pick up the tomato sauce and place it in the basket": 0.75,
    "put the bowl on the stove": 0.75,
    "open the middle drawer of the cabinet": 0.75,
    "turn on the stove": 0.7,
    "pick up the alphabet soup and place it in the basket": 0.8,
    "pick up the ketchup and place it in the basket": 0.8,
}


def _apply_paraphrase_synonyms(text: str) -> str:
    out = text
    for before, after in PARAPHRASE_SYNONYMS:
        if before in out:
            out = out.replace(before, after)
    return out


def _char_ngrams(text: str, n: int = 4) -> dict[str, int]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text)
    counts: dict[str, int] = {}
    for index in range(max(len(cleaned) - n + 1, 0)):
        gram = cleaned[index : index + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(gram, 0) for gram, count in a.items())
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_CANONICAL_NGRAMS = tuple((task, _char_ngrams(task)) for task in CANONICAL_TASKS)


def _paraphrase_features(text: str) -> dict[str, int]:
    """build_paraphrase_model.py の features() と同一の素性を作る。"""
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    feats: dict[str, int] = {}
    for token in tokens:
        key = f"w:{token}"
        feats[key] = feats.get(key, 0) + 1
    for left, right in zip(tokens, tokens[1:]):
        key = f"b:{left}_{right}"
        feats[key] = feats.get(key, 0) + 1
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    for index in range(max(len(cleaned) - 3, 0)):
        key = f"c:{cleaned[index : index + 4]}"
        feats[key] = feats.get(key, 0) + 1
    return feats


def _classify_paraphrase(instruction: str) -> tuple[str | None, float]:
    """学習済みセントロイドで言い換えを正規 task 文字列へ分類する。"""
    if not _PARAPHRASE_CENTROIDS:
        return None, 0.0
    raw = {
        feat: (1.0 + math.log(count)) * _PARAPHRASE_IDF[feat]
        for feat, count in _paraphrase_features(instruction).items()
        if feat in _PARAPHRASE_IDF
    }
    norm = math.sqrt(sum(value * value for value in raw.values()))
    if norm == 0.0:
        return None, 0.0
    best, best_score = None, 0.0
    for task, centroid in _PARAPHRASE_CENTROIDS.items():
        score = sum(
            value * centroid.get(feat, 0.0) for feat, value in raw.items()
        ) / norm
        if score > best_score:
            best, best_score = task, score
    return best, best_score


def _match_canonical_task(instruction: str) -> str | None:
    """言い換え instruction を最も近い学習 task 文字列へ写す。"""
    if not _CANONICAL_NGRAMS:
        return None
    query = _char_ngrams(_apply_paraphrase_synonyms(instruction))
    best, best_score = None, 0.0
    for task, grams in _CANONICAL_NGRAMS:
        score = _cosine(query, grams)
        if score > best_score:
            best, best_score = task, score
    if best is None or best_score < PARAPHRASE_MATCH_MIN_SCORE:
        return None
    return best


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
    key = INSTRUCTION_ALIASES.get(key, key)
    if INSTRUCTION_MODE == "match" and key not in CANONICAL_TASKS:
        predicted, score = _classify_paraphrase(key)
        if predicted is not None and score >= PARAPHRASE_MODEL_MIN_SCORE:
            return predicted
        matched = _match_canonical_task(key)
        if matched is not None:
            return matched
    return key


def _scale_action_motion(action: np.ndarray, scale: float) -> np.ndarray:
    """並進・回転成分のみスケール（gripper=index 6 はそのまま）。"""
    if scale == 1.0:
        return action
    scaled = action.copy()
    scaled[:6] *= scale
    return scaled


def _chunk_scale_for_step(instruction: str, step_in_chunk: int) -> float:
    """新 chunk 内の step 番号（0始まり）に対する並進・回転スケール。"""
    if CHUNK_SCALE_MODE == "off":
        return 1.0
    if CHUNK_SCALE_MODE == "legacy":
        if step_in_chunk != 0:
            return 1.0
        return CHUNK_FIRST_STEP_ACTION_SCALE.get(instruction, 1.0)
    # ramp（既定）: 全タスク共通
    if step_in_chunk < 0:
        return 1.0
    if step_in_chunk >= len(CHUNK_RAMP_SCALES):
        return 1.0
    return CHUNK_RAMP_SCALES[step_in_chunk]


class TemporalEnsembler:
    """ACT の temporal ensembling（https://huggingface.co/papers/2304.13705 Algorithm 2）。

    w_i = exp(-coeff * i) で古い予測ほど重く（coeff>0）、オンライン平均を更新する。
    stride>1 のときは再推論の間は ``consume()`` のみ行い、重み付き平均バッファを進める。
    """

    def __init__(self, coeff: float, chunk_size: int) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.coeff = float(coeff)
        self.ensemble_weights = torch.exp(
            -self.coeff * torch.arange(self.chunk_size, dtype=torch.float32)
        )
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def _ensure_weights(self, device: torch.device) -> None:
        self.ensemble_weights = self.ensemble_weights.to(device=device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=device)

    def consume(self) -> torch.Tensor:
        """新しい予測なしで、バッファ先頭の平均アクションを返す。"""
        if self.ensembled_actions is None or self.ensembled_actions.shape[1] == 0:
            raise RuntimeError("TemporalEnsembler.consume() called with empty buffer")
        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """(batch, chunk, dim) の新予測でバッファを更新し、先頭アクションを返す。"""
        if actions.dim() != 3:
            raise ValueError(f"expected (B, T, D) actions, got {tuple(actions.shape)}")
        if actions.shape[1] < 1:
            raise ValueError("empty action chunk")
        # 予測が設定 chunk より長い場合は先頭のみ使用
        if actions.shape[1] > self.chunk_size:
            actions = actions[:, : self.chunk_size]
        self._ensure_weights(actions.device)

        if self.ensembled_actions is None or self.ensembled_actions.shape[1] == 0:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (actions.shape[1], 1),
                dtype=torch.long,
                device=actions.device,
            )
        else:
            remaining = self.ensembled_actions.shape[1]
            overlap = min(remaining, actions.shape[1])
            if overlap > 0:
                counts = self.ensembled_actions_count[:overlap]
                # ACT online average（stride>1 で remaining < chunk-1 でも可）
                self.ensembled_actions[:, :overlap] *= self.ensemble_weights_cumsum[
                    counts - 1
                ]
                self.ensembled_actions[:, :overlap] += (
                    actions[:, :overlap] * self.ensemble_weights[counts]
                )
                self.ensembled_actions[:, :overlap] /= self.ensemble_weights_cumsum[
                    counts
                ]
                self.ensembled_actions_count[:overlap] = torch.clamp(
                    counts + 1, max=self.chunk_size
                )
            if actions.shape[1] > overlap:
                self.ensembled_actions = torch.cat(
                    [self.ensembled_actions[:, :overlap], actions[:, overlap:]], dim=1
                )
                new_counts = torch.ones(
                    (actions.shape[1] - overlap, 1),
                    dtype=torch.long,
                    device=actions.device,
                )
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count[:overlap], new_counts], dim=0
                )
            else:
                self.ensembled_actions = self.ensembled_actions[:, :overlap]
                self.ensembled_actions_count = self.ensembled_actions_count[:overlap]

        return self.consume()


def _build_lerobot_observation(
    obs: dict[str, np.ndarray],
    instruction: str,
    *,
    visual_layout: str = "chw",
) -> dict:
    """PARC 評価パイプラインの観測を LeRobot / SmolVLA 形式へ変換。"""
    state = np.concatenate(
        [
            obs["robot0_eef_pos"].astype(np.float32),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"].astype(np.float32),
        ]
    )
    return {
        "observation.images.front": _prepare_image(
            obs["agentview_image"], layout=visual_layout
        ),
        "observation.images.wrist": _prepare_image(
            obs["robot0_eye_in_hand_image"], layout=visual_layout
        ),
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

        self._visual_layout = _visual_layout_from_preprocessor(model_path)
        te_chunk = TEMPORAL_ENSEMBLE_CHUNK
        print(
            f"[smolvla] instruction_mode={INSTRUCTION_MODE} "
            f"chunk_scale_mode={CHUNK_SCALE_MODE} "
            f"chunk_ramp={CHUNK_RAMP_SCALES if CHUNK_SCALE_MODE == 'ramp' else '-'} "
            f"temporal_ensemble={int(ENABLE_TEMPORAL_ENSEMBLE)} "
            f"te_coeff={TEMPORAL_ENSEMBLE_COEFF if ENABLE_TEMPORAL_ENSEMBLE else '-'} "
            f"te_stride={TEMPORAL_ENSEMBLE_STRIDE if ENABLE_TEMPORAL_ENSEMBLE else '-'} "
            f"te_chunk={te_chunk if ENABLE_TEMPORAL_ENSEMBLE else '-'} "
            f"visual_layout={self._visual_layout} "
            f"aliases={len(INSTRUCTION_ALIASES)} canonical={len(CANONICAL_TASKS)} "
            f"paraphrase_classes={len(_PARAPHRASE_CENTROIDS)}",
            flush=True,
        )
        # 生成物の同梱漏れは静かに性能を落とすだけなので、採点ログに必ず残す。
        if INSTRUCTION_MODE == "match" and not _PARAPHRASE_CENTROIDS:
            print(
                "[smolvla] WARNING: paraphrase_model.py が読み込めず言い換え分類器が"
                "無効です。zip への同梱漏れの可能性があります（language 摂動で大きく成績が"
                "落ちます）。",
                flush=True,
            )
        if not CANONICAL_TASKS:
            print(
                "[smolvla] WARNING: instruction_aliases.py が読み込めず instruction "
                "対応表が最小構成です。zip への同梱漏れの可能性があります。",
                flush=True,
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = model_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        digest = hashlib.sha256()
        with weights_path.open("rb") as weights_file:
            for block in iter(lambda: weights_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        self.model_path = str(model_path.resolve())
        print(
            f"[smolvla] model_path={self.model_path} "
            f"model_sha256={self.model_sha256} device={self.device}",
            flush=True,
        )
        self.policy = SmolVLAPolicy.from_pretrained(str(model_path)).to(self.device)
        self.policy.eval()

        self._residual_policy = None
        # The bundled residual passed the deployment gate and is enabled when its
        # weights are present.  PARC_RESIDUAL_POLICY=0 remains an explicit opt-out.
        residual_setting = os.environ.get("PARC_RESIDUAL_POLICY", "auto").strip().lower()
        residual_dir = Path(
            os.environ.get(
                "PARC_RESIDUAL_MODEL_DIR", str(model_path.parent / "residual_policy")
            )
        )
        if residual_setting not in ("", "0", "false", "no", "off"):
            weights_exist = (residual_dir / "config.json").is_file() and (
                residual_dir / "model.safetensors"
            ).is_file()
            if weights_exist:
                from residual_policy import load_residual_policy

                manifest_path = residual_dir / "training_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    trained_base = manifest.get("base_model_sha256")
                    if trained_base and trained_base != self.model_sha256:
                        raise RuntimeError(
                            "Residual policy base-model mismatch: "
                            f"trained={trained_base}, loaded={self.model_sha256}"
                        )

                self._residual_policy = load_residual_policy(residual_dir, self.device)
                print(f"[smolvla] residual policy enabled: {residual_dir}", flush=True)
            elif residual_setting not in ("", "auto"):
                raise FileNotFoundError(f"residual policy weights not found: {residual_dir}")

        self._collision_critic = None
        critic_setting = os.environ.get("PARC_COLLISION_CRITIC", "off").strip().lower()
        critic_dir = Path(
            os.environ.get(
                "PARC_COLLISION_CRITIC_DIR", str(model_path.parent / "collision_critic")
            )
        )
        if critic_setting not in ("", "0", "false", "no", "off"):
            weights_exist = (critic_dir / "config.json").is_file() and (
                critic_dir / "model.safetensors"
            ).is_file()
            if weights_exist:
                from collision_critic import CollisionCritic

                self._collision_critic = CollisionCritic(critic_dir, self.device)
                print(f"[smolvla] collision critic enabled: {critic_dir}", flush=True)
            elif critic_setting not in ("", "auto"):
                raise FileNotFoundError(f"collision critic weights not found: {critic_dir}")

        self.preprocess, self.postprocess = _load_pre_post_processors(
            str(model_path), str(self.device)
        )

        self.instruction = ""
        self._steps_into_chunk = 0
        self._te_steps_since_infer = 0
        self._episode_step = 0
        self._temporal_ensembler: TemporalEnsembler | None = None
        if ENABLE_TEMPORAL_ENSEMBLE:
            chunk = te_chunk or int(self.policy.config.chunk_size)
            chunk = max(1, min(chunk, int(self.policy.config.chunk_size)))
            self._temporal_ensembler = TemporalEnsembler(
                TEMPORAL_ENSEMBLE_COEFF, chunk
            )
            print(
                f"[smolvla] temporal ensemble enabled "
                f"coeff={TEMPORAL_ENSEMBLE_COEFF} stride={TEMPORAL_ENSEMBLE_STRIDE} "
                f"chunk={chunk}",
                flush=True,
            )

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.instruction = _normalize_instruction(instruction)
        self._steps_into_chunk = 0
        self._te_steps_since_infer = 0
        self._episode_step = 0
        self.policy.reset()
        if self._temporal_ensembler is not None:
            self._temporal_ensembler.reset()
        self.last_base_action = None
        self.last_residual = None
        self.last_collision_risk = None
        self.last_base_collision_risk = None
        self.last_collision_fallback = False

    def _apply_residual_after_ensemble(
        self, obs: dict[str, np.ndarray], base_action: np.ndarray
    ) -> np.ndarray:
        """Apply the learned residual after temporal ensemble, before safety shield."""
        base_action = np.asarray(base_action, dtype=np.float32).copy()
        self.last_base_action = base_action.copy()
        self.last_residual = np.zeros(6, dtype=np.float32)
        self.last_collision_risk = None
        self.last_base_collision_risk = None
        self.last_collision_fallback = False
        action = base_action
        if self._residual_policy is not None:
            state = np.concatenate(
                [
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                    _quat_to_axis_angle(obs["robot0_eef_quat"]),
                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                ]
            )
            try:
                action, self.last_residual = self._residual_policy.apply(
                    state,
                    base_action,
                    front=obs["agentview_image"],
                    wrist=obs["robot0_eye_in_hand_image"],
                    instruction=self.instruction,
                )
            except TypeError:
                # Backward compatibility for the original state/action-only residual.
                action, self.last_residual = self._residual_policy.apply(state, base_action)
            if (
                self._collision_critic is not None
                and self.instruction == "put the bowl on the stove"
            ):
                residual_risk, _ = self._collision_critic.risk(
                    state, action, obs.get("agentview_image")
                )
                base_risk, _ = self._collision_critic.risk(
                    state, base_action, obs.get("agentview_image")
                )
                threshold = float(
                    os.environ.get(
                        "PARC_COLLISION_CRITIC_THRESHOLD",
                        str(self._collision_critic.config.threshold),
                    )
                )
                self.last_collision_risk = residual_risk
                self.last_base_collision_risk = base_risk
                self.last_collision_fallback = residual_risk >= threshold
                if self.last_collision_fallback:
                    action = base_action.copy()
        action = self._apply_learned_collision_shield(obs, action)
        return self._apply_collision_shield(obs, action)

    def _apply_learned_collision_shield(
        self, obs: dict[str, np.ndarray], action: np.ndarray
    ) -> np.ndarray:
        """Use the critic to choose a lower-risk local action candidate.

        The unmodified action is always a candidate.  An edit is accepted only when
        the original exceeds the calibrated threshold and the critic predicts at
        least ``risk_margin`` improvement, which keeps unrelated tasks unchanged.
        """
        critic = self._collision_critic
        if critic is None or not critic.config.candidate_deltas:
            return action
        if self.instruction != critic.config.task_gate:
            return action
        step = self._episode_step
        if step < critic.config.shield_start or step > critic.config.shield_end:
            return action
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ])
        phase = int(np.searchsorted(critic.config.phase_boundaries, step, side="right"))
        phase = min(phase, len(critic.config.candidate_deltas) - 1)
        delta = np.asarray(critic.config.candidate_deltas[phase], dtype=np.float32)
        candidates = [np.asarray(action, dtype=np.float32)]
        # Stay inside the demonstrated safe-minus-unsafe correction segment;
        # extrapolation looked safer to the classifier but is not data-supported.
        for scale in (0.5, 0.75, 1.0):
            candidates.append(np.clip(action + scale * delta, -1.0, 1.0))
        candidate_array = np.stack(candidates)
        risks = critic.risks(state, candidate_array, obs.get("agentview_image"))
        threshold = float(os.environ.get(
            "PARC_COLLISION_CRITIC_THRESHOLD", str(critic.config.threshold)))
        margin = float(os.environ.get(
            "PARC_COLLISION_CRITIC_MARGIN", str(critic.config.risk_margin)))
        best = int(np.argmin(risks))
        self.last_base_collision_risk = float(risks[0])
        self.last_collision_risk = float(risks[best])
        accepted = bool(risks[0] >= threshold and best != 0
                        and risks[best] <= risks[0] - margin)
        self.last_collision_fallback = accepted
        if accepted:
            if COLLISION_SHIELD_DEBUG:
                print(
                    f"[smolvla-critic] step={step} phase={phase} candidate={best} "
                    f"risk={risks[0]:.3f}->{risks[best]:.3f}", flush=True,
                )
            return candidate_array[best]
        return action

    @staticmethod
    def _estimate_stove_object_centers(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """front画像から (bowl_xy, cream_cheese_xy) を軽量な色分離で推定する。"""
        rgb = np.asarray(image)
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        rgb = rgb.astype(np.float32, copy=False)
        height, width = rgb.shape[:2]
        scale = min(height, width) / 256.0
        yy, xx = np.mgrid[:height, :width]

        # cream cheese の箱は、このタスクでは青成分が優勢な小矩形として見える。
        cream_mask = (
            (rgb[:, :, 2] > 1.15 * rgb[:, :, 0])
            & (rgb[:, :, 2] > 1.10 * rgb[:, :, 1])
            & (rgb[:, :, 2] > 55.0)
            & (yy < 0.60 * height)
        )
        if int(cream_mask.sum()) < max(8, int(40 * scale * scale)):
            return None
        cream_x = float(xx[cream_mask].mean())
        cream_y = float(yy[cream_mask].mean())

        # ボウルは cream cheese の左下近傍にある低彩度・暗色領域。
        gray = rgb.mean(axis=2)
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        bowl_mask = (
            (xx > cream_x - 75.0 * scale)
            & (xx < cream_x - 15.0 * scale)
            & (yy > cream_y - 25.0 * scale)
            & (yy < cream_y + 35.0 * scale)
            & (gray < 115.0)
            & (chroma < 28.0)
        )
        weights = np.where(bowl_mask, np.maximum(115.0 - gray, 0.0), 0.0)
        weight_sum = float(weights.sum())
        if int(bowl_mask.sum()) < max(20, int(100 * scale * scale)) or weight_sum <= 0.0:
            return None
        bowl_x = float((weights * xx).sum() / weight_sum)
        bowl_y = float((weights * yy).sum() / weight_sum)
        return (bowl_x, bowl_y), (cream_x, cream_y)

    def _apply_collision_shield(self, obs: dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        step = self._episode_step
        self._episode_step += 1
        if COLLISION_SHIELD_DEBUG and ENABLE_COLLISION_SHIELD and step == COLLISION_SHIELD_START:
            debug_image = np.asarray(obs.get("agentview_image"))
            print(
                f"[smolvla-shield] gate instruction={self.instruction!r} "
                f"image_shape={debug_image.shape} range=({debug_image.min():.3f},{debug_image.max():.3f})",
                flush=True,
            )
        if (
            not ENABLE_COLLISION_SHIELD
            or self.instruction != "put the bowl on the stove"
            or step < COLLISION_SHIELD_START
            or step > COLLISION_SHIELD_END
        ):
            return action
        centers = self._estimate_stove_object_centers(obs.get("agentview_image"))
        if centers is None:
            return action
        (bowl_x, _), (cream_x, _) = centers
        image_width = np.asarray(obs["agentview_image"]).shape[-2 if np.asarray(obs["agentview_image"]).shape[0] == 3 else 1]
        if cream_x - bowl_x < 20.0 * (image_width / 256.0):
            return action
        shielded = action.copy()
        shielded[0] = np.clip(shielded[0] - COLLISION_SHIELD_GAIN, -1.0, 1.0)
        if COLLISION_SHIELD_DEBUG and step in (COLLISION_SHIELD_START, 30, COLLISION_SHIELD_END):
            print(
                f"[smolvla-shield] step={step} bowl_x={bowl_x:.1f} cream_x={cream_x:.1f} "
                f"action_x={action[0]:.3f}->{shielded[0]:.3f}", flush=True,
            )
        return shielded

    def _predict_action_chunk(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        """観測から (1, T, 7) の action chunk を推論する（postprocess 済み）。"""
        frame = _to_preprocessor_frame(
            _build_lerobot_observation(
                obs, self.instruction, visual_layout=self._visual_layout
            )
        )
        with torch.inference_mode():
            batch = self.preprocess(frame)
            actions = self.policy.predict_action_chunk(batch)
            # postprocess は単一 action (B, D) 想定のため step ごとに適用
            processed = []
            for t in range(actions.shape[1]):
                step = self.postprocess(actions[:, t])
                if isinstance(step, torch.Tensor) and step.dim() == 1:
                    step = step.unsqueeze(0)
                processed.append(step)
            stacked = torch.stack(processed, dim=1)
        return stacked

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.utils.constants import ACTION

        if self._temporal_ensembler is not None:
            need_infer = (
                self._temporal_ensembler.ensembled_actions is None
                or self._temporal_ensembler.ensembled_actions.shape[1] == 0
                or self._te_steps_since_infer >= TEMPORAL_ENSEMBLE_STRIDE
            )
            with torch.inference_mode():
                if need_infer:
                    chunk = self._predict_action_chunk(obs)
                    # ensembler は chunk_size まで使う
                    action_t = self._temporal_ensembler.update(chunk)
                    self._te_steps_since_infer = 1
                else:
                    action_t = self._temporal_ensembler.consume()
                    self._te_steps_since_infer += 1
            action = action_t.detach().cpu().numpy()
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (7,):
                raise ValueError(f"Expected action shape (7,), got {action.shape}")
            return self._apply_residual_after_ensemble(obs, action)

        # 新 chunk 生成直前かどうか（queue 空 = 次の select_action で推論が走る）
        new_chunk = len(self.policy._queues[ACTION]) == 0
        if new_chunk:
            self._steps_into_chunk = 0

        frame = _to_preprocessor_frame(
            _build_lerobot_observation(
                obs, self.instruction, visual_layout=self._visual_layout
            )
        )
        with torch.inference_mode():
            batch = self.preprocess(frame)
            action = self.policy.select_action(batch)
            action = self.postprocess(action)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        scale = _chunk_scale_for_step(self.instruction, self._steps_into_chunk)
        action = _scale_action_motion(action, scale)
        self._steps_into_chunk += 1

        return self._apply_residual_after_ensemble(obs, action)


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
