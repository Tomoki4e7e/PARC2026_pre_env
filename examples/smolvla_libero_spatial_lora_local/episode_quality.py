"""デモ軌道の品質スコア（jerk / SPARC / EEF 回転総量）計算と選定。

LIBERO-plus デモには成功・衝突ラベルが無いため、オフラインでは
observation.state の EE 位置・姿勢から滑らかさ指標を計算する。
成功・衝突は proxy 評価側（proxy_score.py）でチェックポイント選定に使う。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# pipeline 側の SPARC 実装を再利用（リポジトリルート経由）
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.total_score import compute_sparc  # noqa: E402


@dataclass(frozen=True)
class EpisodeQuality:
    episode_index: int
    n_frames: int
    avg_cartesian_jerk: float
    rms_cartesian_jerk: float
    sparc: float
    orientation_path_length: float
    cartesian_path_length: float
    # 合成品質スコア（高いほど良い）
    quality_score: float
    # 任意: 自前 rollout 由来の場合のみ埋まる
    success: float | None = None
    collided: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _orientation_path_length_axis_angle(axis_angles: np.ndarray) -> float:
    """axis-angle シーケンスの隣接フレーム角度差の累積（rad）。"""
    if len(axis_angles) < 2:
        return 0.0
    total = 0.0
    for i in range(len(axis_angles) - 1):
        a = axis_angles[i]
        b = axis_angles[i + 1]
        # 小さな角では ||a-b|| が近似、大きめでも実用上の代理指標として十分
        total += float(np.linalg.norm(a - b))
    return total


def _compute_jerk(positions: np.ndarray, dt: float) -> tuple[float, float]:
    if len(positions) < 4:
        return float("nan"), float("nan")
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    norms = np.linalg.norm(jerk, axis=1)
    avg = float(np.mean(norms))
    rms = float(math.sqrt(np.mean(norms ** 2)))
    return avg, rms


def metrics_from_state(
    state: np.ndarray,
    *,
    dt: float,
) -> dict[str, float]:
    """observation.state (T, 8) → 軌道メトリクス。"""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 6:
        raise ValueError(f"expected state (T, >=6), got {state.shape}")
    pos = state[:, :3]
    ori = state[:, 3:6]
    cart_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
    ori_len = _orientation_path_length_axis_angle(ori)
    avg_j, rms_j = _compute_jerk(pos, dt)
    sparc = float(compute_sparc(pos, dt)) if len(pos) >= 4 else float("nan")
    return {
        "n_frames": float(len(state)),
        "avg_cartesian_jerk": avg_j,
        "rms_cartesian_jerk": rms_j,
        "sparc": sparc,
        "orientation_path_length": ori_len,
        "cartesian_path_length": cart_len,
    }


def quality_score_from_metrics(
    metrics: dict[str, float],
    *,
    refs: dict[str, float] | None = None,
    success: float | None = None,
    collided: float | None = None,
    w_jerk: float = 0.35,
    w_sparc: float = 0.25,
    w_orient: float = 0.20,
    w_path: float = 0.10,
    w_success: float = 0.30,
    w_collision: float = 0.40,
) -> float:
    """高いほど良い合成スコア。refs は「悪い側」の参照スケール（その値で 0 付近）。"""
    refs = refs or {
        "avg_cartesian_jerk": 15.0,
        "sparc": -8.0,  # SPARC は 0 に近いほど良い（負値）
        "orientation_path_length": 6.0,
        "cartesian_path_length": 3.0,
    }

    def _low_is_good(value: float, ref: float) -> float:
        if not np.isfinite(value) or ref <= 0:
            return 0.0
        return float(np.clip(1.0 - value / ref, 0.0, 1.0))

    def _sparc_good(value: float, ref: float) -> float:
        # ref は典型的な悪い SPARC（例 -8）。0 に近いほど 1。
        if not np.isfinite(value):
            return 0.0
        # value=-8 → 0, value=0 → 1
        return float(np.clip((value - ref) / (0.0 - ref), 0.0, 1.0))

    score = (
        w_jerk * _low_is_good(metrics["avg_cartesian_jerk"], refs["avg_cartesian_jerk"])
        + w_sparc * _sparc_good(metrics["sparc"], refs["sparc"])
        + w_orient * _low_is_good(
            metrics["orientation_path_length"], refs["orientation_path_length"]
        )
        + w_path * _low_is_good(
            metrics["cartesian_path_length"], refs["cartesian_path_length"]
        )
    )
    # デモのみなら success/collided は None → 滑らかさ項だけ
    if success is not None:
        score += w_success * float(np.clip(success, 0.0, 1.0))
    if collided is not None:
        score += w_collision * (1.0 - float(np.clip(collided, 0.0, 1.0)))
    return float(score)


def build_episode_quality(
    episode_index: int,
    state: np.ndarray,
    *,
    dt: float,
    success: float | None = None,
    collided: float | None = None,
    refs: dict[str, float] | None = None,
) -> EpisodeQuality:
    metrics = metrics_from_state(state, dt=dt)
    q = quality_score_from_metrics(
        metrics, refs=refs, success=success, collided=collided
    )
    return EpisodeQuality(
        episode_index=int(episode_index),
        n_frames=int(metrics["n_frames"]),
        avg_cartesian_jerk=float(metrics["avg_cartesian_jerk"]),
        rms_cartesian_jerk=float(metrics["rms_cartesian_jerk"]),
        sparc=float(metrics["sparc"]),
        orientation_path_length=float(metrics["orientation_path_length"]),
        cartesian_path_length=float(metrics["cartesian_path_length"]),
        quality_score=q,
        success=success,
        collided=collided,
    )


def choose_top_quality(
    pool: list[int],
    quality_by_index: dict[int, EpisodeQuality],
    count: int,
    *,
    min_quality: float | None = None,
    oversample_top_frac: float = 0.0,
) -> list[int]:
    """品質降順で上位 count を返す。不足時は品質順のまま足りる分だけ。

    oversample_top_frac > 0 なら、上位 frac を重複させて「重み付け」相当にする
    （エピソードリストに複数回載せる = 学習頻度↑）。
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    ranked = sorted(
        pool,
        key=lambda idx: (
            quality_by_index[idx].quality_score
            if idx in quality_by_index
            else float("-inf"),
            -idx,
        ),
        reverse=True,
    )
    if min_quality is not None:
        filtered = [
            idx
            for idx in ranked
            if idx in quality_by_index
            and quality_by_index[idx].quality_score >= min_quality
        ]
        # 全滅回避: 閾値で足りなければ品質上位にフォールバック
        ranked = filtered if len(filtered) >= max(1, count // 2) else ranked

    picked = ranked[: min(count, len(ranked))]
    if oversample_top_frac <= 0 or len(picked) < 2:
        return picked

    n_boost = max(1, int(round(len(picked) * oversample_top_frac)))
    boost = picked[:n_boost]
    # 上位を 1 回余分に混ぜる（順序は品質順を維持しつつ末尾に追加）
    return picked + boost


def load_quality_cache(path: Path) -> dict[int, EpisodeQuality]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    episodes = raw.get("episodes", raw)
    out: dict[int, EpisodeQuality] = {}
    for key, value in episodes.items():
        ep = EpisodeQuality(**value)
        out[int(key)] = ep
    return out


def save_quality_cache(
    path: Path,
    qualities: Iterable[EpisodeQuality],
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "episodes": {str(q.episode_index): q.to_dict() for q in qualities},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_quality(qualities: dict[int, EpisodeQuality]) -> dict[str, float]:
    if not qualities:
        return {}
    jerks = [q.avg_cartesian_jerk for q in qualities.values() if np.isfinite(q.avg_cartesian_jerk)]
    sparcs = [q.sparc for q in qualities.values() if np.isfinite(q.sparc)]
    oris = [q.orientation_path_length for q in qualities.values()]
    scores = [q.quality_score for q in qualities.values()]
    return {
        "n_episodes": float(len(qualities)),
        "mean_quality_score": float(np.mean(scores)),
        "mean_avg_cartesian_jerk": float(np.mean(jerks)) if jerks else float("nan"),
        "mean_sparc": float(np.mean(sparcs)) if sparcs else float("nan"),
        "mean_orientation_path_length": float(np.mean(oris)) if oris else float("nan"),
    }


def group_indices_by_parquet(
    meta: Any,
    episode_indices: list[int],
) -> dict[Path, list[int]]:
    """episode → parquet パスでグループ化（一括読み込み用）。"""
    grouped: dict[Path, list[int]] = defaultdict(list)
    for ep_idx in episode_indices:
        rel = meta.get_data_file_path(int(ep_idx))
        grouped[meta.root / rel].append(int(ep_idx))
    return grouped
