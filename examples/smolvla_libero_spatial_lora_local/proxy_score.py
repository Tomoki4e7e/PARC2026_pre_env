"""Proxy / checkpoint 選定用の複合スコア。

成功・衝突・滑らかさ（jerk / SPARC / EEF 回転総量）を正規化して合成する。
重みは本番の非公開重みとは独立したローカル proxy 用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ローカル proxy 用の参照スケール（悪い側）。評価 JSON の mean_* をこれで 0〜1 正規化。
DEFAULT_REFS: dict[str, float] = {
    "mean_avg_cartesian_jerk": 20.0,
    "mean_sparc": -8.0,
    "mean_orientation_path_length": 8.0,
    "mean_cartesian_path_length": 4.0,
}

# ゲート: 成功を優先しつつ、衝突・滑らかさを減点
DEFAULT_WEIGHTS: dict[str, float] = {
    "success": 1.0,
    "collision_penalty": 0.35,
    "jerk_penalty": 0.12,
    "sparc_bonus": 0.08,
    "orient_penalty": 0.08,
    "path_penalty": 0.05,
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _low_is_good(value: float | None, ref: float) -> float:
    if value is None or not np.isfinite(value) or ref <= 0:
        return 0.0
    return _clip01(1.0 - float(value) / ref)


def _sparc_good(value: float | None, ref: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    # ref=-8 → 0, value=0 → 1
    return _clip01((float(value) - ref) / (0.0 - ref))


def extract_track_metrics(track: dict[str, Any]) -> dict[str, float | None]:
    overall = track.get("overall_metrics") or {}
    success = track.get("overall_score")
    if success is None:
        success = overall.get("mean_success_rate")
    return {
        "success": float(success) if success is not None else None,
        "collision_rate": _as_float(overall.get("mean_collision_rate")),
        "avg_cartesian_jerk": _as_float(overall.get("mean_avg_cartesian_jerk")),
        "sparc": _as_float(overall.get("mean_sparc")),
        "orientation_path_length": _as_float(overall.get("mean_orientation_path_length")),
        "cartesian_path_length": _as_float(overall.get("mean_cartesian_path_length")),
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def composite_from_metrics(
    metrics: dict[str, float | None],
    *,
    weights: dict[str, float] | None = None,
    refs: dict[str, float] | None = None,
) -> dict[str, float]:
    """metrics → composite と内訳。"""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    r = {**DEFAULT_REFS, **(refs or {})}

    success = float(metrics["success"] or 0.0)
    collision = float(metrics["collision_rate"] or 0.0)
    jerk_good = _low_is_good(metrics["avg_cartesian_jerk"], r["mean_avg_cartesian_jerk"])
    sparc_good = _sparc_good(metrics["sparc"], r["mean_sparc"])
    orient_good = _low_is_good(
        metrics["orientation_path_length"], r["mean_orientation_path_length"]
    )
    path_good = _low_is_good(
        metrics["cartesian_path_length"], r["mean_cartesian_path_length"]
    )

    composite = (
        w["success"] * success
        - w["collision_penalty"] * collision
        + w["jerk_penalty"] * (jerk_good - 0.5) * 2.0  # [-w, +w] 付近
        + w["sparc_bonus"] * (sparc_good - 0.5) * 2.0
        + w["orient_penalty"] * (orient_good - 0.5) * 2.0
        + w["path_penalty"] * (path_good - 0.5) * 2.0
    )
    return {
        "success": success,
        "collision_rate": collision,
        "jerk_good": jerk_good,
        "sparc_good": sparc_good,
        "orient_good": orient_good,
        "path_good": path_good,
        "composite": float(composite),
    }


def score_pipeline_json(
    path: Path | str,
    *,
    weights: dict[str, float] | None = None,
    refs: dict[str, float] | None = None,
) -> dict[str, float]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    track = result["tracks"][0]
    metrics = extract_track_metrics(track)
    return composite_from_metrics(metrics, weights=weights, refs=refs)


def blend_proxy_scores(
    inset: dict[str, float],
    holdout: dict[str, float],
    *,
    inset_weight: float = 0.4,
    holdout_weight: float = 0.6,
) -> dict[str, float]:
    """inset / hold-out の composite を従来どおり 0.4/0.6 で合成。"""
    proxy = inset_weight * inset["composite"] + holdout_weight * holdout["composite"]
    return {
        "inset_success": inset["success"],
        "inset_collision_rate": inset["collision_rate"],
        "inset_composite": inset["composite"],
        "holdout_success": holdout["success"],
        "holdout_collision_rate": holdout["collision_rate"],
        "holdout_composite": holdout["composite"],
        "proxy_score": float(proxy),
        # 互換: 旧コードが inset_score / holdout_avg を success 前提で見ていた箇所向け
        "inset_score": inset["success"],
        "holdout_avg": holdout["success"],
    }
