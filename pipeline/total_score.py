from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_BUILTIN_TOTAL_SCORE_DEFAULT: dict[str, Any] = {

    "control_dt": 0.05,
    "use_total_score_as_overall": False,
    "weights": {},
    "collision": {"enabled": True, "threshold_m": 0.001},
}

_BUILTIN_NORMALIZATION_DEFAULT: dict[str, Any] = {

    "references": {},
}


def _candidate_config_paths(filename: str) -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return [
        root / filename,
        Path(f"/workspace/{filename}"),
        Path.cwd() / filename,
    ]


def _load_json_config(
    filename: str,
    env_var: str,
    builtin_default: dict[str, Any],
    explicit_path: str | Path | None,
    merge_keys: tuple[str, ...],
) -> dict[str, Any]:
    explicit = explicit_path or os.environ.get(env_var)
    candidates = [Path(explicit)] if explicit else _candidate_config_paths(filename)

    for candidate in candidates:
        if candidate.is_file():
            try:
                cfg = json.loads(candidate.read_text())
                logger.info("%s を読み込み: %s", filename, candidate)
                merged = {**builtin_default, **cfg}
                for k in merge_keys:
                    merged[k] = {**builtin_default.get(k, {}), **cfg.get(k, {})}
                return merged
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "%s の読み込み失敗 (%s): %s。次候補/内蔵にフォールバック。", filename, candidate, e,
                )

    logger.warning("%s が見つからないため内蔵デフォルトを使用。", filename)
    return dict(builtin_default)


def load_scoring_config(
    total_score_config_path: str | Path | None = None,
    normalization_config_path: str | Path | None = None,
) -> dict[str, Any]:
    total_score_cfg = _load_json_config(
        "total_score_config.json", "TOTAL_SCORE_CONFIG", _BUILTIN_TOTAL_SCORE_DEFAULT,
        total_score_config_path, merge_keys=("weights", "collision"),
    )
    normalization_cfg = _load_json_config(
        "normalization_config.json", "NORMALIZATION_CONFIG", _BUILTIN_NORMALIZATION_DEFAULT,
        normalization_config_path, merge_keys=(),
    )

    merged = dict(total_score_cfg)
    merged["references"] = {
        **_BUILTIN_NORMALIZATION_DEFAULT["references"],
        **normalization_cfg.get("references", {}),
    }
    return merged


def compute_sparc(
    eef_pos_seq: np.ndarray,
    dt: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
    min_speed: float = 1e-6,
) -> float:
    if len(eef_pos_seq) < 2:
        return float("nan")
    velocity = np.gradient(eef_pos_seq, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    if np.max(np.abs(speed)) < min_speed:
        return float("nan")
    n = len(speed)
    nfft = int(2 ** np.ceil(np.log2(n)) * padlevel)
    fft_mag = np.abs(np.fft.rfft(speed, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=dt)
    mag_max = fft_mag.max()
    if mag_max == 0.0:
        return float("nan")
    magnitude = fft_mag / mag_max
    above_th = np.where(magnitude >= amp_th)[0]
    if len(above_th) == 0:
        return 0.0
    fc_adaptive = min(float(freq[above_th[-1]]), fc)
    freq_mask = freq <= fc_adaptive
    if np.sum(freq_mask) < 2:
        return 0.0
    freq_sel = freq[freq_mask]
    mag_sel = magnitude[freq_mask]
    d_freq = np.diff(freq_sel)
    d_mag = np.diff(mag_sel)
    return -float(np.sum(np.sqrt((d_freq / fc_adaptive) ** 2 + d_mag ** 2)))
