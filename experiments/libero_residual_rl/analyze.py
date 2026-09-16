from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import json

import numpy as np
from scipy import stats


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rate_interval(values: list[bool], seed: int = 42) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(array, size=len(array), replace=True).mean() for _ in range(2_000)]
    )
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["base_task"], record["axis"], int(record["level"]))].append(record)
    output: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        rate, low, high = _rate_interval([bool(row["grasp_success"]) for row in rows])
        output["/".join(map(str, key))] = {
            "episodes": len(rows),
            "gsr": rate,
            "gsr_ci95": [low, high],
            "tsr": float(np.mean([row["task_success"] for row in rows])),
            "contact_force_mean": float(np.mean([row["max_contact_force"] for row in rows])),
            "grasp_point_error_mean": float(np.mean([row["grasp_point_error"] for row in rows])),
            "residual_norm_mean": float(np.mean([row["residual_norm_mean"] for row in rows])),
        }
    return output


def compare_paired(
    base_records: list[dict[str, Any]], residual_records: list[dict[str, Any]]
) -> dict[str, Any]:
    key_fields = ("base_task", "task_name", "axis", "level", "seed")

    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {tuple(row[field] for field in key_fields): row for row in rows}

    base = keyed(base_records)
    residual = keyed(residual_records)
    common = sorted(base.keys() & residual.keys(), key=str)
    base_success = np.asarray([base[key]["grasp_success"] for key in common], dtype=bool)
    residual_success = np.asarray([residual[key]["grasp_success"] for key in common], dtype=bool)
    base_only = int(np.sum(base_success & ~residual_success))
    residual_only = int(np.sum(~base_success & residual_success))
    discordant = base_only + residual_only
    mcnemar_p = (
        float(stats.binomtest(min(base_only, residual_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    residual_levels = np.asarray([residual[key]["level"] for key in common], dtype=float)
    residual_norms = np.asarray([residual[key]["residual_norm_mean"] for key in common])
    correlation = stats.spearmanr(residual_levels, residual_norms)
    return {
        "paired_episodes": len(common),
        "base_gsr": float(base_success.mean()) if len(common) else 0.0,
        "residual_gsr": float(residual_success.mean()) if len(common) else 0.0,
        "gsr_delta": float(residual_success.mean() - base_success.mean()) if len(common) else 0.0,
        "base_only_successes": base_only,
        "residual_only_successes": residual_only,
        "mcnemar_exact_p": mcnemar_p,
        "residual_norm_level_spearman": {
            "rho": float(correlation.statistic) if np.isfinite(correlation.statistic) else None,
            "p": float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None,
        },
    }


def write_analysis(
    base_path: str | Path, residual_path: str | Path, output: str | Path
) -> dict[str, Any]:
    base = read_jsonl(base_path)
    residual = read_jsonl(residual_path)
    result = {
        "base": summarize(base),
        "residual": summarize(residual),
        "paired": compare_paired(base, residual),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result

