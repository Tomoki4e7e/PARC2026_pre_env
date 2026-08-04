import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Track
from .rollout import EpisodeResult, TaskResult
from .total_score import (
    compute_sparc,
    load_scoring_config,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    value: float
    description: str


@dataclass
class TaskScore:
    task_name: str
    success_rate: float
    metrics: list[MetricResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "success_rate": self.success_rate,
            "metrics": {m.name: m.value for m in self.metrics},
        }


@dataclass
class TrackScore:
    track: Track
    benchmark_name: str
    task_scores: list[TaskScore]
    overall_score: float
    overall_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track.value,
            "benchmark": self.benchmark_name,
            "overall_score": self.overall_score,
            "overall_metrics": self.overall_metrics,
            "tasks": [ts.to_dict() for ts in self.task_scores],
        }


class Scorer:

    def __init__(self, scoring_config: dict | None = None) -> None:
        self.scoring_config = scoring_config if scoring_config is not None else load_scoring_config()

    def score_task(self, task_result: TaskResult) -> TaskScore:
        metrics = []


        success_rate = task_result.success_rate


        avg_steps = task_result.avg_steps
        if avg_steps > 0:
            metrics.append(MetricResult(
                name="avg_steps_to_success",
                value=avg_steps,
                description="成功エピソードの平均ステップ数",
            ))


        metrics.append(MetricResult(
            name="avg_episode_time_sec",
            value=task_result.avg_time,
            description="エピソードの平均実行時間（秒）",
        ))


        metrics.extend(self._compute_roboeeval_metrics(task_result.episodes))

        return TaskScore(
            task_name=task_result.task_info.name,
            success_rate=success_rate,
            metrics=metrics,
        )

    def score_track(
        self,
        track: Track,
        benchmark_name: str,
        task_results: list[TaskResult],
    ) -> TrackScore:
        task_scores = [self.score_task(tr) for tr in task_results]


        overall_success = np.mean([ts.success_rate for ts in task_scores])


        overall_metrics = {
            "mean_success_rate": float(overall_success),
            "min_success_rate": float(min(ts.success_rate for ts in task_scores)),
            "max_success_rate": float(max(ts.success_rate for ts in task_scores)),
            "n_tasks": len(task_scores),
            "n_total_episodes": sum(
                len(tr.episodes) for tr in task_results
            ),
        }


        all_metric_names = set()
        for ts in task_scores:
            all_metric_names.update(m.name for m in ts.metrics)

        for metric_name in all_metric_names:
            values = []
            for ts in task_scores:
                for m in ts.metrics:
                    if m.name == metric_name:
                        values.append(m.value)
            if values:
                overall_metrics[f"mean_{metric_name}"] = float(np.mean(values))


        overall_score = float(overall_success)

        logger.info(
            "Track %s スコア: overall=%.4f, 総合成功率 %.1f%% (%d タスク)",
            track.value, overall_score, overall_success * 100, len(task_scores),
        )

        return TrackScore(
            track=track,
            benchmark_name=benchmark_name,
            task_scores=task_scores,
            overall_score=overall_score,
            overall_metrics=overall_metrics,
        )

    def _compute_roboeeval_metrics(
        self, episodes: list[EpisodeResult]
    ) -> list[MetricResult]:
        ep_metrics: list[dict[str, float]] = []
        for ep in episodes:
            m = self._compute_episode_metrics(ep)
            if m:
                ep_metrics.append(m)

        results = []


        if episodes:
            collision_rate = float(np.mean([1.0 if ep.collided else 0.0 for ep in episodes]))
            results.append(MetricResult(
                name="collision_rate",
                value=collision_rate,
                description=_METRIC_DESCRIPTIONS["collision_rate"],
            ))

        if not ep_metrics:
            return results


        keys: set[str] = set()
        for m in ep_metrics:
            keys.update(m.keys())
        for key in sorted(keys):
            values = [m[key] for m in ep_metrics if key in m]
            if values:
                results.append(MetricResult(
                    name=key,
                    value=float(np.mean(values)),
                    description=_METRIC_DESCRIPTIONS.get(key, ""),
                ))
        return results

    def _compute_episode_metrics(
        self, ep: EpisodeResult, dt: float = 0.05
    ) -> dict[str, float] | None:
        n = ep.total_steps
        if n < 4:

            return None

        metrics: dict[str, float] = {}


        if ep.ee_positions:
            ee = np.array(ep.ee_positions)
            metrics["cartesian_path_length"] = float(
                np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1))
            )

        if ep.joint_positions:
            jp = np.array(ep.joint_positions)
            metrics["joint_path_length"] = float(
                np.sum(np.linalg.norm(np.diff(jp, axis=0), axis=1))
            )

        if ep.ee_orientations:
            quats = np.array(ep.ee_orientations)
            metrics["orientation_path_length"] = float(
                self._orientation_path_length(quats)
            )


        if ep.ee_positions and len(ep.ee_positions) >= 4:
            ee = np.array(ep.ee_positions)
            avg_j, rms_j = self._compute_jerk(ee, dt)
            metrics["avg_cartesian_jerk"] = avg_j
            metrics["rms_cartesian_jerk"] = rms_j

        if ep.joint_positions and len(ep.joint_positions) >= 4:
            jp = np.array(ep.joint_positions)
            avg_j, rms_j = self._compute_jerk(jp, dt)
            metrics["avg_joint_jerk"] = avg_j
            metrics["rms_joint_jerk"] = rms_j


        if ep.ee_positions and len(ep.ee_positions) >= 4:
            sparc = compute_sparc(np.array(ep.ee_positions), dt)
            if np.isfinite(sparc):
                metrics["sparc"] = float(sparc)

        return metrics

    @staticmethod
    def _compute_jerk(
        positions: np.ndarray, dt: float
    ) -> tuple[float, float]:
        vel = np.diff(positions, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        jerk_norms = np.linalg.norm(jerk, axis=1)
        avg_jerk = float(np.mean(jerk_norms))
        rms_jerk = float(math.sqrt(np.mean(jerk_norms ** 2)))
        return avg_jerk, rms_jerk

    @staticmethod
    def _orientation_path_length(quats: np.ndarray) -> float:
        total = 0.0
        for i in range(len(quats) - 1):
            dot = np.clip(np.abs(np.dot(quats[i], quats[i + 1])), 0.0, 1.0)
            total += 2.0 * math.acos(dot)
        return total


_METRIC_DESCRIPTIONS: dict[str, str] = {
    "cartesian_path_length": "EE位置のユークリッド累積移動量",
    "joint_path_length": "関節空間での累積移動量",
    "orientation_path_length": "EE姿勢の角度累積変化量 (rad)",
    "avg_cartesian_jerk": "EE位置の平均ジャーク (m/s³)",
    "rms_cartesian_jerk": "EE位置のRMSジャーク (m/s³)",
    "avg_joint_jerk": "関節空間の平均ジャーク (rad/s³)",
    "rms_joint_jerk": "関節空間のRMSジャーク (rad/s³)",
    "sparc": "SPARC（軌道の滑らかさ。0 に近いほど滑らか）",
    "collision_rate": "衝突が発生したエピソードの割合",
}


