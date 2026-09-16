from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import time

import numpy as np
import torch

from .base_policy import BasePolicy
from .config import ExperimentConfig
from .env import ResidualLiberoEnv
from .models import ConditionalFlowResidualPolicy
from .task_catalog import TaskSpec
from .trainer import ObservationHistory


def evaluate_policy(
    specs: list[TaskSpec],
    base_policy: BasePolicy,
    config: ExperimentConfig,
    *,
    actor: ConditionalFlowResidualPolicy | None = None,
    output: str | Path,
    condition_name: str,
    device: str | torch.device | None = None,
    terminate_on_grasp: bool = False,
    report_task_success: bool = True,
) -> list[dict[str, Any]]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if actor is not None:
        actor = actor.to(torch_device).eval()
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        env = ResidualLiberoEnv(
            spec,
            base_policy,
            config,
            terminate_on_grasp=terminate_on_grasp,
        )
        residual_norms: list[float] = []
        inference_times: list[float] = []
        base_inference_times: list[float] = []
        final_info: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            obs = env.reset()
            base_action, obs = env.prepare_step()
            base_inference_times.append(env.last_base_latency)
            history = ObservationHistory(config.history)
            history.append(obs, base_action)
            for _ in range(config.max_episode_steps):
                images, vectors = history.arrays()
                inference_start = time.perf_counter()
                if actor is None:
                    residual = np.zeros(7, dtype=np.float32)
                else:
                    with torch.inference_mode():
                        image_tensor = torch.as_tensor(
                            images[None], device=torch_device, dtype=torch.float32
                        ) / 255.0
                        vector_tensor = torch.as_tensor(vectors[None], device=torch_device)
                        residual = actor(image_tensor, vector_tensor)[0, 0].cpu().numpy()
                inference_times.append(time.perf_counter() - inference_start)
                residual_norms.append(float(np.linalg.norm(residual)))
                next_obs, _, done, final_info = env.step(base_action, residual)
                if done:
                    break
                base_action, next_obs = env.prepare_step()
                base_inference_times.append(env.last_base_latency)
                history.append(next_obs, base_action)
        finally:
            env.close()
        tracker = env.tracker
        record = {
            "condition": condition_name,
            "episode_index": index,
            "base_task": spec.base_task,
            "task_name": spec.task_name,
            "axis": spec.axis,
            "level": spec.level,
            "difficulty": spec.difficulty,
            "seed": spec.seed,
            "grasp_success": bool(tracker and tracker.first_success_step is not None),
            "task_success": bool(env.task_success) if report_task_success else None,
            "time_to_grasp_steps": tracker.first_success_step if tracker else None,
            "max_contact_force": tracker.max_contact_force if tracker else None,
            "grasp_point_error": tracker.min_grasp_point_error if tracker else None,
            "failure_mode": tracker.failure_mode() if tracker else "setup_failure",
            "residual_norm_mean": float(np.mean(residual_norms)) if residual_norms else 0.0,
            "residual_norm_max": float(np.max(residual_norms)) if residual_norms else 0.0,
            "steps": env.step_count,
            "elapsed_sec": time.perf_counter() - started,
            "inference_sec_max": max(inference_times, default=0.0),
            "base_inference_sec_max": max(base_inference_times, default=0.0),
            "combined_inference_sec_max": max(
                (
                    base + residual
                    for base, residual in zip(base_inference_times, inference_times)
                ),
                default=0.0,
            ),
            "applied_perturbation": env.applied_perturbation,
            "last_grasp": final_info.get("grasp"),
        }
        records.append(record)
        with target.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    return records


def load_actor(
    checkpoint_dir: str | Path,
    config: ExperimentConfig,
    device: str | torch.device | None = None,
) -> ConditionalFlowResidualPolicy:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    actor = ConditionalFlowResidualPolicy(
        horizon=config.horizon,
        flow_steps=config.flow_steps,
        limits=config.limits,
    )
    payload = torch.load(
        Path(checkpoint_dir) / "checkpoint.pt", map_location=torch_device, weights_only=True
    )
    actor.load_state_dict(payload["actor"])
    return actor.to(torch_device).eval()

