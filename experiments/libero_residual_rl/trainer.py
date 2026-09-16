from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .base_policy import BasePolicy
from .config import ExperimentConfig
from .env import ResidualLiberoEnv
from .models import ConditionalFlowResidualPolicy, TwinQCritic
from .replay import ReplayBatch, ReplayBuffer
from .task_catalog import TaskSpec


class ObservationHistory:
    def __init__(self, length: int):
        self.frames: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=length)
        self.length = length

    @staticmethod
    def frame(obs: dict[str, np.ndarray], base_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(
            (
                np.asarray(obs["agentview_image"]).transpose(2, 0, 1),
                np.asarray(obs["robot0_eye_in_hand_image"]).transpose(2, 0, 1),
            )
        )
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32)[:3],
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                np.asarray(base_action, dtype=np.float32),
            )
        )
        return images.astype(np.uint8), state.astype(np.float32)

    def append(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> None:
        frame = self.frame(obs, base_action)
        if not self.frames:
            self.frames.extend([frame] * self.length)
        else:
            self.frames.append(frame)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        images, vectors = zip(*self.frames)
        return np.stack(images), np.stack(vectors)


def _torch_batch(batch: ReplayBatch, device: torch.device) -> tuple[torch.Tensor, ...]:
    images = torch.as_tensor(batch.images, device=device, dtype=torch.float32) / 255.0
    vectors = torch.as_tensor(batch.vectors, device=device)
    residuals = torch.as_tensor(batch.residuals, device=device)
    rewards = torch.as_tensor(batch.rewards, device=device).reshape(-1, 1)
    next_images = torch.as_tensor(batch.next_images, device=device, dtype=torch.float32) / 255.0
    next_vectors = torch.as_tensor(batch.next_vectors, device=device)
    dones = torch.as_tensor(batch.dones, device=device).reshape(-1, 1)
    return images, vectors, residuals, rewards, next_images, next_vectors, dones


class OnlineResidualTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        device: str | torch.device | None = None,
    ):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = ConditionalFlowResidualPolicy(
            horizon=config.horizon,
            flow_steps=config.flow_steps,
            limits=config.limits,
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQCritic(residual_dim=config.horizon * 7).to(self.device)
        self.critic_target = self.critic.target_copy().to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed)
        self.rng = random.Random(config.seed)
        self.updates = 0
        self.metrics: list[dict[str, float]] = []

    def act(self, images: np.ndarray, vectors: np.ndarray, explore: bool = True) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            image_tensor = torch.as_tensor(images[None], device=self.device, dtype=torch.float32) / 255.0
            vector_tensor = torch.as_tensor(vectors[None], device=self.device)
            chunk = self.actor(image_tensor, vector_tensor)[0].cpu().numpy()
        self.actor.train()
        if explore:
            limits = np.asarray(self.config.limits.vector(), dtype=np.float32)
            chunk += np.random.normal(0, self.config.exploration_std, chunk.shape) * limits
            chunk = np.clip(chunk, -limits, limits)
        return chunk.astype(np.float32)

    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.lerp_(source_parameter, self.config.tau)

    def update(self) -> dict[str, float]:
        batch = _torch_batch(self.replay.sample(self.config.batch_size), self.device)
        images, vectors, residuals, rewards, next_images, next_vectors, dones = batch
        with torch.no_grad():
            next_condition = self.actor_target.condition(next_images, next_vectors)
            next_residual = self.actor_target.sample_from_condition(next_condition)
            target_q1, target_q2 = self.critic_target(next_condition, next_residual)
            target_q = rewards + self.config.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)
            condition = self.actor.condition(images, vectors)
        q1, q2 = self.critic(condition, residuals)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        cfm_loss_value = 0.0
        if self.updates % self.config.policy_delay == 0:
            condition = self.actor.condition(images, vectors)
            sampled = self.actor.sample_from_condition(condition)
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            q_loss = -self.critic.q1_only(condition, sampled).mean()
            residual_penalty = sampled.square().mean()
            smoothness = (sampled[:, 1:] - sampled[:, :-1]).square().mean()
            positive = rewards[:, 0] > 0.05
            cfm_loss = (
                self.actor.flow_matching_loss(condition[positive], residuals[positive])
                if positive.any()
                else torch.zeros((), device=self.device)
            )
            actor_loss = (
                q_loss
                + self.config.cfm_weight * cfm_loss
                + self.config.residual_weight * residual_penalty
                + self.config.smoothness_weight * smoothness
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            self._soft_update(self.actor, self.actor_target)
            actor_loss_value = float(actor_loss.detach())
            cfm_loss_value = float(cfm_loss.detach())
        self._soft_update(self.critic, self.critic_target)
        self.updates += 1
        result = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": actor_loss_value,
            "cfm_loss": cfm_loss_value,
            "mean_reward": float(rewards.mean()),
        }
        self.metrics.append(result)
        return result

    def train(
        self,
        specs: list[TaskSpec],
        base_policy_factory: Callable[[], BasePolicy],
        *,
        episodes: int | None = None,
        log_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        episode_logs: list[dict[str, Any]] = []
        count = episodes or self.config.train_episodes
        base_policy = base_policy_factory()
        for episode in range(count):
            spec = specs[episode % len(specs)]
            env = ResidualLiberoEnv(
                spec, base_policy, self.config, terminate_on_grasp=True
            )
            started = time.time()
            total_reward = 0.0
            updates_before = self.updates
            final_info: dict[str, Any] = {}
            try:
                obs = env.reset()
                base_action, obs = env.prepare_step()
                history = ObservationHistory(self.config.history)
                history.append(obs, base_action)
                images, vectors = history.arrays()
                for _ in range(self.config.max_episode_steps):
                    residual_chunk = self.act(images, vectors, explore=True)
                    next_obs, reward, done, final_info = env.step(base_action, residual_chunk[0])
                    next_base, next_obs = env.prepare_step() if not done else (base_action, next_obs)
                    next_history = deepcopy(history)
                    next_history.append(next_obs, next_base)
                    next_images, next_vectors = next_history.arrays()
                    self.replay.add(
                        images,
                        vectors,
                        residual_chunk,
                        reward,
                        next_images,
                        next_vectors,
                        done,
                    )
                    total_reward += reward
                    if len(self.replay) >= max(self.config.batch_size, self.config.warmup_steps):
                        self.update()
                    history = next_history
                    images, vectors = next_images, next_vectors
                    base_action = next_base
                    if done:
                        break
            finally:
                env.close()
            record = {
                "episode": episode,
                "task": spec.task_name,
                "axis": spec.axis,
                "level": spec.level,
                "seed": spec.seed,
                "reward": total_reward,
                "steps": env.step_count,
                "grasp_success": bool(final_info.get("grasp_success", False)),
                "task_success": bool(final_info.get("task_success", False)),
                "updates": self.updates - updates_before,
                "elapsed_sec": time.time() - started,
                "applied_perturbation": env.applied_perturbation,
            }
            episode_logs.append(record)
            if log_path:
                target = Path(log_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        return episode_logs

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "updates": self.updates,
            },
            target / "checkpoint.pt",
        )
        (target / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")
        (target / "training_metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")

    def load(self, directory: str | Path) -> None:
        payload = torch.load(Path(directory) / "checkpoint.pt", map_location=self.device, weights_only=True)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_target.load_state_dict(payload["actor"])
        self.critic_target.load_state_dict(payload["critic"])
        self.updates = int(payload["updates"])

