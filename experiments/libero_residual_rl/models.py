from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .config import ResidualLimits


class TemporalObservationEncoder(nn.Module):
    def __init__(self, vector_dim: int = 15, visual_dim: int = 96, condition_dim: int = 256):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, visual_dim),
            nn.LayerNorm(visual_dim),
        )
        self.frame = nn.Sequential(
            nn.Linear(visual_dim + vector_dim, condition_dim),
            nn.SiLU(),
            nn.LayerNorm(condition_dim),
        )
        self.temporal = nn.GRU(condition_dim, condition_dim, batch_first=True)

    def forward(self, images: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6 or images.shape[2:4] != (2, 3):
            raise ValueError(f"Expected images (B,L,2,3,H,W), got {tuple(images.shape)}")
        batch, length, _, _, height, width = images.shape
        visual = images.reshape(batch * length, 6, height, width)
        visual = self.visual(visual).reshape(batch, length, -1)
        frames = self.frame(torch.cat((visual, vectors), dim=-1))
        _, hidden = self.temporal(frames)
        return hidden[-1]


class ConditionalVelocity(nn.Module):
    def __init__(self, sample_dim: int, condition_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(sample_dim + condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sample_dim),
        )

    def forward(self, sample: torch.Tensor, flow_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if flow_time.ndim == 1:
            flow_time = flow_time[:, None]
        return self.network(torch.cat((sample, flow_time, condition), dim=-1))


class ConditionalFlowResidualPolicy(nn.Module):
    def __init__(
        self,
        *,
        horizon: int = 4,
        action_dim: int = 7,
        condition_dim: int = 256,
        flow_steps: int = 8,
        limits: ResidualLimits | None = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.sample_dim = horizon * action_dim
        self.flow_steps = flow_steps
        self.encoder = TemporalObservationEncoder(condition_dim=condition_dim)
        self.velocity = ConditionalVelocity(self.sample_dim, condition_dim)
        limit = torch.tensor((limits or ResidualLimits()).vector(), dtype=torch.float32)
        self.register_buffer("limits", limit.repeat(horizon))

    def condition(self, images: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        return self.encoder(images, vectors)

    def sample_from_condition(
        self,
        condition: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        flow_steps: int | None = None,
    ) -> torch.Tensor:
        steps = flow_steps or self.flow_steps
        state = torch.randn(
            condition.shape[0], self.sample_dim, device=condition.device, dtype=condition.dtype
        ) if noise is None else noise
        dt = 1.0 / steps
        for index in range(steps):
            flow_time = torch.full(
                (condition.shape[0],), index / steps, device=condition.device, dtype=condition.dtype
            )
            state = state + dt * self.velocity(state, flow_time, condition)
        bounded = torch.tanh(state) * self.limits
        return bounded.reshape(-1, self.horizon, self.action_dim)

    def forward(
        self, images: torch.Tensor, vectors: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.sample_from_condition(self.condition(images, vectors), noise)

    def flow_matching_loss(
        self, condition: torch.Tensor, target_residual: torch.Tensor
    ) -> torch.Tensor:
        target = target_residual.reshape(-1, self.sample_dim)
        normalized = torch.clamp(target / self.limits, -0.999, 0.999)
        target_latent = torch.atanh(normalized)
        noise = torch.randn_like(target_latent)
        flow_time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        path = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * target_latent
        prediction = self.velocity(path, flow_time, condition)
        return F.mse_loss(prediction, target_latent - noise)


class TwinQCritic(nn.Module):
    def __init__(self, condition_dim: int = 256, residual_dim: int = 28):
        super().__init__()
        self.q1 = self._head(condition_dim + residual_dim)
        self.q2 = self._head(condition_dim + residual_dim)

    @staticmethod
    def _head(input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, condition: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((condition, residual.reshape(residual.shape[0], -1)), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def q1_only(self, condition: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((condition, residual.reshape(residual.shape[0], -1)), dim=-1)
        return self.q1(inputs)

    def target_copy(self) -> "TwinQCritic":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target

