"""Small task-agnostic residual action policy.

The frozen SmolVLA action and 8-D robot state are the only inputs.  The network
changes translation/rotation after temporal ensembling; gripper is untouched.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualPolicyConfig:
    state_dim: int = 8
    action_dim: int = 7
    hidden_dims: tuple[int, ...] = (128, 128)
    translation_limit: float = 0.02
    rotation_limit: float = 0.03
    output_scale: float = 1.0
    state_mean: tuple[float, ...] = (0.0,) * 8
    state_std: tuple[float, ...] = (1.0,) * 8
    base_action_mean: tuple[float, ...] = (0.0,) * 7
    base_action_std: tuple[float, ...] = (1.0,) * 7

    @classmethod
    def from_json(cls, path: Path) -> "ResidualPolicyConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hidden_dims", "state_mean", "state_std", "base_action_mean", "base_action_std"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "translation_limit": self.translation_limit,
            "rotation_limit": self.rotation_limit,
            "output_scale": self.output_scale,
            "state_mean": list(self.state_mean),
            "state_std": list(self.state_std),
            "base_action_mean": list(self.base_action_mean),
            "base_action_std": list(self.base_action_std),
        }


class ResidualActionNetwork(torch.nn.Module):
    def __init__(self, config: ResidualPolicyConfig) -> None:
        super().__init__()
        dims = [config.state_dim + config.action_dim, *config.hidden_dims, 6]
        layers: list[torch.nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(torch.nn.SiLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(inputs))


class ResidualActionPolicy:
    def __init__(self, model_dir: str | Path, device: torch.device) -> None:
        from safetensors.torch import load_file

        root = Path(model_dir)
        self.config = ResidualPolicyConfig.from_json(root / "config.json")
        self.model = ResidualActionNetwork(self.config).to(device)
        self.model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)))
        self.model.eval()
        self.device = device
        self.state_mean = torch.tensor(self.config.state_mean, device=device)
        self.state_std = torch.tensor(self.config.state_std, device=device).clamp_min(1.0e-6)
        self.action_mean = torch.tensor(self.config.base_action_mean, device=device)
        self.action_std = torch.tensor(self.config.base_action_std, device=device).clamp_min(1.0e-6)
        self.limits = torch.tensor(
            [self.config.translation_limit] * 3 + [self.config.rotation_limit] * 3,
            device=device,
        )
        self.output_scale = float(
            os.environ.get("PARC_RESIDUAL_SCALE", str(self.config.output_scale))
        )
        if not 0.0 <= self.output_scale <= 1.0:
            raise ValueError("PARC_RESIDUAL_SCALE must be in [0, 1]")

    def correction(self, state: np.ndarray, base_action: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(base_action, dtype=torch.float32, device=self.device)
        inputs = torch.cat(
            [(state_t - self.state_mean) / self.state_std,
             (action_t - self.action_mean) / self.action_std]
        ).unsqueeze(0)
        with torch.inference_mode():
            correction = self.model(inputs)[0] * self.limits * self.output_scale
        return correction.cpu().numpy().astype(np.float32)

    def apply(self, state: np.ndarray, base_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correction = self.correction(state, base_action)
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[:6] = np.clip(action[:6] + correction, -1.0, 1.0)
        return action, correction
