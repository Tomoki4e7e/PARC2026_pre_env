"""Lightweight action-conditioned collision critic for the stove task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def stove_geometry_features(image: np.ndarray) -> np.ndarray:
    """Return detection and normalized bowl/obstacle image geometry (8-D)."""
    rgb = np.asarray(image)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    missing = np.zeros(8, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return missing
    rgb = rgb.astype(np.float32, copy=False)
    height, width = rgb.shape[:2]
    scale = min(height, width) / 256.0
    yy, xx = np.mgrid[:height, :width]
    cream_mask = (
        (rgb[:, :, 2] > 1.15 * rgb[:, :, 0])
        & (rgb[:, :, 2] > 1.10 * rgb[:, :, 1])
        & (rgb[:, :, 2] > 55.0)
        & (yy < 0.60 * height)
    )
    if int(cream_mask.sum()) < max(8, int(40 * scale * scale)):
        return missing
    cream_x = float(xx[cream_mask].mean())
    cream_y = float(yy[cream_mask].mean())
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
        return missing
    bowl_x = float((weights * xx).sum() / weight_sum)
    bowl_y = float((weights * yy).sum() / weight_sum)
    bx, by, cx, cy = bowl_x / width, bowl_y / height, cream_x / width, cream_y / height
    dx, dy = cx - bx, cy - by
    return np.asarray([1.0, bx, by, cx, cy, dx, dy, np.hypot(dx, dy)], dtype=np.float32)


@dataclass(frozen=True)
class CollisionCriticConfig:
    state_dim: int = 8
    action_dim: int = 7
    visual_dim: int = 8
    hidden_dims: tuple[int, ...] = (128, 128)
    input_mean: tuple[float, ...] = (0.0,) * 23
    input_std: tuple[float, ...] = (1.0,) * 23
    threshold: float = 0.5
    risk_margin: float = 0.05
    horizon: int = 12

    @classmethod
    def from_json(cls, path: Path) -> "CollisionCriticConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hidden_dims", "input_mean", "input_std"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "visual_dim": self.visual_dim,
            "hidden_dims": list(self.hidden_dims),
            "input_mean": list(self.input_mean),
            "input_std": list(self.input_std),
            "threshold": self.threshold,
            "risk_margin": self.risk_margin,
            "horizon": self.horizon,
        }


class CollisionCriticNetwork(torch.nn.Module):
    def __init__(self, config: CollisionCriticConfig) -> None:
        super().__init__()
        dims = [config.state_dim + config.action_dim + config.visual_dim,
                *config.hidden_dims, 1]
        layers: list[torch.nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(torch.nn.SiLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class CollisionCritic:
    def __init__(self, model_dir: str | Path, device: torch.device) -> None:
        from safetensors.torch import load_file

        root = Path(model_dir)
        self.config = CollisionCriticConfig.from_json(root / "config.json")
        self.model = CollisionCriticNetwork(self.config).to(device)
        self.model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)))
        self.model.eval()
        self.device = device
        self.mean = torch.tensor(self.config.input_mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(self.config.input_std, dtype=torch.float32, device=device).clamp_min(1e-6)

    def risk(self, state: np.ndarray, action: np.ndarray, image: np.ndarray) -> tuple[float, bool]:
        visual = stove_geometry_features(image)
        detected = bool(visual[0] > 0.5)
        features = np.concatenate([
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            visual,
        ])
        inputs = (torch.as_tensor(features, device=self.device) - self.mean) / self.std
        with torch.inference_mode():
            risk = torch.sigmoid(self.model(inputs.unsqueeze(0)))[0]
        return float(risk.cpu()), detected
