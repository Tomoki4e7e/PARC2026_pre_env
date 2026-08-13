"""Small task-agnostic residual action policy.

The frozen SmolVLA action and 8-D robot state are the only inputs.  The network
changes translation/rotation after temporal ensembling; gripper is untouched.
"""

from __future__ import annotations

import json
import os
import hashlib
import re
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


def visual_features(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Compact, deterministic visual context for the residual gate."""
    values = []
    for image in (front, wrist):
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim != 3:
            raise ValueError(f"expected an RGB image, got {rgb.shape}")
        if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb[..., :3] / (255.0 if rgb.max(initial=0.0) > 1.5 else 1.0)
        height, width = rgb.shape[:2]
        values.extend(rgb.mean((0, 1)))
        values.extend(rgb.std((0, 1)))
        for ys, xs in (
            (slice(0, height // 2), slice(0, width // 2)),
            (slice(0, height // 2), slice(width // 2, width)),
            (slice(height // 2, height), slice(0, width // 2)),
            (slice(height // 2, height), slice(width // 2, width)),
        ):
            values.extend(rgb[ys, xs].mean((0, 1)))
    return np.asarray(values, dtype=np.float32)


def language_features(instruction: str, dimensions: int = 64) -> np.ndarray:
    """Stable signed feature hashing; unlike Python hash this is reproducible."""
    output = np.zeros(dimensions, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", instruction.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        output[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(output))
    return output / norm if norm > 0.0 else output


@dataclass(frozen=True)
class GatedResidualPolicyConfig(ResidualPolicyConfig):
    policy_type: str = "visual_language_gated_residual"
    visual_dim: int = 36
    language_dim: int = 64
    allowed_instructions: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: Path) -> "GatedResidualPolicyConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hidden_dims", "state_mean", "state_std", "base_action_mean", "base_action_std", "allowed_instructions"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "policy_type": self.policy_type,
            "visual_dim": self.visual_dim,
            "language_dim": self.language_dim,
            "allowed_instructions": list(self.allowed_instructions),
        }


class VisualLanguageGatedResidualNetwork(torch.nn.Module):
    def __init__(self, config: GatedResidualPolicyConfig) -> None:
        super().__init__()
        input_dim = config.state_dim + config.action_dim + config.visual_dim + config.language_dim
        dims = [input_dim, *config.hidden_dims]
        layers = []
        for source, target in zip(dims, dims[1:]):
            layers.extend([torch.nn.Linear(source, target), torch.nn.SiLU()])
        self.backbone = torch.nn.Sequential(*layers)
        self.correction_head = torch.nn.Linear(dims[-1], 6)
        self.gate_head = torch.nn.Linear(dims[-1], 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        return torch.tanh(self.correction_head(hidden)), torch.sigmoid(self.gate_head(hidden))


class VisualLanguageGatedResidualPolicy:
    def __init__(self, model_dir: str | Path, device: torch.device) -> None:
        from safetensors.torch import load_file

        root = Path(model_dir)
        self.config = GatedResidualPolicyConfig.from_json(root / "config.json")
        self.model = VisualLanguageGatedResidualNetwork(self.config).to(device)
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
        self.output_scale = float(os.environ.get("PARC_RESIDUAL_SCALE", str(self.config.output_scale)))

    def apply(
        self,
        state: np.ndarray,
        base_action: np.ndarray,
        *,
        front: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_instruction = " ".join(instruction.lower().split())
        if self.config.allowed_instructions and normalized_instruction not in self.config.allowed_instructions:
            return np.asarray(base_action, dtype=np.float32).copy(), np.zeros(6, dtype=np.float32)
        context = np.concatenate([
            (np.asarray(state, dtype=np.float32) - np.asarray(self.config.state_mean))
            / np.maximum(np.asarray(self.config.state_std), 1.0e-6),
            (np.asarray(base_action, dtype=np.float32) - np.asarray(self.config.base_action_mean))
            / np.maximum(np.asarray(self.config.base_action_std), 1.0e-6),
            visual_features(front, wrist),
            language_features(instruction, self.config.language_dim),
        ]).astype(np.float32)
        with torch.inference_mode():
            direction, gate = self.model(torch.from_numpy(context).to(self.device).unsqueeze(0))
            correction = direction[0] * gate[0] * self.limits * self.output_scale
        correction_np = correction.cpu().numpy().astype(np.float32)
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[:6] = np.clip(action[:6] + correction_np, -1.0, 1.0)
        return action, correction_np


def load_residual_policy(model_dir: str | Path, device: torch.device):
    root = Path(model_dir)
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if raw.get("policy_type") == "visual_language_gated_residual":
        return VisualLanguageGatedResidualPolicy(root, device)
    return ResidualActionPolicy(root, device)


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
