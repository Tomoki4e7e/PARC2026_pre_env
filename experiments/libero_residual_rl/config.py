from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json


ROOT = Path(__file__).resolve().parents[2]
LIBERO_ROOT = ROOT / "LIBERO-plus" / "libero" / "libero"
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class ResidualLimits:
    translation: float = 0.05
    rotation: float = 0.08
    gripper: float = 0.20
    rate: float = 0.04

    def vector(self) -> tuple[float, ...]:
        return (
            self.translation,
            self.translation,
            self.translation,
            self.rotation,
            self.rotation,
            self.rotation,
            self.gripper,
        )


@dataclass
class ExperimentConfig:
    seed: int = 42
    control_freq: int = 20
    history: int = 4
    horizon: int = 4
    image_size: int = 96
    flow_steps: int = 8
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    train_episodes: int = 500
    max_episode_steps: int = 300
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    exploration_std: float = 0.15
    cfm_weight: float = 0.10
    residual_weight: float = 0.02
    smoothness_weight: float = 0.02
    grasp_hold_seconds: float = 0.5
    limits: ResidualLimits = field(default_factory=ResidualLimits)
    artifact_root: Path = ARTIFACT_ROOT

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifact_root"] = str(self.artifact_root)
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

