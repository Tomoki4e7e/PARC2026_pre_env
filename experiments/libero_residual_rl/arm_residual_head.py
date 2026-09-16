from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .gripper_head import GripperHeadConfig, TemporalGripperHead


@dataclass(frozen=True)
class ArmHeadConfig:
    history: int = 4
    image_size: int = 64
    hidden_dim: int = 128
    translation_limit: float = 0.05
    rotation_limit: float = 0.08
    rate_limit: float = 0.04

    def limits(self) -> tuple[float, ...]:
        return (self.translation_limit,) * 3 + (self.rotation_limit,) * 3


class TemporalArmResidualHead(nn.Module):
    def __init__(self, config: ArmHeadConfig):
        super().__init__()
        self.config = config
        self.visual = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.frame = nn.Sequential(
            nn.Linear(64 + 15, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.temporal = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.residual = nn.Linear(config.hidden_dim, 6)
        self.approach = nn.Linear(config.hidden_dim, 1)
        self.contact = nn.Linear(config.hidden_dim, 1)
        self.register_buffer("limits", torch.tensor(config.limits(), dtype=torch.float32))

    def forward(
        self, images: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, history = vectors.shape[:2]
        visual = self.visual(images.reshape(batch * history, *images.shape[2:]))
        visual = visual.reshape(batch, history, -1)
        frames = self.frame(torch.cat((visual, vectors), dim=-1))
        _, hidden = self.temporal(frames)
        state = hidden[-1]
        residual = torch.tanh(self.residual(state)) * self.limits
        return (
            residual,
            self.approach(state).squeeze(-1),
            self.contact(state).squeeze(-1),
        )

    def applied(
        self, images: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, approach, contact = self(images, vectors)
        return residual * torch.sigmoid(approach)[:, None], approach, contact


class ArmCorrectionDataset(Dataset):
    def __init__(self, paths: list[Path], config: ArmHeadConfig):
        self.config = config
        self.episodes = [dict(np.load(path)) for path in paths]
        self.index: list[tuple[int, int]] = []
        self.active_labels: list[float] = []
        for episode_index, episode in enumerate(self.episodes):
            required = {"front", "state", "action", "base_action", "phase", "contact"}
            if not required.issubset(episode):
                continue
            for frame, phase in enumerate(episode["phase"]):
                active = int(phase) <= 1
                if active or frame % 4 == 0:
                    self.index.append((episode_index, frame))
                    self.active_labels.append(float(active))
        if not self.index:
            raise RuntimeError("No arm correction samples found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        episode_index, frame = self.index[item]
        episode = self.episodes[episode_index]
        indices = np.clip(
            np.arange(frame - self.config.history + 1, frame + 1),
            0,
            len(episode["action"]) - 1,
        )
        images = torch.from_numpy(episode["front"][indices].copy()).permute(0, 3, 1, 2)
        images = F.interpolate(
            images.float(),
            (self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        ) / 255.0
        vectors = np.concatenate(
            (episode["state"][indices], episode["base_action"][indices]), axis=-1
        ).astype(np.float32)
        active = int(episode["phase"][frame]) <= 1
        target = np.zeros(6, dtype=np.float32)
        if active:
            target = episode["action"][frame, :6] - episode["base_action"][frame, :6]
            target = np.clip(target, -np.asarray(self.config.limits()), self.config.limits())
        return {
            "images": images,
            "vectors": torch.from_numpy(vectors),
            "target": torch.from_numpy(target),
            "approach": torch.tensor(active, dtype=torch.float32),
            "contact": torch.tensor(bool(episode["contact"][frame]), dtype=torch.float32),
        }


def train_arm_head(
    train_root: Path,
    validation_root: Path,
    output: Path,
    config: ArmHeadConfig,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    train_set = ArmCorrectionDataset(sorted(train_root.glob("*.npz")), config)
    validation_set = ArmCorrectionDataset(sorted(validation_root.glob("*.npz")), config)
    labels = np.asarray(train_set.active_labels)
    positive, negative = max(1, int(labels.sum())), max(1, int((1 - labels).sum()))
    weights = np.where(labels > 0, 0.7 / positive, 0.3 / negative)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TemporalArmResidualHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            applied, approach, contact = model.applied(
                batch["images"].to(device), batch["vectors"].to(device)
            )
            loss = (
                F.smooth_l1_loss(applied, batch["target"].to(device), beta=0.01)
                + 0.2
                * F.binary_cross_entropy_with_logits(
                    approach, batch["approach"].to(device)
                )
                + 0.1
                * F.binary_cross_entropy_with_logits(
                    contact, batch["contact"].to(device)
                )
                + 0.02 * applied.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        absolute_errors = []
        approach_correct = []
        with torch.inference_mode():
            for batch in validation_loader:
                applied, approach, _ = model.applied(
                    batch["images"].to(device), batch["vectors"].to(device)
                )
                absolute_errors.append(
                    (applied - batch["target"].to(device)).abs().cpu().numpy()
                )
                approach_correct.extend(
                    ((approach >= 0).cpu() == batch["approach"].bool()).tolist()
                )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_mae": float(np.concatenate(absolute_errors).mean()),
                "approach_accuracy": float(np.mean(approach_correct)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    report = {
        "train_episodes": len(train_set.episodes),
        "validation_episodes": len(validation_set.episodes),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "metrics": history,
    }
    (output / "training_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class LearnedFrontGraspPolicy:
    """Composes front-only arm and gripper residual heads over one Base call."""

    def __init__(
        self,
        base_policy,
        arm_checkpoint: str | Path,
        gripper_checkpoint: str | Path,
        arm_refinement_checkpoint: str | Path | None = None,
        arm_scale: float = 1.0,
    ):
        self.base_policy = base_policy
        if not 0.0 <= arm_scale <= 1.0:
            raise ValueError("arm_scale must be in [0, 1]")
        self.arm_scale = arm_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arm_root = Path(arm_checkpoint)
        self.arm_config = ArmHeadConfig(**json.loads((arm_root / "config.json").read_text()))
        self.arm = TemporalArmResidualHead(self.arm_config).to(self.device)
        self.arm.load_state_dict(torch.load(arm_root / "model.pt", map_location=self.device))
        self.arm.eval()
        self.arm_refinement = None
        if arm_refinement_checkpoint:
            refinement_root = Path(arm_refinement_checkpoint)
            refinement_config = ArmHeadConfig(
                **json.loads((refinement_root / "config.json").read_text())
            )
            if refinement_config != self.arm_config:
                raise ValueError("Primary and refinement arm configs must match")
            self.arm_refinement = TemporalArmResidualHead(refinement_config).to(
                self.device
            )
            self.arm_refinement.load_state_dict(
                torch.load(refinement_root / "model.pt", map_location=self.device)
            )
            self.arm_refinement.eval()
        gripper_root = Path(gripper_checkpoint)
        self.gripper_config = GripperHeadConfig(
            **json.loads((gripper_root / "config.json").read_text())
        )
        if self.gripper_config.mode != "front":
            raise ValueError("Combined policy requires a front gripper checkpoint")
        self.gripper = TemporalGripperHead(self.gripper_config).to(self.device)
        self.gripper.load_state_dict(
            torch.load(gripper_root / "model.pt", map_location=self.device)
        )
        self.gripper.eval()
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=max(self.arm_config.history, self.gripper_config.history)
        )
        self.refinement_history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.arm_config.history
        )
        self.last_primary_residual = np.zeros(6, dtype=np.float32)
        self.last_arm_residual = np.zeros(6, dtype=np.float32)
        self.gripper_hold_remaining = 0

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.base_policy.reset(instruction, seed=seed)
        self.history.clear()
        self.refinement_history.clear()
        self.last_primary_residual.fill(0)
        self.last_arm_residual.fill(0)
        self.gripper_hold_remaining = 0

    def _inputs(self, obs: dict[str, np.ndarray], base_action: np.ndarray):
        from .collect_grasp_demos import _flip_libero_image, _quat_to_axis_angle

        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                base_action,
            )
        ).astype(np.float32)
        self.history.append((_flip_libero_image(obs["agentview_image"]), state))
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(self.history[0])
        records = list(self.history)

        def tensors(length: int):
            selected = records[-length:]
            images = torch.from_numpy(np.stack([row[0] for row in selected])).permute(
                0, 3, 1, 2
            )
            images = F.interpolate(
                images.float(),
                (64, 64),
                mode="bilinear",
                align_corners=False,
            )[None].to(self.device) / 255.0
            vectors = torch.from_numpy(np.stack([row[1] for row in selected]))[
                None
            ].to(self.device)
            return images, vectors

        return tensors

    def _refinement_inputs(
        self, image: np.ndarray, state_action: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.refinement_history.append((image, state_action))
        while len(self.refinement_history) < self.refinement_history.maxlen:
            self.refinement_history.appendleft(self.refinement_history[0])
        images = torch.from_numpy(
            np.stack([row[0] for row in self.refinement_history])
        ).permute(0, 3, 1, 2)
        images = F.interpolate(
            images.float(),
            (self.arm_config.image_size, self.arm_config.image_size),
            mode="bilinear",
            align_corners=False,
        )[None].to(self.device) / 255.0
        vectors = torch.from_numpy(
            np.stack([row[1] for row in self.refinement_history])
        )[None].to(self.device)
        return images, vectors

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        base_action = np.asarray(self.base_policy.get_action(obs), dtype=np.float32)
        tensors = self._inputs(obs, base_action)
        with torch.inference_mode():
            arm_images, arm_vectors = tensors(self.arm_config.history)
            residual, _, _ = self.arm.applied(arm_images, arm_vectors)
            gripper_images, gripper_vectors = tensors(self.gripper_config.history)
            close_logit, _, _ = self.gripper(gripper_images, gripper_vectors)
        target = residual[0].cpu().numpy()
        rate = self.arm_config.rate_limit
        target = self.last_primary_residual + np.clip(
            target - self.last_primary_residual, -rate, rate
        )
        target = np.clip(
            target,
            -np.asarray(self.arm_config.limits()),
            self.arm_config.limits(),
        ).astype(np.float32)
        self.last_primary_residual = target
        action = base_action.copy()
        if float(torch.sigmoid(close_logit)[0]) >= self.gripper_config.threshold:
            self.gripper_hold_remaining = self.gripper_config.hold_steps
        if self.gripper_hold_remaining > 0:
            action[6] = 1.0
            self.gripper_hold_remaining -= 1
        total_target = target
        if self.arm_refinement is not None:
            from .collect_grasp_demos import _flip_libero_image, _quat_to_axis_angle

            refinement_vector = np.concatenate(
                (
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                    _quat_to_axis_angle(obs["robot0_eef_quat"]),
                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                    np.concatenate(
                        (
                            np.clip(base_action[:6] + target, -1.0, 1.0),
                            action[6:],
                        )
                    ),
                )
            ).astype(np.float32)
            refinement_inputs = self._refinement_inputs(
                _flip_libero_image(obs["agentview_image"]), refinement_vector
            )
            with torch.inference_mode():
                correction, _, _ = self.arm_refinement.applied(*refinement_inputs)
            total_target = target + correction[0].cpu().numpy()
        total_target = np.clip(
            total_target * self.arm_scale,
            -np.asarray(self.arm_config.limits()),
            self.arm_config.limits(),
        )
        total_target = self.last_arm_residual + np.clip(
            total_target - self.last_arm_residual, -rate, rate
        )
        self.last_arm_residual = total_target.astype(np.float32)
        action[:6] = np.clip(
            base_action[:6] + self.last_arm_residual, -1.0, 1.0
        )
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()
    report = train_arm_head(
        args.train_root,
        args.validation_root,
        args.output,
        ArmHeadConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
