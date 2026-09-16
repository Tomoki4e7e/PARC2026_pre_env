from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


HeadMode = Literal["vector", "front", "wrist", "two_camera"]


@dataclass(frozen=True)
class GripperHeadConfig:
    mode: HeadMode = "two_camera"
    history: int = 4
    image_size: int = 64
    hidden_dim: int = 128
    threshold: float = 0.5
    hold_steps: int = 12


class TemporalGripperHead(nn.Module):
    def __init__(self, config: GripperHeadConfig):
        super().__init__()
        self.config = config
        channels = (
            0
            if config.mode == "vector"
            else (6 if config.mode == "two_camera" else 3)
        )
        if channels:
            self.visual = nn.Sequential(
                nn.Conv2d(channels, 24, 5, stride=2),
                nn.SiLU(),
                nn.Conv2d(24, 48, 3, stride=2),
                nn.SiLU(),
                nn.Conv2d(48, 64, 3, stride=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            visual_dim = 64
        else:
            self.visual = None
            visual_dim = 0
        self.frame = nn.Sequential(
            nn.Linear(15 + visual_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.temporal = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.close = nn.Linear(config.hidden_dim, 1)
        self.contact = nn.Linear(config.hidden_dim, 1)
        self.phase = nn.Linear(config.hidden_dim, 5)

    def forward(
        self, images: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, history = vectors.shape[:2]
        if self.visual is not None:
            visual = self.visual(images.reshape(batch * history, *images.shape[2:]))
            visual = visual.reshape(batch, history, -1)
            vectors = torch.cat((vectors, visual), dim=-1)
        frames = self.frame(vectors)
        _, hidden = self.temporal(frames)
        state = hidden[-1]
        return self.close(state).squeeze(-1), self.contact(state).squeeze(-1), self.phase(state)


class LearnedGripperResidualPolicy:
    """Overrides only the gripper coordinate of a frozen base policy."""

    def __init__(
        self,
        base_policy,
        checkpoint: str | Path,
        *,
        post_close_execution_horizon: int | None = None,
        post_close_arm_scale: float = 1.0,
    ):
        self.base_policy = base_policy
        self.post_close_execution_horizon = post_close_execution_horizon
        if not 0.0 < post_close_arm_scale <= 1.0:
            raise ValueError("post_close_arm_scale must be in (0, 1]")
        self.post_close_arm_scale = float(post_close_arm_scale)
        root = Path(checkpoint)
        self.config = GripperHeadConfig(**json.loads((root / "config.json").read_text()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalGripperHead(self.config).to(self.device)
        self.model.load_state_dict(torch.load(root / "model.pt", map_location=self.device))
        self.model.eval()
        self.history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.config.history
        )
        self.hold_remaining = 0
        self.last_probability = 0.0
        self._post_close_horizon_applied = False

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.base_policy.reset(instruction, seed=seed)
        self.history.clear()
        self.hold_remaining = 0
        self._post_close_horizon_applied = False

    @staticmethod
    def _state(obs: dict[str, np.ndarray]) -> np.ndarray:
        from .collect_grasp_demos import _quat_to_axis_angle

        return np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from .collect_grasp_demos import _flip_libero_image

        base_action = np.asarray(self.base_policy.get_action(obs), dtype=np.float32)
        frame = (
            _flip_libero_image(obs["agentview_image"]),
            _flip_libero_image(obs["robot0_eye_in_hand_image"]),
            np.concatenate((self._state(obs), base_action)).astype(np.float32),
        )
        self.history.append(frame)
        while len(self.history) < self.config.history:
            self.history.appendleft(frame)
        records = list(self.history)
        if self.config.mode == "vector":
            images = torch.empty(
                (1, self.config.history, 0, 1, 1), device=self.device
            )
        else:
            keys = (
                (0, 1)
                if self.config.mode == "two_camera"
                else ((0,) if self.config.mode == "front" else (1,))
            )
            arrays = []
            for key in keys:
                value = torch.from_numpy(
                    np.stack([record[key] for record in records])
                ).permute(0, 3, 1, 2)
                value = F.interpolate(
                    value.float(),
                    (self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    align_corners=False,
                ) / 255.0
                arrays.append(value)
            images = torch.cat(arrays, dim=1)[None].to(self.device)
        vectors = torch.from_numpy(np.stack([record[2] for record in records]))[
            None
        ].to(self.device)
        with torch.inference_mode():
            close_logit, _, _ = self.model(images, vectors)
            self.last_probability = float(torch.sigmoid(close_logit)[0])
        if self.last_probability >= self.config.threshold:
            self.hold_remaining = self.config.hold_steps
            if (
                self.post_close_execution_horizon is not None
                and not self._post_close_horizon_applied
                and hasattr(self.base_policy, "set_execution_horizon")
            ):
                self.base_policy.set_execution_horizon(
                    self.post_close_execution_horizon
                )
                self._post_close_horizon_applied = True
        if self.hold_remaining > 0:
            base_action[:6] *= self.post_close_arm_scale
            base_action[6] = 1.0
            self.hold_remaining -= 1
        return base_action


class GraspWindowDataset(Dataset):
    def __init__(self, paths: list[Path], config: GripperHeadConfig):
        self.config = config
        self.episodes = [dict(np.load(path)) for path in paths]
        self.index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            if "phase" not in episode or "base_action" not in episode:
                continue
            phase = episode["phase"]
            useful = np.flatnonzero(
                (phase >= 1)
                | (np.arange(len(phase)) % 8 == 0)
            )
            self.index.extend((episode_index, int(frame)) for frame in useful)
        if not self.index:
            raise RuntimeError("No labeled gripper samples found")
        self.labels = np.asarray(
            [
                self.episodes[ep]["phase"][frame] >= 2
                and self.episodes[ep]["action"][frame, 6] > 0.5
                for ep, frame in self.index
            ],
            dtype=np.float32,
        )

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
        if self.config.mode == "vector":
            images = np.empty((self.config.history, 0, 1, 1), dtype=np.float32)
        else:
            keys = (
                ("front", "wrist")
                if self.config.mode == "two_camera"
                else (self.config.mode,)
            )
            image_parts = []
            for key in keys:
                value = torch.from_numpy(episode[key][indices].copy()).permute(0, 3, 1, 2)
                value = F.interpolate(
                    value.float(),
                    (self.config.image_size, self.config.image_size),
                    mode="bilinear",
                    align_corners=False,
                ) / 255.0
                image_parts.append(value)
            images = torch.cat(image_parts, dim=1)
        vectors = np.concatenate(
            (episode["state"][indices], episode["base_action"][indices]), axis=-1
        ).astype(np.float32)
        phase = int(episode["phase"][frame])
        close = phase >= 2 and episode["action"][frame, 6] > 0.5
        return {
            "images": images,
            "vectors": torch.from_numpy(vectors),
            "close": torch.tensor(close, dtype=torch.float32),
            "contact": torch.tensor(bool(episode["contact"][frame]), dtype=torch.float32),
            "phase": torch.tensor(phase, dtype=torch.long),
        }


def train_head(
    data_root: Path,
    output: Path,
    config: GripperHeadConfig,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    extra_data_roots: list[Path] | None = None,
) -> dict:
    paths = sorted(data_root.glob("*.npz"))
    for root in extra_data_roots or []:
        paths.extend(sorted(root.glob("*.npz")))
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    split = max(1, round(len(paths) * 0.8))
    train_paths, validation_paths = paths[:split], paths[split:]
    if not validation_paths:
        raise RuntimeError("At least two episodes are required")
    train_set = GraspWindowDataset(train_paths, config)
    validation_set = GraspWindowDataset(validation_paths, config)
    positive = max(1, int(train_set.labels.sum()))
    negative = max(1, len(train_set) - positive)
    weights = np.where(train_set.labels > 0, 0.5 / positive, 0.5 / negative)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_set),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TemporalGripperHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    metrics = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            images = batch["images"].to(device)
            vectors = batch["vectors"].to(device)
            close, contact, phase = model(images, vectors)
            loss = (
                F.binary_cross_entropy_with_logits(close, batch["close"].to(device))
                + 0.3
                * F.binary_cross_entropy_with_logits(
                    contact, batch["contact"].to(device)
                )
                + 0.2 * F.cross_entropy(phase, batch["phase"].to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        probabilities, labels = [], []
        with torch.inference_mode():
            for batch in validation_loader:
                logits, _, _ = model(
                    batch["images"].to(device), batch["vectors"].to(device)
                )
                probabilities.extend(torch.sigmoid(logits).cpu().tolist())
                labels.extend(batch["close"].tolist())
        metrics.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=bool)
    candidates = np.linspace(0.2, 0.8, 25)
    scores = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        tp = int((prediction & labels).sum())
        fp = int((prediction & ~labels).sum())
        fn = int((~prediction & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        scores.append((f1 - 0.1 * fp / max(1, (~labels).sum()), threshold, tp, fp, fn))
    _, threshold, tp, fp, fn = max(scores)
    final_config = GripperHeadConfig(**{**asdict(config), "threshold": float(threshold)})
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "config.json").write_text(json.dumps(asdict(final_config), indent=2) + "\n")
    report = {
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "metrics": metrics,
    }
    (output / "training_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("vector", "front", "wrist", "two_camera"),
        default="two_camera",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--extra-data-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = train_head(
        args.data_root,
        args.output,
        GripperHeadConfig(mode=args.mode),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        extra_data_roots=args.extra_data_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
