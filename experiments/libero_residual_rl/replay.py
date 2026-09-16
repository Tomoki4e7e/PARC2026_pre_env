from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    images: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray
    rewards: np.ndarray
    next_images: np.ndarray
    next_vectors: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.storage: list[tuple[np.ndarray, ...]] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        images: np.ndarray,
        vectors: np.ndarray,
        residuals: np.ndarray,
        reward: float,
        next_images: np.ndarray,
        next_vectors: np.ndarray,
        done: bool,
    ) -> None:
        item = (
            images.astype(np.uint8),
            vectors.astype(np.float32),
            residuals.astype(np.float32),
            np.asarray(reward, dtype=np.float32),
            next_images.astype(np.uint8),
            next_vectors.astype(np.float32),
            np.asarray(done, dtype=np.float32),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> ReplayBatch:
        if len(self.storage) < batch_size:
            raise ValueError(f"Need {batch_size} samples, have {len(self.storage)}")
        indexes = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        fields = list(zip(*(self.storage[int(index)] for index in indexes)))
        arrays = [np.stack(field) for field in fields]
        return ReplayBatch(*arrays)

