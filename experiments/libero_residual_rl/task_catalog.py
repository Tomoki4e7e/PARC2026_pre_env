from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import hashlib
import json
import random

from .config import LIBERO_ROOT


MIDDLE_DRAWER = "open_the_middle_drawer_of_the_cabinet"
TOP_DRAWER = "open_the_top_drawer_and_put_the_bowl_inside"
MICROWAVE = "KITCHEN_SCENE7_open_the_microwave"


@dataclass(frozen=True)
class TaskSpec:
    base_task: str
    task_name: str
    language: str
    bddl_file: str
    axis: str
    level: int
    difficulty: int | None
    seed: int
    runtime_parameters: dict[str, float]


def _language_from_bddl(path: Path) -> str:
    import re

    match = re.search(r"\(:language\s+([^)]+)\)", path.read_text())
    if not match:
        raise ValueError(f"No :language field in {path}")
    return " ".join(match.group(1).split())


class TaskCatalog:
    """Builds immutable L0/L1/L2 splits without relying on missing plus assets."""

    def __init__(self, libero_root: str | Path = LIBERO_ROOT):
        self.root = Path(libero_root)
        path = self.root / "benchmark" / "task_classification.json"
        self.classification = json.loads(path.read_text())

    def base_bddl(self, base_task: str) -> Path:
        if base_task == MICROWAVE:
            return self.root / "bddl_files" / "libero_90" / f"{base_task}.bddl"
        return self.root / "bddl_files" / "libero_goal" / f"{base_task}.bddl"

    def layout_variants(self, base_task: str, level: int) -> list[tuple[Path, int]]:
        if base_task == MICROWAVE or level == 0:
            return [(self.base_bddl(base_task), 0)]
        difficulties = {1, 2} if level == 1 else {4, 5}
        rows = [
            row
            for row in self.classification["libero_goal"]
            if row["name"].startswith(base_task + "_")
            and row["category"] == "Objects Layout"
            and row["difficulty_level"] in difficulties
        ]
        result = []
        for row in rows:
            path = self.root / "bddl_files" / "libero_goal" / f"{row['name']}.bddl"
            if path.is_file():
                result.append((path, int(row["difficulty_level"])))
        if not result:
            raise RuntimeError(f"No layout variants for {base_task=} {level=}")
        return result

    @staticmethod
    def _runtime_parameters(axis: str, level: int, rng: random.Random) -> dict[str, float]:
        if level == 0 or axis == "layout":
            return {}
        if axis == "camera":
            bound = 10.0 if level == 1 else 25.0
            return {
                "yaw_deg": rng.uniform(-bound, bound),
                "pitch_deg": rng.uniform(-bound, bound),
            }
        if axis == "robot":
            bound = 0.05 if level == 1 else 0.15
            return {f"joint_fraction_{i}": rng.uniform(-bound, bound) for i in range(7)}
        raise ValueError(f"Unknown perturbation axis: {axis}")

    def build(
        self,
        *,
        base_tasks: Iterable[str] = (MIDDLE_DRAWER, TOP_DRAWER),
        axes: Iterable[str] = ("layout", "camera", "robot"),
        levels: Iterable[int] = (0, 1, 2),
        episodes_per_condition: int = 50,
        seed: int = 42,
    ) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        for base_task in base_tasks:
            for axis in axes:
                for level in levels:
                    variants = self.layout_variants(base_task, level) if axis == "layout" else [
                        (self.base_bddl(base_task), None)
                    ]
                    condition_seed = int.from_bytes(
                        hashlib.sha256(f"{seed}:{base_task}:{axis}:{level}".encode()).digest()[:8],
                        "little",
                    )
                    rng = random.Random(condition_seed)
                    shuffled = list(variants)
                    rng.shuffle(shuffled)
                    for episode in range(episodes_per_condition):
                        path, difficulty = shuffled[episode % len(shuffled)]
                        episode_seed = rng.randrange(2**31)
                        specs.append(
                            TaskSpec(
                                base_task=base_task,
                                task_name=path.stem,
                                language=_language_from_bddl(path),
                                bddl_file=str(path),
                                axis=axis,
                                level=level,
                                difficulty=difficulty,
                                seed=episode_seed,
                                runtime_parameters=self._runtime_parameters(axis, level, rng),
                            )
                        )
        return specs

    @staticmethod
    def save_manifest(specs: list[TaskSpec], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([asdict(spec) for spec in specs], indent=2) + "\n")

