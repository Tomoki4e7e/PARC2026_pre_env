"""学習スクリプト共通ユーティリティ。"""

from __future__ import annotations

import math
import os

# huggingface_hub import 前に設定すること
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import random
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

T = TypeVar("T")


def ensure_work_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_hf_env(hf_home: Path, lerobot_cache: Path) -> None:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["DIFFUSERS_VERBOSITY"] = "error"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_LEROBOT_HOME"] = str(lerobot_cache)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def run_quiet(
    command: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    if check and result.returncode != 0:
        tail = result.stdout[-6000:] if result.stdout else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{tail}")
    return result


def run_hf_with_retry(operation: Callable[[], T]) -> T:
    last_error: BaseException | None = None
    for attempt in range(6):
        try:
            return operation()
        except (HfHubHTTPError, httpx.HTTPStatusError) as error:
            last_error = error
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status != 429 and "429" not in str(error):
                raise
            if attempt == 5:
                break
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = float(headers.get("Retry-After", 15)) + 1
            except (TypeError, ValueError):
                delay = min(120, 15 * (2**attempt) + random.random())
            time.sleep(delay)
    raise RuntimeError("Hugging Face からの取得に失敗しました。") from last_error


def cached_or_downloaded_snapshot(
    repo_id: str,
    revision: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        return Path(
            run_hf_with_retry(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=False,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=1,
                )
            )
        )


def normalize_task_name(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def task_name_from_cell(value) -> str:
    if isinstance(value, str):
        return value
    try:
        if len(value) > 0:
            return str(value[0])
    except TypeError:
        pass
    return str(value)


def choose_evenly_spaced(episode_indices: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count == 1:
        return [episode_indices[0]]
    positions = [
        round(index * (len(episode_indices) - 1) / (count - 1))
        for index in range(count)
    ]
    return [episode_indices[position] for position in positions]


def _suggest_close_tasks(query: str, available: dict[str, str]) -> list[str]:
    q_tokens = set(normalize_task_name(query).split())
    scored: list[tuple[int, str]] = []
    for norm, original in available.items():
        overlap = len(q_tokens & set(norm.split()))
        if overlap:
            scored.append((overlap, original))
    scored.sort(reverse=True)
    return [name for _, name in scored[:10]]


def _episodes_for_task(
    task_name: str,
    task_to_episodes: dict[str, list[int]],
    available_by_normalized: dict[str, str],
    episodes_per_task: int,
    *,
    suite_name: str,
) -> list[int]:
    actual_task = available_by_normalized.get(normalize_task_name(task_name))
    if actual_task is None:
        close = _suggest_close_tasks(task_name, available_by_normalized)
        hint = f"  Close matches: {close[:5]}" if close else ""
        raise RuntimeError(
            f"[{suite_name}] Task not found: {task_name!r}\n{hint}"
        )
    pool = task_to_episodes[actual_task]
    if len(pool) < episodes_per_task:
        raise RuntimeError(
            f"[{suite_name}] {actual_task!r}: need {episodes_per_task} episodes, "
            f"only {len(pool)} available"
        )
    return choose_evenly_spaced(pool, episodes_per_task)


def select_training_episodes(
    dataset_repo: str,
    dataset_revision: str,
    suites: list,
    *,
    task_episode_overrides: dict[str, int] | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    """multi-suite 学習用エピソード index を返す。"""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    overrides = {
        normalize_task_name(key): value
        for key, value in (task_episode_overrides or {}).items()
    }

    dataset_metadata = run_hf_with_retry(
        lambda: LeRobotDatasetMetadata(dataset_repo, revision=dataset_revision)
    )

    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode_index, task_cell in enumerate(dataset_metadata.episodes["tasks"]):
        task_to_episodes[task_name_from_cell(task_cell)].append(int(episode_index))

    available_by_normalized = {
        normalize_task_name(task_name): task_name for task_name in task_to_episodes
    }

    selected_by_task: dict[str, list[int]] = {}
    all_indices: set[int] = set()

    for suite in suites:
        for task_name in suite.task_names:
            count = overrides.get(
                normalize_task_name(task_name),
                suite.episodes_per_task,
            )
            picked = _episodes_for_task(
                task_name,
                task_to_episodes,
                available_by_normalized,
                count,
                suite_name=suite.name,
            )
            selected_by_task[task_name] = picked
            all_indices.update(picked)

    return sorted(all_indices), selected_by_task


def print_episode_selection_summary(
    selected_by_task: dict[str, list[int]],
    suites: list,
) -> None:
    suite_lookup = {
        normalize_task_name(task_name): suite.name
        for suite in suites
        for task_name in suite.task_names
    }
    by_suite: dict[str, int] = defaultdict(int)
    for task_name, indices in selected_by_task.items():
        suite_name = suite_lookup.get(normalize_task_name(task_name), "?")
        by_suite[suite_name] += len(indices)
    print("Episode selection summary:")
    for suite_name in sorted(by_suite):
        print(f"  {suite_name}: {by_suite[suite_name]} episodes")
    print(f"  total (unique): {sum(by_suite.values())} episodes")


def select_spatial_episodes(
    dataset_repo: str,
    dataset_revision: str,
    spatial_task_names: list[str],
    episodes_per_task: int,
) -> list[int]:
    from config import TrainSuiteSpec

    spec = TrainSuiteSpec("spatial", spatial_task_names, episodes_per_task)
    episode_indices, _ = select_training_episodes(dataset_repo, dataset_revision, [spec])
    return episode_indices


def prefetch_dataset(
    dataset_repo: str,
    dataset_revision: str,
    episode_indices: list[int],
    lerobot_src: Path,
) -> None:
    """学習用エピソードの動画・メタデータを Hugging Face から事前取得する。"""
    import sys

    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def _download() -> LeRobotDataset:
        return LeRobotDataset(
            dataset_repo,
            revision=dataset_revision,
            episodes=episode_indices,
        )

    run_hf_with_retry(_download)
    print(f"Dataset ready: {dataset_repo} ({len(episode_indices)} episodes)")


def quat_to_axis_angle(quat) -> "np.ndarray":
    """robosuite 形式 (x, y, z, w) のクォータニオンを axis-angle へ変換。"""
    import numpy as np

    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def flip_libero_image(img) -> "np.ndarray":
    """LIBERO カメラ画像を 180° 反転（LeRobot LiberoProcessorStep と同一）。"""
    import numpy as np

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[-1] < arr.shape[0]:
        return arr[::-1, ::-1].copy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return arr[:, ::-1, ::-1].copy()
    return arr


def prepare_image(img) -> "np.ndarray":
    """PARC の HWC uint8 画像を LeRobot 前処理向け CHW float32 [0, 1] へ変換。"""
    import numpy as np

    arr = flip_libero_image(img)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[-1] < arr.shape[0]:
        arr = arr.transpose(2, 0, 1)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def build_lerobot_observation(obs: dict, instruction: str) -> dict:
    """PARC 評価パイプラインの観測を LeRobot / SmolVLA 形式へ変換。"""
    import numpy as np

    state = np.concatenate(
        [
            obs["robot0_eef_pos"].astype(np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"].astype(np.float32),
        ]
    )
    return {
        "observation.images.front": prepare_image(obs["agentview_image"]),
        "observation.images.wrist": prepare_image(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": instruction,
    }
