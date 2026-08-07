#!/usr/bin/env python3
"""LIBERO-90 多様タスク + 既知40タスク replay の学習データを準備する。

既定では metadata だけを取得して分割・schema・episode 数を検証する。
``--prepare`` を付けると必要な4 suiteを取得してローカル集約データセットを作る。
固定 OOD 12タスクは manifest に記録するだけで、選択 episode には含めない。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config

# huggingface_hub は import 時に cache 位置と XET 設定を確定するため、
# 取り込みより前に work/ 配下のキャッシュへ固定する。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(config.HF_HOME))
os.environ.setdefault("HF_LEROBOT_HOME", str(config.LEROBOT_CACHE))

from diverse_retrain_tasks import (
    OOD_TASK_NAMES,
    TRAIN_TASK_NAMES,
    TRAIN_TASKS_BY_CATEGORY,
    normalize_task_name,
    validate_task_split,
)
from utils import (
    choose_evenly_spaced,
    ensure_work_dirs,
    run_hf_with_retry,
    set_hf_env,
    task_name_from_cell,
)

DATASET_REPO = "nvidia/LIBERO_LeRobot_v3"
DATASET_REVISION = "e5907374380b8f96511957e6ba5582be52a1e179"
AGGREGATED_REPO_ID = "local/libero_diverse_retrain"
AGGREGATED_ROOT = config.WORK_DIR / "datasets" / "libero_diverse_retrain"
NORMALIZED_LIBERO90_ROOT = config.WORK_DIR / "datasets" / "libero90_core_schema"
MANIFEST_PATH = config.WORK_DIR / "diverse_retrain_manifest.json"

LIBERO90_EPISODES_PER_TASK = 20
REPLAY_EPISODES_PER_TASK = 7

LIBERO10_TASK_NAMES = (
    "turn on the stove and put the moka pot on it",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "pick up the book and place it in the back compartment of the caddy",
    "put the yellow and white mug in the microwave and close it",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the cream cheese box and the butter in the basket",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
)

CANONICAL_MOTOR_NAMES: dict[str, tuple[str, ...]] = {
    "observation.state": (
        "x",
        "y",
        "z",
        "axis_angle1",
        "axis_angle2",
        "axis_angle3",
        "gripper",
        "gripper",
    ),
    "action": (
        "x",
        "y",
        "z",
        "axis_angle1",
        "axis_angle2",
        "axis_angle3",
        "gripper",
    ),
}

REQUIRED_FEATURES: dict[str, tuple[str, tuple[int, ...]]] = {
    "observation.images.image": ("video", (256, 256, 3)),
    "observation.images.wrist_image": ("video", (256, 256, 3)),
    "observation.state": ("float32", (8,)),
    "action": ("float32", (7,)),
}


@dataclass(frozen=True)
class SourceSpec:
    suite: str
    task_names: tuple[str, ...]
    episodes_per_task: int
    role: str


SOURCES = (
    SourceSpec(
        "libero_90",
        TRAIN_TASK_NAMES,
        LIBERO90_EPISODES_PER_TASK,
        "diverse_train",
    ),
    SourceSpec(
        "libero_spatial",
        tuple(config.SPATIAL_TASK_NAMES),
        REPLAY_EPISODES_PER_TASK,
        "replay",
    ),
    SourceSpec(
        "libero_object",
        tuple(config.OBJECT_TASK_NAMES),
        REPLAY_EPISODES_PER_TASK,
        "replay",
    ),
    SourceSpec(
        "libero_goal",
        tuple(config.GOAL_TASK_NAMES),
        REPLAY_EPISODES_PER_TASK,
        "replay",
    ),
    SourceSpec(
        "libero_10",
        LIBERO10_TASK_NAMES,
        REPLAY_EPISODES_PER_TASK,
        "replay",
    ),
)


def download_snapshot(*, metadata_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    patterns = (
        [f"{source.suite}/meta/**" for source in SOURCES]
        if metadata_only
        else [f"{source.suite}/**" for source in SOURCES]
    )
    return Path(
        run_hf_with_retry(
            lambda: snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=DATASET_REVISION,
                allow_patterns=patterns,
                token=False,
                max_workers=2,
            )
        )
    )


def load_metadata(snapshot_root: Path) -> dict[str, object]:
    lerobot_src = config.LEROBOT_DIR / "src"
    if not lerobot_src.is_dir():
        raise RuntimeError(
            "LeRobot が未セットアップです。setup_lerobot.sh を実行してください。"
        )
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata: dict[str, object] = {}
    for source in SOURCES:
        root = snapshot_root / source.suite
        metadata[source.suite] = LeRobotDatasetMetadata(
            f"{DATASET_REPO}/{source.suite}",
            root=root,
        )
    return metadata


def canonicalize_features(features: dict[str, dict]) -> dict[str, dict]:
    """suite 間で表記のみ異なる feature metadata を揃える。

    同一ソース repo でも suite ごとに補助 key（scalar の ``fps``）や
    state/action の ``names`` 表記が異なり、集約時の等価判定を通らない。
    dtype と shape は変更しない。
    """
    from lerobot.utils.constants import DEFAULT_FEATURES

    for key, default in DEFAULT_FEATURES.items():
        if key in features:
            features[key] = deepcopy(default)
    for key, motor_names in CANONICAL_MOTOR_NAMES.items():
        if key in features:
            features[key]["names"] = {"motors": list(motor_names)}
    return features


def core_features(features: dict[str, dict]) -> dict[str, dict]:
    core = metadata_core_feature_names()
    return {key: value for key, value in features.items() if key in core}


def validate_schema(metadata: dict[str, object]) -> None:
    from lerobot.datasets.feature_utils import features_equal_for_merge

    baseline_suite: str | None = None
    for source in SOURCES:
        meta = metadata[source.suite]
        features = meta.features
        for key, (dtype, shape) in REQUIRED_FEATURES.items():
            if key not in features:
                raise ValueError(f"[{source.suite}] 必須 feature がありません: {key}")
            actual = features[key]
            if actual["dtype"] != dtype or tuple(actual["shape"]) != shape:
                raise ValueError(
                    f"[{source.suite}] {key} schema mismatch: "
                    f"dtype={actual['dtype']}, shape={actual['shape']}"
                )
        canonicalize_features(features)

        if baseline_suite is None:
            baseline_suite = source.suite
            continue
        baseline = metadata[baseline_suite]
        if meta.fps != baseline.fps:
            raise ValueError(
                f"[{source.suite}] fps mismatch: {meta.fps} != {baseline.fps}"
            )
        if meta.robot_type != baseline.robot_type:
            raise ValueError(
                f"[{source.suite}] robot_type mismatch: "
                f"{meta.robot_type} != {baseline.robot_type}"
            )
        # libero_90 のみ補助 state を持つ。除去は normalize_libero90 で行うため、
        # ここでは学習に使う core feature だけを比較する。
        if not features_equal_for_merge(
            core_features(baseline.features), core_features(features)
        ):
            raise ValueError(
                f"[{source.suite}] canonical 化後も {baseline_suite} と feature が不一致です。"
            )


def task_episode_map(meta: object) -> dict[str, tuple[str, list[int]]]:
    result: dict[str, tuple[str, list[int]]] = {}
    grouped: dict[str, list[int]] = defaultdict(list)
    actual_names: dict[str, str] = {}
    for episode_index, task_cell in enumerate(meta.episodes["tasks"]):
        actual = task_name_from_cell(task_cell)
        normalized = normalize_task_name(actual)
        grouped[normalized].append(episode_index)
        actual_names[normalized] = actual
    for normalized, episode_indices in grouped.items():
        result[normalized] = (actual_names[normalized], episode_indices)
    return result


def select_source_episodes(
    source: SourceSpec,
    meta: object,
) -> tuple[list[int], dict[str, list[int]]]:
    available = task_episode_map(meta)
    selected_by_task: dict[str, list[int]] = {}
    selected: list[int] = []
    for requested in source.task_names:
        normalized = normalize_task_name(requested)
        if normalized not in available:
            raise ValueError(f"[{source.suite}] task がありません: {requested}")
        actual, pool = available[normalized]
        if len(pool) < source.episodes_per_task:
            raise ValueError(
                f"[{source.suite}] {actual}: {source.episodes_per_task} ep 必要、"
                f"{len(pool)} ep のみ"
            )
        picked = choose_evenly_spaced(pool, source.episodes_per_task)
        selected_by_task[actual] = picked
        selected.extend(picked)
    return sorted(set(selected)), selected_by_task


def build_manifest(
    metadata: dict[str, object],
    *,
    aggregated: bool,
) -> dict:
    source_entries: list[dict] = []
    aggregated_episode_indices: list[int] = []
    offset = 0
    diverse_count = 0
    replay_count = 0

    available_90 = {
        actual for actual, _ in task_episode_map(metadata["libero_90"]).values()
    }
    validate_task_split(available_90)

    ood_norm = {normalize_task_name(task) for task in OOD_TASK_NAMES}
    for source in SOURCES:
        meta = metadata[source.suite]
        selected, selected_by_task = select_source_episodes(source, meta)
        if source.suite == "libero_90":
            selected_norm = {
                normalize_task_name(task) for task in selected_by_task
            }
            leaked = selected_norm & ood_norm
            if leaked:
                raise ValueError(f"OOD episode が学習へ混入しています: {sorted(leaked)}")
            diverse_count += len(selected)
        else:
            replay_count += len(selected)

        aggregated_selected = [offset + episode for episode in selected]
        aggregated_episode_indices.extend(aggregated_selected)
        source_entries.append(
            {
                "suite": source.suite,
                "role": source.role,
                "episode_offset": offset,
                "total_source_episodes": meta.total_episodes,
                "episodes_per_task": source.episodes_per_task,
                "selected_episode_count": len(selected),
                "selected_source_episode_indices": selected,
                "selected_aggregated_episode_indices": aggregated_selected,
                "selected_by_task": selected_by_task,
            }
        )
        offset += meta.total_episodes

    total_selected = diverse_count + replay_count
    return {
        "dataset": {
            "source_repo": DATASET_REPO,
            "source_revision": DATASET_REVISION,
            "license": "OpenMDW-1.1",
            "aggregated_repo_id": AGGREGATED_REPO_ID,
            "aggregated_root": str(AGGREGATED_ROOT) if aggregated else None,
            "camera_key_mapping_for_policy": {
                "agentview_image": "observation.images.image",
                "robot0_eye_in_hand_image": "observation.images.wrist_image",
            },
        },
        "split": {
            "train_task_count": len(TRAIN_TASK_NAMES),
            "train_categories": {
                category: list(tasks)
                for category, tasks in TRAIN_TASKS_BY_CATEGORY.items()
            },
            "ood_task_count": len(OOD_TASK_NAMES),
            "ood_tasks_never_train": list(OOD_TASK_NAMES),
        },
        "mixture": {
            "diverse_episode_count": diverse_count,
            "replay_episode_count": replay_count,
            "total_episode_count": total_selected,
            "replay_fraction": replay_count / total_selected,
            "aggregated_episode_indices": sorted(aggregated_episode_indices),
        },
        "sources": source_entries,
    }


def normalize_libero90(snapshot_root: Path, *, force: bool) -> Path:
    """LIBERO-90 固有の補助stateを除き、replay suite と同じschemaにする。"""
    if NORMALIZED_LIBERO90_ROOT.exists():
        if not force:
            return NORMALIZED_LIBERO90_ROOT
        shutil.rmtree(NORMALIZED_LIBERO90_ROOT)

    lerobot_src = config.LEROBOT_DIR / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.dataset_tools import modify_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(
        repo_id=f"{DATASET_REPO}/libero_90",
        root=snapshot_root / "libero_90",
        video_backend=config.VIDEO_BACKEND,
    )
    extras = sorted(set(source.meta.features) - set(metadata_core_feature_names()))
    if extras:
        print(f"  LIBERO-90 の補助 feature を除外: {extras}")
        modify_features(
            source,
            remove_features=extras,
            output_dir=NORMALIZED_LIBERO90_ROOT,
            repo_id=f"{AGGREGATED_REPO_ID}_libero90_core",
        )
    else:
        raise ValueError(
            "LIBERO-90 に除外対象 feature がありません。source schema の変更を確認してください。"
        )
    return NORMALIZED_LIBERO90_ROOT


def metadata_core_feature_names() -> set[str]:
    return {
        *REQUIRED_FEATURES,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }


def aggregate(snapshot_root: Path, *, force: bool) -> None:
    if AGGREGATED_ROOT.exists():
        if not force:
            raise FileExistsError(
                f"{AGGREGATED_ROOT} は既に存在します。再作成は --force を指定してください。"
            )
        shutil.rmtree(AGGREGATED_ROOT)

    lerobot_src = config.LEROBOT_DIR / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets import aggregate as aggregate_module
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    class CanonicalFeatureMetadata(LeRobotDatasetMetadata):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            canonicalize_features(self.info.features)

    normalized_90 = normalize_libero90(snapshot_root, force=force)
    roots = [normalized_90, *(snapshot_root / source.suite for source in SOURCES[1:])]
    repo_ids = [f"{DATASET_REPO}/{source.suite}" for source in SOURCES]
    AGGREGATED_ROOT.parent.mkdir(parents=True, exist_ok=True)

    # aggregate_datasets は内部で metadata を読み直すため、そこにも canonical 化を挟む。
    aggregate_module.LeRobotDatasetMetadata = CanonicalFeatureMetadata
    try:
        aggregate_datasets(
            repo_ids=repo_ids,
            roots=roots,
            aggr_repo_id=AGGREGATED_REPO_ID,
            aggr_root=AGGREGATED_ROOT,
            concatenate_videos=False,
            concatenate_data=False,
        )
    finally:
        aggregate_module.LeRobotDatasetMetadata = LeRobotDatasetMetadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO-90多様タスク + replay データ準備"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="全データを取得し、学習用ローカル集約データセットを作る",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存の集約データセットを削除して再作成（--prepare と併用）",
    )
    args = parser.parse_args()
    if args.force and not args.prepare:
        parser.error("--force は --prepare と併用してください")

    ensure_work_dirs(
        config.WORK_DIR,
        config.HF_HOME,
        config.LEROBOT_CACHE,
        AGGREGATED_ROOT.parent,
    )
    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)

    print("1/4 metadata 取得")
    snapshot_root = download_snapshot(metadata_only=not args.prepare)
    print("2/4 task 分割・schema・episode 数を検証")
    metadata = load_metadata(snapshot_root)
    validate_schema(metadata)
    manifest = build_manifest(metadata, aggregated=args.prepare)

    if args.prepare:
        print("3/4 4 suite をローカル集約")
        aggregate(snapshot_root, force=args.force)
        from align_diverse_dataset_cameras import align_dataset

        print("3b/4 camera key を SmolVLA 向けに整列")
        align_dataset(AGGREGATED_ROOT)
        manifest["dataset"]["camera_key_mapping_for_policy"] = {
            "agentview_image": "observation.images.front",
            "robot0_eye_in_hand_image": "observation.images.wrist",
        }
    else:
        print("3/4 metadata-only（--prepare で全データを集約）")

    print("4/4 manifest 保存")
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mix = manifest["mixture"]
    print(
        f"train={len(TRAIN_TASK_NAMES)} tasks / OOD={len(OOD_TASK_NAMES)} tasks / "
        f"episodes={mix['total_episode_count']} "
        f"(replay={mix['replay_fraction']:.1%})"
    )
    print(f"manifest: {MANIFEST_PATH}")
    if args.prepare:
        print(f"dataset: {AGGREGATED_ROOT}")


if __name__ == "__main__":
    main()
