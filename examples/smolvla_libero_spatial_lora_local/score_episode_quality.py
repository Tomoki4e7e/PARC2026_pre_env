#!/usr/bin/env python3
"""LIBERO-plus / ローカル集約データセットの episode 品質キャッシュを作成する。

例:
  # Track1 stage2 対象タスクのみ
  python score_episode_quality.py --track1-stage2

  # Phase5 用（Track1 + Phase3 弱点）
  python score_episode_quality.py --phase5

  # 任意タスク名を指定
  python score_episode_quality.py --tasks "put the bowl on the stove"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from episode_quality import (
    build_episode_quality,
    group_indices_by_parquet,
    save_quality_cache,
    summarize_quality,
)
from utils import normalize_task_name, set_hf_env, task_name_from_cell


def _require_lerobot() -> Path:
    src = config.LEROBOT_DIR / "src"
    if not (src / "lerobot").is_dir():
        raise SystemExit(f"lerobot がありません: {src}")
    sys.path.insert(0, str(src))
    return src


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    if args.phase5:
        names: list[str] = []
        for suite in list(config.TRACK1_STAGE2_TRAIN_SUITES) + list(
            config.PHASE3_TRAIN_SUITES
        ):
            names.extend(suite.task_names)
        # 重複除去（順序維持）
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            key = normalize_task_name(name)
            if key not in seen:
                seen.add(key)
                out.append(name)
        return out
    if args.track1_stage2:
        names = []
        for suite in config.TRACK1_STAGE2_TRAIN_SUITES:
            names.extend(suite.task_names)
        return names
    if args.phase3:
        names = []
        for suite in config.PHASE3_TRAIN_SUITES:
            names.extend(suite.task_names)
        return names
    raise SystemExit("--tasks / --track1-stage2 / --phase3 / --phase5 のいずれかを指定してください")


def load_metadata(dataset_repo: str, dataset_revision: str | None, dataset_root: Path | None):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if dataset_root is not None:
        return LeRobotDatasetMetadata(dataset_repo, root=dataset_root, revision=dataset_revision)
    return LeRobotDatasetMetadata(dataset_repo, revision=dataset_revision)


def collect_task_pools(meta, task_names: list[str]) -> dict[str, list[int]]:
    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode_index, task_cell in enumerate(meta.episodes["tasks"]):
        task_to_episodes[task_name_from_cell(task_cell)].append(int(episode_index))

    available = {normalize_task_name(k): k for k in task_to_episodes}
    pools: dict[str, list[int]] = {}
    for requested in task_names:
        actual = available.get(normalize_task_name(requested))
        if actual is None:
            raise RuntimeError(f"task not found in dataset: {requested!r}")
        pools[requested] = task_to_episodes[actual]
    return pools


def score_episodes(
    meta,
    episode_indices: list[int],
    *,
    dt: float,
) -> list:
    grouped = group_indices_by_parquet(meta, episode_indices)
    wanted = set(episode_indices)
    qualities = []
    for parquet_path, eps in grouped.items():
        if not parquet_path.is_file():
            raise FileNotFoundError(f"parquet missing: {parquet_path}")
        table = pq.read_table(
            parquet_path, columns=["episode_index", "observation.state"]
        )
        df = table.to_pandas()
        for ep_idx in eps:
            if ep_idx not in wanted:
                continue
            sub = df[df["episode_index"] == ep_idx]
            if len(sub) == 0:
                # dataset_from/to でスライスするフォールバック
                ep_meta = meta.episodes[ep_idx]
                fr = int(ep_meta["dataset_from_index"])
                to = int(ep_meta["dataset_to_index"])
                # parquet 内の相対位置は不明なため skip
                print(f"[warn] episode {ep_idx} not in {parquet_path.name}; skip")
                continue
            states = np.stack(sub["observation.state"].to_numpy())
            qualities.append(
                build_episode_quality(ep_idx, states, dt=dt)
            )
    return qualities


def main() -> None:
    parser = argparse.ArgumentParser(description="Score demo episode trajectory quality")
    parser.add_argument("--track1-stage2", action="store_true")
    parser.add_argument("--phase3", action="store_true")
    parser.add_argument("--phase5", action="store_true", help="Track1+Phase3 弱点タスク")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 JSON（省略時は work/episode_quality_*.json）",
    )
    parser.add_argument("--local-phase4", action="store_true", help="Phase4 集約データセットを対象")
    args = parser.parse_args()

    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)
    _require_lerobot()

    task_names = resolve_tasks(args)
    if args.local_phase4:
        config.apply_phase4_config()
        dataset_repo = config.DATASET_REPO
        dataset_revision = config.DATASET_REVISION
        dataset_root = config.DATASET_ROOT
        default_name = "episode_quality_phase4.json"
    else:
        dataset_repo = config.LIBERO_PLUS_DATASET_REPO
        dataset_revision = config.LIBERO_PLUS_DATASET_REVISION
        dataset_root = None
        if args.phase5:
            default_name = "episode_quality_phase5.json"
        elif args.track1_stage2:
            default_name = "episode_quality_track1_stage2.json"
        elif args.phase3:
            default_name = "episode_quality_phase3.json"
        else:
            default_name = "episode_quality_custom.json"

    out_path = args.output or (config.WORK_DIR / default_name)
    print(f"[quality] dataset={dataset_repo} root={dataset_root}")
    print(f"[quality] tasks={len(task_names)} → {out_path}")

    meta = load_metadata(dataset_repo, dataset_revision, dataset_root)
    dt = 1.0 / float(meta.fps)
    pools = collect_task_pools(meta, task_names)
    all_indices: list[int] = sorted({idx for pool in pools.values() for idx in pool})
    print(f"[quality] scoring {len(all_indices)} episodes ...", flush=True)

    qualities = score_episodes(meta, all_indices, dt=dt)
    by_idx = {q.episode_index: q for q in qualities}
    missing = [i for i in all_indices if i not in by_idx]
    if missing:
        print(f"[quality] warning: {len(missing)} episodes missing metrics")

    meta_payload = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "fps": meta.fps,
        "dt": dt,
        "tasks": {
            name: {
                "n_pool": len(pool),
                "episode_indices": pool,
            }
            for name, pool in pools.items()
        },
        "summary": summarize_quality(by_idx),
    }
    save_quality_cache(out_path, qualities, meta=meta_payload)
    print(f"[quality] wrote {out_path} ({len(qualities)} episodes)")
    print(f"[quality] summary: {meta_payload['summary']}")


if __name__ == "__main__":
    main()
