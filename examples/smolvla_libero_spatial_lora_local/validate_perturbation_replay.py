#!/usr/bin/env python3
"""既存LIBEROデモのactionをtable/light変種へ移植できるか検証する。

外観だけが異なる変種はベースタスクとinit stateを共有する。各タスクのデモを
同じordinalのinit stateから再生し、成功と競技用collision判定を記録する。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATASET_ROOT = (SCRIPT_DIR / "work/lerobot_cache/hub/datasets--lerobot--libero_plus/"
                "snapshots/f3f49f426d75030177b18778374005bc12ccd588")
DEFAULT_CSV = REPO_ROOT / "results/track1_distribution_audit/tasks.csv"
RESULT_PATH = REPO_ROOT / "results/track1_perturbation_replay/summary.json"
os.environ.setdefault("PARC_ENABLE_PROXY_SUITE", "1")


def load_dataset_tables():
    import pandas as pd
    tasks = pd.read_parquet(DATASET_ROOT / "meta/tasks.parquet")
    episodes = pd.read_parquet(DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet")
    frames = pd.concat(
        [pd.read_parquet(path) for path in sorted((DATASET_ROOT / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    return tasks, episodes, frames


def build_lerobot_dataset(record_dir: Path, dataset_root: Path) -> None:
    """採用済みnpz軌道をローカルLeRobotDatasetへ変換する。"""
    recordings = sorted(record_dir.glob("*.npz"))
    if not recordings:
        raise RuntimeError(f"accepted recordings not found: {record_dir}")
    if dataset_root.exists():
        raise FileExistsError(
            f"dataset already exists (refusing to overwrite): {dataset_root}"
        )
    lerobot_src = SCRIPT_DIR / "work/lerobot/src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.front": {
            "dtype": "image", "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image", "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (8,),
            "names": [f"state_{index}" for index in range(8)],
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": [f"action_{index}" for index in range(7)],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/libero_track1_perturbation_replay",
        fps=20, features=features, root=dataset_root, robot_type="panda",
        use_videos=False, image_writer_threads=4,
    )
    for recording in recordings:
        with np.load(recording, allow_pickle=False) as episode:
            task = str(episode["task"].item())
            length = len(episode["action"])
            for index in range(length):
                dataset.add_frame({
                    "observation.images.front": episode["front"][index],
                    "observation.images.wrist": episode["wrist"][index],
                    "observation.state": episode["state"][index],
                    "action": episode["action"][index],
                    "task": task,
                })
            dataset.save_episode(parallel_encoding=False)
            print(f"[dataset] added {recording.name}: {length} frames", flush=True)
    dataset.finalize()
    print(f"[dataset] finalized {len(recordings)} episodes: {dataset_root}")


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = np.sqrt(max(0.0, 1.0 - w * w))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


def replay(
    task_info, init_state: np.ndarray, actions: list[np.ndarray], record_path: Path | None = None
) -> dict:
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    manager = EnvironmentManager(EvalConfig(camera_height=256, camera_width=256))
    env = manager.create_env(task_info)
    try:
        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()
        action_dim = env.robots[0].action_dim
        obs = None
        for _ in range(10):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1])[:action_dim])
        for robot in env.robots:
            robot.controller.use_delta = True

        interested = manager.get_obj_of_interest(task_info)
        diagnostic_positions = {
            key[:-4]: np.asarray(value).tolist()
            for key, value in obs.items()
            if key.endswith("_pos") and not key.startswith("robot0")
            and not key.endswith("_to_robot0_eef_pos")
            and ("bowl" in key or "cream_cheese" in key)
        }
        initial = {
            key[:-4]: np.asarray(value).copy()
            for key, value in obs.items()
            if key.endswith("_pos") and not key.startswith("robot0")
            and not key.endswith("_to_robot0_eef_pos") and key[:-4] not in interested
        }
        max_disp: dict[str, float] = {}
        first_collision: dict[str, float | int | str] | None = None
        done = False
        steps = 0
        front_images, wrist_images, states, recorded_actions = [], [], [], []
        for action in actions:
            front_images.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
            wrist_images.append(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
            states.append(np.concatenate([
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            ]))
            recorded_actions.append(np.asarray(action, dtype=np.float32))
            obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
            steps += 1
            for name, p0 in initial.items():
                if name + "_pos" in obs:
                    displacement = float(np.abs(np.asarray(obs[name + "_pos"]) - p0).sum())
                    max_disp[name] = max(max_disp.get(name, 0.0), displacement)
                    if first_collision is None and displacement > 0.001:
                        first_collision = {
                            "object": name, "step": steps, "displacement": displacement,
                        }
            if done:
                break
        collided = any(value > 0.001 for value in max_disp.values())
        accepted = bool(done) and not collided
        if accepted and record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                record_path,
                front=np.stack(front_images), wrist=np.stack(wrist_images),
                state=np.stack(states).astype(np.float32),
                action=np.stack(recorded_actions).astype(np.float32),
                task=np.asarray(task_info.language), task_id=np.asarray(task_info.name),
            )
        return {"done": bool(done), "collided": collided,
                "accepted": accepted, "steps": steps,
                "record_path": str(record_path) if accepted and record_path else None,
                "max_non_target_displacement": max(max_disp.values(), default=0.0),
                "first_collision": first_collision,
                "diagnostic_positions": diagnostic_positions}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--variants-per-group", type=int, default=1)
    parser.add_argument("--episodes-per-variant", type=int, default=3)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--episode-ordinals", default="")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_payload:
        payload = json.loads(args.worker_payload.read_text(encoding="utf-8"))
        os.environ["PARC_PROXY_TASKS_CSV"] = payload["tasks_csv"]
        from pipeline.config import EvalConfig, PerturbationConfig
        from pipeline.environment import EnvironmentManager
        manager = EnvironmentManager(EvalConfig())
        infos = {info.name: info for info in manager.get_task_infos("libero_proxy")}
        info = infos[payload["task_id"]]
        states = manager.get_perturbed_init_states(
            info, PerturbationConfig(), payload["init_ordinal"] + 1
        )
        record_path = Path(payload["record_path"]) if payload.get("record_path") else None
        outcome = replay(
            info, states[payload["init_ordinal"]], payload["actions"], record_path
        )
        args.worker_output.write_text(json.dumps(outcome), encoding="utf-8")
        return
    os.environ["PARC_PROXY_TASKS_CSV"] = str(args.tasks_csv.resolve())

    with args.tasks_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requested_ids = {value.strip() for value in args.task_ids.split(",") if value.strip()}
    if requested_ids:
        rows = [row for row in rows if row["task_id"] in requested_ids]
        missing = requested_ids - {row["task_id"] for row in rows}
        if missing:
            raise ValueError(f"requested task IDs not found: {sorted(missing)}")
    selected = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["base_task"], row["category"])
        if counts.get(key, 0) < args.variants_per_group:
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1

    tasks_meta, episodes_meta, frames = load_dataset_tables()
    task_index = {name.replace(" ", "_"): int(row.task_index)
                  for name, row in tasks_meta.iterrows()}
    results = []
    for row in selected:
        base = row["base_task"]
        source_index = task_index[base]
        source_episode_ids = [
            int(ep.episode_index) for _, ep in episodes_meta.iterrows()
            if source_index in [int(v) for v in frames.loc[
                frames.episode_index == int(ep.episode_index), "task_index"
            ].unique()]
        ]
        requested_ordinals = (
            [int(value) for value in args.episode_ordinals.split(",") if value.strip()]
            if args.episode_ordinals else list(range(args.episodes_per_variant))
        )
        for ordinal in requested_ordinals:
            if ordinal >= len(source_episode_ids):
                raise ValueError(f"episode ordinal out of range: {base} ordinal={ordinal}")
            episode_id = source_episode_ids[ordinal]
            episode_frames = frames[frames.episode_index == episode_id].sort_values("frame_index")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload_path = args.output.parent / "worker_payload.json"
            worker_output = args.output.parent / "worker_output.json"
            payload_path.write_text(json.dumps({
                "tasks_csv": str(args.tasks_csv.resolve()),
                "task_id": row["task_id"],
                "init_ordinal": ordinal,
                "actions": [np.asarray(action).tolist() for action in episode_frames.action],
                "record_path": str(
                    args.record_dir / f'{row["task_id"]}__ep{ordinal:03d}.npz'
                ) if args.record_dir else None,
            }), encoding="utf-8")
            worker_env = os.environ.copy()
            worker_env["PYTHONPATH"] = (
                f"{REPO_ROOT / 'LIBERO-plus'}:{REPO_ROOT}:{REPO_ROOT / 'compe'}"
            )
            subprocess.run([
                "/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python",
                str(Path(__file__).resolve()), "--worker-payload", str(payload_path),
                "--worker-output", str(worker_output),
            ], cwd=REPO_ROOT, env=worker_env, check=True)
            outcome = json.loads(worker_output.read_text(encoding="utf-8"))
            results.append({**row, "source_episode": episode_id, "init_ordinal": ordinal, **outcome})
            print(row["task_id"], ordinal, outcome, flush=True)

    report = {
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "tasks_csv": str(args.tasks_csv), "output": str(args.output),
            "record_dir": str(args.record_dir) if args.record_dir else None,
            "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        },
        "n_replays": len(results),
        "accepted_rate": sum(item["accepted"] for item in results) / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.build_dataset:
        if not args.record_dir or not args.dataset_root:
            parser.error("--build-dataset requires --record-dir and --dataset-root")
        build_lerobot_dataset(args.record_dir, args.dataset_root)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
