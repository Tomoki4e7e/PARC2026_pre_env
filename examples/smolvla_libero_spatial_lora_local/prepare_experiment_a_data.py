"""Download only the 100 drawer demonstrations used by Experiment A."""

from __future__ import annotations

import json
import sys
import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

import config
from utils import choose_evenly_spaced, normalize_task_name, set_hf_env


def _as_image(value):
    import numpy as np
    import torch
    from PIL import Image

    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if array.shape[:2] != (128, 128):
        array = np.asarray(Image.fromarray(array).resize((128, 128), Image.Resampling.BILINEAR))
    return array


def build_recovery_dataset() -> None:
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    source_manifest = json.loads(
        (config.EXPERIMENT_A_DATASET_ROOT.parent / "experiment_a_manifest.json").read_text()
    )
    oracle_root = config.EXPERIMENT_A_ORACLE_ROOT
    output = config.EXPERIMENT_A_RECOVERY_DATASET_ROOT
    if not oracle_root.is_dir():
        raise FileNotFoundError(f"Oracle demos are absent: {oracle_root}")
    if output.exists():
        shutil.rmtree(output)
    target = LeRobotDataset.create(
        repo_id=config.EXPERIMENT_A_RECOVERY_DATASET_REPO,
        fps=20,
        root=output,
        robot_type="panda",
        use_videos=True,
        streaming_encoding=True,
        batch_encoding_size=1,
        encoder_queue_maxsize=120,
        encoder_threads=4,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264", preset="ultrafast", crf=28, g=2
        ),
        features={
            "observation.images.front": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "observation.images.wrist": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
        },
    )
    selected = source_manifest["selected_episode_indices"]
    source = LeRobotDataset(
        "local/experiment_a_source",
        root=config.EXPERIMENT_A_DATASET_ROOT,
        episodes=selected,
        video_backend=config.VIDEO_BACKEND,
        return_uint8=True,
    )
    current_episode = None
    selected_by_task: dict[str, list[int]] = defaultdict(list)
    for index in range(len(source)):
        row = source[index]
        episode = int(row["episode_index"])
        if current_episode is not None and episode != current_episode:
            target.save_episode()
        current_episode = episode
        action = np.asarray(row["action"], dtype=np.float32).copy()
        action[6] = 1.0 - 2.0 * np.clip(action[6], 0.0, 1.0)
        task = str(row["task"])
        if not selected_by_task[task] or selected_by_task[task][-1] != target.meta.total_episodes:
            selected_by_task[task].append(target.meta.total_episodes)
        target.add_frame(
            {
                "observation.images.front": _as_image(
                    row["observation.images.image"]
                ),
                "observation.images.wrist": _as_image(
                    row["observation.images.wrist_image"]
                ),
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": action,
                "task": task,
            }
        )
    if current_episode is not None:
        target.save_episode()

    for path in sorted(oracle_root.glob("*.npz")):
        episode = np.load(path)
        task = str(episode["task"])
        output_episode = target.meta.total_episodes
        selected_by_task[task].append(output_episode)
        for frame in range(len(episode["action"])):
            target.add_frame(
                {
                    # Collector sidecars were stored in policy-ready orientation;
                    # datasets must retain raw LIBERO orientation because the
                    # training processor performs the 180-degree transform.
                    "observation.images.front": episode["front"][frame][::-1, ::-1],
                    "observation.images.wrist": episode["wrist"][frame][::-1, ::-1],
                    "observation.state": episode["state"][frame],
                    "action": episode["action"][frame],
                    "task": task,
                }
            )
        target.save_episode()
    target.finalize()
    manifest = {
        "dataset_root": str(output),
        "selected_episode_indices": list(range(target.meta.total_episodes)),
        "selected_by_task": selected_by_task,
        "source_episode_count": len(selected),
        "oracle_episode_count": len(list(oracle_root.glob("*.npz"))),
        "gripper_encoding": "libero_signed",
    }
    config.EXPERIMENT_A_RECOVERY_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: len(value) for key, value in selected_by_task.items()}, indent=2))


def append_dagger_dataset() -> None:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig

    manifest = json.loads(config.EXPERIMENT_A_RECOVERY_MANIFEST.read_text())
    if manifest.get("dagger_episode_count"):
        raise RuntimeError("DAgger episodes were already appended")
    target = LeRobotDataset.resume(
        config.EXPERIMENT_A_RECOVERY_DATASET_REPO,
        root=config.EXPERIMENT_A_RECOVERY_DATASET_ROOT,
        video_backend=config.VIDEO_BACKEND,
        image_writer_threads=8,
        batch_encoding_size=1,
        encoder_threads=4,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264", preset="ultrafast", crf=28, g=2
        ),
    )
    selected_by_task = manifest["selected_by_task"]
    paths = sorted(config.EXPERIMENT_A_DAGGER_ROOT.glob("*.npz"))
    for path in paths:
        episode = np.load(path)
        task = str(episode["task"])
        selected_by_task.setdefault(task, []).append(target.meta.total_episodes)
        for frame in range(len(episode["action"])):
            target.add_frame(
                {
                    "observation.images.front": episode["front"][frame][::-1, ::-1],
                    "observation.images.wrist": episode["wrist"][frame][::-1, ::-1],
                    "observation.state": episode["state"][frame],
                    "action": episode["action"][frame],
                    "task": task,
                }
            )
        target.save_episode()
    target.finalize()
    manifest["selected_episode_indices"] = list(range(target.meta.total_episodes))
    manifest["dagger_episode_count"] = len(paths)
    config.EXPERIMENT_A_RECOVERY_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Appended {len(paths)} DAgger episodes")


def _v5_quality(path: Path, *, anchor: bool) -> tuple[bool, str]:
    import numpy as np

    episode = np.load(path)
    required = {
        "front",
        "wrist",
        "state",
        "action",
        "base_action",
        "contact",
        "phase",
        "task",
    }
    if not required.issubset(episode.files):
        return False, "schema"
    if len(episode["action"]) > 600:
        return False, "overlong"
    if not np.isin(episode["action"][:, 6], (-1.0, 1.0)).all():
        return False, "gripper_encoding"
    contact = np.flatnonzero(episode["contact"])
    if not len(contact) or not np.any(episode["phase"] == 4):
        return False, "strict_grasp_labels"
    task = str(episode["task"])
    if task not in {
        "open the middle drawer of the cabinet",
        "open the top drawer and put the bowl inside",
    }:
        return False, "instruction"
    if "top drawer" in task:
        close = np.flatnonzero(episode["action"][: contact[0], 6] > 0.5)
        if len(close) and contact[0] - close[0] > 30:
            return False, "early_close"
    if anchor and not np.array_equal(
        episode["action"][:, :6], episode["base_action"][:, :6]
    ):
        return False, "anchor_arm_changed"
    return True, "accepted"


def _v5_seed(path: Path) -> int:
    root_name = path.parent.name
    if root_name.endswith("_dagger_train"):
        root_name = root_name.removesuffix("_train")
    manifest = path.parent.parent / f"{root_name}_manifest.json"
    data = json.loads(manifest.read_text())
    task_key = (
        "open_the_middle_drawer_of_the_cabinet"
        if "middle" in path.name
        else "open_the_top_drawer_and_put_the_bowl_inside"
    )
    index = int(path.stem.rsplit("_", 1)[1])
    accepted = [row for row in data["tasks"][task_key]["episodes"] if row["accepted"]]
    return int(accepted[index]["seed"])


def build_arm_lora_v5_dataset() -> None:
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    roots = [
        config.WORK_DIR / "datasets" / "experiment_a_arm_corrections_train",
        config.WORK_DIR / "datasets" / "experiment_a_arm_corrections_dagger_train",
    ]
    anchor_root = config.WORK_DIR / "datasets" / "experiment_a_gripper_labeled"
    protocol = json.loads(
        (
            config.REPO_ROOT
            / "experiments/libero_residual_rl/artifacts/manifests/arm_reach_seed_protocol.json"
        ).read_text()
    )
    held_out = set()
    for section in ("candidate", "candidate_after_dagger", "final"):
        for seeds in protocol.get(section, {}).get("tasks", {}).values():
            held_out.update(seeds)

    selected: list[tuple[Path, str, int]] = []
    excluded: list[dict[str, str]] = []
    for root in roots:
        for path in sorted(root.glob("*.npz")):
            accepted, reason = _v5_quality(path, anchor=False)
            seed = _v5_seed(path)
            if seed in held_out:
                accepted, reason = False, "held_out_seed"
            if accepted:
                selected.append((path, root.name, seed))
            else:
                excluded.append({"path": str(path), "reason": reason})

    anchors_by_task: dict[str, list[tuple[Path, str, int]]] = {
        "middle": [],
        "top": [],
    }
    for path in sorted(anchor_root.glob("*.npz")):
        accepted, reason = _v5_quality(path, anchor=True)
        task = "middle" if "middle" in path.name else "top"
        if accepted and len(anchors_by_task[task]) < 50:
            anchors_by_task[task].append((path, "clean_anchor", _v5_seed(path)))
        elif not accepted:
            excluded.append({"path": str(path), "reason": f"anchor_{reason}"})
    if any(len(paths) < 50 for paths in anchors_by_task.values()):
        raise RuntimeError(
            f"Insufficient clean anchors: { {k: len(v) for k, v in anchors_by_task.items()} }"
        )
    selected.extend(anchors_by_task["middle"])
    selected.extend(anchors_by_task["top"])
    seeds = [seed for _, _, seed in selected]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Duplicate v5 training seeds")
    if set(seeds) & held_out:
        raise RuntimeError("v5 training data overlaps candidate/final seeds")

    output = config.EXPERIMENT_A_V5_DATASET_ROOT
    shutil.rmtree(output, ignore_errors=True)
    target = LeRobotDataset.create(
        repo_id=config.EXPERIMENT_A_V5_DATASET_REPO,
        fps=20,
        root=output,
        robot_type="panda",
        use_videos=True,
        streaming_encoding=True,
        batch_encoding_size=1,
        encoder_queue_maxsize=120,
        encoder_threads=4,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264", preset="ultrafast", crf=28, g=2
        ),
        features={
            "observation.images.front": {
                "dtype": "video",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (7,), "names": None},
        },
    )
    selected_by_task: dict[str, list[int]] = defaultdict(list)
    rows = []
    for path, source, seed in selected:
        episode = np.load(path)
        task = str(episode["task"])
        output_episode = target.meta.total_episodes
        selected_by_task[task].append(output_episode)
        for frame in range(len(episode["action"])):
            target.add_frame(
                {
                    "observation.images.front": episode["front"][frame][::-1, ::-1],
                    "observation.images.wrist": episode["wrist"][frame][::-1, ::-1],
                    "observation.state": episode["state"][frame],
                    "action": episode["action"][frame],
                    "task": task,
                }
            )
        target.save_episode()
        rows.append(
            {
                "episode_index": output_episode,
                "source": source,
                "source_path": str(path),
                "seed": seed,
                "task": task,
                "frames": len(episode["action"]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    target.finalize()
    manifest = {
        "dataset_root": str(output),
        "selected_episode_indices": list(range(target.meta.total_episodes)),
        "selected_by_task": selected_by_task,
        "episodes": rows,
        "excluded": excluded,
        "held_out_seeds": sorted(held_out),
        "gripper_encoding": "libero_signed",
        "base_model": str(config.EXPERIMENT_A_V5_BASE_MODEL_DIR),
    }
    config.EXPERIMENT_A_V5_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "selected_by_task": {
                    key: len(value) for key, value in selected_by_task.items()
                },
                "excluded": len(excluded),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-recovery", action="store_true")
    parser.add_argument("--append-dagger", action="store_true")
    parser.add_argument("--build-arm-v5", action="store_true")
    args = parser.parse_args()
    set_hf_env(config.HF_HOME, config.LEROBOT_CACHE)
    lerobot_src = config.LEROBOT_DIR / "src"
    sys.path.insert(0, str(lerobot_src))
    if args.build_arm_v5:
        build_arm_lora_v5_dataset()
        return
    elif args.build_recovery:
        build_recovery_dataset()
        return
    if args.append_dagger:
        append_dagger_dataset()
        return
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata_snapshot = Path(
        snapshot_download(
            repo_id="nvidia/LIBERO_LeRobot_v3",
            repo_type="dataset",
            revision=config.EXPERIMENT_A_DATASET_REVISION,
            allow_patterns=["libero_goal/meta/**"],
            token=False,
        )
    )
    metadata = LeRobotDatasetMetadata(
        "local/experiment_a_source",
        root=metadata_snapshot / "libero_goal",
        revision=config.EXPERIMENT_A_DATASET_REVISION,
    )
    by_task: dict[str, list[int]] = defaultdict(list)
    actual_names: dict[str, str] = {}
    for index, cell in enumerate(metadata.episodes["tasks"]):
        name = cell[0] if isinstance(cell, list) else str(cell)
        normalized = normalize_task_name(name)
        by_task[normalized].append(index)
        actual_names[normalized] = name
    selected: list[int] = []
    selected_by_task: dict[str, list[int]] = {}
    for task in config.EXPERIMENT_A_TRAIN_SUITES[0].task_names:
        normalized = normalize_task_name(task)
        if normalized not in by_task:
            raise RuntimeError(f"Task is absent from source metadata: {task}")
        available = by_task[normalized]
        indices = choose_evenly_spaced(available, min(50, len(available)))
        selected.extend(indices)
        selected_by_task[actual_names[normalized]] = indices

    files = {"libero_goal/meta/**"}
    for episode in selected:
        files.add(f"libero_goal/{metadata.get_data_file_path(episode)}")
        for video_key in metadata.video_keys:
            files.add(f"libero_goal/{metadata.get_video_file_path(episode, video_key)}")
    destination = config.EXPERIMENT_A_DATASET_ROOT.parent
    snapshot_download(
        repo_id="nvidia/LIBERO_LeRobot_v3",
        repo_type="dataset",
        revision=config.EXPERIMENT_A_DATASET_REVISION,
        allow_patterns=sorted(files),
        local_dir=destination,
        token=False,
        max_workers=2,
    )
    manifest = {
        "source_repo": "nvidia/LIBERO_LeRobot_v3",
        "source_revision": config.EXPERIMENT_A_DATASET_REVISION,
        "dataset_root": str(config.EXPERIMENT_A_DATASET_ROOT),
        "selected_episode_indices": sorted(selected),
        "selected_by_task": selected_by_task,
    }
    path = destination / "experiment_a_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected)} episodes at {config.EXPERIMENT_A_DATASET_ROOT}")
    print(path)


if __name__ == "__main__":
    main()

