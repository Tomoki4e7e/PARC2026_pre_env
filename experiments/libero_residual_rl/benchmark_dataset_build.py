from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


FEATURES = {
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
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


def _write_variant(name: str, paths: list[Path], root: Path, encoder, streaming: bool) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = root / name
    shutil.rmtree(output, ignore_errors=True)
    start = time.perf_counter()
    dataset = LeRobotDataset.create(
        repo_id=f"local/benchmark_{name}",
        fps=20,
        root=output,
        robot_type="panda",
        use_videos=True,
        streaming_encoding=streaming,
        batch_encoding_size=1,
        encoder_queue_maxsize=120,
        image_writer_threads=8,
        encoder_threads=4,
        rgb_encoder=encoder,
        features=FEATURES,
    )
    expected_frames = 0
    action_sum = np.zeros(7, dtype=np.float64)
    for path in paths:
        episode = np.load(path)
        expected_frames += len(episode["action"])
        action_sum += episode["action"].sum(axis=0)
        for index in range(len(episode["action"])):
            dataset.add_frame(
                {
                    "observation.images.front": episode["front"][index][::-1, ::-1],
                    "observation.images.wrist": episode["wrist"][index][::-1, ::-1],
                    "observation.state": episode["state"][index],
                    "action": episode["action"][index],
                    "task": str(episode["task"]),
                }
            )
        dataset.save_episode()
    dataset.finalize()
    elapsed = time.perf_counter() - start
    decoded = LeRobotDataset(
        f"local/benchmark_{name}",
        root=output,
        video_backend="pyav",
        return_uint8=True,
    )
    decoded_action_sum = np.zeros(7, dtype=np.float64)
    decoded_first = None
    for index in range(len(decoded)):
        row = decoded[index]
        decoded_action_sum += np.asarray(row["action"])
        if index == 0:
            decoded_first = np.asarray(row["observation.images.front"])
    if len(decoded) != expected_frames:
        raise RuntimeError(f"{name}: expected {expected_frames} frames, got {len(decoded)}")
    if not np.allclose(decoded_action_sum, action_sum, atol=1e-4):
        raise RuntimeError(f"{name}: action values changed")
    if decoded_first is None:
        raise RuntimeError(f"{name}: no decoded image")
    if decoded_first.shape[0] == 3:
        decoded_first = np.moveaxis(decoded_first, 0, -1)
    if decoded_first.dtype != np.uint8:
        decoded_first = np.clip(decoded_first * 255.0, 0, 255).astype(np.uint8)
    expected_first = np.load(paths[0])["front"][0][::-1, ::-1]
    mse = float(
        np.mean(
            (
                decoded_first.astype(np.float32)
                - expected_first.astype(np.float32)
            )
            ** 2
        )
    )
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)
    size_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return {
        "variant": name,
        "episodes": len(paths),
        "frames": expected_frames,
        "elapsed_sec": elapsed,
        "frames_per_sec": expected_frames / elapsed,
        "size_bytes": size_bytes,
        "codec": encoder.vcodec,
        "streaming": streaming,
        "first_frame_psnr_db": psnr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/grasp_dataset_benchmark"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lerobot_src))
    from lerobot.configs.video import RGBEncoderConfig

    paths = sorted(args.npz_root.glob("*.npz"))[: args.episodes]
    if len(paths) != args.episodes:
        raise RuntimeError(f"Requested {args.episodes} episodes, found {len(paths)}")
    variants = [
        ("svtav1_batch1", RGBEncoderConfig(vcodec="libsvtav1", preset=12, crf=30), False),
        ("h264_cpu", RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=28), False),
        ("h264_cpu_stream", RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=28), True),
        ("h264_nvenc_batch", RGBEncoderConfig(vcodec="h264_nvenc", preset=4, crf=23), False),
        ("h264_nvenc_stream", RGBEncoderConfig(vcodec="h264_nvenc", preset=4, crf=23), True),
    ]
    results = []
    for name, encoder, streaming in variants:
        try:
            results.append(
                _write_variant(name, paths, args.output_root, encoder, streaming)
            )
        except Exception as exc:
            results.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})
    valid = [row for row in results if "error" not in row]
    if not valid:
        raise RuntimeError(f"All variants failed: {results}")
    winner = max(valid, key=lambda row: row["frames_per_sec"])
    report = {"results": results, "winner": winner["variant"]}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
