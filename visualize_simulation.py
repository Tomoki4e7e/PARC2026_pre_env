#!/usr/bin/env python3
"""Record a short, self-contained LIBERO / MuJoCo robot motion demo."""

from __future__ import annotations

import os
import sys

# robosuite 1.4's numba cache cannot be created from some mounted filesystems.
# This must be set before importing LIBERO / robosuite.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# OSMesa works both on GPU-less CI machines and normal Linux desktops.  Put
# system Mesa before the conda ImageMagick libraries added by env.sh; otherwise
# conda's libGL can shadow the system OSMesa driver.
os.environ.setdefault("MUJOCO_GL", "osmesa")
_system_gl = "/lib/x86_64-linux-gnu"
_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_system_gl) and not _ld_library_path.startswith(_system_gl):
    os.environ["LD_LIBRARY_PATH"] = f"{_system_gl}:{_ld_library_path}".rstrip(":")
    # The dynamic loader snapshots its search path at process start, so restart
    # once before importing MuJoCo.
    if os.environ.get("PARC_VISUALIZE_REEXEC") != "1":
        os.environ["PARC_VISUALIZE_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/parc2026-matplotlib")

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LIBERO / MuJoCo上でロボットを動かし、映像をMP4に保存します。"
    )
    parser.add_argument("--task-id", type=int, default=1, help="libero_t1のタスク番号 (0-3)")
    parser.add_argument("--steps", type=int, default=180, help="記録する物理ステップ数")
    parser.add_argument("--fps", type=float, default=20.0, help="出力動画のフレームレート")
    parser.add_argument("--size", type=int, default=512, help="各カメラ画像の一辺のpx数")
    parser.add_argument(
        "--camera",
        choices=("agent", "wrist", "both"),
        default="agent",
        help="記録するカメラ。bothは2画面を横に並べます",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/simulation_demo.mp4"),
        help="出力MP4パス",
    )
    return parser.parse_args()


def scripted_action(step: int, action_dim: int = 7) -> np.ndarray:
    """A gentle Cartesian sweep that visibly moves the arm without needing a model."""
    action = np.zeros(action_dim, dtype=np.float64)
    phase = (step // 30) % 6
    # Lift, sweep left/right, sweep forward/back, then lower a little.
    directions = (
        (2, 0.25), (0, 0.22), (0, -0.22),
        (1, 0.22), (1, -0.22), (2, -0.12),
    )
    axis, value = directions[phase]
    action[axis] = value
    if action_dim >= 7:
        action[6] = -1.0 if (step // 15) % 2 == 0 else 1.0
    return action


def _camera_frame(obs: dict[str, np.ndarray], camera: str) -> np.ndarray:
    keys = {
        "agent": ("agentview_image",),
        "wrist": ("robot0_eye_in_hand_image",),
        "both": ("agentview_image", "robot0_eye_in_hand_image"),
    }[camera]
    frames = []
    for key in keys:
        # MuJoCo's off-screen framebuffer is bottom-up; OpenCV expects top-down BGR.
        rgb = np.asarray(obs[key])[::-1]
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return np.concatenate(frames, axis=1)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.fps <= 0 or args.size <= 0:
        raise SystemExit("--steps, --fps, --size は正の値を指定してください")

    # Delayed imports keep the environment variables above effective.
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    config = EvalConfig(camera_height=args.size, camera_width=args.size, seed=42)
    manager = EnvironmentManager(config)
    tasks = manager.get_task_infos("libero_t1")
    if not 0 <= args.task_id < len(tasks):
        raise SystemExit(f"--task-id は 0 から {len(tasks) - 1} の範囲です")
    task = tasks[args.task_id]
    env = manager.create_env(task)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width = args.size * (2 if args.camera == "both" else 1)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, args.size)
    )
    if not writer.isOpened():
        env.close()
        raise RuntimeError(f"動画エンコーダーを開けませんでした: {output}")

    try:
        env.reset()
        init_state = np.asarray(task.init_states[0])
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()
        action_dim = env.robots[0].action_dim
        dummy = np.zeros(action_dim, dtype=np.float64)
        if action_dim >= 7:
            dummy[6] = -1.0
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)
        for robot in env.robots:
            robot.controller.use_delta = True

        for step in range(args.steps):
            obs, _, _, _ = env.step(scripted_action(step, action_dim))
            frame = _camera_frame(obs, args.camera)
            cv2.rectangle(frame, (0, 0), (width, 66), (0, 0, 0), -1)
            cv2.putText(frame, "LIBERO + MuJoCo", (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"task {args.task_id} | step {step + 1}/{args.steps}",
                        (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (100, 220, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        env.close()

    print(f"動画を保存しました: {output}")
    print(f"タスク: {task.language}")


if __name__ == "__main__":
    main()
