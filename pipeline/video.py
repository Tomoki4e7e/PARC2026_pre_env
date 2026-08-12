"""Video recording for local policy rollouts."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120]


class EpisodeVideoRecorder:
    """Stream simulator observations to MP4 without retaining them in memory."""

    def __init__(self, output_dir: Path, task_name: str, episode_id: int,
                 instruction: str, camera: str, fps: float, frame_skip: int):
        self.camera = camera
        self.frame_skip = frame_skip
        self.fps = fps
        self.instruction = instruction
        self.path = output_dir / f"{_safe_name(task_name)}__ep{episode_id:03d}.mp4"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.last_obs = None
        self.frames = 0

    def _frame(self, obs: dict[str, np.ndarray], step: int, status: str) -> np.ndarray:
        keys = {
            "agent": ("agentview_image",),
            "wrist": ("robot0_eye_in_hand_image",),
            "both": ("agentview_image", "robot0_eye_in_hand_image"),
        }[self.camera]
        frames = [cv2.cvtColor(np.asarray(obs[k])[::-1], cv2.COLOR_RGB2BGR) for k in keys]
        frame = np.concatenate(frames, axis=1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(frame, f"SmolVLA rollout | step {step} | {status}", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        instruction = self.instruction[:78]
        cv2.putText(frame, instruction, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.39, (100, 220, 255), 1, cv2.LINE_AA)
        return frame

    def add(self, obs: dict[str, np.ndarray], step: int, force: bool = False) -> None:
        self.last_obs = obs
        if not force and step % self.frame_skip:
            return
        frame = self._frame(obs, step, "running")
        if self.writer is None:
            size = (frame.shape[1], frame.shape[0])
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"動画エンコーダーを開けませんでした: {self.path}")
        self.writer.write(frame)
        self.frames += 1

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        self._fps = value

    def close(self, step: int, success: bool, collided: bool) -> Path | None:
        if self.writer is None:
            return None
        if self.last_obs is not None:
            status = "SUCCESS" if success else ("FAIL: collision" if collided else "FAIL")
            self.writer.write(self._frame(self.last_obs, step, status))
            self.frames += 1
        self.writer.release()
        return self.path
