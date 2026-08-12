from pathlib import Path

import cv2
import numpy as np

from pipeline.video import EpisodeVideoRecorder


def _obs(value: int = 80) -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.full((32, 32, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((32, 32, 3), value + 20, dtype=np.uint8),
    }


def test_episode_video_records_both_cameras_and_final_status(tmp_path: Path):
    recorder = EpisodeVideoRecorder(
        tmp_path, "task/name", 2, "pick the object", "both", 20.0, 2
    )
    recorder.add(_obs(), 0, force=True)
    recorder.add(_obs(90), 1)
    recorder.add(_obs(100), 2)
    path = recorder.close(2, success=False, collided=True)

    assert path is not None and path.exists()
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 32
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
