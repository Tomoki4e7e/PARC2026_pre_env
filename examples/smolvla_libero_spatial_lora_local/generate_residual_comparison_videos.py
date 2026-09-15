#!/usr/bin/env python3
"""Record matched before/after videos for the 0.33443 residual policy."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission_template"
TASK = "open_the_middle_drawer_of_the_cabinet_light_32"


def record(name: str, residual_enabled: bool) -> None:
    os.environ["PARC_RESIDUAL_POLICY"] = "1" if residual_enabled else "0"
    os.environ["PARC_COLLISION_CRITIC"] = "off"
    os.environ["PARC_ENABLE_PROXY_SUITE"] = "1"

    # Load the simulator (and MagickWand/libffi) before torch.  Loading torch
    # first can bind an incompatible system libffi in the local environment.
    from pipeline.config import EvalConfig, Track
    from pipeline.pipeline import EvaluationPipeline

    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    from smolvla_mypolicy import SmolVLAMyPolicy

    previous_cwd = Path.cwd()
    try:
        os.chdir(SUBMISSION)
        policy = SmolVLAMyPolicy("model_weights/smolvla_spatial_lora")
    finally:
        os.chdir(previous_cwd)

    config = EvalConfig(
        n_eval_episodes=1,
        max_steps_per_episode=200,
        seed=31415,
        task_ids=[TASK],
        video_dir=ROOT / "docs" / "assets" / "residual_comparison" / name,
        video_camera="agent",
        video_fps=20.0,
        video_frame_skip=1,
        output_dir=ROOT / "results" / "residual_video_comparison" / name,
    )
    pipeline = EvaluationPipeline(eval_config=config)
    result = pipeline.run(
        SUBMISSION,
        policy,
        tracks=[Track.TRACK1],
        submission_id=name,
        benchmark_override="libero_proxy",
        skip_validation=True,
    )
    score = result.track_scores[0]
    if not score.task_scores:
        raise RuntimeError(f"evaluation produced no task result: {result.to_dict()}")


def main() -> None:
    record("before", residual_enabled=False)
    record("after", residual_enabled=True)


if __name__ == "__main__":
    main()
