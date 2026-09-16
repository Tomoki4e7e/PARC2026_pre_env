from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental frozen SmolVLA server")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--gripper-encoding",
        choices=("zero_one", "libero_signed"),
        default=None,
        help="required only for legacy checkpoints without action_schema.json",
    )
    args = parser.parse_args()

    submission_dir = Path(__file__).resolve().parents[2] / "submission_template"
    sys.path.insert(0, str(submission_dir))
    import uvicorn
    from policy_server import app, set_policy

    from .base_policy import FrozenSmolVLAPolicy

    set_policy(
        FrozenSmolVLAPolicy(
            args.model_dir,
            gripper_encoding=args.gripper_encoding,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

