from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental frozen SmolVLA server")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gripper-head")
    parser.add_argument("--arm-head")
    parser.add_argument("--arm-refinement-head")
    parser.add_argument("--arm-scale", type=float, default=1.0)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help="decoded actions executed before replanning (default: checkpoint value)",
    )
    parser.add_argument("--middle-execution-horizon", type=int)
    parser.add_argument("--top-execution-horizon", type=int)
    parser.add_argument("--post-close-execution-horizon", type=int)
    parser.add_argument("--post-close-arm-scale", type=float, default=1.0)
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
    from .arm_residual_head import LearnedFrontGraspPolicy
    from .gripper_head import LearnedGripperResidualPolicy

    task_horizons = {
        task: horizon
        for task, horizon in (
            ("open the middle drawer of the cabinet", args.middle_execution_horizon),
            (
                "open the top drawer and put the bowl inside",
                args.top_execution_horizon,
            ),
        )
        if horizon is not None
    }
    policy = FrozenSmolVLAPolicy(
        args.model_dir,
        gripper_encoding=args.gripper_encoding,
        execution_horizon=args.execution_horizon,
        execution_horizons=task_horizons,
    )
    print(
        f"[base-server] default_execution_horizon={policy.execution_horizon} "
        f"task_horizons={task_horizons}",
        flush=True,
    )
    if args.arm_refinement_head and not args.arm_head:
        parser.error("--arm-refinement-head requires --arm-head")
    if args.arm_head:
        if not args.gripper_head:
            parser.error("--arm-head requires --gripper-head")
        policy = LearnedFrontGraspPolicy(
            policy,
            arm_checkpoint=args.arm_head,
            gripper_checkpoint=args.gripper_head,
            arm_refinement_checkpoint=args.arm_refinement_head,
            arm_scale=args.arm_scale,
        )
    elif args.gripper_head:
        policy = LearnedGripperResidualPolicy(
            policy,
            args.gripper_head,
            post_close_execution_horizon=args.post_close_execution_horizon,
            post_close_arm_scale=args.post_close_arm_scale,
        )
    set_policy(policy)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

