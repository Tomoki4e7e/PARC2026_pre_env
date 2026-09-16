from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import write_analysis
from .base_policy import FrozenSmolVLAPolicy, RemoteBasePolicy
from .config import ARTIFACT_ROOT, ExperimentConfig
from .evaluate import evaluate_policy, load_actor
from .task_catalog import MICROWAVE, MIDDLE_DRAWER, TOP_DRAWER, TaskCatalog
from .trainer import OnlineResidualTrainer


def _specs(args: argparse.Namespace, *, training: bool = False):
    catalog = TaskCatalog()
    if training:
        return catalog.build(
            base_tasks=(MIDDLE_DRAWER, TOP_DRAWER),
            axes=("layout", "camera", "robot"),
            levels=(1,),
            episodes_per_condition=args.episodes_per_condition,
            seed=args.seed,
        )
    specs = catalog.build(
        base_tasks=(MIDDLE_DRAWER, TOP_DRAWER),
        axes=("layout", "camera", "robot"),
        levels=(0, 1, 2),
        episodes_per_condition=args.episodes_per_condition,
        seed=args.seed,
    )
    if args.include_microwave:
        specs.extend(
            catalog.build(
                base_tasks=(MICROWAVE,),
                axes=("layout",),
                levels=(0,),
                episodes_per_condition=args.episodes_per_condition,
                seed=args.seed,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="LIBERO online flow-residual RL experiment")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--episodes-per-condition", type=int, default=50)
    manifest.add_argument("--include-microwave", action="store_true")
    manifest.add_argument("--output", type=Path)

    train = subparsers.add_parser("train")
    train_base = train.add_mutually_exclusive_group(required=True)
    train_base.add_argument("--model-dir", type=Path)
    train_base.add_argument("--server-url")
    train.add_argument("--episodes", type=int, default=500)
    train.add_argument("--episodes-per-condition", type=int, default=50)
    train.add_argument("--warmup-steps", type=int, default=1_000)

    evaluate = subparsers.add_parser("evaluate")
    evaluate_base = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_base.add_argument("--model-dir", type=Path)
    evaluate_base.add_argument("--server-url")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--condition", choices=("base", "residual"), required=True)
    evaluate.add_argument("--episodes-per-condition", type=int, default=5)
    evaluate.add_argument("--include-microwave", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--base", type=Path, required=True)
    analyze.add_argument("--residual", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig(seed=args.seed, artifact_root=args.artifact_root)
    if args.command == "manifest":
        specs = _specs(args)
        output = args.output or args.artifact_root / "manifests" / "evaluation.json"
        TaskCatalog.save_manifest(specs, output)
        print(json.dumps({"manifest": str(output), "episodes": len(specs)}))
        return
    if args.command == "analyze":
        print(json.dumps(write_analysis(args.base, args.residual, args.output), indent=2))
        return

    base_policy = (
        RemoteBasePolicy(args.server_url)
        if args.server_url
        else FrozenSmolVLAPolicy(args.model_dir)
    )
    if args.command == "train":
        config.train_episodes = args.episodes
        config.warmup_steps = args.warmup_steps
        trainer = OnlineResidualTrainer(config)
        specs = _specs(args, training=True)
        run_dir = args.artifact_root / "runs" / "online_l1"
        TaskCatalog.save_manifest(specs, run_dir / "training_manifest.json")
        trainer.train(
            specs,
            lambda: base_policy,
            episodes=args.episodes,
            log_path=run_dir / "episodes.jsonl",
        )
        trainer.save(run_dir / "checkpoint")
        print(json.dumps({"checkpoint": str(run_dir / "checkpoint"), "episodes": args.episodes}))
        return

    specs = _specs(args)
    actor = load_actor(args.checkpoint, config) if args.checkpoint else None
    if args.condition == "residual" and actor is None:
        parser.error("--checkpoint is required for residual evaluation")
    output = args.artifact_root / "evaluation" / f"{args.condition}.jsonl"
    TaskCatalog.save_manifest(specs, args.artifact_root / "manifests" / "staged_evaluation.json")
    records = evaluate_policy(
        specs,
        base_policy,
        config,
        actor=actor,
        output=output,
        condition_name=args.condition,
    )
    print(json.dumps({"output": str(output), "episodes": len(records)}))


if __name__ == "__main__":
    main()

