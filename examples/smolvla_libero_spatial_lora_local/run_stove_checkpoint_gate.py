#!/usr/bin/env python3
"""修正LoRA checkpointをstove 20 episodeで一次選別する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_phase4_unified_select as runner

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORK = SCRIPT_DIR / "work"
MODEL_ROOT = WORK / "stove_corrective_checkpoint_models"
RESULT_ROOT = REPO_ROOT / "results/stove_corrective_checkpoint_gate"
TASK = "put_the_bowl_on_the_stove_light_11"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="", help="カンマ区切り候補名（baseは自動追加）")
    args = parser.parse_args()
    candidates = {
        "base": WORK / "smolvla_track1_perturbation_replay_merged",
        **{f"ckpt_{step}": MODEL_ROOT / f"ckpt_{step}"
           for step in (500, 1000, 1500, 2000, 2500)},
        **{f"blend_a{tag}": MODEL_ROOT / f"blend_a{tag}"
           for tag in ("01", "02", "03", "05")},
    }
    if args.only:
        selected = {value.strip() for value in args.only.split(",") if value.strip()}
        selected.add("base")
        candidates = {name: path for name, path in candidates.items() if name in selected}
    results = {}
    for name, model in candidates.items():
        out = RESULT_ROOT / name
        runner.deploy_src(model)
        proc = runner.start_server(args.port, out / "server.log")
        try:
            result = runner.run_pipeline(
                port=args.port, output_dir=out, n_episodes=args.n_episodes,
                seed=args.seed, tasks=[TASK],
            )
            metrics = runner.summarize(result)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            runner.stop_port(args.port)
        results[name] = {"model_dir": str(model.resolve()), **metrics}
        print(f"[stove-gate] {name}: success={metrics['success']:.3f} "
              f"collision={metrics['collision']:.3f}", flush=True)
    passed = [name for name, row in results.items() if name != "base"
              and row["success"] >= results["base"]["success"]
              and row["collision"] < results["base"]["collision"]]
    payload = {"protocol": vars(args), "task": TASK, "results": results,
               "passed": passed}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_name = "blend_summary.json" if args.only else "summary.json"
    (RESULT_ROOT / summary_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    runner.deploy_src(candidates["base"])
    print(f"[stove-gate] passed={passed}; restored base", flush=True)


if __name__ == "__main__":
    main()
