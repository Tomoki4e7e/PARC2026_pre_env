#!/usr/bin/env python3
"""画像ベース bowl-on-stove 衝突回避シールドの同一条件 A/B 評価。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import run_phase4_unified_select as runner

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/track1_collision_shield_ab"
TASK = "put_the_bowl_on_the_stove_light_11"


def start_server(port: int, gain: float, log_path: Path) -> subprocess.Popen:
    runner.stop_port(port)
    env = os.environ.copy()
    env.update({
        "PARC_INSTRUCTION_MODE": "generic",
        "PARC_CHUNK_SCALE": "0",
        "PARC_CHUNK_SCALE_MODE": "off",
        "PARC_TEMPORAL_ENSEMBLE": "1",
        "PARC_COLLISION_SHIELD": "1" if gain > 0 else "0",
        "PARC_SHIELD_GAIN": str(gain),
        "PARC_SHIELD_START": "18",
        "PARC_SHIELD_END": "40",
        "PARC_SHIELD_DEBUG": os.environ.get("PARC_SHIELD_DEBUG", "0"),
    })
    env.pop("VIRTUAL_ENV", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "run_policy_server.sh"), "--port", str(port)],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gains", default="0,0.05,0.10,0.15")
    parser.add_argument("--tag", default=DEFAULT_RESULT_ROOT.name)
    args = parser.parse_args()
    result_root = REPO_ROOT / "results" / args.tag
    gains = [float(value) for value in args.gains.split(",")]
    results = {}
    for gain in gains:
        name = "off" if gain == 0 else f"gain_{gain:.2f}"
        out = result_root / name
        proc = start_server(args.port, gain, out / "server.log")
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
            except subprocess.TimeoutExpired:
                proc.kill()
            runner.stop_port(args.port)
        results[name] = {"gain": gain, **metrics}
        print(f"[shield-ab] {name}: success={metrics['success']:.3f} collision={metrics['collision']:.3f}", flush=True)

    base = results["off"]
    passed = [
        name for name, row in results.items()
        if name != "off" and row["success"] >= base["success"] and row["collision"] < base["collision"]
    ]
    payload = {"protocol": vars(args), "task": TASK, "results": results, "passed": passed}
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[shield-ab] passed={passed}", flush=True)


if __name__ == "__main__":
    main()
