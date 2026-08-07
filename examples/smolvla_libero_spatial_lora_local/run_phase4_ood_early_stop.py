#!/usr/bin/env python3
"""Phase4 checkpoint を OOD-90 固定セットで評価し、early stopping する。

各 checkpoint を merge → deploy → policy server 再起動 → run_ood90_eval し、
overall_score 最大の checkpoint を採用する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config


def list_checkpoints(output_dir: Path) -> list[int]:
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    steps = []
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit():
            adapter = path / "pretrained_model" / "adapter_model.safetensors"
            if adapter.is_file():
                steps.append(int(path.name))
    return sorted(steps)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd or REPO_ROOT, check=True)


def stop_policy_server() -> None:
    subprocess.run(
        ["pkill", "-f", "policy_server.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def start_policy_server(port: int) -> subprocess.Popen:
    import os

    log_path = config.WORK_DIR / "logs" / f"policy_server_phase4_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "bash",
            str(SCRIPT_DIR / "run_policy_server.sh"),
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"policy server exited early (code={process.returncode}). "
                f"See {log_path}"
            )
        try:
            import urllib.request

            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    print(f"[early-stop] server ready on :{port}", flush=True)
                    return process
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"policy server did not become ready: {log_path}")


def evaluate_checkpoint(
    step: int,
    *,
    port: int,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    tag = f"phase4_s2_ckpt_{step:06d}"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            "--phase4",
            "--steps",
            str(step),
        ]
    )
    run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "phase4"])
    stop_policy_server()
    server = start_policy_server(port)
    try:
        # pipeline / LIBERO は parc2026 venv。SmolVLA conda では import できない。
        run(
            [
                "bash",
                "-lc",
                (
                    f"source '{REPO_ROOT}/env.sh' && "
                    f"python '{SCRIPT_DIR / 'run_ood90_eval.py'}' "
                    f"--tag {tag} "
                    f"--server-url http://localhost:{port} "
                    f"--n-episodes {n_episodes} "
                    f"--max-steps {max_steps} "
                    f"--seed {seed}"
                ),
            ]
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        stop_policy_server()

    summary_path = REPO_ROOT / "results" / f"ood90_{tag}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_step"] = step
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase4 OOD early stopping")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoints",
        default="",
        help="評価する step をカンマ区切り指定（空なら全 checkpoint）",
    )
    args = parser.parse_args()

    config.apply_phase4_config()
    if args.checkpoints.strip():
        steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    else:
        steps = list_checkpoints(config.OUTPUT_DIR)
    if not steps:
        raise SystemExit(
            f"checkpoint がありません: {config.OUTPUT_DIR / 'checkpoints'}"
        )

    print(f"[early-stop] checkpoints: {steps}")
    results = []
    for step in steps:
        print(f"\n=== evaluate checkpoint {step:06d} ===", flush=True)
        summary = evaluate_checkpoint(
            step,
            port=args.port,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        results.append(summary)
        print(
            f"[early-stop] step={step:06d} overall={summary['overall_score']:.3f}",
            flush=True,
        )

    best = max(
        results,
        key=lambda item: (item["overall_score"], item["checkpoint_step"]),
    )
    out_dir = REPO_ROOT / "results" / "ood90_phase4_from_stage2_early_stop"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoints": results,
        "best_checkpoint_step": best["checkpoint_step"],
        "best_overall_score": best["overall_score"],
        "n_episodes": args.n_episodes,
        "seed": args.seed,
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 最良 checkpoint を merge + deploy して残す
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            "--phase4",
            "--steps",
            str(best["checkpoint_step"]),
        ]
    )
    run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "phase4"])

    print("\n=== early stopping result ===")
    print(f"best checkpoint: {best['checkpoint_step']:06d}")
    print(f"best OOD overall: {best['overall_score']:.3f}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
