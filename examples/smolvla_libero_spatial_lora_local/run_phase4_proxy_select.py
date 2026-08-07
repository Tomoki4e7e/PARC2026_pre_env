#!/usr/bin/env python3
"""Phase4 checkpoint を example4 + hold-out proxy で選定する。

OOD-90 は補助指標。提出選定は
  proxy_score = 0.4 * inset + 0.6 * holdout_avg
を最大にする checkpoint を採用する。
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

HOLDOUT_TASKS = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_2",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_5",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_11",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1",
        "pick_up_the_ketchup_and_place_it_in_the_basket_light_2",
        "pick_up_the_butter_and_place_it_in_the_basket_table_5",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet_table_1",
        "turn_on_the_stove_light_2",
        "put_the_bowl_on_the_plate_table_3",
    ],
}


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

    log_path = config.WORK_DIR / "logs" / f"policy_server_phase4_proxy_{port}.log"
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
                    print(f"[proxy-select] server ready on :{port}", flush=True)
                    return process
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"policy server did not become ready: {log_path}")


def _pipeline_score(output_json: Path) -> float:
    result = json.loads(output_json.read_text(encoding="utf-8"))
    return float(result["tracks"][0]["overall_score"])


def evaluate_suite(
    *,
    port: int,
    benchmark: str,
    tasks: list[str],
    n_episodes: int,
    seed: int,
    output_dir: Path,
    tag: str,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_args = " ".join(f"'{name}'" for name in tasks)
    run(
        [
            "bash",
            "-lc",
            (
                f"source '{REPO_ROOT}/env.sh' && "
                f"python -m pipeline "
                f"--server-url 'http://localhost:{port}' "
                f"--track track1 "
                f"--benchmark '{benchmark}' "
                f"--n-episodes {n_episodes} "
                f"--seed {seed} "
                f"--output-dir '{output_dir}' "
                f"--tasks {task_args}"
            ),
        ]
    )
    candidates = sorted(output_dir.glob("server_*.json"))
    if not candidates:
        raise RuntimeError(f"評価結果がありません: {output_dir}")
    renamed = output_dir / f"{tag}.json"
    candidates[0].replace(renamed)
    return _pipeline_score(renamed)


def evaluate_checkpoint(
    step: int,
    *,
    port: int,
    inset_episodes: int,
    holdout_episodes: int,
    seed: int,
) -> dict:
    tag = f"phase4_v2_ckpt_{step:06d}"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            "--phase4",
            "--steps",
            str(step),
        ]
    )
    run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "phase4_v2"])
    stop_policy_server()
    server = start_policy_server(port)
    try:
        out_root = REPO_ROOT / "results" / f"proxy_select_{tag}"
        inset_dir = out_root / "inset"
        inset_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "bash",
                "-lc",
                (
                    f"source '{REPO_ROOT}/env.sh' && "
                    f"python -m pipeline "
                    f"--server-url 'http://localhost:{port}' "
                    f"--track track1 "
                    f"--n-episodes {inset_episodes} "
                    f"--seed {seed} "
                    f"--output-dir '{inset_dir}'"
                ),
            ]
        )
        inset_json = sorted(inset_dir.glob("server_*.json"))[0]
        inset_path = inset_dir / "example4.json"
        inset_json.replace(inset_path)
        inset = _pipeline_score(inset_path)

        holdout_scores = {}
        holdout_dir = out_root / "holdout"
        for benchmark, tasks in HOLDOUT_TASKS.items():
            suite = benchmark.replace("libero_", "")
            holdout_scores[suite] = evaluate_suite(
                port=port,
                benchmark=benchmark,
                tasks=tasks,
                n_episodes=holdout_episodes,
                seed=seed,
                output_dir=holdout_dir,
                tag=suite,
            )
        holdout_avg = sum(holdout_scores.values()) / len(holdout_scores)
        proxy = 0.4 * inset + 0.6 * holdout_avg
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        stop_policy_server()

    return {
        "checkpoint_step": step,
        "tag": tag,
        "inset_score": inset,
        "holdout_scores": holdout_scores,
        "holdout_avg": holdout_avg,
        "proxy_score": proxy,
        "inset_episodes": inset_episodes,
        "holdout_episodes": holdout_episodes,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase4 proxy-based checkpoint selection")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--inset-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoints",
        default="",
        help="評価する step をカンマ区切り（空なら全 checkpoint）",
    )
    parser.add_argument(
        "--stage2-baseline",
        action="store_true",
        help="選定後に stage2 を同条件で評価して比較する",
    )
    args = parser.parse_args()

    config.apply_phase4_config()
    if args.checkpoints.strip():
        steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    else:
        steps = list_checkpoints(config.OUTPUT_DIR)
    if not steps:
        raise SystemExit(f"checkpoint がありません: {config.OUTPUT_DIR / 'checkpoints'}")

    print(f"[proxy-select] checkpoints: {steps}")
    results = []
    for step in steps:
        print(f"\n=== evaluate checkpoint {step:06d} ===", flush=True)
        summary = evaluate_checkpoint(
            step,
            port=args.port,
            inset_episodes=args.inset_episodes,
            holdout_episodes=args.holdout_episodes,
            seed=args.seed,
        )
        results.append(summary)
        print(
            f"[proxy-select] step={step:06d} "
            f"inset={summary['inset_score']:.3f} "
            f"holdout={summary['holdout_avg']:.3f} "
            f"proxy={summary['proxy_score']:.3f}",
            flush=True,
        )

    best = max(
        results,
        key=lambda item: (item["proxy_score"], item["inset_score"], item["checkpoint_step"]),
    )
    out_dir = REPO_ROOT / "results" / "phase4_v2_proxy_select"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_baseline = None
    if args.stage2_baseline:
        print("\n=== stage2 baseline ===", flush=True)
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"])
        stop_policy_server()
        server = start_policy_server(args.port)
        try:
            # reuse evaluate path by temporary deploy already done
            inset_dir = out_dir / "stage2_inset"
            inset_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "bash",
                    "-lc",
                    (
                        f"source '{REPO_ROOT}/env.sh' && "
                        f"python -m pipeline "
                        f"--server-url 'http://localhost:{args.port}' "
                        f"--track track1 "
                        f"--n-episodes {args.inset_episodes} "
                        f"--seed {args.seed} "
                        f"--output-dir '{inset_dir}'"
                    ),
                ]
            )
            inset_json = sorted(inset_dir.glob("server_*.json"))[0]
            inset_path = inset_dir / "example4.json"
            inset_json.replace(inset_path)
            inset = _pipeline_score(inset_path)
            holdout_scores = {}
            for benchmark, tasks in HOLDOUT_TASKS.items():
                suite = benchmark.replace("libero_", "")
                holdout_scores[suite] = evaluate_suite(
                    port=args.port,
                    benchmark=benchmark,
                    tasks=tasks,
                    n_episodes=args.holdout_episodes,
                    seed=args.seed,
                    output_dir=out_dir / "stage2_holdout",
                    tag=suite,
                )
            holdout_avg = sum(holdout_scores.values()) / len(holdout_scores)
            stage2_baseline = {
                "inset_score": inset,
                "holdout_scores": holdout_scores,
                "holdout_avg": holdout_avg,
                "proxy_score": 0.4 * inset + 0.6 * holdout_avg,
            }
            print(
                f"[proxy-select] stage2 inset={inset:.3f} "
                f"holdout={holdout_avg:.3f} proxy={stage2_baseline['proxy_score']:.3f}",
                flush=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
            stop_policy_server()

    # 最良を deploy（stage2 より悪ければ stage2 を推奨表示）
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            "--phase4",
            "--steps",
            str(best["checkpoint_step"]),
        ]
    )
    if stage2_baseline is not None and stage2_baseline["proxy_score"] > best["proxy_score"]:
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"])
        selected = "stage2"
    else:
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "phase4_v2"])
        selected = f"phase4_v2@{best['checkpoint_step']:06d}"

    report = {
        "checkpoints": results,
        "best_checkpoint_step": best["checkpoint_step"],
        "best_proxy_score": best["proxy_score"],
        "best_inset_score": best["inset_score"],
        "best_holdout_avg": best["holdout_avg"],
        "stage2_baseline": stage2_baseline,
        "selected_deploy": selected,
        "inset_episodes": args.inset_episodes,
        "holdout_episodes": args.holdout_episodes,
        "seed": args.seed,
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n=== proxy selection result ===")
    print(f"best checkpoint: {best['checkpoint_step']:06d}")
    print(f"best proxy: {best['proxy_score']:.3f}")
    print(f"deployed: {selected}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
