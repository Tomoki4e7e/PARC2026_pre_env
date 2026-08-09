#!/usr/bin/env python3
"""Phase4/Phase5 checkpoint を example4 + hold-out proxy で選定する。

選定指標:
  proxy_score = 0.4 * inset_composite + 0.6 * holdout_composite
  composite = success - collision_penalty + smoothness terms
            （jerk / SPARC / EEF 回転総量 / 経路長）

hold-out は **その段階の学習タスクと排他**（clean）でなければ選定に使わない。
Phase5 で 5/9 が学習済みタスクだったため暗記モデルを選び、本番 0.000 を出した
事故を踏まえ、既定で clean hold-out を自動構成し、汚染時は停止する。

出荷の最終決定は本スクリプト単体では行わない。
Phase4 系の出荷は `run_phase4_unified_select.py` が唯一の決定点
（proxy / OOD early-stop の二重系統を統合）。
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
import proxy_holdout
from proxy_score import blend_proxy_scores, score_pipeline_json


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


def start_policy_server(port: int, *, log_tag: str) -> subprocess.Popen:
    import os

    log_path = config.WORK_DIR / "logs" / f"policy_server_{log_tag}_{port}.log"
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


def _pipeline_metrics(output_json: Path) -> dict[str, float]:
    return score_pipeline_json(output_json)


def evaluate_suite(
    *,
    port: int,
    benchmark: str,
    tasks: list[str],
    n_episodes: int,
    seed: int,
    output_dir: Path,
    tag: str,
) -> dict[str, float]:
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
    return _pipeline_metrics(renamed)


def evaluate_ood(
    *,
    port: int,
    deploy_variant: str,
    tag: str,
    n_episodes: int,
    seed: int,
) -> float:
    """現在の deploy 対象で OOD-90（未学習 12 タスク）を評価し overall を返す。"""
    run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), deploy_variant])
    stop_policy_server()
    server = start_policy_server(port, log_tag=f"ood_{tag}")
    try:
        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_ood90_eval.py"),
                "--tag",
                tag,
                "--server-url",
                f"http://localhost:{port}",
                "--n-episodes",
                str(n_episodes),
                "--seed",
                str(seed),
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
    return float(json.loads(summary_path.read_text(encoding="utf-8"))["overall_score"])


def evaluate_checkpoint(
    step: int,
    *,
    port: int,
    inset_episodes: int,
    holdout_episodes: int,
    seed: int,
    merge_flag: str,
    deploy_variant: str,
    tag_prefix: str,
    holdout_tasks: dict[str, list[str]],
) -> dict:
    tag = f"{tag_prefix}_ckpt_{step:06d}"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            merge_flag,
            "--steps",
            str(step),
        ]
    )
    run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), deploy_variant])
    stop_policy_server()
    server = start_policy_server(port, log_tag=f"{tag_prefix}_proxy")
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
        inset = _pipeline_metrics(inset_path)

        holdout_scores = {}
        holdout_metrics = []
        holdout_dir = out_root / "holdout"
        for benchmark, tasks in holdout_tasks.items():
            suite = benchmark.replace("libero_", "")
            metrics = evaluate_suite(
                port=port,
                benchmark=benchmark,
                tasks=tasks,
                n_episodes=holdout_episodes,
                seed=seed,
                output_dir=holdout_dir,
                tag=suite,
            )
            holdout_scores[suite] = metrics["success"]
            holdout_metrics.append(metrics)

        holdout_avg_metrics = {
            key: float(sum(m[key] for m in holdout_metrics) / len(holdout_metrics))
            for key in holdout_metrics[0]
        }
        blended = blend_proxy_scores(inset, holdout_avg_metrics)
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
        "inset_score": blended["inset_success"],
        "inset_collision_rate": blended["inset_collision_rate"],
        "inset_composite": blended["inset_composite"],
        "holdout_scores": holdout_scores,
        "holdout_avg": blended["holdout_success"],
        "holdout_collision_rate": blended["holdout_collision_rate"],
        "holdout_composite": blended["holdout_composite"],
        "proxy_score": blended["proxy_score"],
        "inset_episodes": inset_episodes,
        "holdout_episodes": holdout_episodes,
        "seed": seed,
        "inset_detail": inset,
        "holdout_detail": holdout_avg_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proxy-based checkpoint selection (success+collision+smoothness)"
    )
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
    parser.add_argument(
        "--phase5",
        action="store_true",
        help="Phase5 quality LoRA の checkpoint を選定する",
    )
    parser.add_argument(
        "--holdout-mode",
        choices=("auto_clean", "legacy"),
        default="auto_clean",
        help="auto_clean: 学習タスクと排他な hold-out を自動構成（既定）"
        " / legacy: 従来の固定タスク（汚染していれば警告のみ）",
    )
    parser.add_argument(
        "--ood-gate",
        action="store_true",
        help="OOD-90（未学習 12 タスク）で stage2 を下回る候補を採用しない",
    )
    parser.add_argument("--ood-episodes", type=int, default=3)
    args = parser.parse_args()

    if args.phase5:
        config.apply_phase5_config()
        merge_flag = "--phase5"
        deploy_variant = "phase5"
        tag_prefix = "phase5_quality"
        stage_name = "phase5"
        report_dir = REPO_ROOT / "results" / "phase5_quality_proxy_select"
    else:
        config.apply_phase4_config()
        merge_flag = "--phase4"
        deploy_variant = "phase4_v2"
        tag_prefix = "phase4_v2"
        stage_name = "phase4"
        report_dir = REPO_ROOT / "results" / "phase4_v2_proxy_select"

    trained_bases = proxy_holdout.trained_bases_for_stage(stage_name)
    print(f"[proxy-select] {stage_name} 学習ベースタスク: {len(trained_bases)} 件")
    if args.holdout_mode == "auto_clean":
        holdout_tasks, holdout_report = proxy_holdout.select_clean_holdout(
            trained_bases, per_suite=3
        )
        print("[proxy-select] clean hold-out（学習タスクと排他）:")
        for benchmark, names in holdout_tasks.items():
            print(f"  {benchmark}: {len(names)} 件")
            for name in names:
                print(f"    - {name}")
    else:
        holdout_tasks = proxy_holdout.LEGACY_HOLDOUT_TASKS
        holdout_report = {"mode": "legacy"}
        contaminated = proxy_holdout.contamination_report(holdout_tasks, trained_bases)
        if contaminated:
            total = sum(len(v) for v in contaminated.values())
            print(
                f"[proxy-select] 警告: legacy hold-out の {total} タスクが学習済みです。"
                " 汎化指標として信頼できません:"
            )
            for benchmark, names in contaminated.items():
                for name in names:
                    print(f"    - {benchmark}: {name}")

    if args.checkpoints.strip():
        steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    else:
        steps = list_checkpoints(config.OUTPUT_DIR)
    if not steps:
        raise SystemExit(f"checkpoint がありません: {config.OUTPUT_DIR / 'checkpoints'}")

    print(f"[proxy-select] variant={deploy_variant} checkpoints: {steps}")
    results = []
    for step in steps:
        print(f"\n=== evaluate checkpoint {step:06d} ===", flush=True)
        summary = evaluate_checkpoint(
            step,
            port=args.port,
            inset_episodes=args.inset_episodes,
            holdout_episodes=args.holdout_episodes,
            seed=args.seed,
            merge_flag=merge_flag,
            deploy_variant=deploy_variant,
            tag_prefix=tag_prefix,
            holdout_tasks=holdout_tasks,
        )
        results.append(summary)
        print(
            f"[proxy-select] step={step:06d} "
            f"inset_s={summary['inset_score']:.3f} "
            f"holdout_s={summary['holdout_avg']:.3f} "
            f"coll_i={summary['inset_collision_rate']:.3f} "
            f"proxy={summary['proxy_score']:.3f}",
            flush=True,
        )

    best = max(
        results,
        key=lambda item: (
            item["proxy_score"],
            item["inset_composite"],
            item["checkpoint_step"],
        ),
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    stage2_baseline = None
    if args.stage2_baseline:
        print("\n=== stage2 baseline ===", flush=True)
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"])
        stop_policy_server()
        server = start_policy_server(args.port, log_tag="stage2_baseline")
        try:
            inset_dir = report_dir / "stage2_inset"
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
            inset = _pipeline_metrics(inset_path)
            holdout_metrics = []
            holdout_scores = {}
            for benchmark, tasks in holdout_tasks.items():
                suite = benchmark.replace("libero_", "")
                metrics = evaluate_suite(
                    port=args.port,
                    benchmark=benchmark,
                    tasks=tasks,
                    n_episodes=args.holdout_episodes,
                    seed=args.seed,
                    output_dir=report_dir / "stage2_holdout",
                    tag=suite,
                )
                holdout_scores[suite] = metrics["success"]
                holdout_metrics.append(metrics)
            holdout_avg_metrics = {
                key: float(sum(m[key] for m in holdout_metrics) / len(holdout_metrics))
                for key in holdout_metrics[0]
            }
            blended = blend_proxy_scores(inset, holdout_avg_metrics)
            stage2_baseline = {
                "inset_score": blended["inset_success"],
                "holdout_scores": holdout_scores,
                "holdout_avg": blended["holdout_success"],
                "proxy_score": blended["proxy_score"],
                "inset_composite": blended["inset_composite"],
                "holdout_composite": blended["holdout_composite"],
            }
            print(
                f"[proxy-select] stage2 inset={blended['inset_success']:.3f} "
                f"holdout={blended['holdout_success']:.3f} "
                f"proxy={blended['proxy_score']:.3f}",
                flush=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
            stop_policy_server()

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "merge_lora.py"),
            merge_flag,
            "--steps",
            str(best["checkpoint_step"]),
        ]
    )

    ood_scores: dict[str, float] = {}
    if args.ood_gate:
        print("\n=== OOD-90 gate（未学習 12 タスク）===", flush=True)
        ood_scores["candidate"] = evaluate_ood(
            port=args.port,
            deploy_variant=deploy_variant,
            tag=f"{tag_prefix}_gate_{best['checkpoint_step']:06d}",
            n_episodes=args.ood_episodes,
            seed=args.seed,
        )
        ood_scores["stage2"] = evaluate_ood(
            port=args.port,
            deploy_variant="stage2",
            tag="stage2_gate",
            n_episodes=args.ood_episodes,
            seed=args.seed,
        )
        print(
            f"[proxy-select] OOD-90 candidate={ood_scores['candidate']:.3f} "
            f"stage2={ood_scores['stage2']:.3f}",
            flush=True,
        )

    # ゲート: 1 つでも落ちたら候補を採用せず stage2 を維持する
    gates: dict[str, bool] = {}
    if stage2_baseline is not None:
        gates["proxy_beats_stage2"] = (
            best["proxy_score"] > stage2_baseline["proxy_score"]
        )
        gates["clean_holdout_not_worse"] = (
            best["holdout_composite"] >= stage2_baseline["holdout_composite"]
        )
    if ood_scores:
        gates["ood_not_worse"] = ood_scores["candidate"] >= ood_scores["stage2"]

    failed = [name for name, ok in gates.items() if not ok]
    if failed:
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"])
        selected = "stage2"
        print(f"[proxy-select] ゲート不通過 {failed} → stage2 を維持", flush=True)
    else:
        run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), deploy_variant])
        selected = f"{deploy_variant}@{best['checkpoint_step']:06d}"

    report = {
        "checkpoints": results,
        "best_checkpoint_step": best["checkpoint_step"],
        "best_proxy_score": best["proxy_score"],
        "best_inset_score": best["inset_score"],
        "best_holdout_avg": best["holdout_avg"],
        "scoring": "composite(success,collision,jerk,sparc,orientation,path)",
        "holdout_mode": args.holdout_mode,
        "holdout_tasks": holdout_tasks,
        "holdout_report": holdout_report,
        "trained_base_task_count": len(trained_bases),
        "stage2_baseline": stage2_baseline,
        "ood_scores": ood_scores or None,
        "gates": gates,
        "failed_gates": failed,
        "selected_deploy": selected,
        "inset_episodes": args.inset_episodes,
        "holdout_episodes": args.holdout_episodes,
        "seed": args.seed,
    }
    report_path = report_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n=== proxy selection result ===")
    print(f"best checkpoint: {best['checkpoint_step']:06d}")
    print(f"best proxy: {best['proxy_score']:.3f}")
    print(f"gates: {gates or '(baseline 未評価のためゲートなし)'}")
    print(f"deployed: {selected}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
