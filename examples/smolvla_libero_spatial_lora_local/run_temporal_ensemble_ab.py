#!/usr/bin/env python3
"""Temporal ensemble（ACT 式）の A/B 評価（stage2 + generic + chunk off）。

Temporal ensemble は推論時手法のため追加学習は行わない。
同一 stage2 重みで off / TE 設定を比較する。

使い方:
    /home/cm-pc/.local/smolvla-conda-env/bin/python \\
      examples/smolvla_libero_spatial_lora_local/run_temporal_ensemble_ab.py \\
      --variants off,te_s5_c0.01,te_s10_c0.01 \\
      --tag temporal_ensemble_ab_stage2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import proxy_holdout
from proxy_score import blend_proxy_scores, score_pipeline_json


def parse_variant(label: str) -> dict:
    """'off' | 'te_s{stride}_c{coeff}' | 'te_s{stride}_c{coeff}_k{chunk}'"""
    label = label.strip()
    if label == "off":
        return {"label": "off", "enabled": False, "stride": 5, "coeff": 0.01, "chunk": None}
    match = re.fullmatch(
        r"te_s(\d+)_c([0-9.]+)(?:_k(\d+))?",
        label,
    )
    if not match:
        raise ValueError(
            f"unknown variant {label!r}; use off or te_s5_c0.01 or te_s5_c0.01_k25"
        )
    return {
        "label": label,
        "enabled": True,
        "stride": int(match.group(1)),
        "coeff": float(match.group(2)),
        "chunk": int(match.group(3)) if match.group(3) else None,
    }


def stop_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for line in out.splitlines():
        if f":{port} " not in line:
            continue
        for part in line.split(","):
            if part.strip().startswith("pid="):
                pid = int(part.strip().split("=")[1].split(")")[0])
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    time.sleep(2)


def start_server(port: int, variant: dict, log_path: Path) -> subprocess.Popen:
    stop_port(port)
    env = os.environ.copy()
    env["PARC_INSTRUCTION_MODE"] = "generic"
    env["PARC_CHUNK_SCALE"] = "0"
    env["PARC_CHUNK_SCALE_MODE"] = "off"
    if variant["enabled"]:
        env["PARC_TEMPORAL_ENSEMBLE"] = "1"
        env["PARC_TE_STRIDE"] = str(variant["stride"])
        env["PARC_TE_COEFF"] = str(variant["coeff"])
        if variant["chunk"] is not None:
            env["PARC_TE_CHUNK"] = str(variant["chunk"])
        else:
            env.pop("PARC_TE_CHUNK", None)
    else:
        env["PARC_TEMPORAL_ENSEMBLE"] = "0"
        env.pop("PARC_TE_STRIDE", None)
        env.pop("PARC_TE_COEFF", None)
        env.pop("PARC_TE_CHUNK", None)
    env.pop("VIRTUAL_ENV", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "run_policy_server.sh"), "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print(
                        f"[ab] server ready variant={variant['label']} :{port}",
                        flush=True,
                    )
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready; see {log_path}")


def safe_dirname(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def run_pipeline(
    *,
    port: int,
    output_dir: Path,
    n_episodes: int,
    seed: int,
    benchmark: str | None = None,
    tasks: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"source '{REPO_ROOT}/env.sh' && "
        f"python -m pipeline "
        f"--server-url 'http://localhost:{port}' "
        f"--track track1 "
        f"--n-episodes {n_episodes} "
        f"--seed {seed} "
        f"--output-dir '{output_dir}'"
    )
    if benchmark:
        cmd += f" --benchmark {benchmark}"
    if tasks:
        quoted = " ".join(f"'{t}'" for t in tasks)
        cmd += f" --tasks {quoted}"
    log_path = output_dir / "eval.log"
    with log_path.open("w") as log:
        subprocess.run(
            ["bash", "-lc", cmd],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    candidates = sorted(output_dir.glob("server_*.json"))
    if not candidates:
        raise RuntimeError(f"no result json in {output_dir}; see {log_path}")
    dest = output_dir / "result.json"
    candidates[0].replace(dest)
    return dest


def summarize_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    track = data["tracks"][0]
    metrics = track.get("overall_metrics", {})
    scored = score_pipeline_json(path)
    return {
        "success": float(scored["success"]),
        "collision": float(scored["collision_rate"]),
        "composite": float(scored["composite"]),
        "jerk": float(metrics.get("mean_avg_cartesian_jerk") or 0.0),
        "sparc": float(metrics.get("mean_sparc") or 0.0),
        "elapsed": float(data.get("total_elapsed_sec") or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--variants",
        default="off,te_s5_c0.01,te_s10_c0.01",
        help="カンマ区切り。例: off,te_s5_c0.01,te_s10_c0.01,te_s5_c0.0",
    )
    parser.add_argument("--tag", default="temporal_ensemble_ab_stage2")
    parser.add_argument("--inset-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    variants = [parse_variant(v) for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise SystemExit("評価バリアントが空です")

    subprocess.run(
        ["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"],
        cwd=REPO_ROOT,
        check=True,
    )

    trained = proxy_holdout.trained_bases_for_stage("track1_stage2")
    holdout, _ = proxy_holdout.select_clean_holdout(trained, per_suite=3)
    print("[ab] clean hold-out:")
    for suite, names in holdout.items():
        for name in names:
            print(f"  - {suite}: {name}")
    print("[ab] variants:")
    for v in variants:
        print(f"  - {v}")

    report_dir = REPO_ROOT / "results" / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for variant in variants:
        label = variant["label"]
        print(f"\n######## {label} ########", flush=True)
        mode_dir = report_dir / safe_dirname(label)
        mode_dir.mkdir(parents=True, exist_ok=True)
        proc = start_server(args.port, variant, mode_dir / "server.log")
        try:
            time.sleep(1)
            log_text = (mode_dir / "server.log").read_text(
                encoding="utf-8", errors="replace"
            )
            expected = f"temporal_ensemble={int(variant['enabled'])}"
            if expected not in log_text:
                print(f"[ab] WARNING: {expected} not found in server log", flush=True)

            inset_path = run_pipeline(
                port=args.port,
                output_dir=mode_dir / "inset",
                n_episodes=args.inset_episodes,
                seed=args.seed,
            )
            inset = summarize_json(inset_path)
            print(
                f"[ab] {label} inset success={inset['success']:.3f} "
                f"coll={inset['collision']:.3f} jerk={inset['jerk']:.4f} "
                f"elapsed={inset['elapsed']:.1f}s",
                flush=True,
            )

            hold_metrics = []
            hold_scores = {}
            for benchmark, tasks in holdout.items():
                suite = benchmark.replace("libero_", "")
                path = run_pipeline(
                    port=args.port,
                    output_dir=mode_dir / f"holdout_{suite}",
                    n_episodes=args.holdout_episodes,
                    seed=args.seed,
                    benchmark=benchmark,
                    tasks=tasks,
                )
                metrics = summarize_json(path)
                hold_scores[suite] = metrics
                hold_metrics.append(metrics)
                print(
                    f"[ab] {label} holdout_{suite} success={metrics['success']:.3f} "
                    f"coll={metrics['collision']:.3f}",
                    flush=True,
                )

            hold_avg = {
                key: float(sum(m[key] for m in hold_metrics) / len(hold_metrics))
                for key in hold_metrics[0]
            }
            blended = blend_proxy_scores(
                {
                    "success": inset["success"],
                    "collision_rate": inset["collision"],
                    "composite": inset["composite"],
                },
                {
                    "success": hold_avg["success"],
                    "collision_rate": hold_avg["collision"],
                    "composite": hold_avg["composite"],
                },
            )
            results[label] = {
                "variant": variant,
                "inset": inset,
                "holdout_by_suite": hold_scores,
                "holdout_avg": hold_avg,
                "proxy_score": blended["proxy_score"],
                "inset_composite": blended["inset_composite"],
                "holdout_composite": blended["holdout_composite"],
            }
            print(
                f"[ab] {label} proxy={blended['proxy_score']:.3f} "
                f"holdout_avg={hold_avg['success']:.3f}",
                flush=True,
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            stop_port(args.port)

    if "off" not in results:
        baseline_name = max(results, key=lambda n: results[n]["proxy_score"])
    else:
        baseline_name = "off"
    baseline = results[baseline_name]

    te_gates: dict[str, dict[str, bool]] = {}
    for label, row in results.items():
        if label == baseline_name:
            continue
        te_gates[label] = {
            "proxy_beats_baseline": row["proxy_score"] > baseline["proxy_score"] + 1e-9,
            "holdout_not_worse": (
                row["holdout_composite"] >= baseline["holdout_composite"] - 1e-9
            ),
            "inset_not_worse": (
                row["inset"]["success"] >= baseline["inset"]["success"] - 1e-9
            ),
            "collision_not_worse": (
                row["holdout_avg"]["collision"]
                <= baseline["holdout_avg"]["collision"] + 1e-9
            ),
            "jerk_not_worse": (
                row["holdout_avg"]["jerk"] <= baseline["holdout_avg"]["jerk"] + 1e-9
            ),
        }

    adopted = [label for label, gates in te_gates.items() if all(gates.values())]
    selected = baseline_name
    if adopted:
        selected = max(adopted, key=lambda n: results[n]["proxy_score"])

    summary = {
        "model": "stage2",
        "instruction_mode": "generic",
        "chunk_scale_mode": "off",
        "note": "temporal ensemble is inference-only; no additional training",
        "variants": results,
        "holdout_tasks": holdout,
        "baseline_variant": baseline_name,
        "te_gates": te_gates,
        "adopted_variants": adopted,
        "selected_variant": selected,
        "inset_episodes": args.inset_episodes,
        "holdout_episodes": args.holdout_episodes,
        "seed": args.seed,
    }
    out = report_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\n=== temporal ensemble A/B (stage2) ===")
    print(
        f"{'variant':22s} {'inset':>7s} {'hold':>7s} {'coll_h':>7s} "
        f"{'jerk_h':>7s} {'proxy':>7s}"
    )
    for label, row in results.items():
        print(
            f"{label:22s} {row['inset']['success']:7.3f} "
            f"{row['holdout_avg']['success']:7.3f} "
            f"{row['holdout_avg']['collision']:7.3f} "
            f"{row['holdout_avg']['jerk']:7.3f} "
            f"{row['proxy_score']:7.3f}"
        )
    print(f"baseline: {baseline_name}")
    print(f"te gates: {json.dumps(te_gates, ensure_ascii=False)}")
    print(f"adopted: {adopted or '(none)'}")
    print(f"selected: {selected}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
