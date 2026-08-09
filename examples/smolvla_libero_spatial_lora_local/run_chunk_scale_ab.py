#!/usr/bin/env python3
"""chunk 境界スケール方式の A/B 評価（stage2 + generic）。

比較対象:
  - off    : スケールなし
  - legacy : タスク辞書の先頭 1 step のみ（旧挙動）
  - ramp   : 全タスク共通ランプ（--ramp / --extra-ramps）

使い方:
    /home/cm-pc/.local/smolvla-conda-env/bin/python \\
      examples/smolvla_libero_spatial_lora_local/run_chunk_scale_ab.py \\
      --modes off --extra-ramps '0.85,0.95,1.0;0.9,1.0' \\
      --tag chunk_scale_ab_mild
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

MODES = ("off", "legacy", "ramp")


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


def start_server(
    port: int,
    scale_mode: str,
    log_path: Path,
    *,
    ramp: str = "0.6,0.8,1.0",
) -> subprocess.Popen:
    stop_port(port)
    env = os.environ.copy()
    env["PARC_INSTRUCTION_MODE"] = "generic"
    if scale_mode == "off":
        env["PARC_CHUNK_SCALE"] = "0"
        env["PARC_CHUNK_SCALE_MODE"] = "off"
    else:
        env["PARC_CHUNK_SCALE"] = "1"
        env["PARC_CHUNK_SCALE_MODE"] = scale_mode
        env["PARC_CHUNK_SCALE_RAMP"] = ramp
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
                        f"[ab] server ready scale_mode={scale_mode} "
                        f"ramp={ramp if scale_mode == 'ramp' else '-'} :{port}",
                        flush=True,
                    )
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready; see {log_path}")


def build_jobs(modes: list[str], default_ramp: str, extra_ramps: list[str]) -> list[dict]:
    jobs: list[dict] = []
    for mode in modes:
        if mode == "ramp":
            jobs.append(
                {
                    "label": f"ramp[{default_ramp}]",
                    "scale_mode": "ramp",
                    "ramp": default_ramp,
                }
            )
        else:
            jobs.append({"label": mode, "scale_mode": mode, "ramp": default_ramp})
    for ramp in extra_ramps:
        jobs.append({"label": f"ramp[{ramp}]", "scale_mode": "ramp", "ramp": ramp})
    seen: set[str] = set()
    unique: list[dict] = []
    for job in jobs:
        if job["label"] in seen:
            continue
        seen.add(job["label"])
        unique.append(job)
    return unique


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
    parser.add_argument("--modes", default="off")
    parser.add_argument(
        "--ramp",
        default="0.6,0.8,1.0",
        help="modes に ramp が含まれるときの既定ランプ",
    )
    parser.add_argument(
        "--extra-ramps",
        default="",
        help="追加 milder ramp（セミコロン区切り）。例: '0.85,0.95,1.0;0.9,1.0'",
    )
    parser.add_argument("--tag", default="chunk_scale_ab_mild")
    parser.add_argument("--inset-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    extra_ramps = [r.strip() for r in args.extra_ramps.split(";") if r.strip()]
    jobs = build_jobs(modes, args.ramp, extra_ramps)
    if not jobs:
        raise SystemExit("評価ジョブが空です")

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
    print("[ab] jobs:")
    for job in jobs:
        print(
            f"  - {job['label']} "
            f"(scale_mode={job['scale_mode']}, ramp={job['ramp']})"
        )

    report_dir = REPO_ROOT / "results" / args.tag
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for job in jobs:
        label = job["label"]
        scale_mode = job["scale_mode"]
        ramp = job["ramp"]
        print(f"\n######## {label} ########", flush=True)
        mode_dir = report_dir / safe_dirname(label)
        mode_dir.mkdir(parents=True, exist_ok=True)
        proc = start_server(
            args.port, scale_mode, mode_dir / "server.log", ramp=ramp
        )
        try:
            time.sleep(1)
            log_text = (mode_dir / "server.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if f"chunk_scale_mode={scale_mode}" not in log_text:
                print(
                    f"[ab] WARNING: chunk_scale_mode={scale_mode} not found",
                    flush=True,
                )

            inset_path = run_pipeline(
                port=args.port,
                output_dir=mode_dir / "inset",
                n_episodes=args.inset_episodes,
                seed=args.seed,
            )
            inset = summarize_json(inset_path)
            print(
                f"[ab] {label} inset success={inset['success']:.3f} "
                f"coll={inset['collision']:.3f} jerk={inset['jerk']:.4f}",
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
                "scale_mode": scale_mode,
                "ramp": ramp if scale_mode == "ramp" else None,
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

    non_ramp = [n for n, r in results.items() if r["scale_mode"] != "ramp"]
    baseline_name = max(
        non_ramp or list(results),
        key=lambda name: results[name]["proxy_score"],
    )
    baseline = results[baseline_name]

    ramp_gates: dict[str, dict[str, bool]] = {}
    for label, row in results.items():
        if row["scale_mode"] != "ramp":
            continue
        ramp_gates[label] = {
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
        }

    adopted = [label for label, gates in ramp_gates.items() if all(gates.values())]
    selected = max(results.items(), key=lambda kv: kv[1]["proxy_score"])[0]
    if not adopted:
        selected = baseline_name

    summary = {
        "model": "stage2",
        "instruction_mode": "generic",
        "modes": results,
        "holdout_tasks": holdout,
        "baseline_mode": baseline_name,
        "ramp_gates": ramp_gates,
        "adopted_ramps": adopted,
        "selected_mode": selected,
        "inset_episodes": args.inset_episodes,
        "holdout_episodes": args.holdout_episodes,
        "seed": args.seed,
        "default_ramp": args.ramp,
        "extra_ramps": extra_ramps,
    }
    out = report_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\n=== chunk scale mild A/B (stage2) ===")
    print(f"{'mode':28s} {'inset':>7s} {'hold':>7s} {'coll_h':>7s} {'proxy':>7s}")
    for label, row in results.items():
        print(
            f"{label:28s} {row['inset']['success']:7.3f} "
            f"{row['holdout_avg']['success']:7.3f} "
            f"{row['holdout_avg']['collision']:7.3f} "
            f"{row['proxy_score']:7.3f}"
        )
    print(f"baseline: {baseline_name}")
    print(f"ramp gates: {json.dumps(ramp_gates, ensure_ascii=False)}")
    print(f"adopted ramps: {adopted or '(none)'}")
    print(f"selected: {selected}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
