#!/usr/bin/env python3
"""stage2 で instruction 正規化モード (alias/generic/match) を A/B 比較する。

評価セット:
  - inset: example 4（alias 表で既にカバーされる想定 → 差は小さいはず）
  - holdout: clean spatial 3 タスク（instruction 末尾に "table N"/"light N" が付く
    → generic/match のサフィックス除去が効くはず）
  - language: proxy language カテゴリ（既知言い換えは alias 表でカバー済み）

使い方:
    /home/cm-pc/.local/smolvla-conda-env/bin/python \\
      examples/smolvla_libero_spatial_lora_local/run_instruction_mode_ab.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

SMOLVLA_PY = Path.home() / ".local" / "smolvla-conda-env" / "bin" / "python"
PROXY_CSV = REPO_ROOT / "compe" / "proxy" / "PROXY_TASKS.csv"

MODES = ("alias", "generic", "match")


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


def start_server(port: int, mode: str, log_path: Path) -> subprocess.Popen:
    stop_port(port)
    env = os.environ.copy()
    env["PARC_INSTRUCTION_MODE"] = mode
    env.pop("VIRTUAL_ENV", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            "bash",
            str(SCRIPT_DIR / "run_policy_server.sh"),
            "--port",
            str(port),
        ],
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
                    print(f"[ab] server ready mode={mode} :{port}", flush=True)
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready; see {log_path}")


def run_pipeline(
    *,
    port: int,
    output_dir: Path,
    n_episodes: int,
    seed: int,
    benchmark: str | None = None,
    tasks: list[str] | None = None,
    enable_proxy: bool = False,
) -> dict:
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
    env = os.environ.copy()
    if enable_proxy:
        env["PARC_ENABLE_PROXY_SUITE"] = "1"
    log_path = output_dir / "eval.log"
    with log_path.open("w") as log:
        subprocess.run(
            ["bash", "-lc", cmd],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    candidates = sorted(output_dir.glob("server_*.json"))
    if not candidates:
        raise RuntimeError(f"no result json in {output_dir}; see {log_path}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def score_of(result: dict) -> dict[str, float]:
    track = result["tracks"][0]
    metrics = track.get("overall_metrics", {})
    return {
        "success": float(track["overall_score"]),
        "collision": float(metrics.get("mean_collision_rate") or 0.0),
        "elapsed": float(result.get("total_elapsed_sec") or 0.0),
        "n_tasks": float(len(track.get("tasks") or [])),
    }


def language_tasks() -> list[str]:
    with PROXY_CSV.open(newline="", encoding="utf-8") as handle:
        return [
            row["task_id"]
            for row in csv.DictReader(handle)
            if row["category"] == "language"
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--inset-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--language-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-language",
        action="store_true",
        help="language カテゴリ評価を省略",
    )
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # stage2 をデプロイ（提出候補）
    subprocess.run(
        ["bash", str(SCRIPT_DIR / "deploy_model.sh"), "stage2"],
        cwd=REPO_ROOT,
        check=True,
    )

    trained = proxy_holdout.trained_bases_for_stage("track1_stage2")
    holdout, _ = proxy_holdout.select_clean_holdout(trained, per_suite=3)
    spatial_holdout = holdout["libero_spatial"]
    print("[ab] clean spatial hold-out:")
    for name in spatial_holdout:
        print(f"  - {name}")

    report_dir = REPO_ROOT / "results" / "instruction_mode_ab_stage2"
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for mode in modes:
        print(f"\n######## mode={mode} ########", flush=True)
        mode_dir = report_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        proc = start_server(args.port, mode, mode_dir / "server.log")
        try:
            # Confirm mode from server log
            time.sleep(1)
            log_text = (mode_dir / "server.log").read_text(encoding="utf-8", errors="replace")
            if f"instruction_mode={mode}" not in log_text:
                print(
                    f"[ab] WARNING: server log に instruction_mode={mode} が見つかりません",
                    flush=True,
                )

            inset = score_of(
                run_pipeline(
                    port=args.port,
                    output_dir=mode_dir / "inset",
                    n_episodes=args.inset_episodes,
                    seed=args.seed,
                )
            )
            print(
                f"[ab] {mode} inset success={inset['success']:.3f} "
                f"coll={inset['collision']:.3f}",
                flush=True,
            )

            hold = score_of(
                run_pipeline(
                    port=args.port,
                    output_dir=mode_dir / "holdout_spatial",
                    n_episodes=args.holdout_episodes,
                    seed=args.seed,
                    benchmark="libero_spatial",
                    tasks=spatial_holdout,
                )
            )
            print(
                f"[ab] {mode} holdout_spatial success={hold['success']:.3f} "
                f"coll={hold['collision']:.3f}",
                flush=True,
            )

            lang = None
            if not args.skip_language:
                lang = score_of(
                    run_pipeline(
                        port=args.port,
                        output_dir=mode_dir / "language",
                        n_episodes=args.language_episodes,
                        seed=args.seed,
                        benchmark="libero_proxy",
                        tasks=language_tasks(),
                        enable_proxy=True,
                    )
                )
                print(
                    f"[ab] {mode} language success={lang['success']:.3f} "
                    f"coll={lang['collision']:.3f}",
                    flush=True,
                )

            results[mode] = {
                "inset": inset,
                "holdout_spatial": hold,
                "language": lang,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            stop_port(args.port)

    summary = {
        "model": "stage2",
        "modes": results,
        "holdout_spatial_tasks": spatial_holdout,
        "inset_episodes": args.inset_episodes,
        "holdout_episodes": args.holdout_episodes,
        "language_episodes": args.language_episodes,
        "seed": args.seed,
        "notes": {
            "alias_vs_generic": (
                "generic は摂動サフィックス (table/light) を除去する。"
                "hold-out で差が出るのが本命。"
            ),
            "match_risk": (
                "未知 instruction を既知 40 タスクへ強制写像する。"
                "OOD false-accept が高いため本番採用は慎重に。"
            ),
        },
    }
    out = report_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\n=== instruction mode A/B (stage2) ===")
    header = f"{'mode':8s} {'inset':>8s} {'holdout':>8s} {'language':>8s}"
    print(header)
    for mode in modes:
        row = results[mode]
        lang_s = (
            f"{row['language']['success']:.3f}" if row["language"] else "  n/a"
        )
        print(
            f"{mode:8s} {row['inset']['success']:8.3f} "
            f"{row['holdout_spatial']['success']:8.3f} {lang_s:>8s}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    if not SMOLVLA_PY.is_file():
        raise SystemExit(f"SmolVLA python not found: {SMOLVLA_PY}")
    main()
