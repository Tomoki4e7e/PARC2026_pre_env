#!/usr/bin/env python3
"""Phase4 系候補を一本の選定プロトコルで横並びし、出荷先を明示固定する。

背景:
  run_phase4_proxy_select.py（→ phase4_v2）と run_phase4_ood_early_stop.py（→ phase4）
  が別指標・別 variant で走り、deploy_model.sh の silent fallback が意図しない重みを
  出荷し得た。本スクリプトが唯一の出荷決定点になる。

プロトコル（現行推論設定: instruction=generic, chunk_scale=off）:
  1. inset (example4)
  2. clean hold-out（候補の学習タスクと排他。Phase4 は suite 全学習のため N/A）
  3. OOD-90（未学習 12 タスク）
  4. stage2 を baseline。ゲートを1つでも落とせば stage2 を維持

使い方:
    /home/cm-pc/.local/smolvla-conda-env/bin/python \\
      examples/smolvla_libero_spatial_lora_local/run_phase4_unified_select.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
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

WORK = config.WORK_DIR
SELECTION_DIR = REPO_ROOT / "results" / "phase4_unified_select"
SELECTION_JSON = SELECTION_DIR / "selection.json"
SELECTED_MERGED = WORK / "smolvla_libero_phase4_selected_merged"

# 評価対象の merged モデル（再学習なしで横並び）
CANDIDATES: dict[str, Path] = {
    "stage2": WORK / "smolvla_libero_plus_track1_stage2_merged",
    "phase4_v2": WORK / "smolvla_libero_diverse_phase4_v2_merged",
    "phase4_from_stage2": WORK / "smolvla_libero_diverse_phase4_from_stage2_merged",
    "phase4_legacy": WORK / "smolvla_libero_diverse_phase4_merged",
}

# clean hold-out を試みるときの学習タスク集合（汚染判定用）
TRAINED_STAGE_HINT: dict[str, str] = {
    "stage2": "track1_stage2",
    "phase4_v2": "phase4",
    "phase4_from_stage2": "phase4",
    "phase4_legacy": "phase4",
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


def start_server(port: int, log_path: Path) -> subprocess.Popen:
    stop_port(port)
    env = os.environ.copy()
    env["PARC_INSTRUCTION_MODE"] = "generic"
    env["PARC_CHUNK_SCALE_MODE"] = "off"
    env["PARC_CHUNK_SCALE"] = "0"
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
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print(f"[unified] server ready :{port}", flush=True)
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready; see {log_path}")


def deploy_src(src: Path) -> None:
    dest = REPO_ROOT / "submission_template" / "model_weights" / "smolvla_spatial_lora"
    dest.mkdir(parents=True, exist_ok=True)
    find_cmd = ["find", str(dest), "-mindepth", "1", "-delete"]
    subprocess.run(find_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        [
            "rsync",
            "-aL",
            "--delete",
            "--exclude",
            "train_config.json",
            "--exclude",
            "interpolation.json",
            f"{src}/",
            f"{dest}/",
        ],
        check=True,
    )
    print(f"[unified] deployed {src.name} → submission_template", flush=True)


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


def summarize(path: Path) -> dict[str, float]:
    scored = score_pipeline_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "success": float(scored["success"]),
        "collision": float(scored["collision_rate"]),
        "composite": float(scored["composite"]),
        "elapsed": float(data.get("total_elapsed_sec") or 0.0),
    }


def run_ood(port: int, tag: str, n_episodes: int, seed: int) -> float:
    subprocess.run(
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
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    summary = REPO_ROOT / "results" / f"ood90_{tag}" / "summary.json"
    return float(json.loads(summary.read_text(encoding="utf-8"))["overall_score"])


def try_clean_holdout(candidate: str) -> dict[str, list[str]] | None:
    stage = TRAINED_STAGE_HINT.get(candidate, "phase4")
    trained = proxy_holdout.trained_bases_for_stage(stage)
    try:
        holdout, _report = proxy_holdout.select_clean_holdout(trained, per_suite=3)
        return holdout
    except RuntimeError as exc:
        print(f"[unified] {candidate}: clean hold-out N/A ({exc})", flush=True)
        return None


def evaluate_candidate(
    name: str,
    src: Path,
    *,
    port: int,
    inset_episodes: int,
    holdout_episodes: int,
    ood_episodes: int,
    seed: int,
) -> dict:
    out_root = SELECTION_DIR / name
    out_root.mkdir(parents=True, exist_ok=True)
    deploy_src(src)
    proc = start_server(port, out_root / "server.log")
    try:
        inset = summarize(
            run_pipeline(
                port=port,
                output_dir=out_root / "inset",
                n_episodes=inset_episodes,
                seed=seed,
            )
        )
        print(
            f"[unified] {name} inset={inset['success']:.3f} coll={inset['collision']:.3f}",
            flush=True,
        )

        holdout_tasks = try_clean_holdout(name)
        holdout_avg = None
        holdout_by_suite = None
        if holdout_tasks is not None:
            hold_metrics = []
            holdout_by_suite = {}
            for benchmark, tasks in holdout_tasks.items():
                suite = benchmark.replace("libero_", "")
                metrics = summarize(
                    run_pipeline(
                        port=port,
                        output_dir=out_root / f"holdout_{suite}",
                        n_episodes=holdout_episodes,
                        seed=seed,
                        benchmark=benchmark,
                        tasks=tasks,
                    )
                )
                holdout_by_suite[suite] = metrics
                hold_metrics.append(metrics)
                print(
                    f"[unified] {name} holdout_{suite}={metrics['success']:.3f}",
                    flush=True,
                )
            holdout_avg = {
                key: float(sum(m[key] for m in hold_metrics) / len(hold_metrics))
                for key in hold_metrics[0]
            }

        if ood_episodes > 0:
            ood = run_ood(port, f"unified_{name}", ood_episodes, seed)
            print(f"[unified] {name} ood90={ood:.3f}", flush=True)
        else:
            ood = None
            print(f"[unified] {name} ood90=SKIP", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        stop_port(port)

    if holdout_avg is not None:
        blended = blend_proxy_scores(
            {
                "success": inset["success"],
                "collision_rate": inset["collision"],
                "composite": inset["composite"],
            },
            {
                "success": holdout_avg["success"],
                "collision_rate": holdout_avg["collision"],
                "composite": holdout_avg["composite"],
            },
        )
        proxy = blended["proxy_score"]
        holdout_composite = blended["holdout_composite"]
    else:
        # suite hold-out が無い候補は inset composite を proxy の代替にする
        proxy = inset["composite"]
        holdout_composite = None

    return {
        "name": name,
        "model_dir": str(src),
        "inset": inset,
        "holdout_available": holdout_avg is not None,
        "holdout_tasks": holdout_tasks,
        "holdout_by_suite": holdout_by_suite,
        "holdout_avg": holdout_avg,
        "holdout_composite": holdout_composite,
        "ood90": ood,
        "proxy_score": proxy,
    }


def decide(results: dict[str, dict]) -> tuple[str, dict[str, dict[str, bool]]]:
    baseline = results["stage2"]
    gates: dict[str, dict[str, bool]] = {}
    for name, row in results.items():
        if name == "stage2":
            continue
        g = {
            "inset_not_worse": row["inset"]["success"]
            >= baseline["inset"]["success"] - 1e-9,
            "proxy_beats_stage2": row["proxy_score"] > baseline["proxy_score"] + 1e-9,
        }
        if row["ood90"] is not None and baseline["ood90"] is not None:
            g["ood_not_worse"] = row["ood90"] >= baseline["ood90"] - 1e-9
        if row["holdout_available"] and baseline["holdout_available"]:
            g["clean_holdout_not_worse"] = (
                float(row["holdout_composite"])
                >= float(baseline["holdout_composite"]) - 1e-9
            )
        gates[name] = g

    winners = [name for name, g in gates.items() if all(g.values())]
    if not winners:
        return "stage2", gates
    # 複数通過なら proxy 最大
    best = max(winners, key=lambda name: results[name]["proxy_score"])
    return best, gates


def write_selected(name: str, src: Path) -> None:
    if SELECTED_MERGED.exists() or SELECTED_MERGED.is_symlink():
        if SELECTED_MERGED.is_dir() and not SELECTED_MERGED.is_symlink():
            shutil.rmtree(SELECTED_MERGED)
        else:
            SELECTED_MERGED.unlink()
    # コピーではなく参照を残す（容量節約）。deploy は selection.json の path を正とする。
    try:
        SELECTED_MERGED.symlink_to(src.resolve())
    except OSError:
        shutil.copytree(src, SELECTED_MERGED)
    print(f"[unified] selected → {SELECTED_MERGED} ({name})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase4 統一選定")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATES),
        help="カンマ区切り候補名",
    )
    parser.add_argument("--inset-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--ood-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    if "stage2" not in names:
        names.insert(0, "stage2")

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name in names:
        src = CANDIDATES.get(name)
        if src is None:
            raise SystemExit(f"unknown candidate: {name}")
        if not (src / "model.safetensors").is_file():
            print(f"[unified] SKIP {name}: missing {src}", flush=True)
            continue
        print(f"\n======== evaluate {name} ========", flush=True)
        results[name] = evaluate_candidate(
            name,
            src,
            port=args.port,
            inset_episodes=args.inset_episodes,
            holdout_episodes=args.holdout_episodes,
            ood_episodes=args.ood_episodes,
            seed=args.seed,
        )

    if "stage2" not in results:
        raise SystemExit("stage2 baseline が評価できませんでした")

    selected, gates = decide(results)
    selected_src = Path(results[selected]["model_dir"])
    write_selected(selected, selected_src)

    # 最終提出テンプレートにも選定結果を反映
    deploy_src(selected_src)

    payload = {
        "protocol": {
            "instruction_mode": "generic",
            "chunk_scale_mode": "off",
            "inset_episodes": args.inset_episodes,
            "holdout_episodes": args.holdout_episodes,
            "ood_episodes": args.ood_episodes,
            "seed": args.seed,
            "note": (
                "Phase4 は libero_plus 内 clean hold-out が成立しない。"
                "その場合は inset + OOD-90 + stage2 ゲートで決める。"
            ),
        },
        "candidates": results,
        "gates": gates,
        "selected": selected,
        "selected_model_dir": str(selected_src.resolve()),
        "selected_symlink": str(SELECTED_MERGED),
        "ship_decision": (
            "stage2 を維持"
            if selected == "stage2"
            else f"{selected} を出荷（全ゲート通過）"
        ),
    }
    SELECTION_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (SELECTION_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n=== Phase4 unified selection ===")
    print(f"{'name':22s} {'inset':>7s} {'hold':>7s} {'ood':>7s} {'proxy':>7s}")
    for name, row in results.items():
        hold = (
            f"{row['holdout_avg']['success']:.3f}"
            if row["holdout_available"]
            else "  N/A"
        )
        ood = f"{row['ood90']:.3f}" if row["ood90"] is not None else "N/A"
        print(
            f"{name:22s} {row['inset']['success']:7.3f} {hold:>7s} "
            f"{ood:>7s} {row['proxy_score']:7.3f}"
        )
    print(f"gates: {json.dumps(gates, ensure_ascii=False)}")
    print(f"selected: {selected}")
    print(f"selection: {SELECTION_JSON}")


if __name__ == "__main__":
    main()
