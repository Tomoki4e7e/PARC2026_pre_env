#!/usr/bin/env python3
"""Select a drawer interpolation only if all six protected series are preserved."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_phase4_unified_select as runner
import run_track1_distribution_audit as audit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORK = SCRIPT_DIR / "work"
RESULT_ROOT = REPO_ROOT / "results/drawer_candidate_gate_v3"
DRAWER_PREFIX = "pick_up_the_black_bowl_in_the_top_drawer"


def candidate_paths(root: Path, only: set[str]) -> dict[str, Path]:
    result = {"stage2": WORK / "smolvla_track1_perturbation_replay_merged"}
    for path in sorted(root.glob("ckpt_*_a*")):
        if not only or path.name in only:
            result[path.name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=WORK / "drawer_checkpoint_models_v3_033443")
    parser.add_argument("--only", default="")
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--protected-tolerance", type=float, default=0.0)
    parser.add_argument("--drawer-only", action="store_true",
                        help="一次screen用: 引き出し系列だけ評価する")
    args = parser.parse_args()
    only = {value.strip() for value in args.only.split(",") if value.strip()}
    candidates = candidate_paths(args.model_root, only)
    missing = [str(path) for path in candidates.values() if not (path / "model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"candidate models missing: {missing}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "tasks.csv"
    rows = audit.build_csv(csv_path, args.per_category, ["table", "light"])
    if args.drawer_only:
        rows = [row for row in rows if row["base_task"].startswith(DRAWER_PREFIX)]
        audit_rows_path = RESULT_ROOT / "drawer_only_tasks.csv"
        import csv
        with audit_rows_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        csv_path = audit_rows_path
    task_ids = [row["task_id"] for row in rows]
    old_proxy = os.environ.get("PARC_PROXY_TASKS_CSV")
    old_residual = os.environ.get("PARC_RESIDUAL_POLICY")
    os.environ["PARC_ENABLE_PROXY_SUITE"] = "1"
    os.environ["PARC_PROXY_TASKS_CSV"] = str(csv_path)
    # Candidate hashes necessarily differ from the residual's training base.
    # Compare the base policy delta fairly; residual is gated separately later.
    os.environ["PARC_RESIDUAL_POLICY"] = "off"
    reports = {}
    try:
        for name, model in candidates.items():
            out = RESULT_ROOT / name
            cached = out / "result.json"
            if cached.is_file():
                raw = json.loads(cached.read_text(encoding="utf-8"))
                reports[name] = audit.summarize(raw, rows)
                print(f"[drawer-gate] resume {name}: {reports[name]['overall']}", flush=True)
                continue
            runner.deploy_src(model)
            proc = runner.start_server(args.port, out / "server.log")
            try:
                result_path = runner.run_pipeline(
                    port=args.port, output_dir=out, n_episodes=args.n_episodes,
                    seed=args.seed, benchmark="libero_proxy", tasks=task_ids,
                )
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                reports[name] = audit.summarize(raw, rows)
            finally:
                proc.terminate(); proc.wait(timeout=20); runner.stop_port(args.port)
            print(f"[drawer-gate] {name}: {reports[name]['overall']}", flush=True)
    finally:
        runner.deploy_src(candidates["stage2"])
        if old_proxy is None:
            os.environ.pop("PARC_PROXY_TASKS_CSV", None)
        else:
            os.environ["PARC_PROXY_TASKS_CSV"] = old_proxy
        if old_residual is None:
            os.environ.pop("PARC_RESIDUAL_POLICY", None)
        else:
            os.environ["PARC_RESIDUAL_POLICY"] = old_residual

    base = reports["stage2"]["by_base_and_category"]
    gates = {}
    for name, report in reports.items():
        if name == "stage2":
            continue
        groups = report["by_base_and_category"]
        drawer_keys = [key for key in base if key.startswith(DRAWER_PREFIX)]
        protected_keys = [key for key in base if key not in drawer_keys]
        gates[name] = {
            "drawer_improved": sum(groups[k]["success"] for k in drawer_keys)
            > sum(base[k]["success"] for k in drawer_keys),
            "all_six_protected_not_worse": all(
                groups[key]["success"] + args.protected_tolerance >= base[key]["success"]
                for key in protected_keys
            ),
            "protected_groups": {
                key: {"base": base[key]["success"], "candidate": groups[key]["success"]}
                for key in protected_keys
            },
        }
    passed = [name for name, gate in gates.items() if gate["drawer_improved"]
              and gate["all_six_protected_not_worse"]]
    selected = max(
        passed,
        key=lambda name: sum(value["success"] for value in
                             reports[name]["by_base_and_category"].values()),
        default="stage2",
    )
    payload = {"protocol": vars(args), "reports": reports, "gates": gates,
               "passed": passed, "selected": selected,
               "decision": "promote" if selected != "stage2" else "keep_stage2"}
    (RESULT_ROOT / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8"
    )
    print(f"[drawer-gate] selected={selected}; stage2 restored")


if __name__ == "__main__":
    main()
