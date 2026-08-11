#!/usr/bin/env python3
"""Track 1 の既知4タスクを多数の table/light 変種で監査する。

既存の PROXY_TASKS.csv は変更せず、実験専用CSVを生成して libero_proxy に
読み込ませる。結果は成功、衝突を伴う失敗、非衝突の未完了に分解する。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus" / "libero" / "libero" / "bddl_files"
RESULT_ROOT = REPO_ROOT / "results" / "track1_distribution_audit"

BASE_TASKS = {
    "libero_spatial": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "libero_object": [
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
    ],
    "libero_goal": "put_the_bowl_on_the_stove",
}
PATTERNS = {
    "table": re.compile(r"_table_\d+$"),
    "light": re.compile(r"_light_\d+$"),
}
MANDATORY_VARIANTS = {
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_2",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket_table_27",
    "pick_up_the_milk_and_place_it_in_the_basket_light_15",
    "put_the_bowl_on_the_stove_light_11",
}
LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")


def _bases():
    for suite, value in BASE_TASKS.items():
        for base in value if isinstance(value, list) else [value]:
            yield suite, base


def _evenly(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def build_csv(path: Path, per_category: int, categories: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for suite, base in _bases():
        for category in categories:
            candidates = [
                p for p in sorted((BDDL_ROOT / suite).glob(f"{base}*.bddl"))
                if " copy" not in p.stem and PATTERNS[category].search(p.stem)
            ]
            selected_paths = _evenly([str(p) for p in candidates], per_category)
            selected_stems = {Path(value).stem for value in selected_paths}
            selected_paths.extend(
                str(path) for path in candidates
                if path.stem in MANDATORY_VARIANTS and path.stem not in selected_stems
            )
            for bddl in selected_paths:
                bddl_path = Path(bddl)
                match = LANGUAGE_RE.search(bddl_path.read_text(encoding="utf-8"))
                if not match:
                    raise RuntimeError(f"language not found: {bddl_path}")
                rows.append({
                    "task_num": str(len(rows) + 1),
                    "task_id": bddl_path.stem,
                    "instruction": " ".join(match.group(1).split()),
                    "suite": suite,
                    "category": category,
                    "base_task": base,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def start_server(port: int, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "run_policy_server.sh"), "--port", str(port)],
        cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return proc
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"policy server did not become ready; see {log_path}")


def summarize(result: dict, rows: list[dict[str, str]]) -> dict:
    metadata = {row["task_id"]: row for row in rows}
    groups: dict[str, list[dict[str, float]]] = {}
    task_details = []
    for task in result["tracks"][0]["tasks"]:
        meta = metadata[task["task_name"]]
        success = float(task["success_rate"])
        collision = float(task.get("metrics", {}).get("collision_rate", 0.0))
        # success と collision は排他的。残りが非衝突の未完了。
        noncompletion = max(0.0, 1.0 - success - collision)
        item = {"success": success, "collision_failure": collision,
                "noncompletion_failure": noncompletion}
        key = f'{meta["base_task"]}:{meta["category"]}'
        groups.setdefault(key, []).append(item)
        task_details.append({**meta, **item, "metrics": task.get("metrics", {})})

    def mean(items, name):
        return sum(item[name] for item in items) / len(items)

    aggregate = {
        key: {name: mean(items, name) for name in
              ("success", "collision_failure", "noncompletion_failure")}
        for key, items in groups.items()
    }
    all_items = [item for items in groups.values() for item in items]
    return {
        "n_tasks": len(task_details),
        "n_episodes_per_task": result["tracks"][0]["overall_metrics"]["n_total_episodes"] / len(task_details),
        "overall": {name: mean(all_items, name) for name in
                    ("success", "collision_failure", "noncompletion_failure")},
        "by_base_and_category": aggregate,
        "tasks": task_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--categories", default="table,light")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args()
    categories = [value.strip() for value in args.categories.split(",") if value.strip()]
    unknown = set(categories) - set(PATTERNS)
    if unknown:
        parser.error(f"unknown categories: {sorted(unknown)}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "tasks.csv"
    rows = build_csv(csv_path, args.per_category, categories)
    if not args.no_deploy:
        subprocess.run(["bash", str(SCRIPT_DIR / "deploy_model.sh"), "augmentation_selected"],
                       cwd=REPO_ROOT, check=True)
    proc = start_server(args.port, RESULT_ROOT / "server.log")
    try:
        output_dir = RESULT_ROOT / "raw"
        output_dir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["PARC_ENABLE_PROXY_SUITE"] = "1"
        env["PARC_PROXY_TASKS_CSV"] = str(csv_path)
        completed_tasks: list[dict] = []
        total_elapsed = 0.0
        task_ids = [row["task_id"] for row in rows]
        for start in range(0, len(task_ids), args.batch_size):
            batch_index = start // args.batch_size
            batch_dir = output_dir / f"batch_{batch_index:03d}"
            raw_paths = sorted(batch_dir.glob("server_*.json"))
            if raw_paths:
                print(f"[audit] resume batch {batch_index + 1}", flush=True)
                batch_result = json.loads(raw_paths[0].read_text(encoding="utf-8"))
            else:
                batch_dir.mkdir(parents=True, exist_ok=True)
                selected = task_ids[start:start + args.batch_size]
                quoted_tasks = " ".join(f"'{task}'" for task in selected)
                command = (
                    f"source '{REPO_ROOT}/env.sh' && python -m pipeline "
                    f"--server-url http://localhost:{args.port} --track track1 "
                    f"--benchmark libero_proxy --n-episodes {args.n_episodes} "
                    f"--max-steps {args.max_steps} --seed {args.seed} "
                    f"--output-dir '{batch_dir}' --tasks {quoted_tasks}"
                )
                with (RESULT_ROOT / f"eval_batch_{batch_index:03d}.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                raw_path = sorted(batch_dir.glob("server_*.json"))[0]
                batch_result = json.loads(raw_path.read_text(encoding="utf-8"))
                print(f"[audit] completed batch {batch_index + 1}", flush=True)
            completed_tasks.extend(batch_result["tracks"][0]["tasks"])
            total_elapsed += float(batch_result.get("total_elapsed_sec", 0.0))

        combined = {
            "total_elapsed_sec": total_elapsed,
            "tracks": [{
                "tasks": completed_tasks,
                "overall_metrics": {
                    "n_total_episodes": len(completed_tasks) * args.n_episodes,
                },
            }],
        }
        report = summarize(combined, rows)
        report["protocol"] = vars(args)
        (RESULT_ROOT / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["overall"], indent=2))
        print(f"saved: {RESULT_ROOT / 'summary.json'}")
    finally:
        os.killpg(proc.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
