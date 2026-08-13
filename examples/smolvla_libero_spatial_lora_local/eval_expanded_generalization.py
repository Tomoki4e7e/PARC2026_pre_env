#!/usr/bin/env python3
"""Evaluate the expanded model on three non-overlapping generalization slices."""

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
from collections import defaultdict
from pathlib import Path

import config

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BDDL_ROOT = REPO_ROOT / "LIBERO-plus/libero/libero/bddl_files"
RESULT_ROOT = REPO_ROOT / "results/expanded_generalization"
LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")
VARIANT_RE = re.compile(r"_(table|light)_(\d+)$")

KNOWN_BASES = {
    "libero_spatial": [config.TRACK1_STAGE2_SPATIAL_TASK_NAMES[0]],
    "libero_object": config.TRACK1_STAGE2_OBJECT_TASK_NAMES,
    "libero_goal": config.TRACK1_STAGE2_GOAL_TASK_NAMES,
}

# LIBERO-Long tasks contain objects absent from the 30-task Spatial/Object/Goal
# training set (moka pots, mugs and books), while retaining familiar table/light
# perturbation IDs.
UNSEEN_OBJECT_BASES = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]

# These exact perturbation suffixes occur in the successful local replay set.
TRAINED_SUFFIXES = {
    "table_1", "table_3", "table_9", "table_15", "table_16", "table_21",
    "table_22", "table_28", "light_1", "light_9", "light_20", "light_31",
    "light_43",
}


def language(path: Path) -> str:
    match = LANGUAGE_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"language not found: {path}")
    return " ".join(match.group(1).split())


def suffix(stem: str) -> str | None:
    match = VARIANT_RE.search(stem)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def candidates(suite: str, base: str, *, trained_texture: bool) -> list[Path]:
    paths = []
    file_base = base.replace(" ", "_")
    for path in sorted((BDDL_ROOT / suite).glob(f"{file_base}_*.bddl")):
        value = suffix(path.stem)
        if value is None:
            continue
        if (value in TRAINED_SUFFIXES) == trained_texture:
            paths.append(path)
    return paths


def evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"need {count} variants, only {len(paths)} available")
    return [paths[round(i * (len(paths) - 1) / (count - 1))] for i in range(count)] if count > 1 else [paths[len(paths) // 2]]


def build_tasks_csv(path: Path, *, include_spatial_clean: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(group: str, suite: str, base: str, bddl: Path) -> None:
        rows.append({
            "task_num": str(len(rows) + 1), "task_id": bddl.stem,
            "instruction": language(bddl), "suite": suite, "group": group,
            "base_task": base, "texture_suffix": suffix(bddl.stem) or "clean",
        })

    # The ten canonical initial configurations are the only source used to
    # decide whether the optional Spatial corrective stage is needed.
    if include_spatial_clean:
        for base in config.SPATIAL_TASK_NAMES:
            bddl = BDDL_ROOT / "libero_spatial" / f"{base.replace(' ', '_')}.bddl"
            if not bddl.is_file():
                raise FileNotFoundError(f"clean Spatial task not found: {bddl}")
            add("spatial_clean", "libero_spatial", base, bddl)

    # A: exact four learned/evaluation semantics, perturbation IDs never replayed.
    for suite, bases in KNOWN_BASES.items():
        for base in bases:
            for bddl in evenly(candidates(suite, base, trained_texture=False), 3):
                add("known_task_unknown_texture", suite, base, bddl)

    # B: novel objects from LIBERO-Long under texture IDs present in training.
    per_base = [3, 3, 2, 2, 2]
    for base, count in zip(UNSEEN_OBJECT_BASES, per_base, strict=True):
        for bddl in evenly(candidates("libero_10", base, trained_texture=True), count):
            add("unseen_object_trained_texture", "libero_10", base, bddl)

    # C: learned objects/tasks beyond the four Track1 tasks, held-out textures.
    remaining = []
    for suite, tasks in (
        ("libero_spatial", config.SPATIAL_TASK_NAMES),
        ("libero_object", config.OBJECT_TASK_NAMES),
        ("libero_goal", config.GOAL_TASK_NAMES),
    ):
        known = set(KNOWN_BASES.get(suite, []))
        remaining.extend((suite, task) for task in tasks if task not in known)
    for suite, base in evenly(remaining, 12):
        add("trained_object_unknown_texture", suite, base,
            evenly(candidates(suite, base, trained_texture=False), 1)[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["group"]] += 1
    expected = {
        "known_task_unknown_texture": 12,
        "unseen_object_trained_texture": 12,
        "trained_object_unknown_texture": 12,
    }
    if include_spatial_clean:
        expected["spatial_clean"] = 10
    if dict(counts) != expected:
        raise RuntimeError(f"unexpected evaluation split: {dict(counts)}")
    return rows


def start_server(
    port: int,
    model_dir: Path,
    residual_dir: Path | None,
    log_path: Path,
    *,
    residual_enabled: bool,
    critic_dir: Path | None = None,
    critic_enabled: bool = False,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "PARC_MODEL_DIR": str(model_dir.resolve()),
        "PARC_RESIDUAL_POLICY": "1" if residual_enabled else "0",
        "PARC_COLLISION_CRITIC": "1" if critic_enabled else "0",
        "POLICY_SERVER_LOG": str(log_path.resolve()),
    })
    if residual_dir is not None:
        env["PARC_RESIDUAL_MODEL_DIR"] = str(residual_dir.resolve())
    if critic_dir is not None:
        env["PARC_COLLISION_CRITIC_DIR"] = str(critic_dir.resolve())
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "run_policy_server.sh"), "--port", str(port)],
        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                health = json.loads(response.read())
                if (response.status == 200
                        and bool(health.get("residual_enabled")) == residual_enabled
                        and bool(health.get("collision_critic_enabled")) == critic_enabled):
                    return proc
                # A just-terminated server can briefly answer on the reused port.
                # Do not burn through all retries while its health flags mismatch.
                time.sleep(2)
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"policy server did not become ready; see {log_path}")


def main() -> None:
    global RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=config.EXPANDED_MERGED_MODEL_DIR)
    parser.add_argument("--residual-dir", type=Path)
    parser.add_argument(
        "--disable-residual", action="store_true",
        help="Evaluate the merged SmolVLA checkpoint without a residual policy",
    )
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--include-spatial-clean", action="store_true",
        help="同じmanifestへSpatialの標準10配置を追加（失敗時補正の判定用）",
    )
    args = parser.parse_args()
    if not args.disable_residual and args.residual_dir is None:
        parser.error("--residual-dir is required unless --disable-residual is set")
    RESULT_ROOT = args.result_root.resolve()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "tasks.csv"
    rows = build_tasks_csv(csv_path, include_spatial_clean=args.include_spatial_clean)
    proc = start_server(
        args.port, args.model_dir, args.residual_dir, RESULT_ROOT / "server.log",
        residual_enabled=not args.disable_residual,
    )
    try:
        metadata = {row["task_id"]: row for row in rows}
        task_ids = list(metadata)
        completed = []
        eval_env = os.environ.copy()
        eval_env["PARC_ENABLE_PROXY_SUITE"] = "1"
        eval_env["PARC_PROXY_TASKS_CSV"] = str(csv_path.resolve())
        eval_env["PYTHONPATH"] = ":".join([
            str(REPO_ROOT / "LIBERO-plus"), str(REPO_ROOT), str(REPO_ROOT / "compe"),
            eval_env.get("PYTHONPATH", ""),
        ])
        eval_env["LIBERO_ROOT"] = str(REPO_ROOT / "LIBERO-plus")
        conda_lib = "/home/cm-pc/miniconda3/envs/parc2026/lib"
        eval_env["LD_LIBRARY_PATH"] = ":".join(
            value for value in (conda_lib, eval_env.get("LD_LIBRARY_PATH", "")) if value
        )
        for start in range(0, len(task_ids), args.batch_size):
            batch = start // args.batch_size
            batch_dir = RESULT_ROOT / "raw" / f"batch_{batch:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            selected = task_ids[start:start + args.batch_size]
            command = [
                "/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python", "-m", "pipeline",
                "--server-url", f"http://127.0.0.1:{args.port}", "--track", "track1",
                "--benchmark", "libero_proxy", "--n-episodes", str(args.n_episodes),
                "--max-steps", str(args.max_steps), "--seed", str(args.seed),
                "--output-dir", str(batch_dir), "--tasks", *selected,
            ]
            with (RESULT_ROOT / f"eval_batch_{batch:03d}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO_ROOT, env=eval_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result_path = sorted(batch_dir.glob("server_*.json"))[-1]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            completed.extend(result["tracks"][0]["tasks"])
            print(f"[expanded-eval] batch {batch + 1} complete", flush=True)

        groups: dict[str, list[dict]] = defaultdict(list)
        details = []
        for task in completed:
            meta = metadata[task["task_name"]]
            item = {
                **meta,
                "success_rate": float(task["success_rate"]),
                "collision_rate": float(task.get("metrics", {}).get("collision_rate", 0.0)),
                "metrics": task.get("metrics", {}),
            }
            groups[meta["group"]].append(item)
            details.append(item)
        protocol = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        summary = {
            "protocol": protocol,
            "groups": {
                name: {
                    "n_tasks": len(items),
                    "n_episodes": len(items) * args.n_episodes,
                    "success_rate": sum(x["success_rate"] for x in items) / len(items),
                    "collision_rate": sum(x["collision_rate"] for x in items) / len(items),
                }
                for name, items in groups.items()
            },
            "tasks": details,
        }
        (RESULT_ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary["groups"], indent=2, ensure_ascii=False))
    finally:
        os.killpg(proc.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
