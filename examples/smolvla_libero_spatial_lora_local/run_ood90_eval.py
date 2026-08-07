#!/usr/bin/env python3
"""LIBERO-90由来の真のOOD固定セットを評価する。

lerobot/libero_plus の既知40タスクと意味的に同一のタスクを避け、未知の物体、
受け皿、関節操作、関係指定、複合操作を含む12タスクを固定した。
policy server は事前に起動しておくこと。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diverse_retrain_tasks import OOD_BENCHMARK_TASKS_BY_CATEGORY

TASKS = OOD_BENCHMARK_TASKS_BY_CATEGORY


def main() -> None:
    parser = argparse.ArgumentParser(description="LIBERO-90 OOD固定セット評価")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = REPO_ROOT / "results" / f"ood90_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_names = [name for names in TASKS.values() for name in names]
    # pipeline / LIBERO は parc2026 venv が必要。呼び出し元が SmolVLA conda でも動くよう
    # env.sh 経由で起動する。
    task_args = " ".join(f"'{name}'" for name in task_names)
    command = [
        "bash",
        "-lc",
        (
            f"source '{REPO_ROOT}/env.sh' && "
            f"python -m pipeline "
            f"--server-url '{args.server_url}' "
            f"--track track1 "
            f"--benchmark libero_90 "
            f"--n-episodes {args.n_episodes} "
            f"--max-steps {args.max_steps} "
            f"--seed {args.seed} "
            f"--output-dir '{output_dir}' "
            f"--tasks {task_args}"
        ),
    ]
    log_path = output_dir / "eval.log"
    with log_path.open("w") as log:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    candidates = sorted(output_dir.glob("server_*.json"))
    if not candidates:
        raise RuntimeError(f"評価結果がありません。ログを確認: {log_path}")
    raw_path = candidates[0]
    result = json.loads(raw_path.read_text())
    track = result["tracks"][0]
    task_scores = {
        item["task_name"]: item["success_rate"] for item in track["tasks"]
    }

    categories: dict[str, dict] = {}
    for category, names in TASKS.items():
        scores = [task_scores[name] for name in names]
        categories[category] = {
            "score": sum(scores) / len(scores),
            "tasks": {name: task_scores[name] for name in names},
        }

    summary = {
        "tag": args.tag,
        "benchmark": "libero_90",
        "n_tasks": len(task_names),
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "overall_score": track["overall_score"],
        "categories": categories,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"=== ood90_{args.tag} ===")
    print(f"overall              {track['overall_score']:.3f}")
    for category, value in categories.items():
        print(f"{category:20s} {value['score']:.3f}")
    print(f"保存: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
