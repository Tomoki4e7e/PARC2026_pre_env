#!/usr/bin/env python3
"""摂動カテゴリ別プロキシ評価を実行し、カテゴリ別スコアを集計する。

compe/proxy/PROXY_TASKS.csv（本番と同じ CSV 登録方式）の libero_proxy スイートを
カテゴリごとに評価する。instruction も本番同様 bddl の言い回しが渡る。

前提: ポリシーサーバーが起動していること（run_policy_server.sh）。
      評価は parc2026 venv で実行すること（source env.sh）。

使い方:
    source env.sh
    python examples/smolvla_libero_spatial_lora_local/run_proxy_eval.py --tag multisuite_legacy
    python examples/smolvla_libero_spatial_lora_local/run_proxy_eval.py --tag base_generic \
        --categories light,language
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_CSV = REPO_ROOT / "compe" / "proxy" / "PROXY_TASKS.csv"

# カテゴリごとのエピソード数。add / level は init state が 1 本のみなので少なくする。
EPISODES_BY_CATEGORY: dict[str, int] = {
    "table": 10,
    "tb": 10,
    "light": 10,
    "language": 10,
    "add": 2,
    "level": 2,
}


def load_tasks() -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    with open(PROXY_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            by_category[row["category"]].append(row["task_id"])
    return dict(by_category)


def run_category(
    category: str,
    task_ids: list[str],
    *,
    output_dir: Path,
    server_url: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
    camera: int,
) -> dict:
    result_path = output_dir / f"{category}.json"
    command = [
        sys.executable,
        "-m",
        "pipeline",
        "--server-url",
        server_url,
        "--track",
        "track1",
        "--benchmark",
        "libero_proxy",
        "--n-episodes",
        str(n_episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--tasks",
        *task_ids,
    ]
    env = os.environ.copy()
    env["PARC_ENABLE_PROXY_SUITE"] = "1"
    env["LIBERO_EVAL_CAMERA"] = str(camera)

    log_path = output_dir / f"{category}.log"
    with open(log_path, "w") as log:
        subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )

    raw = output_dir / "server_8000.json"
    if not raw.is_file():
        candidates = sorted(output_dir.glob("server_*.json"))
        if not candidates:
            raise RuntimeError(f"評価結果が見つかりません: {output_dir}")
        raw = candidates[0]
    raw.replace(result_path)
    return json.loads(result_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="摂動カテゴリ別プロキシ評価")
    parser.add_argument("--tag", required=True, help="出力先 results/proxy_<tag>/")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument(
        "--categories",
        default=",".join(EPISODES_BY_CATEGORY),
        help="評価カテゴリ（カンマ区切り）",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--camera", type=int, default=128, help="描画解像度（本番は 128）"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="全カテゴリ共通のエピソード数（既定はカテゴリ別）",
    )
    args = parser.parse_args()

    by_category = load_tasks()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    missing = [c for c in categories if c not in by_category]
    if missing:
        parser.error(f"PROXY_TASKS.csv に無いカテゴリ: {missing}")

    output_dir = REPO_ROOT / "results" / f"proxy_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for category in categories:
        task_ids = by_category[category]
        n_episodes = args.n_episodes or EPISODES_BY_CATEGORY.get(category, 10)
        print(
            f"[proxy] {category}: {len(task_ids)} タスク × {n_episodes} ep",
            flush=True,
        )
        result = run_category(
            category,
            task_ids,
            output_dir=output_dir,
            server_url=args.server_url,
            n_episodes=n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            camera=args.camera,
        )
        track = result["tracks"][0]
        per_task = {t["task_name"]: t["success_rate"] for t in track["tasks"]}
        summary[category] = {
            "score": track["overall_score"],
            "n_tasks": len(per_task),
            "n_episodes": n_episodes,
            "collision_rate": track["overall_metrics"].get("mean_collision_rate"),
            "elapsed_sec": result["total_elapsed_sec"],
            "tasks": per_task,
        }
        print(
            f"[proxy] {category}: score={track['overall_score']:.3f} "
            f"({result['total_elapsed_sec']:.0f}s)",
            flush=True,
        )

    scores = [v["score"] for v in summary.values()]
    overall = sum(scores) / len(scores) if scores else 0.0
    payload = {
        "tag": args.tag,
        "camera": args.camera,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "instruction_mode": os.environ.get("PARC_INSTRUCTION_MODE", "(server 側設定)"),
        "category_mean": overall,
        "categories": summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== proxy_{args.tag} ===")
    for category, value in summary.items():
        print(f"  {category:9s} {value['score']:.3f}")
    print(f"  {'平均':9s} {overall:.3f}")
    print(f"保存: {summary_path}")


if __name__ == "__main__":
    main()
