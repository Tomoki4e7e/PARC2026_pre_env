"""選定用 hold-out タスクを「その段階の学習タスクと排他」に保つ仕組み。

背景: Phase5 の選定では hold-out 9 タスクのうち 5 タスクが Phase5 の学習
データそのものだったため、暗記したモデルが最良と判定され、本番（非公開
タスク）でスコア 0 になった。以降は hold-out を自動で clean 化し、汚染時は
明示的に停止する。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import config
from utils import normalize_task_name

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_POOL_PATH = config.WORK_DIR / "holdout_task_pool.json"

HOLDOUT_BENCHMARKS = ("libero_spatial", "libero_object", "libero_goal")

# `view_*` / `noise_*` 系のバリアントは bddl_files が存在せず評価が落ちるため、
# bddl が確実にある table / light 摂動だけを候補にする。
SAFE_VARIANT_RE = re.compile(r"_(?:table|light)_\d+$")

# 従来（Phase4 まで）使っていた固定 hold-out。clean な限り継続利用して
# 過去の数値と比較可能にする。
LEGACY_HOLDOUT_TASKS: dict[str, list[str]] = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_2",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_5",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_11",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1",
        "pick_up_the_ketchup_and_place_it_in_the_basket_light_2",
        "pick_up_the_butter_and_place_it_in_the_basket_table_5",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet_table_1",
        "turn_on_the_stove_light_2",
        "put_the_bowl_on_the_plate_table_3",
    ],
}

_SUITE_BASE_NAMES: dict[str, list[str]] = {
    "libero_spatial": config.SPATIAL_TASK_NAMES,
    "libero_object": config.OBJECT_TASK_NAMES,
    "libero_goal": config.GOAL_TASK_NAMES,
}


def known_bases(benchmark: str | None = None) -> list[str]:
    """正規化済みベースタスク名（摂動サフィックスなし）。"""
    if benchmark is not None:
        names = _SUITE_BASE_NAMES[benchmark]
    else:
        names = [n for v in _SUITE_BASE_NAMES.values() for n in v]
    return [normalize_task_name(n) for n in names]


def base_of(task_name: str, bases: list[str]) -> str | None:
    """摂動バリアント名 → ベースタスク名。最長前方一致で判定する。"""
    normalized = normalize_task_name(task_name)
    matched: str | None = None
    for base in bases:
        if normalized == base or normalized.startswith(base + " "):
            if matched is None or len(base) > len(matched):
                matched = base
    return matched


def trained_bases_for_stage(stage: str) -> set[str]:
    """その段階の LoRA 学習に使われたベースタスク（正規化済み）。"""
    if stage in ("stage1", "multisuite"):
        suites = config.TRAIN_SUITES
    elif stage == "track1_stage2":
        suites = config.TRACK1_STAGE2_TRAIN_SUITES
    elif stage == "phase3":
        suites = config.PHASE3_TRAIN_SUITES
    elif stage == "phase5":
        suites = config.PHASE5_TRAIN_SUITES
    elif stage == "phase4":
        return _phase4_trained_bases()
    else:
        raise ValueError(f"unknown stage: {stage!r}")
    return {
        normalize_task_name(name) for suite in suites for name in suite.task_names
    }


def _phase4_trained_bases() -> set[str]:
    """Phase4 は replay で 4 suite の全タスクを再学習しているため manifest を読む。"""
    manifest_path = config.PHASE4_MANIFEST_PATH
    if not manifest_path.is_file():
        # manifest が無い場合は安全側（全 suite 学習済み扱い）
        return set(known_bases())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained: set[str] = set()
    for source in manifest.get("sources", []):
        for task_name in source.get("selected_by_task", {}):
            trained.add(normalize_task_name(task_name))
    return trained


def refresh_task_pool(*, path: Path | None = None, timeout: int = 900) -> dict[str, list[str]]:
    """pipeline 環境（env.sh）でタスク名一覧を取得してキャッシュする。"""
    out_path = path or TASK_POOL_PATH
    benches = json.dumps(list(HOLDOUT_BENCHMARKS))
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from pipeline.environment import EnvironmentManager\n"
        "from pipeline.config import EvalConfig\n"
        "em = EnvironmentManager(EvalConfig())\n"
        f"benches = json.loads('{benches}')\n"
        "out = {b: [t.name for t in em.get_task_infos(b)] for b in benches}\n"
        f"Path(r'{out_path}').write_text(json.dumps(out))\n"
    )
    subprocess.run(
        ["bash", "-lc", f"source '{REPO_ROOT}/env.sh' && python - <<'PY'\n{script}PY"],
        cwd=REPO_ROOT,
        check=True,
        timeout=timeout,
        stdout=subprocess.DEVNULL,
    )
    return load_task_pool(path=out_path)


def load_task_pool(*, path: Path | None = None, auto_refresh: bool = True) -> dict[str, list[str]]:
    pool_path = path or TASK_POOL_PATH
    if not pool_path.is_file():
        if not auto_refresh:
            raise FileNotFoundError(
                f"タスク名キャッシュがありません: {pool_path}\n"
                "proxy_holdout.refresh_task_pool() を実行してください。"
            )
        return refresh_task_pool(path=pool_path)
    return json.loads(pool_path.read_text(encoding="utf-8"))


def select_clean_holdout(
    trained: set[str],
    *,
    pool: dict[str, list[str]] | None = None,
    per_suite: int = 3,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """学習タスクと排他な hold-out を suite ごとに per_suite 件選ぶ。

    従来の固定タスクが clean ならそれを優先し、不足分を同 suite の未学習
    ベースタスクから補う。補えない場合は RuntimeError。
    """
    pool = pool if pool is not None else load_task_pool()
    selected: dict[str, list[str]] = {}
    report: dict[str, Any] = {"per_suite": per_suite, "suites": {}, "excluded": {}}

    for benchmark in HOLDOUT_BENCHMARKS:
        bases = known_bases(benchmark)
        available = pool.get(benchmark, [])
        if not available:
            raise RuntimeError(f"タスク一覧が空です: {benchmark}")

        variants_by_base: dict[str, list[str]] = {}
        for name in available:
            base = base_of(name, bases)
            if base is None:
                continue
            variants_by_base.setdefault(base, []).append(name)

        excluded = sorted(b for b in variants_by_base if b in trained)
        clean_bases = sorted(b for b in variants_by_base if b not in trained)

        picked: list[str] = []
        used_bases: set[str] = set()

        for legacy in LEGACY_HOLDOUT_TASKS.get(benchmark, []):
            base = base_of(legacy, bases)
            if base is None or base in trained or base in used_bases:
                continue
            if legacy not in available:
                continue
            picked.append(legacy)
            used_bases.add(base)
            if len(picked) >= per_suite:
                break

        for base in clean_bases:
            if len(picked) >= per_suite:
                break
            if base in used_bases:
                continue
            variants = sorted(
                v for v in variants_by_base[base] if SAFE_VARIANT_RE.search(v)
            )
            if not variants:
                continue
            picked.append(variants[len(variants) // 2])
            used_bases.add(base)

        if len(picked) < per_suite:
            raise RuntimeError(
                f"[{benchmark}] clean な hold-out が {per_suite} 件そろいません "
                f"(clean base={len(clean_bases)}, 学習済み base={len(excluded)})。\n"
                "この段階は当該 suite をほぼ全学習しているため、libero_plus 内の "
                "hold-out は汎化指標になりません。OOD-90 を使ってください。"
            )

        selected[benchmark] = picked
        report["suites"][benchmark] = {
            "picked": picked,
            "n_clean_bases": len(clean_bases),
            "n_trained_bases": len(excluded),
        }
        report["excluded"][benchmark] = excluded

    return selected, report


def contamination_report(
    holdout: dict[str, list[str]], trained: set[str]
) -> dict[str, list[str]]:
    """hold-out のうち学習済みタスク（汚染）を suite ごとに返す。"""
    contaminated: dict[str, list[str]] = {}
    for benchmark, names in holdout.items():
        bases = known_bases(benchmark)
        bad = [n for n in names if (base_of(n, bases) or "") in trained]
        if bad:
            contaminated[benchmark] = bad
    return contaminated
