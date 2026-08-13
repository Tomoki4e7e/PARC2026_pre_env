from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "examples/smolvla_libero_spatial_lora_local"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

builder = importlib.import_module("build_curriculum_datasets")
comparison = importlib.import_module("compare_curriculum_order")
config = importlib.import_module("config")
evaluator = importlib.import_module("eval_expanded_generalization")


def test_failed_spatial_tasks_selects_only_below_threshold(tmp_path: Path) -> None:
    first, second = config.SPATIAL_TASK_NAMES[:2]
    summary = {
        "tasks": [
            {"group": "spatial_clean", "base_task": first, "success_rate": 0.5},
            {"group": "spatial_clean", "base_task": second, "success_rate": 1.0},
            {"group": "trained_object_unknown_texture", "base_task": first, "success_rate": 0.0},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert builder.failed_spatial_tasks(path, 0.75) == [first]


def test_representative_replay_is_stratified_and_excludes_targets() -> None:
    pools = {"target": list(range(10)), "a": [10, 11], "b": [20, 21], "c": [30, 31]}
    selected = builder.stratified_replay(pools, {"target"}, 5)
    assert [task for task, _ in selected] == ["a", "b", "c", "a", "b"]
    assert all(task != "target" for task, _ in selected)


def test_track1_replay_ratio_is_twenty_percent() -> None:
    target = sum(config.TRACK1_STAGE2_TASK_EPISODE_OVERRIDES.values())
    ratio = config.TRACK1_REPLAY_EPISODES / (target + config.TRACK1_REPLAY_EPISODES)
    assert target == 170
    assert 0.20 <= ratio <= 0.21


def test_eval_manifest_adds_exactly_ten_clean_spatial_tasks(tmp_path: Path) -> None:
    rows = evaluator.build_tasks_csv(tmp_path / "tasks.csv", include_spatial_clean=True)
    clean = [row for row in rows if row["group"] == "spatial_clean"]
    assert len(rows) == 46
    assert {row["base_task"] for row in clean} == set(config.SPATIAL_TASK_NAMES)


def test_comparison_rejects_different_task_manifests() -> None:
    left = {"protocol": {"n_episodes": 2, "max_steps": 600, "seed": 42},
            "tasks": [{"task_id": "a"}]}
    right = {"protocol": {"n_episodes": 2, "max_steps": 600, "seed": 42},
             "tasks": [{"task_id": "b"}]}
    try:
        comparison.assert_same_protocol([left, right])
    except ValueError as error:
        assert "manifests differ" in str(error)
    else:
        raise AssertionError("mismatched A/B manifest must be rejected")
