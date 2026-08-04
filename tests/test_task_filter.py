import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@dataclass
class _Task:
    name: str


def _make_pipeline(task_ids):
    from pipeline.pipeline import EvaluationPipeline

    obj = EvaluationPipeline.__new__(EvaluationPipeline)
    obj.config = types.SimpleNamespace(task_ids=task_ids)
    return obj


TASKS = [_Task("task_a"), _Task("task_b"), _Task("task_c")]


def test_filters_to_requested_tasks():
    p = _make_pipeline(["task_c", "task_a"])
    got = p._filter_by_task_ids(TASKS, "libero_t1")

    assert [t.name for t in got] == ["task_a", "task_c"]


def test_none_keeps_all_tasks():
    p = _make_pipeline(None)
    assert p._filter_by_task_ids(TASKS, "libero_t1") == TASKS


def test_unknown_task_raises_with_available_names():
    p = _make_pipeline(["task_a", "no_such_task"])
    with pytest.raises(RuntimeError) as exc:
        p._filter_by_task_ids(TASKS, "libero_t1")
    assert "no_such_task" in str(exc.value)
    assert "task_b" in str(exc.value)
