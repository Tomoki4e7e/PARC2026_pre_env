from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import torch

_SUITE = "libero_proxy"
_PKG_DIR = Path(__file__).resolve().parent
_CSV = _PKG_DIR / "PROXY_TASKS.csv"

# 変種ごとの init state 解決規則（LIBERO-plus Benchmark.get_task_init_states と同じ）
_BASE_SUFFIX_RE = re.compile(r"_(?:table|tb|light|language)_\d+")
_NEWOBJ_RE = re.compile(r"_add_\d+|_level\d+_sample\d+")

_REGISTERED = False


def _load_rows() -> list[dict]:
    with open(_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{_CSV} is empty; cannot register {_SUITE}.")
    return rows


def _init_states_for(task, get_libero_path):
    root = get_libero_path("init_states")
    name = task.init_states_file
    if _NEWOBJ_RE.search(name):
        # 物体追加系（add / level）は専用ディレクトリに変種ごとの init state がある
        path = os.path.join(root, "libero_newobj", task.problem_folder, name)
    else:
        path = os.path.join(root, task.problem_folder, _BASE_SUFFIX_RE.sub("", name))
    init_states = torch.load(path, weights_only=False)
    # add / level は init state が 1 本しか無く 1 次元で保存されている
    if getattr(init_states, "ndim", 2) == 1:
        init_states = init_states.reshape(1, -1)
    return init_states


def register_proxy():
    """PROXY_TASKS.csv から libero_proxy スイートを登録する（compe/t1 と同方式）。"""
    global _REGISTERED
    from libero.libero import benchmark as _b
    from libero.libero import get_libero_path

    from compe.t1.register import _install_support_contact_guard

    _install_support_contact_guard()
    rows = _load_rows()
    _b.task_maps[_SUITE] = {
        row["task_id"]: _b.Task(
            name=row["task_id"],
            language=row["instruction"],
            problem="Libero",
            problem_folder=row["suite"],
            bddl_file=f"{row['task_id']}.bddl",
            init_states_file=f"{row['task_id']}.pruned_init",
        )
        for row in rows
    }
    if _SUITE not in _b.libero_suites:
        _b.libero_suites.append(_SUITE)

    if _REGISTERED:
        return _b.get_benchmark(_SUITE)

    @_b.register_benchmark
    class LIBERO_PROXY(_b.Benchmark):
        def __init__(self, task_order_index: int = 0):
            assert task_order_index == 0, (
                f"{_SUITE} has a variable task count; only task_order_index=0 supported."
            )
            super().__init__(task_order_index=task_order_index)
            self.name = _SUITE
            self._make_benchmark()

        def _make_benchmark(self):
            self.tasks = list(_b.task_maps[self.name].values())
            self.n_tasks = len(self.tasks)

        def get_task_init_states(self, i):
            return _init_states_for(self.tasks[i], get_libero_path)

    _REGISTERED = True

    return _b.get_benchmark(_SUITE)
