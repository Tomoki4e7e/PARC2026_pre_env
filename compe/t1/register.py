from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import torch

_SUITE = "libero_t1"
_PKG_DIR = Path(__file__).resolve().parent
_CSV = _PKG_DIR / "T1_TASKS.csv"


_SUFFIX_RE = re.compile(r"_light_[^.]*|_table_\d+")

_REGISTERED = False
_SUPPORT_GUARD_INSTALLED = False


def _resting_on_static_scene(env, obj_name) -> bool:
    sim = env.sim
    skip = getattr(env, "_support_guard_excluded_geom_ids", None)
    if skip is None:
        skip = set()
        for gid in range(sim.model.ngeom):
            name = sim.model.geom_id2name(gid) or ""
            if name.startswith(("gripper", "robot", "mount")):
                skip.add(gid)
        for obj in getattr(env, "objects_dict", {}).values():
            for geom in getattr(obj, "contact_geoms", []):
                try:
                    skip.add(sim.model.geom_name2id(geom))
                except Exception:
                    pass
        env._support_guard_excluded_geom_ids = skip

    object_geom_ids = set()
    for geom in getattr(env.get_object(obj_name), "contact_geoms", []):
        try:
            object_geom_ids.add(sim.model.geom_name2id(geom))
        except Exception:
            pass
    if not object_geom_ids:
        return True

    for contact_index in range(sim.data.ncon):
        contact = sim.data.contact[contact_index]
        geom1, geom2 = contact.geom1, contact.geom2
        if (
            geom1 in object_geom_ids
            and geom2 not in skip
            and geom2 not in object_geom_ids
        ):
            return True
        if (
            geom2 in object_geom_ids
            and geom1 not in skip
            and geom1 not in object_geom_ids
        ):
            return True
    return False


def _install_support_contact_guard() -> None:
    global _SUPPORT_GUARD_INSTALLED
    if _SUPPORT_GUARD_INSTALLED:
        return

    from libero.libero.envs.object_states.base_object_states import SiteObjectState

    if getattr(SiteObjectState, "_support_contact_guard", False):
        _SUPPORT_GUARD_INSTALLED = True
        return

    original_check_ontop = SiteObjectState.check_ontop

    def check_ontop(self, other):
        base = original_check_ontop(self, other)
        if not base or self.env.get_object(self.parent_name) is not None:
            return base
        try:
            return _resting_on_static_scene(self.env, other.object_name)
        except Exception:
            return base

    SiteObjectState.check_ontop = check_ontop
    SiteObjectState._support_contact_guard = True
    _SUPPORT_GUARD_INSTALLED = True


def _load_rows() -> list[dict]:
    with open(_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{_CSV} is empty; cannot register {_SUITE}.")
    return rows


def _init_states_for(task, get_libero_path):
    fn = task.init_states_file
    root = get_libero_path("init_states")
    stem, ext = os.path.splitext(fn)
    stripped = _SUFFIX_RE.sub("", stem) + ext
    p = os.path.join(root, task.problem_folder, stripped)
    return torch.load(p, weights_only=False)


def register_t1():
    global _REGISTERED
    from libero.libero import benchmark as _b
    from libero.libero import get_libero_path

    _install_support_contact_guard()
    rows = _load_rows()
    _b.task_maps[_SUITE] = {
        r["task_id"]: _b.Task(
            name=r["task_id"],
            language=r["instruction"],
            problem="Libero",
            problem_folder=r["suite"],
            bddl_file=f"{r['task_id']}.bddl",
            init_states_file=f"{r['task_id']}.pruned_init",
        )
        for r in rows
    }
    if _SUITE not in _b.libero_suites:
        _b.libero_suites.append(_SUITE)

    if _REGISTERED:
        return _b.get_benchmark(_SUITE)

    @_b.register_benchmark
    class LIBERO_T1(_b.Benchmark):
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
