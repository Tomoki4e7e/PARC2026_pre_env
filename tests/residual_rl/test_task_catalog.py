from pathlib import Path

from experiments.libero_residual_rl.task_catalog import (
    MIDDLE_DRAWER,
    TOP_DRAWER,
    TaskCatalog,
)


def test_catalog_is_reproducible_and_assets_exist():
    catalog = TaskCatalog()
    first = catalog.build(episodes_per_condition=2, seed=7)
    second = catalog.build(episodes_per_condition=2, seed=7)

    assert first == second
    assert len(first) == 2 * 3 * 3 * 2
    assert {spec.base_task for spec in first} == {MIDDLE_DRAWER, TOP_DRAWER}
    assert all(Path(spec.bddl_file).is_file() for spec in first)


def test_l2_is_disjoint_from_l1_layout_difficulties():
    catalog = TaskCatalog()
    specs = catalog.build(
        base_tasks=(MIDDLE_DRAWER,),
        axes=("layout",),
        levels=(1, 2),
        episodes_per_condition=20,
    )
    low = {spec.difficulty for spec in specs if spec.level == 1}
    high = {spec.difficulty for spec in specs if spec.level == 2}

    assert low <= {1, 2}
    assert high <= {4, 5}
    assert low.isdisjoint(high)


def test_runtime_levels_respect_requested_ranges():
    specs = TaskCatalog().build(
        base_tasks=(MIDDLE_DRAWER,),
        axes=("camera", "robot"),
        levels=(1, 2),
        episodes_per_condition=10,
    )
    for spec in specs:
        if spec.axis == "camera":
            bound = 10 if spec.level == 1 else 25
            assert abs(spec.runtime_parameters["yaw_deg"]) <= bound
            assert abs(spec.runtime_parameters["pitch_deg"]) <= bound
        else:
            bound = 0.05 if spec.level == 1 else 0.15
            assert all(abs(value) <= bound for value in spec.runtime_parameters.values())

