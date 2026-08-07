"""LIBERO-90 多様タスク再学習用の固定 train/OOD 分割。

OOD_TASK_NAMES は checkpoint 選定専用であり、学習データへ混入させない。
学習側は新規物体、受け皿、関節操作、複合・関係操作を各10タスク含む。
"""

from __future__ import annotations

import re


def normalize_task_name(value: str) -> str:
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


OOD_TASKS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "articulated": (
        "close the top drawer of the cabinet",
        "close the microwave",
        "open the microwave",
        "turn off the stove",
    ),
    "novel_placement": (
        "put the frying pan on the stove",
        "put the frying pan on the cabinet shelf",
        "pick up the alphabet soup and put it in the tray",
        "pick up the red mug and place it to the right of the caddy",
        "pick up the book on the left and place it on top of the shelf",
    ),
    "relational_multistep": (
        "put the butter at the back in the top drawer of the cabinet and close it",
        "stack the black bowl at the front on the black bowl in the middle",
        "stack the left bowl on the right bowl and place them in the tray",
    ),
}

# pipeline の libero_90 benchmark で使う BDDL task ID。
# OOD_TASKS_BY_CATEGORY とカテゴリ内の順序を一致させる。
OOD_BENCHMARK_TASKS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "articulated": (
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE8_turn_off_the_stove",
    ),
    "novel_placement": (
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
    ),
    "relational_multistep": (
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    ),
}

TRAIN_TASKS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "articulated": (
        "open the bottom drawer of the cabinet",
        "open the top drawer of the cabinet",
        "open the top drawer of the cabinet and put the bowl in it",
        "close the bottom drawer of the cabinet",
        "close the bottom drawer of the cabinet and open the top drawer",
        "turn on the stove",
        "turn on the stove and put the frying pan on it",
        "close the top drawer of the cabinet and put the black bowl on top of it",
        "put the ketchup in the top drawer of the cabinet",
        "put the black bowl in the bottom drawer of the cabinet",
    ),
    "novel_object": (
        "put the right moka pot on the stove",
        "put the moka pot on the stove",
        "put the red mug on the plate",
        "put the yellow and white mug on the right plate",
        "put the white mug on the plate",
        "pick up the book in the middle and place it on the cabinet shelf",
        "pick up the book on the right and place it under the cabinet shelf",
        "pick up the cream cheese and put it in the tray",
        "pick up the tomato sauce and put it in the tray",
        "put the wine bottle on the wine rack",
    ),
    "novel_receptacle": (
        "pick up the book and place it in the right compartment of the caddy",
        "pick up the book and place it in the front compartment of the caddy",
        "pick up the book and place it in the left compartment of the caddy",
        "pick up the book and place it in the back compartment of the caddy",
        "pick up the yellow and white mug and place it to the right of the caddy",
        "pick up the white mug and place it to the right of the caddy",
        "pick up the chocolate pudding and put it in the tray",
        "pick up the salad dressing and put it in the tray",
        "pick up the butter and put it in the tray",
        "pick up the ketchup and put it in the tray",
    ),
    "relational_multistep": (
        "put the chocolate pudding in the top drawer of the cabinet and close it",
        "put the butter at the front in the top drawer of the cabinet and close it",
        "stack the right bowl on the left bowl and place them in the tray",
        "stack the middle black bowl on the back black bowl",
        "put the chocolate pudding to the right of the plate",
        "put the chocolate pudding to the left of the plate",
        "put the black bowl at the front on the plate",
        "put the black bowl at the back on the plate",
        "put the white bowl to the right of the plate",
        "put the yellow and white mug to the front of the white mug",
    ),
}

OOD_TASK_NAMES = tuple(
    task for tasks in OOD_TASKS_BY_CATEGORY.values() for task in tasks
)
TRAIN_TASK_NAMES = tuple(
    task for tasks in TRAIN_TASKS_BY_CATEGORY.values() for task in tasks
)


def validate_task_split(available_tasks: set[str] | None = None) -> None:
    """分割の重複と、任意のデータセット task 一覧との整合性を検査する。"""
    train_norm = {normalize_task_name(task) for task in TRAIN_TASK_NAMES}
    ood_norm = {normalize_task_name(task) for task in OOD_TASK_NAMES}

    if len(train_norm) != len(TRAIN_TASK_NAMES):
        raise ValueError("TRAIN_TASK_NAMES に正規化後の重複があります。")
    if len(ood_norm) != len(OOD_TASK_NAMES):
        raise ValueError("OOD_TASK_NAMES に正規化後の重複があります。")
    if OOD_TASKS_BY_CATEGORY.keys() != OOD_BENCHMARK_TASKS_BY_CATEGORY.keys():
        raise ValueError("OOD language と benchmark ID のカテゴリが一致しません。")
    for category in OOD_TASKS_BY_CATEGORY:
        if len(OOD_TASKS_BY_CATEGORY[category]) != len(
            OOD_BENCHMARK_TASKS_BY_CATEGORY[category]
        ):
            raise ValueError(f"OOD task 対応数が一致しません: {category}")

    overlap = train_norm & ood_norm
    if overlap:
        raise ValueError(f"OOD task が学習側へ混入しています: {sorted(overlap)}")

    if available_tasks is None:
        return

    available_norm = {normalize_task_name(task) for task in available_tasks}
    missing = (train_norm | ood_norm) - available_norm
    if missing:
        raise ValueError(f"データセットに存在しない task があります: {sorted(missing)}")


validate_task_split()
