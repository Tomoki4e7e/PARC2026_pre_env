#!/usr/bin/env python3
"""Generate a lightweight success/failure matrix in LIBERO + MuJoCo."""

from __future__ import annotations

# Importing this module configures the headless renderer before LIBERO is loaded.
from visualize_simulation import _camera_frame

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SCENARIO_CATEGORIES = {
    "success": "success",
    "no_grasp": "no_grasp",                 # backward-compatible aliases
    "drop_early": "drop_early",
    "wrong_target": "wrong_target",
    "collision": "collision",
    "no_grasp_open": "no_grasp",
    "no_grasp_hover": "no_grasp",
    "no_grasp_offset": "no_grasp",
    "drop_after_lift": "drop_early",
    "drop_midway": "drop_early",
    "drop_near_target": "drop_early",
    "wrong_target_left": "wrong_target",
    "wrong_target_right": "wrong_target",
    "wrong_target_distractor": "wrong_target",
    "collision_milk": "collision",
    "collision_butter": "collision",
    "collision_orange_juice": "collision",
}
SCENARIOS = tuple(SCENARIO_CATEGORIES)
FAILURE_MATRIX = (
    "no_grasp_open", "no_grasp_hover", "no_grasp_offset",
    "drop_after_lift", "drop_midway", "drop_near_target",
    "wrong_target_left", "wrong_target_right", "wrong_target_distractor",
    "collision_milk", "collision_butter", "collision_orange_juice",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="成功・失敗パターンを一括で動画検証します")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                   default=["success", "no_grasp", "drop_early", "wrong_target", "collision"])
    p.add_argument("--matrix", action="store_true",
                   help="4失敗カテゴリ×3バリアント（12ケース）を一括実行")
    p.add_argument("--output-dir", type=Path, default=Path("results/pattern_validation"))
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--frame-skip", type=int, default=3,
                   help="N物理ステップごとに1フレーム記録（大きいほど軽量）")
    p.add_argument("--camera", choices=("agent", "wrist", "both"), default="agent")
    return p.parse_args()


class ScenarioRunner:
    def __init__(self, env, task, writer, camera: str, frame_skip: int, scenario: str):
        self.env, self.task, self.writer = env, task, writer
        self.camera, self.frame_skip, self.scenario = camera, frame_skip, scenario
        self.step_count = 0
        self.frame_count = 0
        self.obs = None
        self.initial_positions: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.env.reset()
        self.env.sim.set_state_from_flattened(np.asarray(self.task.init_states[0]))
        self.env.sim.forward()
        action = np.zeros(self.env.robots[0].action_dim, dtype=np.float64)
        action[-1] = -1.0
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(action)
        for robot in self.env.robots:
            robot.controller.use_delta = True
        self.initial_positions = {
            k[:-4]: np.asarray(v).copy() for k, v in self.obs.items()
            if k.endswith("_pos") and not k.startswith("robot0")
            and not k.endswith("_to_robot0_eef_pos")
        }
        self.record(force=True)

    def record(self, force: bool = False) -> None:
        if not force and self.step_count % self.frame_skip:
            return
        frame = _camera_frame(self.obs, self.camera)
        width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (width, 62), (0, 0, 0), -1)
        cv2.putText(frame, f"pattern: {self.scenario}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"physics step {self.step_count}", (12, 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 255), 1, cv2.LINE_AA)
        self.writer.write(frame)
        self.frame_count += 1

    def step(self, xyz=(0.0, 0.0, 0.0), gripper=-1.0) -> None:
        action = np.zeros(self.env.robots[0].action_dim, dtype=np.float64)
        action[:3] = np.asarray(xyz)
        action[-1] = gripper
        self.obs, _, _, _ = self.env.step(action)
        self.step_count += 1
        self.record()

    def goto(self, target, gripper: float, max_steps: int = 50,
             tolerance: float = 0.012) -> None:
        """Move the EEF toward a fixed point or a callable moving target."""
        for _ in range(max_steps):
            goal = np.asarray(target() if callable(target) else target)
            delta = goal - np.asarray(self.obs["robot0_eef_pos"])
            if np.linalg.norm(delta) < tolerance:
                break
            self.step(np.clip(delta * 6.0, -0.5, 0.5), gripper)

    def hold(self, steps: int, gripper: float) -> None:
        for _ in range(steps):
            self.step(gripper=gripper)

    def max_displacements(self) -> dict[str, float]:
        result = {}
        for name, initial in self.initial_positions.items():
            current = self.obs.get(name + "_pos")
            if current is not None:
                result[name] = float(np.abs(np.asarray(current) - initial).sum())
        return result


def run_pattern(r: ScenarioRunner, scenario: str) -> dict:
    # Task 1 is deliberately used: it is a compact tabletop pick-and-place task.
    obj_key, target_key = "tomato_sauce_1_pos", "basket_1_pos"
    r.reset()
    target_center = np.asarray(r.obs[target_key]).copy()
    object_start = np.asarray(r.obs[obj_key]).copy()

    category = SCENARIO_CATEGORIES[scenario]
    above_object = lambda: np.asarray(r.obs[obj_key]) + (0.0, 0.0, 0.15)
    grasp_offset = (0.10, 0.0, 0.035) if scenario == "no_grasp_offset" else (0.0, 0.0, 0.035)
    grasp_point = lambda: np.asarray(r.obs[obj_key]) + grasp_offset
    r.goto(above_object, -1.0, 40)
    if scenario == "no_grasp_hover":
        r.hold(25, -1.0)
        r.goto(object_start + (-0.18, 0.12, 0.20), -1.0, 45)
    else:
        r.goto(grasp_point, -1.0, 35, tolerance=0.008)

    if scenario == "no_grasp_hover":
        pass
    elif category == "no_grasp":
        # Keep the gripper open, or close at an intentional lateral offset.
        if scenario == "no_grasp_offset":
            r.hold(15, 1.0)
        r.goto(target_center + (0.0, 0.0, 0.22), -1.0, 55)
        r.hold(25, -1.0)
    else:
        r.hold(15, 1.0)
        r.goto(lambda: np.asarray(r.obs[obj_key]) + (0.0, 0.0, 0.22), 1.0, 40)

        if scenario in ("drop_early", "drop_after_lift"):
            r.hold(8, -1.0)
            r.goto(target_center + (0.0, 0.0, 0.24), -1.0, 50)
        elif scenario == "drop_midway":
            midpoint = (np.asarray(r.obs[obj_key]) + target_center) / 2
            r.goto(midpoint + (0.0, 0.0, 0.22), 1.0, 40)
            r.hold(8, -1.0)
            r.goto(target_center + (0.0, 0.0, 0.24), -1.0, 35)
        elif scenario == "drop_near_target":
            r.goto(target_center + (0.10, 0.0, 0.22), 1.0, 55)
            r.hold(8, -1.0)
        elif category == "wrong_target":
            wrong_offsets = {
                "wrong_target": (0.24, -0.12, 0.24),
                "wrong_target_left": (-0.24, -0.10, 0.24),
                "wrong_target_right": (0.24, -0.10, 0.24),
            }
            if scenario == "wrong_target_distractor":
                wrong_goal = np.asarray(r.obs["bbq_sauce_1_pos"]) + (0.0, 0.0, 0.20)
            else:
                wrong_goal = target_center + wrong_offsets[scenario]
            r.goto(wrong_goal, 1.0, 55)
            r.hold(8, -1.0)
        elif category == "collision":
            # Carry the object toward a distractor instead of the basket.
            distractor_keys = {
                "collision": "milk_1_pos",
                "collision_milk": "milk_1_pos",
                "collision_butter": "butter_1_pos",
                "collision_orange_juice": "orange_juice_1_pos",
            }
            distractor = np.asarray(r.obs[distractor_keys[scenario]]).copy()
            r.goto(distractor + (0.0, 0.0, 0.10), 1.0, 55)
            r.goto(distractor + (0.0, 0.0, 0.025), 1.0, 25)
            r.hold(8, -1.0)
        else:  # success
            # Release above the original basket center, avoiding contact with it.
            r.goto(target_center + (0.0, 0.0, 0.25), 1.0, 60)
            r.hold(10, -1.0)

    r.hold(45, -1.0)
    r.record(force=True)
    displacement = r.max_displacements()
    collateral = {
        k: v for k, v in displacement.items()
        if k not in ("tomato_sauce_1", "basket_1") and v > 0.001
    }
    return {
        "scenario": scenario,
        "category": category,
        "variant": scenario,
        "expected_success": category == "success",
        "goal_success": bool(r.env.check_success()),
        "expectation_matched": bool(r.env.check_success()) == (category == "success"),
        "physics_steps": r.step_count,
        "video_frames": r.frame_count,
        "object_travel_m_l1": float(
            np.abs(np.asarray(r.obs[obj_key]) - object_start).sum()
        ),
        "collateral_displacements_m_l1": collateral,
    }


def main() -> None:
    args = parse_args()
    if args.matrix:
        args.scenarios = list(FAILURE_MATRIX)
    if args.size <= 0 or args.fps <= 0 or args.frame_skip <= 0:
        raise SystemExit("--size, --fps, --frame-skip は正の値を指定してください")
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = EnvironmentManager(EvalConfig(camera_height=args.size, camera_width=args.size))
    task = manager.get_task_infos("libero_t1")[1]
    env = manager.create_env(task)
    width = args.size * (2 if args.camera == "both" else 1)
    results = []
    try:
        for scenario in args.scenarios:
            path = output_dir / f"{scenario}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (width, args.size))
            if not writer.isOpened():
                raise RuntimeError(f"動画エンコーダーを開けませんでした: {path}")
            try:
                runner = ScenarioRunner(env, task, writer, args.camera,
                                        args.frame_skip, scenario)
                result = run_pattern(runner, scenario)
            finally:
                writer.release()
            result["video"] = path.name
            results.append(result)
            print(f"{scenario:12s} goal={result['goal_success']} "
                  f"matched={result['expectation_matched']} -> {path}")
    finally:
        env.close()

    summary = {
        "task": task.language,
        "camera": args.camera,
        "frame_skip": args.frame_skip,
        "all_expectations_matched": all(r["expectation_matched"] for r in results),
        "results": results,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"結果: {manifest}")


if __name__ == "__main__":
    main()
