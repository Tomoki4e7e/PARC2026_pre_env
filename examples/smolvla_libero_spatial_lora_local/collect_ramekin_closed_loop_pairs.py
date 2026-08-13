#!/usr/bin/env python3
"""Collect unique same-init safe ramekin trajectories with feedback correction.

The controller tracks a previously verified safe end-effector path.  At every
simulator step it recomputes a Cartesian correction from the live observation,
so this is closed-loop collection rather than open-loop action replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import validate_perturbation_replay as replay_tool

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "LIBERO-plus", REPO_ROOT / "compe"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
TASKS = {
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_light_41",
}


def metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as episode:
        return json.loads(str(episode["metadata_json"].item()))


def grasp_index(actions: np.ndarray) -> int:
    closed = np.flatnonzero(np.asarray(actions)[:, -1] > 0.25)
    return int(closed[0]) if len(closed) else min(30, len(actions) - 1)


def annotate(path: Path, provenance: dict) -> None:
    with np.load(path, allow_pickle=False) as episode:
        payload = {key: np.asarray(episode[key]) for key in episode.files}
    item = json.loads(str(payload["metadata_json"].item()))
    item["preference_pair"] = provenance
    payload["metadata_json"] = np.asarray(json.dumps(item, ensure_ascii=False))
    temporary = path.with_suffix(".annotating.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--unsafe-root", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--safe-per-init", type=int, default=2)
    parser.add_argument("--max-attempts-per-init", type=int, default=30)
    parser.add_argument("--ordinals", default="0,1,2,3,4")
    parser.add_argument("--confirm-local-generated-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_generated_data:
        parser.error("--confirm-local-generated-data is required")
    if args.output_root.exists() and any(args.output_root.rglob("*.npz")):
        raise FileExistsError(f"refusing to mix with existing records: {args.output_root}")

    rows = {row["task_id"]: row for row in csv.DictReader(
        args.tasks_csv.open(encoding="utf-8")) if row["task_id"] in TASKS}
    if set(rows) != TASKS:
        raise RuntimeError(f"missing task metadata: {TASKS - set(rows)}")
    os.environ["PARC_PROXY_TASKS_CSV"] = str(args.tasks_csv.resolve())
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager

    manager = EnvironmentManager(EvalConfig())
    infos = {info.name: info for info in manager.get_task_infos("libero_proxy")}

    unsafe: dict[tuple[str, int], Path] = {}
    for path in sorted(args.unsafe_root.glob("*.npz")):
        item = metadata(path)
        if item["task_name"] not in TASKS or not item.get("collided"):
            continue
        ordinal = int(item["collection_context"]["init_state_ordinal"])
        unsafe[(item["task_name"], ordinal)] = path
    requested_ordinals = {int(value) for value in args.ordinals.split(",") if value.strip()}
    unsafe = {key: value for key, value in unsafe.items() if key[1] in requested_ordinals}
    expected = len(TASKS) * len(requested_ordinals)
    if len(unsafe) != expected:
        raise RuntimeError(f"expected {expected} collided ramekin initial states, found {len(unsafe)}")

    donors = []
    for path in sorted(args.donor_root.rglob("*.npz")):
        item = metadata(path)
        pair = item.get("corrective_pair", {})
        if (item["task_name"] in TASKS and item.get("success")
                and not item.get("collided") and pair.get("same_initial_state")):
            with np.load(path, allow_pickle=False) as episode:
                donors.append({
                    "path": path,
                    "base_episode_id": int(pair["base_episode_id"]),
                    "state": np.asarray(episode["observation.state"], dtype=np.float32),
                    "action": np.asarray(episode["action"], dtype=np.float32),
                })
    if not donors:
        raise RuntimeError("verified safe donor trajectories not found")

    # Gain converts Cartesian tracking error in metres to normalized OSC action.
    gains = (0.25, 0.5, 1.0, 2.0, 4.0)
    jitters = ((0.0, 0.0), (0.001, 0.0), (-0.001, 0.0),
               (0.0, 0.001), (0.0, -0.001))
    results = []
    for pair_index, ((task_id, ordinal), unsafe_path) in enumerate(sorted(unsafe.items())):
        with np.load(unsafe_path, allow_pickle=False) as episode:
            unsafe_state = np.asarray(episode["observation.state"], dtype=np.float32)
            unsafe_action = np.asarray(episode["action"], dtype=np.float32)
        unsafe_grasp = unsafe_state[grasp_index(unsafe_action), :3]
        info = infos[task_id]
        states = manager.get_perturbed_init_states(
            info, PerturbationConfig(), ordinal + 1
        )
        accepted = []
        accepted_action_hashes: set[str] = set()
        attempts = []
        candidates = ([(donor, 0.0, (0.0, 0.0)) for donor in donors] + [
            (donor, gain, jitter)
            for gain in gains for jitter in jitters for donor in donors
        ])[:args.max_attempts_per_init]
        for candidate_index, (donor, gain, jitter) in enumerate(candidates):
            donor_grasp_index = grasp_index(donor["action"])
            offset = unsafe_grasp - donor["state"][donor_grasp_index, :3]
            offset[2] = 0.0
            if donor["base_episode_id"] == ordinal:
                offset[:] = 0.0
            offset[:2] += np.asarray(jitter, dtype=np.float32)

            def controller(step, obs, action, *, _gain=gain, _offset=offset,
                           _states=donor["state"]):
                index = min(step, len(_states) - 1)
                current = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                desired = _states[index, :3] + _offset
                corrected = action.copy()
                corrected[:3] += _gain * (desired - current)
                corrected[:6] = np.clip(corrected[:6], -1.0, 1.0)
                return corrected

            tag = f"{task_id}__init{ordinal:02d}__candidate{candidate_index:03d}"
            record_dir = args.output_root / tag
            outcome = replay_tool.replay(
                info, states[ordinal], donor["action"], rl_record_dir=record_dir,
                episode_id=ordinal, camera_size=256, action_transform=controller,
                collection_context={
                    "method": "closed_loop_safe_path_tracking",
                    "unsafe_pair": str(unsafe_path.resolve()),
                    "donor": str(donor["path"].resolve()),
                    "feedback_gain": gain,
                    "xy_alignment_m": offset[:2].tolist(),
                    "xy_jitter_m": list(jitter),
                    "same_initial_state": True,
                    "init_state_ordinal": ordinal,
                },
            )
            attempt = {
                "candidate": candidate_index, "donor": str(donor["path"]),
                "gain": gain, "jitter": jitter, **outcome,
            }
            attempts.append(attempt)
            print(f"[closed-loop] {pair_index + 1}/{expected} init={ordinal} "
                  f"candidate={candidate_index} gain={gain:g} jitter={jitter} "
                  f"success={outcome['accepted']} collision={outcome['collided']}", flush=True)
            if outcome["accepted"] and not outcome["collided"]:
                safe_path = Path(outcome["rl_record_path"])
                with np.load(safe_path, allow_pickle=False) as safe_episode:
                    action_hash = hashlib.sha256(
                        np.asarray(safe_episode["action"], dtype=np.float32).tobytes()
                    ).hexdigest()
                if action_hash in accepted_action_hashes:
                    attempt["duplicate_safe_action"] = True
                    continue
                provenance = {
                    "unsafe": str(unsafe_path.resolve()),
                    "safe": str(safe_path.resolve()),
                    "same_initial_state": True,
                    "task_id": task_id, "init_state_ordinal": ordinal,
                    "method": "closed_loop_safe_path_tracking",
                    "donor": str(donor["path"].resolve()),
                    "feedback_gain": gain, "xy_jitter_m": list(jitter),
                }
                annotate(safe_path, provenance)
                accepted_action_hashes.add(action_hash)
                accepted.append(provenance)
                if len(accepted) >= args.safe_per_init:
                    break
        results.append({
            "task_id": task_id, "init_state_ordinal": ordinal,
            "unsafe": str(unsafe_path), "accepted": accepted, "attempts": attempts,
        })

    summary = {
        "protocol": "closed_loop_same_init_ramekin_safe_collection",
        "requested_safe_per_init": args.safe_per_init,
        "n_initial_states": len(results),
        "n_initial_states_with_safe": sum(bool(row["accepted"]) for row in results),
        "n_unique_safe_trajectories": sum(len(row["accepted"]) for row in results),
        "results": results,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
