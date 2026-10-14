"""Phase D: small clean-reset recheck of the existing delay and spatial curves.

Scenes were fixed before this run: episode ids 0,60,120,...,540.
Delay keeps reset_seed = 4000000 + episode_id and horizon 100.
Spatial keeps reset_seed = 6000000 + round(L*100)*1000 + episode_id.
Default physics only. This does not replace the full scans.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import phaseB_diagnose as pb
from round2_common import set_goal_side

OLD = pb.ROOT / "data_and_analysis" / "corrective_2026-10-01"
OUT = pb.OUT
EPISODES = (0, 60, 120, 180, 240, 300, 360, 420, 480, 540)
DELAYS = (0, 1, 2, 9)
SIDES = (0.3, 0.8)
HORIZONS = (50, 100)
DEFAULT_PHYSICS = {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}


def spatial_seed(side: float, episode_id: int) -> int:
    return 6_000_000 + int(round(side * 100)) * 1000 + int(episode_id)


def load_delay() -> dict:
    rows = {}
    with (OLD / "delay_sweep_h100_raw.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["delay_steps"]) not in DELAYS:
                continue
            if int(row["episode_id"]) not in EPISODES:
                continue
            rows[(int(row["policy_seed"]), int(row["episode_id"]), int(row["delay_steps"]))] = row
    return rows


def load_spatial() -> dict:
    rows = {}
    for horizon, name in ((100, "spatial_h100_raw.csv"), (50, "spatial_h50_raw.csv")):
        with (OLD / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                tenths = int(round(float(row["L"]) * 100))
                if tenths not in (30, 80):
                    continue
                if int(row["episode_id"]) not in EPISODES:
                    continue
                rows[(int(row["policy_seed"]), int(row["episode_id"]), horizon, tenths)] = row
    return rows


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    found = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            found.add((row["family"], row["policy_seed"], row["episode_id"], row["delay_steps"], row["L"], row["horizon"]))
    return found


def make_env(horizon: int):
    import gymnasium as gym

    env = gym.make("FrankaPushSparse-v0", max_episode_steps=horizon, disable_env_checker=True)
    node = env
    found = False
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            node._max_episode_steps = horizon
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit missing")
    return env, env.unwrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--family", choices=("delay", "spatial", "all"), default="all")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    delay_old = load_delay()
    spatial_old = load_spatial()
    dest = OUT / f"phaseD_recheck_seed{args.seed}.csv"
    finished = done_keys(dest)
    template, template_base = make_env(100)
    pristine = pb.snapshot_model(template_base)
    model = pb.load_policy(args.seed, template)
    ids = {
        "obj_body": int(template_base._model_names.body_name2id["obj"]),
        "obj_geom": int(template_base._model_names.geom_name2id["obj_geom"]),
    }
    fields = [
        "family", "policy_seed", "episode_id", "reset_seed", "delay_steps", "L", "horizon",
        "success", "steps", "final_distance",
        "initial_x", "initial_y", "initial_z", "goal_x", "goal_y", "goal_z",
        "old_success", "old_steps", "old_final_distance",
        "old_initial_x", "old_initial_y", "old_initial_z", "old_goal_x", "old_goal_y", "old_goal_z",
        "success_agree", "steps_agree",
    ]
    new_file = not dest.exists()
    families = ("delay", "spatial") if args.family == "all" else (args.family,)
    with dest.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for family in families:
            for episode_id in EPISODES:
                jobs = []
                if family == "delay":
                    for delay in DELAYS:
                        jobs.append((delay, "", 100, 4_000_000 + episode_id, None))
                else:
                    for side in SIDES:
                        for horizon in HORIZONS:
                            jobs.append((0, f"{side:.1f}", horizon, spatial_seed(side, episode_id), side))
                for delay, side_text, horizon, reset_seed, side in jobs:
                    key = (family, str(args.seed), str(episode_id), str(delay), side_text, str(horizon))
                    if key in finished:
                        continue
                    run_env, run_base = make_env(horizon)
                    if side is not None:
                        set_goal_side(run_base, side)
                    pb.install_clean_reset(run_base, pristine)
                    bundle = pb.run_episode(
                        run_env, run_base, model, ids, reset_seed, delay, DEFAULT_PHYSICS, pb.PROTOCOL_CLEAN, horizon
                    )
                    achieved = bundle["achieved_after_reset"]
                    desired = bundle["desired_after_reset"]
                    if family == "delay":
                        old = delay_old[(args.seed, episode_id, delay)]
                    else:
                        old = spatial_old[(args.seed, episode_id, horizon, int(round(side * 100)))]
                    row = {
                        "family": family,
                        "policy_seed": args.seed,
                        "episode_id": episode_id,
                        "reset_seed": reset_seed,
                        "delay_steps": delay,
                        "L": side_text,
                        "horizon": horizon,
                        "success": int(bundle["success"][0]),
                        "steps": int(bundle["steps"][0]),
                        "final_distance": f"{float(bundle['final_distance'][0]):.16g}",
                        "initial_x": f"{float(achieved[0]):.16g}",
                        "initial_y": f"{float(achieved[1]):.16g}",
                        "initial_z": f"{float(achieved[2]):.16g}",
                        "goal_x": f"{float(desired[0]):.16g}",
                        "goal_y": f"{float(desired[1]):.16g}",
                        "goal_z": f"{float(desired[2]):.16g}",
                        "old_success": int(old["success"]),
                        "old_steps": int(old["steps_used"]),
                        "old_final_distance": old["final_distance"],
                        "old_initial_x": old["initial_object_x"],
                        "old_initial_y": old["initial_object_y"],
                        "old_initial_z": old["initial_object_z"],
                        "old_goal_x": old["target_x"],
                        "old_goal_y": old["target_y"],
                        "old_goal_z": old["target_z"],
                        "success_agree": int(int(bundle["success"][0]) == int(old["success"])),
                        "steps_agree": int(int(bundle["steps"][0]) == int(old["steps_used"])),
                    }
                    writer.writerow(row)
                    handle.flush()
                    run_env.close()
                    print(
                        f"[seed {args.seed}] {family} ep {episode_id} d {delay} L {side_text or '-'} H {horizon} "
                        f"new {row['success']}/{row['steps']} old {row['old_success']}/{row['old_steps']}",
                        flush=True,
                    )
    template.close()


if __name__ == "__main__":
    main()
