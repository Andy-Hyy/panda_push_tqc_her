"""Summarize a completed clean-reset spatial matrix.

Refuses to treat an incomplete CSV as a result. Reads the new rows and the
2026-10-01 spatial_h50_raw.csv / spatial_h100_raw.csv. Does not modify them.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
OLD = OUT.parent / "corrective_2026-10-01"
SIDES = ("0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80")
SEEDS = (0, 42, 123)
COORD_ATOL = 1e-8
DISTANCE_BINS = ((0.0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, math.inf))
RADIUS_BINS = ((0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, math.inf))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 4800:
        raise SystemExit(f"{path.name} has {len(rows)} rows; refusing to summarize an incomplete file")
    return rows


def side_text(value: str) -> str:
    return format(float(value), ".2f")


def index_old(path: Path) -> dict:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["policy_seed"]), side_text(row["L"]), int(row["episode_id"]))] = row
    if len(rows) != 14400:
        raise SystemExit(f"{path.name} has {len(rows)} rows")
    return rows


def fnum(row: dict, key: str) -> float:
    return float(row[key])


def in_bin(value: float, low: float, high: float) -> bool:
    return low <= value < high


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fraction_csv(path: Path, rates: dict) -> None:
    with path.open("w", newline="") as handle:
        handle.write("," + ",".join(str(float(side)) for side in SIDES) + "\n")
        for seed in SEEDS:
            handle.write(str(seed) + "," + ",".join(format(rates[(seed, side)], ".17g") for side in SIDES) + "\n")


def main() -> None:
    grouped = {}
    problems = []
    for seed in SEEDS:
        rows = read_rows(OUT / f"spatial_seed{seed}.csv")
        seen = set()
        for row in rows:
            key = (row["L"], int(row["episode_id"]))
            if key in seen or row["L"] not in SIDES:
                problems.append(f"bad key seed {seed} {key}")
            seen.add(key)
            if row["horizon_source"] != "trajectory_prefix":
                problems.append(f"horizon source {row['horizon_source']}")
            if row["delay_steps"] != "0" or float(row["latency_s"]) != 0.0:
                problems.append("delay is not zero")
            if float(row["time_after_reset"]) != 0.0 or row["control_restored"] != "1":
                problems.append("reset time or control check failed")
            if row["goal_unchanged"] != "1" or row["model_check_ok"] != "1" or row["writer"] != "C":
                problems.append("model or goal audit failed")
            for name in ("mass", "friction", "damping_scale", "gain_scale"):
                if abs(float(row[name]) - 1.0) > 1e-12 or abs(float(row[name + "_read"]) - 1.0) > 1e-8:
                    problems.append(f"{name} is not nominal")
            success_h50 = int(row["success_h50"])
            success_h100 = int(row["success_h100"])
            if success_h50 and not success_h100:
                problems.append("H50 success without H100 success")
            if success_h50 != int(float(row["distance_h50"]) < 0.05):
                problems.append("H50 flag disagrees with distance")
            if success_h100 != int(float(row["distance_h100"]) < 0.05):
                problems.append("H100 flag disagrees with distance")
            if success_h50 == 0:
                state = OUT / row["state_h50"]
                if not row["state_h50"] or not state.exists():
                    problems.append(f"missing step-50 state {row['state_h50']}")
            if success_h100 == 0:
                state = OUT / row["state_h100"]
                if not row["state_h100"] or not state.exists():
                    problems.append(f"missing step-100 state {row['state_h100']}")
            if int(row["initial_near"]) != int(float(row["initial_distance"]) < 0.05):
                problems.append("initial_near disagrees with initial distance")
            grouped[(seed, row["L"], int(row["episode_id"]))] = row
        if len(seen) != 4800:
            problems.append(f"seed {seed} unique rows {len(seen)}")
    for side in SIDES:
        for episode_id in range(600):
            base = grouped[(0, side, episode_id)]
            for seed in (42, 123):
                other = grouped[(seed, side, episode_id)]
                for key in (
                    "initial_object_x",
                    "initial_object_y",
                    "initial_object_z",
                    "target_x",
                    "target_y",
                    "target_z",
                    "reset_seed",
                ):
                    if abs(float(base[key]) - float(other[key])) > 0.0:
                        problems.append(f"policies differ at L={side} ep {episode_id} {key}")
                        break
    old50 = index_old(OLD / "spatial_h50_raw.csv")
    old100 = index_old(OLD / "spatial_h100_raw.csv")
    changes = []
    coord_bad = 0
    cell_rows = []
    summary_rows = []
    added_rows = []
    near_rows = []
    rates50 = {}
    rates100 = {}
    for seed in SEEDS:
        for side in SIDES:
            keys = [(seed, side, episode_id) for episode_id in range(600)]
            new_rows = [grouped[key] for key in keys]
            n50 = sum(int(row["success_h50"]) for row in new_rows)
            n100 = sum(int(row["success_h100"]) for row in new_rows)
            old_n50 = sum(int(old50[key]["success"]) for key in keys)
            old_n100 = sum(int(old100[key]["success"]) for key in keys)
            flips50 = flips100 = step50 = step100 = 0
            for key, row in zip(keys, new_rows):
                for name, new_value, old_value in (
                    ("initial_object_x", row["initial_object_x"], old50[key]["initial_object_x"]),
                    ("initial_object_y", row["initial_object_y"], old50[key]["initial_object_y"]),
                    ("initial_object_z", row["initial_object_z"], old50[key]["initial_object_z"]),
                    ("target_x", row["target_x"], old50[key]["target_x"]),
                    ("target_y", row["target_y"], old50[key]["target_y"]),
                    ("target_z", row["target_z"], old50[key]["target_z"]),
                ):
                    if abs(float(new_value) - float(old_value)) > COORD_ATOL:
                        coord_bad += 1
                        problems.append(f"coordinate {name} at {key} differs by more than 1e-8")
                        break
                if int(row["success_h50"]) != int(old50[key]["success"]):
                    flips50 += 1
                if int(row["success_h100"]) != int(old100[key]["success"]):
                    flips100 += 1
                if int(row["steps_h50"]) != int(old50[key]["steps_used"]):
                    step50 += 1
                if int(row["steps_h100"]) != int(old100[key]["steps_used"]):
                    step100 += 1
                if (
                    int(row["success_h50"]) != int(old50[key]["success"])
                    or int(row["success_h100"]) != int(old100[key]["success"])
                    or int(row["steps_h50"]) != int(old50[key]["steps_used"])
                    or int(row["steps_h100"]) != int(old100[key]["steps_used"])
                ):
                    changes.append(
                        {
                            "policy_seed": seed,
                            "L": side,
                            "episode_id": key[2],
                            "old_success_h50": old50[key]["success"],
                            "new_success_h50": row["success_h50"],
                            "old_steps_h50": old50[key]["steps_used"],
                            "new_steps_h50": row["steps_h50"],
                            "old_success_h100": old100[key]["success"],
                            "new_success_h100": row["success_h100"],
                            "old_steps_h100": old100[key]["steps_used"],
                            "new_steps_h100": row["steps_h100"],
                            "initial_distance": row["initial_distance"],
                        }
                    )
            rates50[(seed, side)] = n50 / 600
            rates100[(seed, side)] = n100 / 600
            for horizon, new_n, old_n, flips, steps in (
                ("H50", n50, old_n50, flips50, step50),
                ("H100", n100, old_n100, flips100, step100),
            ):
                cell_rows.append(
                    {
                        "policy_seed": seed,
                        "L": side,
                        "horizon": horizon,
                        "n": 600,
                        "new_success": new_n,
                        "old_success": old_n,
                        "new_rate": format(new_n / 600, ".17g"),
                        "old_rate": format(old_n / 600, ".17g"),
                        "rate_difference_pp": format((new_n - old_n) / 600 * 100, ".17g"),
                        "success_flips": flips,
                        "step_changes": steps,
                    }
                )
            added = sum(int(row["success_h50"]) == 0 and int(row["success_h100"]) == 1 for row in new_rows)
            near = [row for row in new_rows if int(row["initial_near"]) == 1]
            rest = [row for row in new_rows if int(row["initial_near"]) == 0]
            added_rows.append({"policy_seed": seed, "L": side, "added_successes_h50_to_h100": added, "n": 600})
            for horizon, attr in (("H50", "success_h50"), ("H100", "success_h100")):
                near_rows.append(
                    {
                        "policy_seed": seed,
                        "L": side,
                        "horizon": horizon,
                        "near_n": len(near),
                        "near_success": sum(int(row[attr]) for row in near),
                        "rest_n": len(rest),
                        "rest_success": sum(int(row[attr]) for row in rest),
                        "official_success": sum(int(row[attr]) for row in new_rows),
                        "official_n": 600,
                    }
                )
    for side in SIDES:
        for horizon, rates in (("H50", rates50), ("H100", rates100)):
            values = [rates[(seed, side)] for seed in SEEDS]
            summary_rows.append(
                {
                    "L": side,
                    "horizon": horizon,
                    "seed0_success": int(round(values[0] * 600)),
                    "seed42_success": int(round(values[1] * 600)),
                    "seed123_success": int(round(values[2] * 600)),
                    "mean_rate": format(float(np.mean(values)), ".17g"),
                    "sample_sd_pp": format(sample_sd(values) * 100, ".17g"),
                    "added_h50_to_h100": sum(
                        int(row["added_successes_h50_to_h100"]) for row in added_rows if row["L"] == side
                    )
                    if horizon == "H100"
                    else "",
                }
            )
    distance_rows = []
    radius_rows = []
    direction_rows = []
    for horizon, attr in (("H50", "success_h50"), ("H100", "success_h100")):
        for low, high in DISTANCE_BINS:
            chosen = [
                row
                for row in grouped.values()
                if in_bin(float(row["initial_distance"]), low, high)
            ]
            distance_rows.append(
                {
                    "horizon": horizon,
                    "bin": f"[{low:.2f},{high if math.isfinite(high) else 'inf'})",
                    "n": len(chosen),
                    "success": sum(int(row[attr]) for row in chosen),
                    "rate": format(sum(int(row[attr]) for row in chosen) / len(chosen), ".17g") if chosen else "",
                    "note": "exploratory",
                }
            )
        for low, high in RADIUS_BINS:
            chosen = []
            for row in grouped.values():
                radius = math.hypot(float(row["target_x"]) - 0.6, float(row["target_y"]))
                if in_bin(radius, low, high):
                    chosen.append(row)
            radius_rows.append(
                {
                    "horizon": horizon,
                    "target_radius_bin": f"[{low:.2f},{high if math.isfinite(high) else 'inf'})",
                    "n": len(chosen),
                    "success": sum(int(row[attr]) for row in chosen),
                    "rate": format(sum(int(row[attr]) for row in chosen) / len(chosen), ".17g") if chosen else "",
                    "note": "exploratory",
                }
            )
        quadrants = defaultdict(list)
        for row in grouped.values():
            angle = math.atan2(float(row["target_y"]) - float(row["initial_object_y"]), float(row["target_x"]) - float(row["initial_object_x"]))
            label = ("E", "N", "W", "S")[int((angle + math.pi / 4) % (2 * math.pi) // (math.pi / 2)) % 4]
            quadrants[label].append(row)
        for label in ("E", "N", "W", "S"):
            chosen = quadrants[label]
            direction_rows.append(
                {
                    "horizon": horizon,
                    "push_quadrant": label,
                    "n": len(chosen),
                    "success": sum(int(row[attr]) for row in chosen),
                    "rate": format(sum(int(row[attr]) for row in chosen) / len(chosen), ".17g") if chosen else "",
                    "note": "exploratory",
                }
            )
    fraction_csv(OUT / "spatial_fraction_h50.csv", rates50)
    fraction_csv(OUT / "spatial_fraction_h100.csv", rates100)
    write_csv(
        OUT / "summary_by_L.csv",
        summary_rows,
        ["L", "horizon", "seed0_success", "seed42_success", "seed123_success", "mean_rate", "sample_sd_pp", "added_h50_to_h100"],
    )
    write_csv(
        OUT / "old_new_cell_comparison.csv",
        cell_rows,
        [
            "policy_seed",
            "L",
            "horizon",
            "n",
            "new_success",
            "old_success",
            "new_rate",
            "old_rate",
            "rate_difference_pp",
            "success_flips",
            "step_changes",
        ],
    )
    write_csv(
        OUT / "old_new_episode_changes.csv",
        changes,
        [
            "policy_seed",
            "L",
            "episode_id",
            "old_success_h50",
            "new_success_h50",
            "old_steps_h50",
            "new_steps_h50",
            "old_success_h100",
            "new_success_h100",
            "old_steps_h100",
            "new_steps_h100",
            "initial_distance",
        ],
    )
    write_csv(
        OUT / "sensitivity_initial_near.csv",
        near_rows,
        ["policy_seed", "L", "horizon", "near_n", "near_success", "rest_n", "rest_success", "official_success", "official_n"],
    )
    write_csv(OUT / "h50_to_h100_added.csv", added_rows, ["policy_seed", "L", "added_successes_h50_to_h100", "n"])
    write_csv(OUT / "exploratory_distance.csv", distance_rows, ["horizon", "bin", "n", "success", "rate", "note"])
    write_csv(
        OUT / "exploratory_target_radius.csv",
        radius_rows,
        ["horizon", "target_radius_bin", "n", "success", "rate", "note"],
    )
    write_csv(
        OUT / "exploratory_direction.csv",
        direction_rows,
        ["horizon", "push_quadrant", "n", "success", "rate", "note"],
    )
    unique_problems = list(dict.fromkeys(problems))
    report = {
        "complete": not unique_problems,
        "coordinate_mismatches_vs_old": coord_bad,
        "episode_rows_with_success_or_step_change": len(changes),
        "problems": unique_problems[:50],
        "problem_count": len(unique_problems),
    }
    (OUT / "summary_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if unique_problems:
        raise SystemExit(f"summary checks failed: {len(unique_problems)}")
    print(f"[summary] checks passed, {len(changes)} old/new episode changes", flush=True)


if __name__ == "__main__":
    main()
