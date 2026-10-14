"""Phase A file audit. No new rollouts.

Rebuilds the P2 versus horizon-100 delay-sweep pairing from the saved CSVs
and writes paired_P2_sweep_comparison.csv plus phaseA_stats.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

OLD = Path(__file__).resolve().parents[2] / "corrective_2026-10-01"
OUT = Path(__file__).resolve().parents[1]
ROOT = OLD.parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(rows: list[dict], key_fn, label: str) -> dict:
    out = {}
    dupes = []
    for row in rows:
        key = key_fn(row)
        if key in out:
            dupes.append(key)
        out[key] = row
    if dupes:
        raise SystemExit(f"{label} has {len(dupes)} duplicate keys, first {dupes[:5]}")
    return out


def coord_equal(a: str, b: str) -> bool:
    return a == b


def main() -> None:
    p2_rows = read_csv(OLD / "P2_delay_only.csv")
    sweep_rows = read_csv(OLD / "delay_sweep_h100_raw.csv")
    p0_rows = read_csv(OLD / "P0_baseline.csv")
    p2 = index_unique(
        p2_rows,
        lambda r: (int(r["model_seed"]), int(r["episode_id"])),
        "P2",
    )
    sweep = index_unique(
        sweep_rows,
        lambda r: (int(r["policy_seed"]), int(r["episode_id"]), int(r["delay_steps"])),
        "delay sweep",
    )
    p0 = index_unique(
        p0_rows,
        lambda r: (int(r["model_seed"]), int(r["episode_id"])),
        "P0",
    )
    if len(p2) != 1800 or len(p0) != 1800 or len(sweep) != 18000:
        raise SystemExit(f"unexpected counts P2={len(p2)} P0={len(p0)} sweep={len(sweep)}")

    fields = [
        "policy_seed",
        "episode_id",
        "delay_steps",
        "reset_seed_p2",
        "reset_seed_sweep",
        "reset_seed_expected",
        "horizon_p2",
        "horizon_sweep",
        "success_p2",
        "success_sweep",
        "success_mismatch",
        "steps_p2",
        "steps_sweep",
        "steps_mismatch",
        "initial_object_x_p2",
        "initial_object_y_p2",
        "initial_object_z_p2",
        "initial_object_x_sweep",
        "initial_object_y_sweep",
        "initial_object_z_sweep",
        "goal_x_p2",
        "goal_y_p2",
        "goal_z_p2",
        "target_x_sweep",
        "target_y_sweep",
        "target_z_sweep",
        "stored_coordinates_equal",
        "initial_distance_p2",
        "initial_distance_sweep",
        "final_distance_p2",
        "final_distance_sweep",
        "mass_p2",
        "friction_p2",
        "damping_scale_p2",
        "gain_scale_p2",
        "near_start_p2",
    ]
    paired_path = OUT / "paired_P2_sweep_comparison.csv"
    success_mm = []
    steps_mm = []
    by_seed_success = Counter()
    by_seed_steps = Counter()
    by_delay_success = Counter()
    by_delay_steps = Counter()
    direction = Counter()
    missing = []
    coord_bad = 0
    reset_bad = 0
    horizon_bad = 0
    physics_not_default = 0
    p2_success = 0
    sweep_success = 0
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in (0, 42, 123):
            for episode_id in range(600):
                left = p2[(seed, episode_id)]
                delay = int(float(left["delay_steps"]))
                right = sweep.get((seed, episode_id, delay))
                if right is None:
                    missing.append((seed, episode_id, delay))
                    continue
                coords = [
                    (left["initial_object_x"], right["initial_object_x"]),
                    (left["initial_object_y"], right["initial_object_y"]),
                    (left["initial_object_z"], right["initial_object_z"]),
                    (left["goal_x"], right["target_x"]),
                    (left["goal_y"], right["target_y"]),
                    (left["goal_z"], right["target_z"]),
                ]
                coordinates_equal = all(coord_equal(a, b) for a, b in coords)
                if not coordinates_equal:
                    coord_bad += 1
                expected_reset = 4_000_000 + episode_id
                if int(left["reset_seed"]) != expected_reset or int(right["reset_seed"]) != expected_reset:
                    reset_bad += 1
                if int(left["horizon"]) != 100 or int(right["horizon"]) != 100:
                    horizon_bad += 1
                if not (
                    left["mass"] == "1.00000000"
                    and left["friction"] == "1.00000000"
                    and left["damping_scale"] == "1.00000000"
                    and left["actuator_gain_scale"] == "1.00000000"
                ):
                    physics_not_default += 1
                s_left = int(left["success"])
                s_right = int(right["success"])
                st_left = int(left["steps_used"])
                st_right = int(right["steps_used"])
                p2_success += s_left
                sweep_success += s_right
                sm = int(s_left != s_right)
                tm = int(st_left != st_right)
                if sm:
                    success_mm.append((seed, episode_id, delay))
                    by_seed_success[seed] += 1
                    by_delay_success[delay] += 1
                    direction[f"p2_{s_left}_sweep_{s_right}"] += 1
                if tm:
                    steps_mm.append((seed, episode_id, delay))
                    by_seed_steps[seed] += 1
                    by_delay_steps[delay] += 1
                initial = float(left["initial_object_to_goal_distance"])
                writer.writerow(
                    {
                        "policy_seed": seed,
                        "episode_id": episode_id,
                        "delay_steps": delay,
                        "reset_seed_p2": left["reset_seed"],
                        "reset_seed_sweep": right["reset_seed"],
                        "reset_seed_expected": expected_reset,
                        "horizon_p2": left["horizon"],
                        "horizon_sweep": right["horizon"],
                        "success_p2": s_left,
                        "success_sweep": s_right,
                        "success_mismatch": sm,
                        "steps_p2": st_left,
                        "steps_sweep": st_right,
                        "steps_mismatch": tm,
                        "initial_object_x_p2": left["initial_object_x"],
                        "initial_object_y_p2": left["initial_object_y"],
                        "initial_object_z_p2": left["initial_object_z"],
                        "initial_object_x_sweep": right["initial_object_x"],
                        "initial_object_y_sweep": right["initial_object_y"],
                        "initial_object_z_sweep": right["initial_object_z"],
                        "goal_x_p2": left["goal_x"],
                        "goal_y_p2": left["goal_y"],
                        "goal_z_p2": left["goal_z"],
                        "target_x_sweep": right["target_x"],
                        "target_y_sweep": right["target_y"],
                        "target_z_sweep": right["target_z"],
                        "stored_coordinates_equal": int(coordinates_equal),
                        "initial_distance_p2": left["initial_object_to_goal_distance"],
                        "initial_distance_sweep": right["initial_distance"],
                        "final_distance_p2": left["final_object_to_goal_distance"],
                        "final_distance_sweep": right["final_distance"],
                        "mass_p2": left["mass"],
                        "friction_p2": left["friction"],
                        "damping_scale_p2": left["damping_scale"],
                        "gain_scale_p2": left["actuator_gain_scale"],
                        "near_start_p2": int(initial < 0.05),
                    }
                )

    p0_match = {"n": 0, "success_mismatch": 0, "steps_mismatch": 0, "coord_mismatch": 0, "p0_success": 0, "sweep_success": 0}
    for seed in (0, 42, 123):
        for episode_id in range(600):
            left = p0[(seed, episode_id)]
            right = sweep[(seed, episode_id, 0)]
            p0_match["n"] += 1
            p0_match["p0_success"] += int(left["success"])
            p0_match["sweep_success"] += int(right["success"])
            if int(left["success"]) != int(right["success"]):
                p0_match["success_mismatch"] += 1
            if int(left["steps_used"]) != int(right["steps_used"]):
                p0_match["steps_mismatch"] += 1
            coords = [
                (left["initial_object_x"], right["initial_object_x"]),
                (left["initial_object_y"], right["initial_object_y"]),
                (left["initial_object_z"], right["initial_object_z"]),
                (left["goal_x"], right["target_x"]),
                (left["goal_y"], right["target_y"]),
                (left["goal_z"], right["target_z"]),
            ]
            if not all(a == b for a, b in coords):
                p0_match["coord_mismatch"] += 1

    def partial_agrees(official: Path, partials: list[Path], seed_field: str) -> dict:
        official_rows = read_csv(official)
        partial_rows = []
        for path in partials:
            partial_rows.extend(read_csv(path))
        def key(row):
            seed = int(row.get(seed_field, row.get("policy_seed", row.get("model_seed"))))
            extra = int(row["delay_steps"]) if "delay_steps" in row and seed_field == "policy_seed" else 0
            horizon = int(row["horizon"]) if "horizon" in row and "L" in path.name else 0
            return (seed, int(row["episode_id"]), extra, horizon, path.name if False else "")
        # Compare multiset of full row tuples after aligning seed column name.
        def norm(row, source_name):
            data = dict(row)
            if "model_seed" in data and "policy_seed" not in data:
                data["policy_seed"] = data.pop("model_seed")
            data["_source_name"] = source_name
            return tuple(sorted(data.items()))
        official_norm = [norm(row, "official") for row in official_rows]
        # Official files are concatenations; compare by key equality of the raw dicts
        # after renaming model_seed. For P2, partial uses model_seed and official does too.
        return {
            "official_rows": len(official_rows),
            "partial_rows": len(partial_rows),
            "equal": official_rows == partial_rows,
        }

    p2_partials = [OLD / "partial" / f"P2_seed{s}.csv" for s in (0, 42, 123)]
    # Official P2 is not necessarily partial concatenation order. Compare as keyed maps.
    def rows_equal_keyed(official_rows, partial_paths, key_fields):
        partial_rows = []
        for path in partial_paths:
            partial_rows.extend(read_csv(path))
        def key(row):
            return tuple(row[field] for field in key_fields)
        left = index_unique(official_rows, key, "official")
        right = index_unique(partial_rows, key, "partial")
        only_left = set(left) - set(right)
        only_right = set(right) - set(left)
        value_diff = 0
        for key_value in set(left) & set(right):
            if left[key_value] != right[key_value]:
                value_diff += 1
        return {
            "official": len(left),
            "partial": len(right),
            "only_official": len(only_left),
            "only_partial": len(only_right),
            "value_diff": value_diff,
        }

    p2_vs_partial = rows_equal_keyed(
        p2_rows,
        p2_partials,
        ["model_seed", "episode_id"],
    )
    sweep_partials = [
        OLD / "partial" / "round2" / f"delay_h100_d{d}_seed{s}.csv"
        for s in (0, 42, 123)
        for d in range(10)
    ]
    sweep_vs_partial = rows_equal_keyed(
        sweep_rows,
        sweep_partials,
        ["policy_seed", "episode_id", "delay_steps"],
    )

    log_report = {}
    for path in sorted((OLD / "logs").glob("phase*.log")):
        text = path.read_text(errors="replace")
        starts = re.findall(r"from episode (\d+)", text)
        progresses = [int(x) for x in re.findall(r"(\d+)/600", text)]
        log_report[path.name] = {
            "bytes": path.stat().st_size,
            "from_episode": starts,
            "progress_count": len(progresses),
            "progress_monotonic": progresses == sorted(progresses),
            "ends_at_600_blocks": progresses[-6:] if progresses else [],
            "has_traceback": "Traceback" in text,
        }

    both = set(success_mm) & set(steps_mm)
    success_only = set(success_mm) - set(steps_mm)
    steps_only = set(steps_mm) - set(success_mm)
    stats = {
        "paired_rows": 1800 - len(missing),
        "missing_sweep_keys": missing,
        "p2_successes": p2_success,
        "matched_sweep_successes": sweep_success,
        "success_mismatch": len(success_mm),
        "steps_mismatch": len(steps_mm),
        "success_and_steps": len(both),
        "success_only": len(success_only),
        "steps_only": len(steps_only),
        "by_seed_success": dict(by_seed_success),
        "by_seed_steps": dict(by_seed_steps),
        "by_delay_success": dict(sorted(by_delay_success.items())),
        "by_delay_steps": dict(sorted(by_delay_steps.items())),
        "direction": dict(direction),
        "stored_coordinate_mismatches": coord_bad,
        "reset_seed_mismatches": reset_bad,
        "horizon_mismatches": horizon_bad,
        "p2_physics_not_default": physics_not_default,
        "p0_vs_delay0": p0_match,
        "p2_vs_partial": p2_vs_partial,
        "sweep_vs_partial": sweep_vs_partial,
        "logs": log_report,
    }
    (OUT / "phaseA_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: stats[k] for k in stats if k != "logs"}, indent=2, ensure_ascii=False))
    print("logs", json.dumps(log_report, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
