"""Summaries for the controlled SpatialDR and Fixed Large retraining.

This file is not part of the locked protocol hash. It reads the formal
episode logs and the evaluation CSVs. It does not train, evaluate, or
edit those records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
SUMMARY = OUT / "summary"
METHODS = ("spatial_dr", "fixed_large")
SEEDS = (0, 42, 123)
SIDES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
PRIMARY_SIDES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CHECKPOINTS = (199998, 400002, 499998, 600000, 799998, 1000002)
FINAL_STEPS = 1000002
PROTOCOL_SHA = "6c6d6c39a858fbf32cda4c57bd30f766646e7ec4313c7f8a07156203362e90cd"
CENTER = 0.60
BOX_ATOL = 1e-8
IN_BOX = 1e-9


def side_key(value: float) -> float:
    return round(float(value), 2)


def load_training() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        for seed in SEEDS:
            folder = OUT / "runs" / f"{method}_seed{seed}" / "episodes"
            for path in sorted(folder.glob("env*.csv")):
                frame = pd.read_csv(path)
                frame["source"] = str(path.relative_to(OUT))
                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["L"] = data["L"].astype(float)
    return data


def training_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems = []
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            part = data[(data["method"] == method) & (data["train_seed"] == seed)].copy()
            if part.empty:
                problems.append(f"no training episodes for {method} seed {seed}")
                continue
            span_x = part["goal_high_x"] - part["goal_low_x"]
            span_y = part["goal_high_y"] - part["goal_low_y"]
            center_x = (part["goal_high_x"] + part["goal_low_x"]) / 2
            center_y = (part["goal_high_y"] + part["goal_low_y"]) / 2
            bad_box = (
                (np.abs(span_x - part["L"]) > BOX_ATOL)
                | (np.abs(span_y - part["L"]) > BOX_ATOL)
                | (np.abs(center_x - CENTER) > BOX_ATOL)
                | (np.abs(center_y) > BOX_ATOL)
            )
            outside_own = (
                (part["goal_x"] < part["goal_low_x"] - IN_BOX)
                | (part["goal_x"] > part["goal_high_x"] + IN_BOX)
                | (part["goal_y"] < part["goal_low_y"] - IN_BOX)
                | (part["goal_y"] > part["goal_high_y"] + IN_BOX)
            )
            # Training logs store the sampled object, not a second copy of the box.
            # The protocol box is side 0.30 centered at (0.60, 0).
            outside_object = (
                (part["object_x"] < 0.45 - IN_BOX)
                | (part["object_x"] > 0.75 + IN_BOX)
                | (part["object_y"] < -0.15 - IN_BOX)
                | (part["object_y"] > 0.15 + IN_BOX)
            )
            bad_object_box = outside_object
            outside_030 = (np.abs(part["goal_x"] - CENTER) > 0.15 + IN_BOX) | (np.abs(part["goal_y"]) > 0.15 + IN_BOX)
            if method == "spatial_dr":
                bad_L = (part["L"] < 0.30) | (part["L"] >= 0.80)
            else:
                bad_L = np.abs(part["L"] - 0.80) > 0
            reasons = part["termination_reason"].value_counts().to_dict()
            finished = part[part["termination_reason"] != "training_stopped"]
            meta = json.loads((OUT / "runs" / f"{method}_seed{seed}" / "run_meta.json").read_text())
            step_sum = int(part["steps_used"].sum())
            row = {
                "method": method,
                "train_seed": seed,
                "episodes": int(len(part)),
                "L_min": float(part["L"].min()),
                "L_max": float(part["L"].max()),
                "L_mean": float(part["L"].mean()),
                "n_L_eq_0.80": int((part["L"] == 0.80).sum()),
                "n_L_below_0.30": int((part["L"] < 0.30).sum()),
                "n_L_at_or_above_0.80": int((part["L"] >= 0.80).sum()),
                "bad_L": int(bad_L.sum()),
                "bad_goal_box": int(bad_box.sum()),
                "goals_outside_own_box": int(outside_own.sum()),
                "bad_object_box": int(bad_object_box.sum()),
                "objects_outside_box": int(outside_object.sum()),
                "goals_outside_0.30": int(outside_030.sum()),
                "success_finished": int((finished["success"] == 1).sum()),
                "finished_episodes": int(len(finished)),
                "training_stopped": int(reasons.get("training_stopped", 0)),
                "time_limit": int(reasons.get("time_limit", 0)),
                "success_reason": int(reasons.get("success", 0)),
                "steps_sum": step_sum,
                "num_timesteps": int(meta["num_timesteps"]),
                "steps_minus_timesteps": step_sum - int(meta["num_timesteps"]),
                "final_sha256": meta["final_sha256"],
                "started": meta["started"],
                "ended": meta["ended"],
            }
            rows.append(row)
            if int(bad_L.sum()) or int(bad_box.sum()) or int(outside_own.sum()) or int(bad_object_box.sum()) or int(outside_object.sum()):
                problems.append(f"training distribution mismatch {method} seed {seed}")
            if step_sum != int(meta["num_timesteps"]):
                problems.append(
                    f"episode steps {step_sum} differ from num_timesteps {meta['num_timesteps']} for {method} seed {seed}"
                )
            if int(meta["num_timesteps"]) != FINAL_STEPS:
                problems.append(f"{method} seed {seed} stopped at {meta['num_timesteps']}")
    return pd.DataFrame(rows), problems


def load_eval() -> pd.DataFrame:
    frames = []
    for phase in ("validation", "final"):
        for method in METHODS:
            for seed in SEEDS:
                path = OUT / "records" / f"{phase}_{method}_seed{seed}.csv"
                frame = pd.read_csv(
                    path,
                    usecols=[
                        "phase",
                        "method",
                        "train_seed",
                        "checkpoint_steps",
                        "checkpoint_sha256",
                        "L",
                        "episode_id",
                        "reset_seed",
                        "initial_distance",
                        "initial_near",
                        "mass_read",
                        "friction_read",
                        "damping_scale_read",
                        "gain_scale_read",
                        "success",
                        "steps_used",
                        "termination_reason",
                        "protocol_sha256",
                    ],
                )
                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["L"] = data["L"].astype(float).map(side_key)
    data["near_flag"] = (data["initial_distance"] < 0.05).astype(int)
    return data


def eval_problems(data: pd.DataFrame) -> list[str]:
    problems = []
    if set(data["protocol_sha256"].unique()) != {PROTOCOL_SHA}:
        problems.append(f"protocol sha set is {set(data['protocol_sha256'].unique())}")
    if not np.allclose(data[["mass_read", "friction_read", "damping_scale_read", "gain_scale_read"]], 1.0):
        problems.append("physics readback is not nominal")
    if not (data["near_flag"] == data["initial_near"].astype(int)).all():
        problems.append("initial_near disagrees with initial_distance < 0.05")
    if set(data["success"].unique()) != {0, 1}:
        problems.append(f"success values {set(data['success'].unique())}")
    if set(data["termination_reason"].unique()) != {"success", "time_limit"}:
        problems.append(f"termination values {set(data['termination_reason'].unique())}")
    if (data["steps_used"] < 1).any():
        problems.append("an episode took no step")
    final = data[data["phase"] == "final"]
    validation = data[data["phase"] == "validation"]
    if set(final["checkpoint_steps"].unique()) != {FINAL_STEPS}:
        problems.append("final test used a checkpoint other than 1000002")
    if set(validation["checkpoint_steps"].unique()) != set(CHECKPOINTS):
        problems.append("validation is missing a predetermined checkpoint")
    if len(final) != 28800 or len(validation) != 28800:
        problems.append(f"row counts validation {len(validation)} final {len(final)}")
    for method in METHODS:
        for seed in SEEDS:
            manifest = json.loads((OUT / "runs" / f"{method}_seed{seed}" / "checkpoints_manifest.json").read_text())
            expected = {int(item["steps"]): item["sha256"] for item in manifest["checkpoints"]}
            part = data[(data["method"] == method) & (data["train_seed"] == seed)]
            for steps, sha in part.groupby("checkpoint_steps")["checkpoint_sha256"].unique().items():
                if list(sha) != [expected[int(steps)]]:
                    problems.append(f"sha mismatch {method} seed {seed} steps {steps}")
    near_counts = (
        final.groupby(["L", "method", "train_seed"])["near_flag"].sum().unstack(["method", "train_seed"])
    )
    if near_counts.nunique(axis=1).max() != 1:
        problems.append("near-start counts differ across models at the same L")
    return problems


def rate_table(part: pd.DataFrame) -> pd.DataFrame:
    grouped = part.groupby(["method", "train_seed", "L"], as_index=False).agg(
        successes=("success", "sum"),
        episodes=("success", "size"),
    )
    grouped["rate"] = grouped["successes"] / grouped["episodes"]
    kept = part[part["near_flag"] == 0]
    excluded = kept.groupby(["method", "train_seed", "L"], as_index=False).agg(
        successes_excluding_near=("success", "sum"),
        episodes_excluding_near=("success", "size"),
    )
    merged = grouped.merge(excluded, on=["method", "train_seed", "L"], how="left")
    merged["near_count"] = merged["episodes"] - merged["episodes_excluding_near"]
    merged["rate_excluding_near"] = merged["successes_excluding_near"] / merged["episodes_excluding_near"]
    return merged


def method_table(by_seed: pd.DataFrame, rate_column: str) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        for method in METHODS:
            rates = by_seed[(by_seed["L"] == side) & (by_seed["method"] == method)].sort_values("train_seed")
            values = rates[rate_column].to_numpy(dtype=float)
            rows.append(
                {
                    "L": side,
                    "method": method,
                    "mean_rate": float(values.mean()),
                    "sd_pp": float(values.std(ddof=1) * 100),
                    "pooled_successes": int(rates["successes"].sum()) if rate_column == "rate" else int(rates["successes_excluding_near"].sum()),
                    "pooled_episodes": int(rates["episodes"].sum()) if rate_column == "rate" else int(rates["episodes_excluding_near"].sum()),
                }
            )
    return pd.DataFrame(rows)


def paired(by_seed: pd.DataFrame, rate_column: str, success_column: str, episode_column: str) -> pd.DataFrame:
    rows = []
    for side in list(SIDES) + ["primary"]:
        diffs = []
        for seed in SEEDS:
            if side == "primary":
                dr = by_seed[(by_seed["method"] == "spatial_dr") & (by_seed["train_seed"] == seed) & (by_seed["L"].isin(PRIMARY_SIDES))]
                fixed = by_seed[(by_seed["method"] == "fixed_large") & (by_seed["train_seed"] == seed) & (by_seed["L"].isin(PRIMARY_SIDES))]
                dr_rate = float(dr[rate_column].mean())
                fixed_rate = float(fixed[rate_column].mean())
                dr_success = int(dr[success_column].sum())
                fixed_success = int(fixed[success_column].sum())
                dr_n = int(dr[episode_column].sum())
                fixed_n = int(fixed[episode_column].sum())
                label = "0.30-0.80"
            else:
                dr = by_seed[(by_seed["method"] == "spatial_dr") & (by_seed["train_seed"] == seed) & (by_seed["L"] == side)].iloc[0]
                fixed = by_seed[(by_seed["method"] == "fixed_large") & (by_seed["train_seed"] == seed) & (by_seed["L"] == side)].iloc[0]
                dr_rate = float(dr[rate_column])
                fixed_rate = float(fixed[rate_column])
                dr_success = int(dr[success_column])
                fixed_success = int(fixed[success_column])
                dr_n = int(dr[episode_column])
                fixed_n = int(fixed[episode_column])
                label = f"{side:.2f}"
            diff = (dr_rate - fixed_rate) * 100
            diffs.append(diff)
            rows.append(
                {
                    "L": label,
                    "train_seed": seed,
                    "spatial_dr_successes": dr_success,
                    "spatial_dr_episodes": dr_n,
                    "fixed_large_successes": fixed_success,
                    "fixed_large_episodes": fixed_n,
                    "spatial_dr_rate": dr_rate,
                    "fixed_large_rate": fixed_rate,
                    "difference_pp": diff,
                }
            )
        rows.append(
            {
                "L": label,
                "train_seed": "mean",
                "spatial_dr_successes": "",
                "spatial_dr_episodes": "",
                "fixed_large_successes": "",
                "fixed_large_episodes": "",
                "spatial_dr_rate": "",
                "fixed_large_rate": "",
                "difference_pp": float(np.mean(diffs)),
                "difference_sd_pp": float(np.std(diffs, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def learning_tables(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for steps in CHECKPOINTS:
        part = validation[validation["checkpoint_steps"] == steps]
        for method in METHODS:
            for seed in SEEDS:
                seed_part = part[(part["method"] == method) & (part["train_seed"] == seed)]
                rates = []
                for side in PRIMARY_SIDES:
                    side_part = seed_part[seed_part["L"] == side]
                    rates.append(float(side_part["success"].mean()))
                far = seed_part[seed_part["L"] == 0.80]
                rows.append(
                    {
                        "checkpoint_steps": steps,
                        "method": method,
                        "train_seed": seed,
                        "primary_rate": float(np.mean(rates)),
                        "primary_successes": int(seed_part[seed_part["L"].isin(PRIMARY_SIDES)]["success"].sum()),
                        "primary_episodes": int(seed_part[seed_part["L"].isin(PRIMARY_SIDES)].shape[0]),
                        "L080_successes": int(far["success"].sum()),
                        "L080_episodes": int(len(far)),
                        "L080_rate": float(far["success"].mean()),
                    }
                )
    by_seed = pd.DataFrame(rows)
    method_rows = []
    for steps in CHECKPOINTS:
        for method in METHODS:
            part = by_seed[(by_seed["checkpoint_steps"] == steps) & (by_seed["method"] == method)]
            primary = part["primary_rate"].to_numpy(dtype=float)
            far = part["L080_rate"].to_numpy(dtype=float)
            method_rows.append(
                {
                    "checkpoint_steps": steps,
                    "method": method,
                    "primary_mean": float(primary.mean()),
                    "primary_sd_pp": float(primary.std(ddof=1) * 100),
                    "L080_mean": float(far.mean()),
                    "L080_sd_pp": float(far.std(ddof=1) * 100),
                }
            )
    per_L = (
        validation.groupby(["checkpoint_steps", "method", "train_seed", "L"], as_index=False)
        .agg(successes=("success", "sum"), episodes=("success", "size"))
    )
    per_L["rate"] = per_L["successes"] / per_L["episodes"]
    return by_seed, pd.DataFrame(method_rows), per_L


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    training = load_training()
    train_table, train_problems = training_summary(training)
    records = load_eval()
    problems = train_problems + eval_problems(records)
    final = rate_table(records[records["phase"] == "final"])
    official = method_table(final, "rate")
    excluded = method_table(final, "rate_excluding_near")
    excluded = excluded.rename(columns={"mean_rate": "mean_rate_excluding_near", "sd_pp": "sd_pp_excluding_near", "pooled_successes": "pooled_successes_excluding_near", "pooled_episodes": "pooled_episodes_excluding_near"})
    by_method = official.merge(
        excluded[["L", "method", "mean_rate_excluding_near", "sd_pp_excluding_near", "pooled_successes_excluding_near", "pooled_episodes_excluding_near"]],
        on=["L", "method"],
    )
    pairs = paired(final, "rate", "successes", "episodes")
    pairs_near = paired(final, "rate_excluding_near", "successes_excluding_near", "episodes_excluding_near")
    learning_seed, learning_method, learning_L = learning_tables(records[records["phase"] == "validation"])
    train_table.to_csv(SUMMARY / "training_sampling.csv", index=False)
    final.to_csv(SUMMARY / "final_by_seed_L.csv", index=False)
    by_method.to_csv(SUMMARY / "final_by_method_L.csv", index=False)
    pairs.to_csv(SUMMARY / "final_paired.csv", index=False)
    pairs_near.to_csv(SUMMARY / "final_paired_excluding_near.csv", index=False)
    learning_seed.to_csv(SUMMARY / "learning_by_seed.csv", index=False)
    learning_method.to_csv(SUMMARY / "learning_by_method.csv", index=False)
    learning_L.to_csv(SUMMARY / "learning_by_seed_L.csv", index=False)
    (SUMMARY / "problems.json").write_text(json.dumps(problems, indent=2))
    print(train_table.to_string(index=False))
    print("--- final by method ---")
    print(by_method.to_string(index=False))
    print("--- paired L=0.80 and primary ---")
    print(pairs[pairs["L"].isin(["0.80", "0.30-0.80"])].to_string(index=False))
    print("--- learning ---")
    print(learning_method.to_string(index=False))
    print("--- problems ---")
    print(problems if problems else "none")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
