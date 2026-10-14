"""Verify the locked single-factor CSVs and write the summary tables.

This script does not import MuJoCo and does not rerun episodes. The formal
runner and its protocol file stay untouched. Nominal mass 1 kg, sliding
friction 1, and gain scale 1 are read from the unchanged P0-v3 CSVs through
reuse_map.csv. Those rows are not copied into the new episode files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P0_DIR = ROOT.parent / "corrective_round3_2026-10-01"
PROTOCOL_PATH = ROOT / "single_factor_protocol.json"
REUSE_PATH = ROOT / "reuse_map.csv"
READBACK_PATH = ROOT / "probes" / "readback.json"

SEEDS = (0, 42, 123)
CHECKPOINT = {
    0: "ab38ff9c17dbde348f0d0048858d4753570c04b2e7f486cacf60135717f085bd",
    42: "e7c4840645dae9949b40c2b58045888a354b8193d3d0ffea429ae1a3ba087f74",
    123: "25f64416614872cddcf5dc83075b55fea5f32a81b4ab60caa17454de9f3ea966",
}
P0_PROTOCOL = "e72dd7e83d71991f995dc67b0c679d2b7736a8ffe5c51ac8ab9ddbf067f2f956"
ATOL = 1e-8
RANGE_ATOL = 1e-9

# Float sets of the 29 new cells. Tokens in the CSV are kept as written.
EXPECTED = {
    "object_mass": {0.05, 0.1, 0.5, 2.0, 3.0},
    "object_friction": {0.05, 0.2, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0},
    "joint_damping_scale": {0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3},
    "actuator_gain_scale": {0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 10.0},
}
REUSED = {
    "object_mass": 1.0,
    "object_friction": 1.0,
    "actuator_gain_scale": 1.0,
}
FACTOR_ORDER = (
    "object_mass",
    "object_friction",
    "joint_damping_scale",
    "actuator_gain_scale",
)

# June 2025 summary means. Those files pool overlapping eval seeds inside
# each checkpoint and are not this protocol. Stored so the comparison table
# does not depend on re-reading a second layout.
HISTORICAL = {
    "object_mass": {
        0.05: (99.9, 0.2),
        0.1: (100.0, 0.0),
        0.5: (99.8, 0.3),
        1.0: (99.7, 0.3),
        2.0: (99.6, 0.8),
        3.0: (99.0, 0.3),
    },
    "object_friction": {
        0.05: (99.0, 1.3),
        0.2: (99.0, 1.3),
        0.5: (99.0, 1.3),
        0.8: (99.0, 1.3),
        1.0: (99.7, 0.3),
        1.2: (99.4, 0.1),
        1.5: (99.6, 0.3),
        2.0: (99.6, 0.3),
        2.5: (99.8, 0.2),
        3.0: (98.7, 0.8),
    },
    "joint_damping_scale": {
        0.001: (99.8, 0.3),
        0.01: (99.8, 0.4),
        0.02: (99.8, 0.3),
        0.05: (99.7, 0.3),
        0.08: (99.8, 0.3),
        0.1: (99.8, 0.3),
        0.2: (99.8, 0.3),
        0.3: (99.7, 0.3),
    },
    "actuator_gain_scale": {
        0.0: (98.9, 0.7),
        1.0: (99.7, 0.3),
        2.0: (99.1, 0.8),
        3.0: (99.3, 0.6),
        5.0: (99.3, 0.6),
        6.0: (99.3, 0.6),
        7.0: (99.3, 0.6),
        10.0: (99.3, 0.6),
    },
}
FIELD_NOTE = {
    "object_mass": (
        "Horizon 50 versus 100, overlapping eval seeds versus reset_seed "
        "4000000+episode_id, and a June geom-mass recompile versus writer C. "
        "A current-tree compile at 0.05 kg matches writer C. The June binary "
        "was not recovered. The rate gap is not assigned to one of these."
    ),
    "object_friction": (
        "Horizon and initialization differ. On this XML a one-number friction "
        "attribute and writer C both leave torsion 0.005 and roll 0.0001. "
        "The rate gap is not assigned to one of these."
    ),
    "joint_damping_scale": (
        "The June worker wrote damping only on mocap joints that already had "
        "a damping attribute, leaving the six object DOFs at 0.01. This grid "
        "multiplies the whole pristine dof_damping array, so object DOFs are "
        "0.01 times the scale. Horizon and initialization also differ. A rate "
        "gap cannot be assigned to one of these."
    ),
    "actuator_gain_scale": (
        "Horizon and initialization differ. Both definitions leave biasprm "
        "unscaled. On this model gainprm components 1 and 2 are already 0, "
        "so scaling column 0 matches a one-number gainprm at compile time. "
        "The end effector is mocap-welded. A high success rate does not show "
        "that the gain column is inactive."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, atol: float = ATOL) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def load_new() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = ROOT / f"single_factor_seed{seed}.csv"
        frame = pd.read_csv(path)
        frame["source_csv"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_p0() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = P0_DIR / f"P0_v3_seed{seed}.csv"
        frame = pd.read_csv(path)
        frame["source_csv"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    problems: list[str] = []
    protocol = json.loads(PROTOCOL_PATH.read_text())
    protocol_id = protocol["protocol_id"]
    protocol_sha = protocol["protocol_sha256"]
    if protocol_id != "single_factor_clean_reset_writerC_h100":
        problems.append(f"protocol_id is {protocol_id}")

    df = load_new()
    if len(df) != 52200:
        problems.append(f"new rows {len(df)} != 52200")
    for seed in SEEDS:
        part = df[df["policy_seed"] == seed]
        if len(part) != 17400:
            problems.append(f"seed {seed} rows {len(part)} != 17400")
        keys = part.groupby(["factor", "value", "episode_id"], sort=False).size()
        if int((keys != 1).sum()) or len(keys) != 17400:
            problems.append(f"seed {seed} unique keys {len(keys)}")

    got = {
        factor: set(np.round(part["value"].astype(float), 8))
        for factor, part in df.groupby("factor")
    }
    if set(got) != set(EXPECTED):
        problems.append(f"factors {sorted(got)} != {sorted(EXPECTED)}")
    for factor, expect in EXPECTED.items():
        have = got.get(factor, set())
        if have != expect:
            problems.append(f"{factor} values {sorted(have)} != {sorted(expect)}")

    for factor, nominal in REUSED.items():
        hit = df[(df["factor"] == factor) & np.isclose(df["value"], nominal)]
        if len(hit):
            problems.append(f"nominal {factor}={nominal} was rerun ({len(hit)} rows)")

    constants = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha,
        "evaluation_source": "new_run",
        "writer": "C",
        "delay_steps": 0,
    }
    for column, expected in constants.items():
        values = set(df[column].astype(str).unique())
        if values != {str(expected)}:
            problems.append(f"{column} values {values}")
    if set(df["model_check_ok"].astype(int).unique()) != {1}:
        problems.append(f"model_check_ok {df['model_check_ok'].unique()}")
    if set(df["bias_unchanged"].astype(int).unique()) != {1}:
        problems.append("bias_unchanged is not all 1")
    if df["success"].isna().any() or set(df["success"].astype(int).unique()) - {0, 1}:
        problems.append("success is not a complete 0/1 field")
    if (df["steps_used"] < 1).any() or (df["steps_used"] > 100).any():
        problems.append("steps_used outside 1..100")
    bad_seed = df["reset_seed"] != 4000000 + df["episode_id"]
    if bad_seed.any():
        problems.append(f"reset_seed mismatches {int(bad_seed.sum())}")
    for seed, digest in CHECKPOINT.items():
        have = set(df.loc[df["policy_seed"] == seed, "checkpoint_sha256"])
        if have != {digest}:
            problems.append(f"checkpoint seed {seed} is {have}")

    finite_cols = [
        "initial_distance",
        "final_distance",
        "mass_read",
        "friction_read",
        "damping_scale_read",
        "gain_scale_read",
        "time_after_reset",
        "ctrl_maxabs_vs_neutral",
        "goal_z",
    ]
    for column in finite_cols:
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            problems.append(f"{column} has a non-finite value")
    if not np.allclose(df["time_after_reset"], 0.0, rtol=0.0, atol=ATOL):
        problems.append("time_after_reset is not 0")
    if not np.allclose(df["ctrl_maxabs_vs_neutral"], 0.0, rtol=0.0, atol=ATOL):
        problems.append("ctrl is not neutral after reset")

    for axis, center in (("x", 0.6), ("y", 0.0)):
        for prefix in ("goal", "obj"):
            low = df[f"{prefix}_low_{axis}"]
            high = df[f"{prefix}_high_{axis}"]
            if not np.allclose(high - low, 0.3, rtol=0.0, atol=RANGE_ATOL):
                problems.append(f"{prefix} {axis} side is not 0.30")
            if not np.allclose((high + low) / 2, center, rtol=0.0, atol=RANGE_ATOL):
                problems.append(f"{prefix} {axis} center is not {center}")
    if not (df["goal_z"] > 0.0).all():
        problems.append("goal_z is not the object height")

    if not np.allclose(df["friction_torsion_read"], 0.005, rtol=0.0, atol=ATOL):
        problems.append("friction torsion moved")
    if not np.allclose(df["friction_roll_read"], 0.0001, rtol=0.0, atol=ATOL):
        problems.append("friction roll moved")
    if not np.allclose(df["invweight_trans"] * df["mass_read"], 1.0, rtol=0.0, atol=1e-6):
        problems.append("invweight_trans is not 1/mass")

    mass = df["factor"] == "object_mass"
    friction = df["factor"] == "object_friction"
    damping = df["factor"] == "joint_damping_scale"
    gain = df["factor"] == "actuator_gain_scale"
    checks = [
        (mass, "mass_read", df.loc[mass, "value"], "mass readback"),
        (friction, "friction_read", df.loc[friction, "value"], "friction readback"),
        (damping, "damping_scale_read", df.loc[damping, "value"], "damping readback"),
        (gain, "gain_scale_read", df.loc[gain, "value"], "gain readback"),
        (~mass, "mass_read", 1.0, "mass left at 1"),
        (~friction, "friction_read", 1.0, "friction left at 1"),
        (~damping, "damping_scale_read", 1.0, "damping left at 1"),
        (~gain, "gain_scale_read", 1.0, "gain left at 1"),
    ]
    for mask, column, target, label in checks:
        have = df.loc[mask, column].to_numpy(dtype=float)
        want = target.to_numpy(dtype=float) if isinstance(target, pd.Series) else target
        if not np.allclose(have, want, rtol=0.0, atol=ATOL):
            problems.append(label)
    obj_damp = df.loc[damping, "obj_dof_damping_read"].to_numpy(dtype=float)
    obj_want = 0.01 * df.loc[damping, "value"].to_numpy(dtype=float)
    if not np.allclose(obj_damp, obj_want, rtol=0.0, atol=1e-12):
        problems.append("object dof damping is not 0.01 times the scale")
    if not np.allclose(df.loc[~damping, "obj_dof_damping_read"], 0.01, rtol=0.0, atol=ATOL):
        problems.append("object dof damping moved outside the damping cells")

    # One changed field per cell: inertia scales with mass and stays at the
    # compiled 1 kg value otherwise. The three axes stay equal.
    inertia = df[["inertia_x", "inertia_y", "inertia_z"]].to_numpy(dtype=float)
    if not np.allclose(inertia[:, 0], inertia[:, 1], rtol=0.0, atol=1e-15):
        problems.append("inertia axes differ")
    if not np.allclose(inertia[:, 0], inertia[:, 2], rtol=0.0, atol=1e-15):
        problems.append("inertia z differs")
    base = df.loc[~mass, "inertia_x"]
    if base.nunique() != 1:
        problems.append("non-mass inertia is not a single compiled value")
    else:
        base_value = float(base.iloc[0])
        scaled = df.loc[mass, "inertia_x"].to_numpy(dtype=float)
        want = base_value * df.loc[mass, "value"].to_numpy(dtype=float)
        if not np.allclose(scaled, want, rtol=0.0, atol=1e-15):
            problems.append("mass inertia was not scaled once from the 1 kg value")

    spots = [
        ("object_mass", 0.05, 5, 1, "mass_read", 0.05),
        ("object_friction", 0.05, 4, 1, "friction_read", 0.05),
        ("joint_damping_scale", 0.001, 5, 1, "obj_dof_damping_read", 1e-5),
        ("actuator_gain_scale", 0.0, 4, 1, "gain_scale_read", 0.0),
    ]
    for factor, value, steps, success, column, read in spots:
        row = df[
            (df["policy_seed"] == 0)
            & (df["episode_id"] == 0)
            & (df["factor"] == factor)
            & np.isclose(df["value"], value)
        ]
        if len(row) != 1:
            problems.append(f"spot {factor}={value} missing")
            continue
        item = row.iloc[0]
        if int(item["steps_used"]) != steps or int(item["success"]) != success:
            problems.append(
                f"spot {factor}={value} steps/success "
                f"{item['steps_used']}/{item['success']}"
            )
        if not close(item[column], read, atol=1e-12 if column.endswith("damping_read") else ATOL):
            problems.append(f"spot {factor}={value} {column}={item[column]}")

    pose_cols = ["initial_x", "initial_y", "initial_z", "goal_x", "goal_y", "goal_z"]
    grouped = df.groupby(["factor", "value", "episode_id"])[pose_cols]
    spread = grouped.max() - grouped.min()
    pose_mismatch = int((spread.max(axis=1) > ATOL).sum())
    if pose_mismatch:
        problems.append(f"cross-policy pose mismatches {pose_mismatch}")

    p0 = load_p0()
    if len(p0) != 1800:
        problems.append(f"P0 rows {len(p0)}")
    p0_success = int(p0["success"].sum())
    if p0_success != 1799:
        problems.append(f"P0 successes {p0_success} != 1799")
    reuse = pd.read_csv(REUSE_PATH)
    if len(reuse) != 5400:
        problems.append(f"reuse rows {len(reuse)}")
    if set(reuse["evaluation_source"]) != {"reused_P0"}:
        problems.append("reuse source label changed")
    if set(reuse["source_protocol_sha256"]) != {P0_PROTOCOL}:
        problems.append("reuse protocol hash is not the P0 hash")
    for path in sorted(reuse["source_file"].unique()):
        recorded = set(reuse.loc[reuse["source_file"] == path, "source_file_sha256"])
        actual = sha256(Path(path))
        if recorded != {actual}:
            problems.append(f"P0 file hash mismatch {path}")
    reuse_keys = reuse.groupby(["factor", "value", "policy_seed", "episode_id"]).size()
    if len(reuse_keys) != 5400 or int((reuse_keys != 1).sum()):
        problems.append("reuse keys are not unique")

    p0_key = p0.set_index(["policy_seed", "episode_id"])
    merged = df.join(p0_key[pose_cols + ["success", "initial_distance", "steps_used"]], on=["policy_seed", "episode_id"], rsuffix="_p0")
    pose_vs_p0 = merged[pose_cols].to_numpy(dtype=float) - merged[[c + "_p0" for c in pose_cols]].to_numpy(dtype=float)
    pose_vs_p0_mismatch = int((np.abs(pose_vs_p0).max(axis=1) > ATOL).sum())
    if pose_vs_p0_mismatch:
        problems.append(f"pose mismatches versus P0 {pose_vs_p0_mismatch}")

    fail_rows = df[df["success"].astype(int) == 0]
    fail_files = list((ROOT / "states").glob("fail_*.npz")) if (ROOT / "states").exists() else []
    trace_files = list((ROOT / "traces").glob("*.npz")) if (ROOT / "traces").exists() else []
    warnings = []
    if len(fail_files) != len(fail_rows):
        warnings.append(f"failure npz {len(fail_files)} != failed rows {len(fail_rows)}")
    if len(trace_files) != 29:
        warnings.append(f"seed-0 episode-0 traces {len(trace_files)} != 29")

    # Summaries. Nominal cells reuse P0 success on the same scenes.
    rows = []
    rate_rows = []
    near_rows = []
    compare_rows = []

    def emit(factor: str, value: float, token: str, source: str, part: pd.DataFrame) -> None:
        counts = []
        for seed in SEEDS:
            seed_part = part[part["policy_seed"] == seed]
            n = int(len(seed_part))
            successes = int(seed_part["success"].sum())
            counts.append((seed, n, successes))
            rate_rows.append(
                {
                    "factor": factor,
                    "value": token,
                    "value_float": value,
                    "policy_seed": seed,
                    "n": n,
                    "successes": successes,
                    "success_rate": successes / n,
                    "evaluation_source": source,
                }
            )
        rates = [successes / n for _, n, successes in counts]
        flips = int((part["success"].astype(int) != part["success_p0"].astype(int)).sum()) if "success_p0" in part.columns else ""
        rows.append(
            {
                "factor": factor,
                "value": token,
                "value_float": value,
                "evaluation_source": source,
                "n_seed0": counts[0][1],
                "success_seed0": counts[0][2],
                "n_seed42": counts[1][1],
                "success_seed42": counts[1][2],
                "n_seed123": counts[2][1],
                "success_seed123": counts[2][2],
                "n_pooled": sum(n for _, n, _ in counts),
                "success_pooled": sum(successes for _, _, successes in counts),
                "mean_pct": float(np.mean(rates) * 100),
                "sd_pct": float(np.std(rates, ddof=1) * 100),
                "paired_flips_vs_p0": flips,
            }
        )
        near_mask = part["initial_distance"].to_numpy(dtype=float) < 0.05
        kept = part.loc[~near_mask]
        near_counts = []
        for seed in SEEDS:
            seed_part = kept[kept["policy_seed"] == seed]
            near_counts.append((int(len(seed_part)), int(seed_part["success"].sum())))
        kept_rates = [s / n for n, s in near_counts]
        near_rows.append(
            {
                "factor": factor,
                "value": token,
                "value_float": value,
                "evaluation_source": source,
                "n_excluded": int(near_mask.sum()),
                "n_seed0": near_counts[0][0],
                "success_seed0": near_counts[0][1],
                "n_seed42": near_counts[1][0],
                "success_seed42": near_counts[1][1],
                "n_seed123": near_counts[2][0],
                "success_seed123": near_counts[2][1],
                "n_pooled": sum(n for n, _ in near_counts),
                "success_pooled": sum(s for _, s in near_counts),
                "mean_pct": float(np.mean(kept_rates) * 100),
                "sd_pct": float(np.std(kept_rates, ddof=1) * 100),
            }
        )
        old = HISTORICAL[factor].get(round(value, 8))
        compare_rows.append(
            {
                "factor": factor,
                "value": token,
                "value_float": value,
                "historical_mean_pct": None if old is None else old[0],
                "historical_sd_pct": None if old is None else old[1],
                "new_mean_pct": float(np.mean(rates) * 100),
                "new_sd_pct": float(np.std(rates, ddof=1) * 100),
                "new_success_pooled": sum(successes for _, _, successes in counts),
                "new_n_pooled": sum(n for _, n, _ in counts),
                "delta_mean_pp": None if old is None else float(np.mean(rates) * 100 - old[0]),
                "difference_not_assigned_to_one_change": FIELD_NOTE[factor],
            }
        )

    work = merged.copy()
    for (factor, value), part in work.groupby(["factor", "value"], sort=False):
        token = f"{value:g}"
        emit(str(factor), float(value), token, "new_run", part)

    p0_part = p0.copy()
    p0_part["success_p0"] = p0_part["success"]
    for factor, value in REUSED.items():
        emit(factor, value, f"{value:g}", "reused_P0", p0_part)

    summary = pd.DataFrame(rows).sort_values(["factor", "value_float"])
    rates = pd.DataFrame(rate_rows).sort_values(["factor", "value_float", "policy_seed"])
    near = pd.DataFrame(near_rows).sort_values(["factor", "value_float"])
    compare = pd.DataFrame(compare_rows).sort_values(["factor", "value_float"])
    # Stable factor order.
    for frame in (summary, rates, near, compare):
        frame["factor"] = pd.Categorical(frame["factor"], FACTOR_ORDER, ordered=True)
    summary = summary.sort_values(["factor", "value_float"])
    rates = rates.sort_values(["factor", "value_float", "policy_seed"])
    near = near.sort_values(["factor", "value_float"])
    compare = compare.sort_values(["factor", "value_float"])

    summary.to_csv(ROOT / "single_factor_summary.csv", index=False)
    rates.to_csv(ROOT / "single_factor_rates.csv", index=False)
    near.to_csv(ROOT / "single_factor_near_sensitivity.csv", index=False)
    compare.to_csv(ROOT / "single_factor_old_vs_new.csv", index=False)

    def outcome_diffs(factor: str, anchor: float, others: list[float]) -> dict:
        base = df[(df["factor"] == factor) & np.isclose(df["value"], anchor)].sort_values(
            ["policy_seed", "episode_id"]
        )
        out = {}
        for value in others:
            part = df[(df["factor"] == factor) & np.isclose(df["value"], value)].sort_values(
                ["policy_seed", "episode_id"]
            )
            out[f"{value:g}"] = {
                "success_flips": int((base["success"].to_numpy() != part["success"].to_numpy()).sum()),
                "step_changes": int((base["steps_used"].to_numpy() != part["steps_used"].to_numpy()).sum()),
                "final_distance_changes": int(
                    (np.abs(base["final_distance"].to_numpy() - part["final_distance"].to_numpy()) > ATOL).sum()
                ),
            }
        return out

    p0_fail = p0[p0["success"].astype(int) == 0][["policy_seed", "episode_id"]]
    p0_fail_follow = []
    for _, item in p0_fail.iterrows():
        hit = df[
            (df["policy_seed"] == item["policy_seed"]) & (df["episode_id"] == item["episode_id"])
        ]
        still = hit[hit["success"].astype(int) == 0][["factor", "value"]]
        p0_fail_follow.append(
            {
                "policy_seed": int(item["policy_seed"]),
                "episode_id": int(item["episode_id"]),
                "still_failed": [
                    {"factor": str(row.factor), "value": float(row.value)}
                    for row in still.itertuples(index=False)
                ],
            }
        )

    fail_by_cell = (
        fail_rows.groupby(["factor", "value"]).size().sort_values(ascending=False).head(12)
    )
    step_counts = fail_rows["steps_used"].value_counts().sort_index()
    check_payload = {
        "problem_count": len(problems),
        "problems": problems,
        "warnings": warnings,
        "new_rows": int(len(df)),
        "failed_rows": int(len(fail_rows)),
        "failure_npz": len(fail_files),
        "traces": len(trace_files),
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha,
        "p0_successes": p0_success,
        "pose_mismatch_across_policies": pose_mismatch,
        "pose_mismatch_versus_p0": pose_vs_p0_mismatch,
        "near_starts_new_rows": int((df["initial_distance"] < 0.05).sum()),
        "near_starts_p0": int((p0["initial_distance"] < 0.05).sum()),
        "goal_z_min": float(df["goal_z"].min()),
        "goal_z_max": float(df["goal_z"].max()),
        "failure_steps": {str(int(k)): int(v) for k, v in step_counts.items()},
        "failures_at_100": int((fail_rows["steps_used"] == 100).sum()),
        "highest_failure_cells": [
            {"factor": str(a), "value": float(b), "failures": int(c)}
            for (a, b), c in fail_by_cell.items()
        ],
        "friction_0.05_versus_0.2_0.5_0.8": outcome_diffs(
            "object_friction", 0.05, [0.2, 0.5, 0.8]
        ),
        "gain_3_versus_5_6_7_10": outcome_diffs(
            "actuator_gain_scale", 3.0, [5.0, 6.0, 7.0, 10.0]
        ),
        "damping_0.2_versus_0.3": outcome_diffs("joint_damping_scale", 0.2, [0.3]),
        "p0_failure_followup": p0_fail_follow,
        "failure_final_distance_median": {
            str(factor): float(part["final_distance"].median())
            for factor, part in fail_rows.groupby("factor")
        },
        "lowest_mean_pct": float(summary["mean_pct"].min()),
        "highest_mean_pct": float(summary["mean_pct"].max()),
        "cells_below_98": int((summary["mean_pct"] < 98).sum()),
    }
    (ROOT / "logs" / "summary_checks.json").write_text(
        json.dumps(check_payload, indent=2) + "\n"
    )
    print(json.dumps(check_payload, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
