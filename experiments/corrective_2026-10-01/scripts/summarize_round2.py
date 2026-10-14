"""Merge round-2 partials, check pairing, and write the locked tables and figures.

The curve description in the report is written after these numbers exist.
This script records the rates and the arithmetic. It does not name a cliff.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from round2_common import (
    DELAYS,
    DELAY_RESET_BASE,
    HORIZONS,
    LENGTHS,
    MODEL_SEEDS,
    N_EPISODES,
    OUT,
    P0_CSV,
    P1_CSV,
    P2_CSV,
    P3_CSV,
    PARTIAL,
    SPATIAL_RESET_BASE,
    fmt,
    load_stage0_rows,
    mean_sd,
    median,
    read_csv,
    wilson,
)
from phase2_delay_sweep import FIELDS as DELAY_FIELDS
from phase2_delay_sweep import partial_path as delay_partial
from phase3_physical_ablation import CONDITIONS
from phase3_physical_ablation import FIELDS as ABLATION_FIELDS
from phase3_physical_ablation import partial_path as ablation_partial
from phase4_spatial_horizon import FIELDS as SPATIAL_FIELDS
from phase4_spatial_horizon import partial_path as spatial_partial
from phase4_spatial_horizon import spatial_reset_seed

DISTANCE_BINS = (
    ("0-0.10", 0.0, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.30", 0.20, 0.30),
    ("0.30-0.40", 0.30, 0.40),
    ("0.40-0.50", 0.40, 0.50),
    (">=0.50", 0.50, math.inf),
)
SCENE_KEYS = (
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "target_x",
    "target_y",
    "target_z",
    "initial_distance",
)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_rows(path: Path, n_episodes: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = read_csv(path)
    if len(rows) != n_episodes:
        raise SystemExit(f"{path} has {len(rows)} rows, expected {n_episodes}")
    ids = [int(row["episode_id"]) for row in rows]
    if ids != list(range(n_episodes)):
        raise SystemExit(f"{path} episode ids are not 0..{n_episodes - 1}")
    return rows


def steps_stats(rows: list[dict]) -> tuple[str, str, int]:
    steps = [int(row["steps_used"]) for row in rows if int(row["success"]) == 1]
    if not steps:
        return "", "", 0
    return f"{(sum(steps) / len(steps)):.4f}", f"{median(steps):.4f}", len(steps)


def seed_rates(rows: list[dict]) -> tuple[dict[int, float], int, int]:
    grouped = {seed: [] for seed in MODEL_SEEDS}
    for row in rows:
        grouped[int(row["policy_seed"])].append(int(row["success"]))
    rates = {}
    successes = 0
    for seed in MODEL_SEEDS:
        flags = grouped[seed]
        if len(flags) != len(grouped[MODEL_SEEDS[0]]):
            raise SystemExit("seed groups differ in size")
        rates[seed] = sum(flags) / len(flags)
        successes += sum(flags)
    return rates, successes, len(rows)


def rate_record(rates: dict[int, float], successes: int, n: int, steps: tuple[str, str, int]) -> dict:
    mean, sd = mean_sd([rates[seed] for seed in MODEL_SEEDS])
    low, high = wilson(successes, n)
    return {
        "rate_seed0": f"{rates[0]:.8f}",
        "rate_seed42": f"{rates[42]:.8f}",
        "rate_seed123": f"{rates[123]:.8f}",
        "mean_across_seeds": f"{mean:.8f}",
        "sd_across_seeds": f"{sd:.8f}",
        "total_n": n,
        "total_successes": successes,
        "total_success_rate": f"{(successes / n):.8f}",
        "wilson_low": f"{low:.8f}",
        "wilson_high": f"{high:.8f}",
        "mean_steps_to_success": steps[0],
        "median_steps_to_success": steps[1],
        "n_success": steps[2],
    }


def save_delay_plot(summary_rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latency = [float(row["latency_s"]) for row in summary_rows]
    mean = [float(row["mean_across_seeds"]) for row in summary_rows]
    sd = [float(row["sd_across_seeds"]) for row in summary_rows]
    labels = [f"{float(row['latency_s']):.1f} s\n{int(row['delay_steps'])} steps" for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(latency, mean, yerr=sd, marker="o", capsize=3, color="#1f4e79")
    ax.set_xticks(latency)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Action latency (seconds)\nlower tick label: delay_steps")
    ax.set_ylabel("Success rate")
    ax.set_title("Frozen TQC+HER policies, horizon 100, default physics")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spatial_plots(summary_rows: list[dict], steps_path: Path, rate_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate_rows = [row for row in summary_rows if row["row_type"] == "rate"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for horizon, marker in ((50, "o"), (100, "s")):
        chosen = [row for row in rate_rows if int(row["horizon"]) == horizon]
        ax.errorbar(
            [float(row["L"]) for row in chosen],
            [float(row["mean_across_seeds"]) for row in chosen],
            yerr=[float(row["sd_across_seeds"]) for row in chosen],
            marker=marker,
            capsize=3,
            label=f"H{horizon}",
        )
    ax.set_xlabel("Target-region side length L (m)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.set_title("Same scenes and policies, two episode horizons")
    fig.tight_layout()
    fig.savefig(rate_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for horizon, color in ((50, "#1f4e79"), (100, "#b85c38")):
        chosen = [row for row in rate_rows if int(row["horizon"]) == horizon]
        xs = [float(row["L"]) for row in chosen]
        medians = [float(row["median_steps_to_success"]) if row["median_steps_to_success"] else math.nan for row in chosen]
        means = [float(row["mean_steps_to_success"]) if row["mean_steps_to_success"] else math.nan for row in chosen]
        ax.plot(xs, medians, marker="o", color=color, label=f"H{horizon} median")
        ax.plot(xs, means, marker="x", linestyle="--", color=color, label=f"H{horizon} mean")
    ax.set_xlabel("Target-region side length L (m)")
    ax.set_ylabel("Steps to success, successful episodes")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("Steps used when the object finished within 0.05 m")
    fig.tight_layout()
    fig.savefig(steps_path, dpi=150)
    plt.close(fig)


def save_distance_plot(summary_rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [row for row in summary_rows if row["row_type"] == "distance_bin"]
    centers = {"0-0.10": 0.05, "0.10-0.20": 0.15, "0.20-0.30": 0.25, "0.30-0.40": 0.35, "0.40-0.50": 0.45, ">=0.50": 0.60}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for horizon, marker in ((50, "o"), (100, "s")):
        chosen = [row for row in bins if int(row["horizon"]) == horizon]
        xs, ys, ns = [], [], []
        for row in chosen:
            if int(row["total_n"]) == 0:
                continue
            xs.append(centers[row["distance_bin"]])
            ys.append(float(row["total_success_rate"]))
            ns.append(int(row["total_n"]))
        ax.plot(xs, ys, marker=marker, label=f"H{horizon}")
        if horizon == 100:
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(list(centers.values()))
    ax.set_xticklabels(list(centers))
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Initial object-target distance (m)")
    ax.set_ylabel("Success rate")
    ax.legend()
    ax.set_title("Pooled across L; labels on H100 are episode counts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_delay(partial_dir: Path, n_episodes: int) -> list[dict]:
    raw = []
    by_delay = {}
    for delay_steps in DELAYS:
        block = []
        for seed in MODEL_SEEDS:
            rows = require_rows(delay_partial(partial_dir, delay_steps, seed), n_episodes)
            for row in rows:
                if int(row["policy_seed"]) != seed or int(row["delay_steps"]) != delay_steps:
                    raise SystemExit("delay partial header drifted")
                if int(row["horizon"]) != 100:
                    raise SystemExit("delay sweep horizon is not 100")
                if int(row["reset_seed"]) != DELAY_RESET_BASE + int(row["episode_id"]):
                    raise SystemExit("delay reset seed drifted")
                if float(row["latency_s"]) != delay_steps * 0.5:
                    raise SystemExit("latency is not delay_steps * 0.5 s")
            block.extend(rows)
        by_delay[delay_steps] = block
        raw.extend(block)
    reference = by_delay[0]
    for delay_steps, block in by_delay.items():
        for left, right in zip(reference, block):
            if left["policy_seed"] != right["policy_seed"] or left["episode_id"] != right["episode_id"]:
                raise SystemExit("delay blocks are not aligned")
            for key in SCENE_KEYS:
                if left[key] != right[key]:
                    raise SystemExit(f"scene changed at delay {delay_steps} episode {left['episode_id']}")
    write_csv(OUT / "delay_sweep_h100_raw.csv", DELAY_FIELDS, raw)
    summary = []
    previous = None
    for delay_steps in DELAYS:
        block = by_delay[delay_steps]
        rates, successes, n = seed_rates(block)
        record = rate_record(rates, successes, n, steps_stats(block))
        record["delay_steps"] = delay_steps
        record["latency_s"] = f"{delay_steps * 0.5:.1f}"
        current = successes / n
        record["pooled_change_from_previous"] = "" if previous is None else f"{(current - previous):.8f}"
        previous = current
        summary.append(record)
    fields = [
        "delay_steps",
        "latency_s",
        "rate_seed0",
        "rate_seed42",
        "rate_seed123",
        "mean_across_seeds",
        "sd_across_seeds",
        "total_n",
        "total_successes",
        "total_success_rate",
        "wilson_low",
        "wilson_high",
        "mean_steps_to_success",
        "median_steps_to_success",
        "n_success",
        "pooled_change_from_previous",
    ]
    write_csv(OUT / "delay_sweep_h100_summary.csv", fields, summary)
    save_delay_plot(summary, OUT / "delay_sweep_h100.png")
    write_delay_check(summary)
    p0 = {(int(row["model_seed"]), int(row["episode_id"])): row for row in read_csv(P0_CSV)}
    mismatches = []
    for row in reference:
        key = (int(row["policy_seed"]), int(row["episode_id"]))
        old = p0[key]
        same_scene = (
            old["initial_object_x"] == row["initial_object_x"]
            and old["initial_object_y"] == row["initial_object_y"]
            and old["initial_object_z"] == row["initial_object_z"]
            and old["goal_x"] == row["target_x"]
            and old["goal_y"] == row["target_y"]
            and old["goal_z"] == row["target_z"]
        )
        same_outcome = int(old["success"]) == int(row["success"]) and int(old["steps_used"]) == int(row["steps_used"])
        if not (same_scene and same_outcome):
            mismatches.append(key)
    mismatch_path = OUT / "delay_sweep_h100_p0_mismatches.csv"
    if mismatches:
        write_csv(
            mismatch_path,
            ["policy_seed", "episode_id"],
            [{"policy_seed": seed, "episode_id": episode} for seed, episode in mismatches],
        )
        raise SystemExit(f"delay 0 disagrees with P0 on {len(mismatches)} episodes")
    if mismatch_path.exists():
        mismatch_path.unlink()
    for row in summary:
        print(
            f"delay {row['delay_steps']} latency {row['latency_s']}s "
            f"rate {row['total_success_rate']} change {row['pooled_change_from_previous']}",
            flush=True,
        )
    return summary


def write_delay_check(summary: list[dict]) -> None:
    clean = {int(row["delay_steps"]): row for row in summary}
    uniform_rates = [float(clean[delay]["total_success_rate"]) for delay in range(1, 10)]
    uniform_expected = sum(uniform_rates) / len(uniform_rates)
    p2_rows = read_csv(P2_CSV)
    p3_rows = read_csv(P3_CSV)
    p2_rate = sum(int(row["success"]) for row in p2_rows) / len(p2_rows)
    p3_rate = sum(int(row["success"]) for row in p3_rows) / len(p3_rows)
    counts = defaultdict(int)
    p2_success = defaultdict(int)
    for row in p2_rows:
        delay = int(row["delay_steps"])
        counts[delay] += 1
        p2_success[delay] += int(row["success"])
    weighted = 0.0
    lines = [
        "# 延迟分布核对",
        "",
        "Phase 2 的每个 `delay_steps` 都用同一组 1800 个场景（3 个策略种子 × 600 局，视界 100，物理参数为训练默认值）。",
        "下面的期望把这条清洁曲线按给定的延迟分布加权，不重新抽样。",
        "P2 和 P3 的场景分配跟这条曲线不是同一批配对，所以下面只比较数值是否落在同一量级，不把差额写成协议失败。",
        "",
        "## 清洁曲线",
        "",
        "| delay_steps | 延迟 (s) | 合并成功率 | 相对上一档的变化 |",
        "|---:|---:|---:|---:|",
    ]
    for delay in DELAYS:
        row = clean[delay]
        change = row["pooled_change_from_previous"] or "—"
        lines.append(
            f"| {delay} | {row['latency_s']} | {float(row['total_success_rate']):.4%} | {change} |"
        )
    lines.extend(
        [
            "",
            "## Uniform{1,...,9}",
            "",
            "九个档的合并成功率算术平均：",
            "",
            f"- 期望成功率 = {uniform_expected:.6f} ({uniform_expected:.4%})",
            f"- P2 合并成功率 = {p2_rate:.6f} ({p2_rate:.4%})，差额（期望 − P2）= {uniform_expected - p2_rate:+.6f}",
            f"- P3 合并成功率 = {p3_rate:.6f} ({p3_rate:.4%})，差额（期望 − P3）= {uniform_expected - p3_rate:+.6f}",
            "",
            "P2 保存的延迟频数不是严格均匀的。用 P2 的实际频数给清洁曲线加权，是第二个对照，不是另一条实验曲线。",
            "",
            "| delay_steps | P2 局数 | P2 本档成功率 | 清洁曲线本档成功率 |",
            "|---:|---:|---:|---:|",
        ]
    )
    total = sum(counts.values())
    for delay in range(1, 10):
        weighted += counts[delay] / total * float(clean[delay]["total_success_rate"])
        p2_here = p2_success[delay] / counts[delay]
        lines.append(
            f"| {delay} | {counts[delay]} | {p2_here:.4%} | {float(clean[delay]['total_success_rate']):.4%} |"
        )
    lines.extend(
        [
            "",
            f"- 按 P2 频数加权的清洁曲线 = {weighted:.6f} ({weighted:.4%})",
            f"- 该加权值 − P2 = {weighted - p2_rate:+.6f}",
            f"- 该加权值 − P3 = {weighted - p3_rate:+.6f}",
            "",
            "P3 在 P2 的延迟之外还抽了质量、摩擦、阻尼和增益。它不应该被解释成延迟分布本身的期望。",
            "",
        ]
    )
    (OUT / "delay_distribution_check.md").write_text("\n".join(lines), encoding="utf-8")


def distance_bin(value: float) -> str:
    for label, low, high in DISTANCE_BINS:
        if low <= value < high:
            return label
    raise SystemExit(f"distance {value} did not fall in a bin")


def summarize_spatial(partial_dir: Path, n_episodes: int) -> None:
    stored = {}
    for horizon in HORIZONS:
        for side in LENGTHS:
            block = []
            for seed in MODEL_SEEDS:
                rows = require_rows(spatial_partial(partial_dir, horizon, side, seed), n_episodes)
                for row in rows:
                    if int(row["policy_seed"]) != seed or int(row["horizon"]) != horizon:
                        raise SystemExit("spatial partial drifted")
                    if float(row["L"]) != side:
                        raise SystemExit("L drifted")
                    expected = SPATIAL_RESET_BASE + int(round(side * 100)) * 1000 + int(row["episode_id"])
                    if int(row["reset_seed"]) != expected or expected != spatial_reset_seed(side, int(row["episode_id"])):
                        raise SystemExit("spatial reset seed drifted")
                    half = side / 2
                    if abs(float(row["target_x"]) - 0.6) > half + 1e-6 or abs(float(row["target_y"])) > half + 1e-6:
                        raise SystemExit("target left the requested region")
                    if abs(float(row["initial_object_x"]) - 0.6) > 0.15 + 1e-6 or abs(float(row["initial_object_y"])) > 0.15 + 1e-6:
                        raise SystemExit("object left the 0.30 m spawn")
                block.extend(rows)
            stored[(horizon, side)] = block
    for side in LENGTHS:
        for left, right in zip(stored[(50, side)], stored[(100, side)]):
            if left["policy_seed"] != right["policy_seed"] or left["episode_id"] != right["episode_id"]:
                raise SystemExit("H50 and H100 are not aligned")
            for key in SCENE_KEYS:
                if left[key] != right[key]:
                    raise SystemExit(f"H50/H100 scene mismatch at L={side} episode {left['episode_id']}")
    write_csv(OUT / "spatial_h50_raw.csv", SPATIAL_FIELDS, [row for side in LENGTHS for row in stored[(50, side)]])
    write_csv(OUT / "spatial_h100_raw.csv", SPATIAL_FIELDS, [row for side in LENGTHS for row in stored[(100, side)]])

    summary = []
    rate_fields_tail = [
        "rate_seed0",
        "rate_seed42",
        "rate_seed123",
        "mean_across_seeds",
        "sd_across_seeds",
        "total_n",
        "total_successes",
        "total_success_rate",
        "wilson_low",
        "wilson_high",
        "mean_steps_to_success",
        "median_steps_to_success",
        "n_success",
    ]
    for horizon in HORIZONS:
        for side in LENGTHS:
            block = stored[(horizon, side)]
            rates, successes, n = seed_rates(block)
            record = rate_record(rates, successes, n, steps_stats(block))
            record.update({"row_type": "rate", "horizon": horizon, "L": f"{side:.2f}", "distance_bin": ""})
            summary.append(record)
            print(
                f"spatial H{horizon} L{side:.2f} rate {record['total_success_rate']} "
                f"median_steps {record['median_steps_to_success']}",
                flush=True,
            )
    for horizon in HORIZONS:
        pooled = []
        for side in LENGTHS:
            pooled.extend(stored[(horizon, side)])
        grouped = defaultdict(list)
        for row in pooled:
            grouped[distance_bin(float(row["initial_distance"]))].append(row)
        for label, _, _ in DISTANCE_BINS:
            block = grouped.get(label, [])
            if not block:
                record = {field: "" for field in rate_fields_tail}
                record.update(
                    {
                        "row_type": "distance_bin",
                        "horizon": horizon,
                        "L": "",
                        "distance_bin": label,
                        "total_n": 0,
                        "total_successes": 0,
                        "n_success": 0,
                    }
                )
            else:
                # Counts per seed are not balanced inside a distance bin.
                rates = {}
                successes = sum(int(row["success"]) for row in block)
                for seed in MODEL_SEEDS:
                    flags = [int(row["success"]) for row in block if int(row["policy_seed"]) == seed]
                    rates[seed] = (sum(flags) / len(flags)) if flags else float("nan")
                low, high = wilson(successes, len(block))
                finite = [rates[seed] for seed in MODEL_SEEDS if not math.isnan(rates[seed])]
                mean, sd = mean_sd(finite) if len(finite) >= 2 else (float("nan"), float("nan"))
                steps = steps_stats(block)
                record = {
                    "rate_seed0": "" if math.isnan(rates[0]) else f"{rates[0]:.8f}",
                    "rate_seed42": "" if math.isnan(rates[42]) else f"{rates[42]:.8f}",
                    "rate_seed123": "" if math.isnan(rates[123]) else f"{rates[123]:.8f}",
                    "mean_across_seeds": "" if math.isnan(mean) else f"{mean:.8f}",
                    "sd_across_seeds": "" if math.isnan(sd) else f"{sd:.8f}",
                    "total_n": len(block),
                    "total_successes": successes,
                    "total_success_rate": f"{(successes / len(block)):.8f}",
                    "wilson_low": f"{low:.8f}",
                    "wilson_high": f"{high:.8f}",
                    "mean_steps_to_success": steps[0],
                    "median_steps_to_success": steps[1],
                    "n_success": steps[2],
                    "row_type": "distance_bin",
                    "horizon": horizon,
                    "L": "",
                    "distance_bin": label,
                }
            summary.append(record)
            print(
                f"distance H{horizon} {label} n {record['total_n']} rate {record.get('total_success_rate', '')}",
                flush=True,
            )
    fields = ["row_type", "horizon", "L", "distance_bin", *rate_fields_tail]
    write_csv(OUT / "spatial_horizon_summary.csv", fields, summary)
    save_spatial_plots(summary, OUT / "spatial_steps_to_success.png", OUT / "spatial_horizon_comparison.png")
    save_distance_plot(summary, OUT / "success_vs_initial_distance.png")


def copied_p1_rows(n_episodes: int) -> list[dict]:
    stage0 = load_stage0_rows()
    rows = read_csv(P1_CSV)
    if len(rows) != n_episodes * len(MODEL_SEEDS):
        raise SystemExit(f"P1 has {len(rows)} rows, expected {n_episodes * len(MODEL_SEEDS)}")
    copied = []
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id >= n_episodes:
            continue
        source = stage0[episode_id]
        if int(row["delay_steps"]) != 0 or int(row["horizon"]) != 100 or int(row["goal_noise_applied"]) != 0:
            raise SystemExit("P1 row is not the locked zero-delay horizon-100 protocol")
        if int(row["reset_seed"]) != DELAY_RESET_BASE + episode_id:
            raise SystemExit("P1 reset seed drifted")
        for column, source_column in (
            ("mass", "mass"),
            ("friction", "friction"),
            ("damping_scale", "damping_scale"),
            ("actuator_gain_scale", "gain_scale"),
        ):
            if row[column] != fmt(float(source[source_column])):
                raise SystemExit(f"P1 {column} does not match Stage 0 episode {episode_id}")
        copied.append(
            {
                "condition": "P1-all",
                "source": "copied_from_P1_joint_physical_no_delay.csv",
                "policy_seed": int(row["model_seed"]),
                "episode_id": episode_id,
                "reset_seed": int(row["reset_seed"]),
                "delay_steps": 0,
                "horizon": 100,
                "mass": row["mass"],
                "friction": row["friction"],
                "damping_scale": row["damping_scale"],
                "actuator_gain_scale": row["actuator_gain_scale"],
                "mass_randomized": 1,
                "friction_randomized": 1,
                "damping_randomized": 1,
                "gain_randomized": 1,
                "success": int(row["success"]),
                "steps_used": int(row["steps_used"]),
                "initial_object_x": row["initial_object_x"],
                "initial_object_y": row["initial_object_y"],
                "initial_object_z": row["initial_object_z"],
                "target_x": row["goal_x"],
                "target_y": row["goal_y"],
                "target_z": row["goal_z"],
                "initial_distance": row["initial_object_to_goal_distance"],
                "final_distance": row["final_object_to_goal_distance"],
            }
        )
    return copied


def summarize_ablation(partial_dir: Path, n_episodes: int) -> None:
    copied = copied_p1_rows(n_episodes)
    stored = {"P1-all": copied}
    for condition, flags in CONDITIONS.items():
        block = []
        for seed in MODEL_SEEDS:
            rows = require_rows(ablation_partial(partial_dir, condition, seed), n_episodes)
            for row in rows:
                if row["condition"] != condition or int(row["policy_seed"]) != seed:
                    raise SystemExit("ablation partial drifted")
                if int(row["delay_steps"]) != 0 or int(row["horizon"]) != 100:
                    raise SystemExit("ablation protocol drifted")
                if int(row["reset_seed"]) != DELAY_RESET_BASE + int(row["episode_id"]):
                    raise SystemExit("ablation reset seed drifted")
                observed = (
                    int(row["mass_randomized"]),
                    int(row["friction_randomized"]),
                    int(row["damping_randomized"]),
                    int(row["gain_randomized"]),
                )
                if observed != flags:
                    raise SystemExit(f"{condition} factor flags drifted")
            block.extend(rows)
        for left, right in zip(copied, block):
            if int(left["policy_seed"]) != int(right["policy_seed"]) or int(left["episode_id"]) != int(right["episode_id"]):
                raise SystemExit("ablation alignment drifted")
            for key in SCENE_KEYS:
                if left[key] != right[key]:
                    raise SystemExit(f"{condition} scene is not paired with P1")
        stored[condition] = block
    order = ["P1-all", *CONDITIONS]
    raw = [row for condition in order for row in stored[condition]]
    write_csv(OUT / "physical_ablation_raw.csv", ABLATION_FIELDS, raw)

    p0_rows = read_csv(P0_CSV)
    p0_flags = [int(row["success"]) for row in p0_rows]
    p0_by_seed = {seed: [] for seed in MODEL_SEEDS}
    for row in p0_rows:
        p0_by_seed[int(row["model_seed"])].append(int(row["success"]))
    p0_rates = {seed: sum(p0_by_seed[seed]) / len(p0_by_seed[seed]) for seed in MODEL_SEEDS}

    summary = []
    fields = [
        "condition",
        "source",
        "rate_seed0",
        "rate_seed42",
        "rate_seed123",
        "mean_across_seeds",
        "sd_across_seeds",
        "total_n",
        "total_successes",
        "total_success_rate",
        "wilson_low",
        "wilson_high",
        "mean_steps_to_success",
        "median_steps_to_success",
        "n_success",
        "change_from_p0",
        "change_from_p1_all",
    ]
    p0_record = rate_record(
        p0_rates,
        sum(p0_flags),
        len(p0_flags),
        steps_stats(
            [
                {"success": row["success"], "steps_used": row["steps_used"]}
                for row in p0_rows
            ]
        ),
    )
    p0_record.update(
        {
            "condition": "P0-reference",
            "source": "copied_from_P0_baseline.csv",
            "change_from_p0": "0.00000000",
            "change_from_p1_all": "",
        }
    )
    summary.append(p0_record)
    p1_rate = None
    p1_seed_rates = None
    for condition in order:
        block = stored[condition]
        rates, successes, n = seed_rates(block)
        record = rate_record(rates, successes, n, steps_stats(block))
        pooled = successes / n
        record.update(
            {
                "condition": condition,
                "source": block[0]["source"],
                "change_from_p0": f"{(pooled - (sum(p0_flags) / len(p0_flags))):.8f}",
                "change_from_p1_all": "0.00000000" if condition == "P1-all" else f"{(pooled - p1_rate):.8f}",
            }
        )
        if condition == "P1-all":
            p1_rate = pooled
            p1_seed_rates = rates
            p0_record["change_from_p1_all"] = f"{((sum(p0_flags) / len(p0_flags)) - p1_rate):.8f}"
        summary.append(record)
        print(
            f"ablation {condition} rate {record['total_success_rate']} "
            f"from P1 {record['change_from_p1_all']}",
            flush=True,
        )
    del p1_seed_rates
    write_csv(OUT / "physical_ablation_summary.csv", fields, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", nargs="+", required=True, choices=["delay", "spatial", "ablation"])
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--partial-dir", type=Path, default=PARTIAL)
    args = parser.parse_args()
    if "delay" in args.which:
        summarize_delay(args.partial_dir, args.episodes)
    if "spatial" in args.which:
        summarize_spatial(args.partial_dir, args.episodes)
    if "ablation" in args.which:
        summarize_ablation(args.partial_dir, args.episodes)


if __name__ == "__main__":
    main()
