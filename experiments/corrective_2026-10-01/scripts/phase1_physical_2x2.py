"""Phase 1 locked 2x2. Does not retrain. Does not edit the manuscript.

P0  true zero delay, XML defaults
P1  true zero delay, Stage 0 mass/friction/damping_scale/gain_scale rows
P2  Stage 0 delay rows, XML defaults
P3  Stage 0 delay rows and Stage 0 physical rows

Goal noise from the Stage 0 CSV is not applied. Horizon is 100.
Scene seed is 4000000 + episode_id for every condition and every model.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PARTIAL = OUT / "partial"
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import panda_mujoco_gym  # noqa: E402, F401
from sb3_contrib import TQC  # noqa: E402

MODEL_ROOT = ROOT.parent / "panda_project_physics_generalization" / "logs"
MODELS = {
    0: MODEL_ROOT / "batch512_her4_seed0_2025-06-17_13-50-58" / "final_model.zip",
    42: MODEL_ROOT / "batch512_her4_seed42_2025-06-17_23-01-53" / "final_model.zip",
    123: MODEL_ROOT / "batch512_her4_seed123_2025-06-18_23-21-02" / "final_model.zip",
}
STAGE0 = ROOT / "data_and_analysis" / "physical_element" / "eval_multi_param_seed0.csv"
STAGE0_OTHERS = [
    ROOT / "data_and_analysis" / "physical_element" / "eval_multi_param_seed42.csv",
    ROOT / "data_and_analysis" / "physical_element" / "eval_multi_param_seed123.csv",
]
HORIZON = 100
CONDITIONS = ("P0", "P1", "P2", "P3")
OFFICIAL = {
    "P0": OUT / "P0_baseline.csv",
    "P1": OUT / "P1_joint_physical_no_delay.csv",
    "P2": OUT / "P2_delay_only.csv",
    "P3": OUT / "P3_full_joint_original.csv",
}
FIELDS = [
    "condition",
    "model_seed",
    "episode_id",
    "stage0_rng_seed",
    "stage0_episode",
    "success",
    "delay_steps",
    "mass",
    "friction",
    "damping_scale",
    "actuator_gain_scale",
    "arm_joint_damping",
    "obj_joint_damping",
    "actuator1_gain",
    "actuator3_gain",
    "actuator5_gain",
    "gripper_gain",
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "initial_object_to_goal_distance",
    "steps_used",
    "final_object_to_goal_distance",
    "horizon",
    "reset_seed",
    "goal_noise_applied",
]
T_CRIT_DF2 = 4.302652729911275
Z_95 = 1.959963984540054
PARAM_COLS = [
    "seed",
    "episode",
    "mass",
    "friction",
    "damping_scale",
    "gain_scale",
    "delay_steps",
    "goal_noise_x",
    "goal_noise_y",
    "goal_noise_z",
]


class Delay:
    def __init__(self, delay_steps: int, action_dim: int):
        self.delay_steps = int(delay_steps)
        zero = np.zeros(action_dim, dtype=np.float32)
        self.pending = deque(zero.copy() for _ in range(self.delay_steps))

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self.delay_steps == 0:
            return action
        executed = self.pending.popleft()
        self.pending.append(action)
        return executed


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_stage0_rows() -> list[dict]:
    primary = read_csv(STAGE0)
    primary_key = [tuple(row[col] for col in PARAM_COLS) for row in primary]
    for path in STAGE0_OTHERS:
        other = [tuple(row[col] for col in PARAM_COLS) for row in read_csv(path)]
        if other != primary_key:
            raise SystemExit(f"Stage 0 parameter rows differ: {path}")
    rows = sorted(primary, key=lambda row: (int(float(row["seed"])), int(float(row["episode"]))))
    if len(rows) != 600:
        raise SystemExit(f"expected 600 Stage 0 rows, got {len(rows)}")
    keys = [(int(float(row["seed"])), int(float(row["episode"]))) for row in rows]
    if len(set(keys)) != 600:
        raise SystemExit("Stage 0 seed/episode keys are not unique")
    delays = {int(float(row["delay_steps"])) for row in rows}
    if delays != set(range(1, 10)):
        raise SystemExit(f"Stage 0 delay values are not 1..9: {sorted(delays)}")
    return rows


def completed_episodes(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    rows = read_csv(path)
    ids = [int(row["episode_id"]) for row in rows]
    if ids != list(range(len(ids))):
        raise SystemExit(f"{path} is not a contiguous prefix")
    return len(ids)


def fmt(value: float) -> str:
    return f"{value:.8f}"


def success_flag(info) -> int:
    return int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)


def apply_physics(base, originals, mass, friction, damping_scale, gain_scale) -> None:
    base.model.body_mass[originals["obj_body"]] = mass
    base.model.geom_friction[originals["obj_geom"], 0] = friction
    base.model.dof_damping[:] = originals["damping"] * damping_scale
    base.model.actuator_gainprm[:, 0] = originals["gain"] * gain_scale


def run_model(model_seed: int, n_episodes: int, partial_dir: Path) -> None:
    rows = load_stage0_rows()[:n_episodes]
    model_path = MODELS[model_seed]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    partial_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("FrankaPushSparse-v0", max_episode_steps=HORIZON, disable_env_checker=True)
    base = env.unwrapped
    if int(env.spec.max_episode_steps) != HORIZON:
        raise SystemExit(f"horizon is {env.spec.max_episode_steps}, expected {HORIZON}")
    originals = {
        "obj_body": int(base._model_names.body_name2id["obj"]),
        "obj_geom": int(base._model_names.geom_name2id["obj_geom"]),
        "mass": float(base.model.body_mass[int(base._model_names.body_name2id["obj"])]),
        "friction": float(base.model.geom_friction[int(base._model_names.geom_name2id["obj_geom"]), 0]),
        "damping": base.model.dof_damping.copy(),
        "gain": base.model.actuator_gainprm[:, 0].copy(),
    }
    if not math.isclose(originals["mass"], 1.0) or not math.isclose(originals["friction"], 1.0):
        raise SystemExit(f"unexpected defaults mass={originals['mass']} friction={originals['friction']}")
    model = TQC.load(str(model_path), env=env, device="cpu")
    try:
        for condition in CONDITIONS:
            path = partial_dir / f"{condition}_seed{model_seed}.csv"
            start = completed_episodes(path)
            if start >= n_episodes:
                print(f"[model {model_seed}] {condition} already complete", flush=True)
                continue
            new_file = start == 0
            handle = path.open("a", newline="")
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
                handle.flush()
            print(f"[model {model_seed}] {condition} from episode {start}", flush=True)
            t0 = time.perf_counter()
            for episode_id in range(start, n_episodes):
                source = rows[episode_id]
                delay = 0 if condition in ("P0", "P1") else int(float(source["delay_steps"]))
                if condition in ("P0", "P2"):
                    mass = originals["mass"]
                    friction = originals["friction"]
                    damping_scale = 1.0
                    gain_scale = 1.0
                else:
                    mass = float(source["mass"])
                    friction = float(source["friction"])
                    damping_scale = float(source["damping_scale"])
                    gain_scale = float(source["gain_scale"])
                reset_seed = 4_000_000 + episode_id
                obs, _ = env.reset(seed=reset_seed)
                apply_physics(base, originals, mass, friction, damping_scale, gain_scale)
                achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                delay_buffer = Delay(delay, int(np.asarray(env.action_space.shape)[0]))
                terminated = False
                truncated = False
                info = {}
                steps = 0
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    applied = delay_buffer.apply(action)
                    obs, _, terminated, truncated, info = env.step(applied)
                    steps += 1
                final_achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                final_desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                if not np.allclose(final_desired, desired):
                    raise SystemExit(f"goal changed during episode {episode_id}")
                writer.writerow(
                    {
                        "condition": condition,
                        "model_seed": model_seed,
                        "episode_id": episode_id,
                        "stage0_rng_seed": int(float(source["seed"])),
                        "stage0_episode": int(float(source["episode"])),
                        "success": success_flag(info),
                        "delay_steps": delay,
                        "mass": fmt(mass),
                        "friction": fmt(friction),
                        "damping_scale": fmt(damping_scale),
                        "actuator_gain_scale": fmt(gain_scale),
                        "arm_joint_damping": fmt(float(base.model.dof_damping[0])),
                        "obj_joint_damping": fmt(float(base.model.dof_damping[9])),
                        "actuator1_gain": fmt(float(base.model.actuator_gainprm[0, 0])),
                        "actuator3_gain": fmt(float(base.model.actuator_gainprm[2, 0])),
                        "actuator5_gain": fmt(float(base.model.actuator_gainprm[4, 0])),
                        "gripper_gain": fmt(float(base.model.actuator_gainprm[7, 0])),
                        "initial_object_x": fmt(float(achieved[0])),
                        "initial_object_y": fmt(float(achieved[1])),
                        "initial_object_z": fmt(float(achieved[2])),
                        "goal_x": fmt(float(desired[0])),
                        "goal_y": fmt(float(desired[1])),
                        "goal_z": fmt(float(desired[2])),
                        "initial_object_to_goal_distance": fmt(float(np.linalg.norm(achieved - desired))),
                        "steps_used": steps,
                        "final_object_to_goal_distance": fmt(float(np.linalg.norm(final_achieved - desired))),
                        "horizon": HORIZON,
                        "reset_seed": reset_seed,
                        "goal_noise_applied": 0,
                    }
                )
                handle.flush()
                if (episode_id + 1) % 25 == 0 or episode_id + 1 == n_episodes:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[model {model_seed}] {condition} {episode_id + 1}/{n_episodes} "
                        f"in {elapsed:.0f}s",
                        flush=True,
                    )
            handle.close()
    finally:
        env.close()


def wilson(successes: int, n: int) -> tuple[float, float]:
    p = successes / n
    den = 1 + Z_95**2 / n
    centre = (p + Z_95**2 / (2 * n)) / den
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / den
    return centre - half, centre + half


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(var)
    half = T_CRIT_DF2 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def check_pairing(tables: dict[str, list[dict]]) -> None:
    for condition in CONDITIONS:
        for row in tables[condition]:
            if int(row["horizon"]) != HORIZON or int(row["goal_noise_applied"]) != 0:
                raise SystemExit("horizon or goal noise drifted")
            if int(row["reset_seed"]) != 4_000_000 + int(row["episode_id"]):
                raise SystemExit("reset seed drifted")
    base = tables["P0"]
    for condition in CONDITIONS:
        if len(tables[condition]) != len(base):
            raise SystemExit(f"{condition} row count {len(tables[condition])} != {len(base)}")
        for left, right in zip(base, tables[condition]):
            if left["model_seed"] != right["model_seed"] or left["episode_id"] != right["episode_id"]:
                raise SystemExit("episode alignment drifted")
            for key in ("initial_object_x", "initial_object_y", "initial_object_z", "goal_x", "goal_y", "goal_z"):
                if left[key] != right[key]:
                    raise SystemExit(f"scene not paired at {condition} episode {left['episode_id']}")
    for left, right in zip(tables["P1"], tables["P3"]):
        for key in ("mass", "friction", "damping_scale", "actuator_gain_scale"):
            if left[key] != right[key]:
                raise SystemExit("P1/P3 physical draws are not paired")
        if int(left["delay_steps"]) != 0 or int(right["delay_steps"]) == 0:
            raise SystemExit("P1/P3 delay assignment drifted")
    for left, right in zip(tables["P2"], tables["P3"]):
        if left["delay_steps"] != right["delay_steps"]:
            raise SystemExit("P2/P3 delay draws are not paired")
    for row in tables["P0"] + tables["P2"]:
        if not math.isclose(float(row["mass"]), 1.0) or not math.isclose(float(row["friction"]), 1.0):
            raise SystemExit("default physical cell was randomized")
        if not math.isclose(float(row["damping_scale"]), 1.0) or not math.isclose(float(row["actuator_gain_scale"]), 1.0):
            raise SystemExit("default scale cell was randomized")
    for row in tables["P0"] + tables["P1"]:
        if int(row["delay_steps"]) != 0:
            raise SystemExit("zero-delay cell has nonzero delay")


def summarize(partial_dir: Path, n_episodes: int) -> None:
    tables = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for model_seed in (0, 42, 123):
            path = partial_dir / f"{condition}_seed{model_seed}.csv"
            rows = read_csv(path)
            if len(rows) != n_episodes:
                raise SystemExit(f"{path} has {len(rows)} rows, expected {n_episodes}")
            tables[condition].extend(rows)
        with OFFICIAL[condition].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(tables[condition])
    check_pairing(tables)

    summary_fields = [
        "row_type",
        "condition",
        "model_seed",
        "n",
        "successes",
        "failures",
        "success_rate",
        "mean_across_seeds",
        "sd_across_seeds",
        "ci95_low",
        "ci95_high",
        "ci_method",
        "pooled_wilson_low",
        "pooled_wilson_high",
    ]
    summary_rows = []
    pooled = {}
    per_seed = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        grouped = {0: [], 42: [], 123: []}
        for row in tables[condition]:
            grouped[int(row["model_seed"])].append(int(row["success"]))
        rates = []
        for model_seed in (0, 42, 123):
            flags = grouped[model_seed]
            k = sum(flags)
            n = len(flags)
            rate = k / n
            rates.append(rate)
            per_seed[condition][model_seed] = rate
            summary_rows.append(
                {
                    "row_type": "rate",
                    "condition": condition,
                    "model_seed": model_seed,
                    "n": n,
                    "successes": k,
                    "failures": n - k,
                    "success_rate": f"{rate:.8f}",
                    "mean_across_seeds": "",
                    "sd_across_seeds": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "ci_method": "",
                    "pooled_wilson_low": "",
                    "pooled_wilson_high": "",
                }
            )
        flags = [int(row["success"]) for row in tables[condition]]
        k = sum(flags)
        n = len(flags)
        mean, sd, low, high = mean_sd_ci(rates)
        wlow, whigh = wilson(k, n)
        pooled[condition] = k / n
        summary_rows.append(
            {
                "row_type": "rate",
                "condition": condition,
                "model_seed": "all",
                "n": n,
                "successes": k,
                "failures": n - k,
                "success_rate": f"{(k / n):.8f}",
                "mean_across_seeds": f"{mean:.8f}",
                "sd_across_seeds": f"{sd:.8f}",
                "ci95_low": f"{low:.8f}",
                "ci95_high": f"{high:.8f}",
                "ci_method": "student_t_df2_on_three_seed_rates",
                "pooled_wilson_low": f"{wlow:.8f}",
                "pooled_wilson_high": f"{whigh:.8f}",
            }
        )

    effect_defs = {
        "physical_P0_minus_P1": ("P0", "P1", lambda a, b: a - b),
        "delay_P0_minus_P2": ("P0", "P2", lambda a, b: a - b),
        "full_P0_minus_P3": ("P0", "P3", lambda a, b: a - b),
        "interaction_P3_minus_P2_minus_P1_plus_P0": ("P3", "P2", None),
    }
    for name, (left, right, op) in effect_defs.items():
        if name.startswith("interaction"):
            diffs = [
                per_seed["P3"][s] - per_seed["P2"][s] - per_seed["P1"][s] + per_seed["P0"][s]
                for s in (0, 42, 123)
            ]
            pooled_value = pooled["P3"] - pooled["P2"] - pooled["P1"] + pooled["P0"]
        else:
            diffs = [op(per_seed[left][s], per_seed[right][s]) for s in (0, 42, 123)]
            pooled_value = op(pooled[left], pooled[right])
        mean, sd, low, high = mean_sd_ci(diffs)
        summary_rows.append(
            {
                "row_type": "effect",
                "condition": name,
                "model_seed": "all",
                "n": 3,
                "successes": "",
                "failures": "",
                "success_rate": f"{pooled_value:.8f}",
                "mean_across_seeds": f"{mean:.8f}",
                "sd_across_seeds": f"{sd:.8f}",
                "ci95_low": f"{low:.8f}",
                "ci95_high": f"{high:.8f}",
                "ci_method": "student_t_df2_on_three_seed_differences",
                "pooled_wilson_low": "",
                "pooled_wilson_high": "",
            }
        )

    summary_path = OUT / "physical_2x2_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")
    for row in summary_rows:
        if row["model_seed"] == "all":
            print(
                f"{row['row_type']} {row['condition']} pooled={row['success_rate']} "
                f"mean={row['mean_across_seeds']} sd={row['sd_across_seeds']} "
                f"ci=({row['ci95_low']}, {row['ci95_high']})",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, choices=sorted(MODELS))
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--partial-dir", type=Path, default=PARTIAL)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.partial_dir, args.episodes)
        return
    if args.model_seed is None:
        raise SystemExit("--model-seed is required unless --summarize")
    run_model(args.model_seed, args.episodes, args.partial_dir)


if __name__ == "__main__":
    main()
