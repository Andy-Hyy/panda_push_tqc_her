"""Shared locks for the 2026-10-01 corrective round 2.

Does not retrain. Does not edit the manuscript.
Delay 0 returns the current action. One decision is 0.5 s when the XML
timestep, n_substeps and the 10-step inner loop are the values this tree uses.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PARTIAL = OUT / "partial" / "round2"
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
P0_CSV = OUT / "P0_baseline.csv"
P1_CSV = OUT / "P1_joint_physical_no_delay.csv"
P2_CSV = OUT / "P2_delay_only.csv"
P3_CSV = OUT / "P3_full_joint_original.csv"
DELAYS = tuple(range(10))
LENGTHS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
HORIZONS = (50, 100)
N_EPISODES = 600
MODEL_SEEDS = (0, 42, 123)
DELAY_RESET_BASE = 4_000_000
SPATIAL_RESET_BASE = 6_000_000
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
# Worker constants. Used only to reject rows outside the original sampler.
MASS_RANGE = (0.05, 3.0)
FRICTION_RANGE = (0.05, 3.0)
DAMPING_SCALE_RANGE = (0.001, 0.3)
GAIN_SCALE_RANGE = (0.0, 10.0)


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


def fmt(value: float) -> str:
    return f"{value:.8f}"


def completed_episodes(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    rows = read_csv(path)
    ids = [int(row["episode_id"]) for row in rows]
    if ids != list(range(len(ids))):
        raise SystemExit(f"{path} is not a contiguous episode_id prefix")
    return len(ids)


def open_writer(path: Path, fields: list[str], start: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not path.exists()) or path.stat().st_size == 0
    handle = path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if new_file:
        writer.writeheader()
        handle.flush()
    if start and new_file:
        raise SystemExit(f"{path} resumed from {start} without a header")
    return handle, writer


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        raise SystemExit("Wilson interval needs a positive n")
    p = successes / n
    den = 1 + Z_95**2 / n
    centre = (p + Z_95**2 / (2 * n)) / den
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / den
    return centre - half, centre + half


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


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
    for row in rows:
        mass = float(row["mass"])
        friction = float(row["friction"])
        damping = float(row["damping_scale"])
        gain = float(row["gain_scale"])
        if not (MASS_RANGE[0] <= mass < MASS_RANGE[1]):
            raise SystemExit(f"mass outside Stage 0 range: {mass}")
        if not (FRICTION_RANGE[0] <= friction < FRICTION_RANGE[1]):
            raise SystemExit(f"friction outside Stage 0 range: {friction}")
        if not (DAMPING_SCALE_RANGE[0] <= damping < DAMPING_SCALE_RANGE[1]):
            raise SystemExit(f"damping scale outside Stage 0 range: {damping}")
        if not (GAIN_SCALE_RANGE[0] <= gain < GAIN_SCALE_RANGE[1]):
            raise SystemExit(f"gain scale outside Stage 0 range: {gain}")
    return rows


def success_flag(info) -> int:
    return int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)


def assert_control_period(base) -> float:
    period = float(base.model.opt.timestep) * int(base.n_substeps) * 10
    if not math.isclose(period, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"control period is {period}, expected 0.5 s")
    return period


def assert_default_ranges(base) -> None:
    if not math.isclose(float(base.goal_xy_range), 0.3) or not math.isclose(float(base.obj_xy_range), 0.3):
        raise SystemExit(f"expected 0.30 m ranges, got goal={base.goal_xy_range} obj={base.obj_xy_range}")
    if not np.allclose(base.goal_range_low, [0.45, -0.15, 0.0]) or not np.allclose(base.goal_range_high, [0.75, 0.15, 0.0]):
        raise SystemExit(f"goal box drifted: {base.goal_range_low} {base.goal_range_high}")
    if not np.allclose(base.obj_range_low, [0.45, -0.15, 0.0]) or not np.allclose(base.obj_range_high, [0.75, 0.15, 0.0]):
        raise SystemExit(f"object box drifted: {base.obj_range_low} {base.obj_range_high}")


def set_goal_side(base, side: float) -> None:
    side = float(side)
    base.goal_xy_range = side
    base.goal_range_low = np.array([-side / 2 + base.goal_x_offset, -side / 2, 0.0])
    base.goal_range_high = np.array([side / 2 + base.goal_x_offset, side / 2, base.goal_z_range])
    base.goal_range_low[0] += 0.6
    base.goal_range_high[0] += 0.6
    expected_low = np.array([-side / 2 + 0.6, -side / 2, 0.0])
    expected_high = np.array([side / 2 + 0.6, side / 2, 0.0])
    if not np.allclose(base.goal_range_low, expected_low) or not np.allclose(base.goal_range_high, expected_high):
        raise SystemExit(f"goal box for L={side} is {base.goal_range_low} {base.goal_range_high}")
    if not math.isclose(float(base.obj_xy_range), 0.3):
        raise SystemExit("object spawn range changed while setting the target side")
    if not np.allclose(base.obj_range_low, [0.45, -0.15, 0.0]) or not np.allclose(base.obj_range_high, [0.75, 0.15, 0.0]):
        raise SystemExit("object box changed while setting the target side")


def set_horizon(env, horizon: int) -> None:
    found = False
    node = env
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            node._max_episode_steps = int(horizon)
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit wrapper not found")


def assert_horizon_binds(env, horizon: int) -> None:
    set_horizon(env, horizon)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        steps = 0
        info = {}
        while not done:
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
            if steps > horizon:
                raise SystemExit(f"horizon {horizon} did not stop the episode")
        if success_flag(info):
            continue
        if steps != horizon:
            raise SystemExit(f"failed episode stopped at {steps}, horizon {horizon}")
        return
    raise SystemExit(f"could not confirm horizon {horizon}")


def make_env(horizon: int, goal_xy_range: float = 0.3):
    env = gym.make(
        "FrankaPushSparse-v0",
        max_episode_steps=horizon,
        goal_xy_range=goal_xy_range,
        disable_env_checker=True,
    )
    base = env.unwrapped
    if float(base.distance_threshold) != 0.05:
        raise SystemExit(f"distance threshold is {base.distance_threshold}")
    assert_control_period(base)
    set_horizon(env, horizon)
    return env, base


def load_policy(model_seed: int, env):
    path = MODELS[model_seed]
    if not path.exists():
        raise FileNotFoundError(path)
    return TQC.load(str(path), env=env, device="cpu")


def originals_from(base) -> dict:
    obj_body = int(base._model_names.body_name2id["obj"])
    obj_geom = int(base._model_names.geom_name2id["obj_geom"])
    originals = {
        "obj_body": obj_body,
        "obj_geom": obj_geom,
        "mass": float(base.model.body_mass[obj_body]),
        "friction": float(base.model.geom_friction[obj_geom, 0]),
        "damping": base.model.dof_damping.copy(),
        "gain": base.model.actuator_gainprm[:, 0].copy(),
    }
    if not math.isclose(originals["mass"], 1.0) or not math.isclose(originals["friction"], 1.0):
        raise SystemExit(f"unexpected defaults mass={originals['mass']} friction={originals['friction']}")
    return originals


def apply_physics(base, originals, mass, friction, damping_scale, gain_scale) -> None:
    base.model.body_mass[originals["obj_body"]] = mass
    base.model.geom_friction[originals["obj_geom"], 0] = friction
    base.model.dof_damping[:] = originals["damping"] * damping_scale
    base.model.actuator_gainprm[:, 0] = originals["gain"] * gain_scale


def run_episode(env, model, obs, desired, delay_steps: int, horizon: int) -> tuple[int, int, float]:
    delay = Delay(delay_steps, int(np.asarray(env.action_space.shape)[0]))
    terminated = False
    truncated = False
    info = {}
    steps = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(delay.apply(action))
        steps += 1
        if steps > horizon:
            raise SystemExit(f"episode exceeded horizon {horizon}")
    final_achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    final_desired = np.asarray(obs["desired_goal"], dtype=np.float64)
    if not np.allclose(final_desired, desired):
        raise SystemExit("goal changed during the episode")
    final_distance = float(np.linalg.norm(final_achieved - final_desired))
    info_success = success_flag(info)
    distance_success = int(final_distance < 0.05)
    if info_success != distance_success:
        raise SystemExit(
            f"success flags disagree: info={info_success} distance={final_distance}"
        )
    if not info_success and steps != horizon:
        raise SystemExit(f"failed episode stopped at {steps}, horizon {horizon}")
    return info_success, steps, final_distance
