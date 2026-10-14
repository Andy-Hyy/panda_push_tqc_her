"""Shared implementation for the controlled SpatialDR and Fixed Large retraining.

Both methods use this file. The only behavioral difference is how the goal-box
side is chosen before each reset. Historical trainers and checkpoints are not
modified. The binary scene cache is a copy of the already verified spatial mjb.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

# Force one compute thread before NumPy or torch load in this process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.pop("MUJOCO_GL", None)

import numpy as np

OUT = Path(__file__).resolve().parents[1]
PANDA_FINAL = OUT.parents[1]
ROUND3_SCRIPTS = PANDA_FINAL / "data_and_analysis" / "corrective_round3_2026-10-01" / "scripts"
OLD_SCRIPTS = PANDA_FINAL / "data_and_analysis" / "corrective_2026-10-01" / "scripts"
SPATIAL_MJB = PANDA_FINAL / "data_and_analysis" / "spatial_clean_2026-10-02" / "cache" / "panda_push.mjb"
CANONICAL_PUSH = PANDA_FINAL / "panda_mujoco_gym" / "assets" / "push.xml"
CACHE_MJB = OUT / "cache" / "panda_push.mjb"
PROTOCOL_PATH = OUT / "training_protocol.json"

METHODS = ("spatial_dr", "fixed_large")
TRAIN_SEEDS = (0, 42, 123)
N_ENVS = 6
TOTAL_TIMESTEPS = 1_000_000
HORIZON = 100
FIXED_SIDE = 0.80
DR_LOW = 0.30
DR_HIGH = 0.80
SIDES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
PHYSICS = {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}
# Multiples of n_envs nearest to the requested budgets. The learn loop stops at
# the first multiple that is at least TOTAL_TIMESTEPS, which is 1_000_002.
CHECKPOINT_STEPS = (199_998, 400_002, 499_998, 600_000, 799_998, 1_000_002)
SMOKE_TIMESTEPS = 20_000
SMOKE_EXPECTED_STEPS = 20_004
PROTOCOL_ID = "spatial_train_controlled_clean_reset_h100"
# numpy Generator.uniform is half-open on the right. Fixed Large assigns 0.80
# and does not draw from this generator.
SAMPLING = {
    "spatial_dr": "each episode draws L with numpy Generator.uniform(0.30, 0.80), so L is in [0.30, 0.80)",
    "fixed_large": "each episode assigns L=0.80; the draw generator is not consumed",
    "given_L": "object and goal are then uniform in their boxes via the environment np_random",
    "l_seed": "8100000 + train_seed * 100 + env_index, PCG64, independent of env.np_random",
    "env_seed": "SB3 VecEnv.seed(train_seed) passes train_seed + env_index on the first reset only",
    "goal_z": "goal_z_range is 0, so sampled goal z is the object height; the box check is xy",
}

TRAIN_CONFIG = {
    "policy": "MultiInputPolicy",
    "learning_rate": 0.001,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 512,
    "tau": 0.05,
    "gamma": 0.95,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "top_quantiles_to_drop_per_net": 2,
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "optimize_memory_usage": False,
    "stats_window_size": 100,
    "action_noise": None,
    "device": "cpu",
    "n_envs": N_ENVS,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "net_arch_pi": [256, 256, 256],
    "net_arch_qf": [256, 256, 256],
    "n_quantiles": 25,
    "reward_type": "sparse",
    "success_distance": "strictly < 0.05",
    "reward_rule": "environment compute_reward: 0 when distance <= 0.05, else -1",
    "horizon": HORIZON,
    "total_timesteps_requested": TOTAL_TIMESTEPS,
    "timestep_count": "sum of environment transitions across the six environments",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def import_helpers():
    import sys

    for path in (OLD_SCRIPTS, ROUND3_SCRIPTS, PANDA_FINAL):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def install_binary_loader() -> None:
    import mujoco

    canonical_xml = os.path.realpath(CANONICAL_PUSH)
    mjb = str(CACHE_MJB)

    def from_xml_path(filename, *args, **kwargs):
        if os.path.realpath(filename) == canonical_xml:
            return mujoco.MjModel.from_binary_path(mjb)
        raise SystemExit(f"unexpected XML load outside the verified cache: {filename}")

    mujoco.MjModel.from_xml_path = from_xml_path


def ensure_mjb() -> str:
    expected = "ff842fd8407c1edde8058bf004774523283bcdc6b81ed33240c945380798fddd"
    source = sha256_file(SPATIAL_MJB)
    if source != expected:
        raise SystemExit("spatial mjb hash changed; refusing to copy an unverified model")
    CACHE_MJB.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_MJB.exists() or sha256_file(CACHE_MJB) != expected:
        CACHE_MJB.write_bytes(SPATIAL_MJB.read_bytes())
    if sha256_file(CACHE_MJB) != expected:
        raise SystemExit("copied mjb hash does not match the spatial cache")
    return expected


def l_seed(train_seed: int, env_index: int) -> int:
    return 8_100_000 + int(train_seed) * 100 + int(env_index)


def env_reset_seed(train_seed: int, env_index: int) -> int:
    """Seed SB3 applies on the first reset of environment env_index."""
    return int(train_seed) + int(env_index)


def validation_reset_seed(side: float, episode_id: int) -> int:
    return 7_000_000 + int(round(side * 100)) * 1000 + int(episode_id)


def final_reset_seed(side: float, episode_id: int) -> int:
    return 6_000_000 + int(round(side * 100)) * 1000 + int(episode_id)


def prepare_base(base) -> tuple[dict, dict]:
    import_helpers()
    from phaseC_mass_implementation import install_episode_reset, object_ids, snapshot

    pristine = snapshot(base)
    ids = object_ids(base)
    install_episode_reset(base, pristine, ids, PHYSICS, "C")
    return pristine, ids


def make_push_env(side: float = 0.30):
    import gymnasium as gym

    ensure_mjb()
    install_binary_loader()
    import panda_mujoco_gym  # noqa: F401  registers FrankaPushSparse-v0

    env = gym.make(
        "FrankaPushSparse-v0",
        max_episode_steps=HORIZON,
        goal_xy_range=float(side),
        disable_env_checker=True,
    )
    found = False
    node = env
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            if int(node._max_episode_steps) != HORIZON:
                raise SystemExit(f"TimeLimit is {node._max_episode_steps}")
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit missing")
    base = env.unwrapped
    if float(base.distance_threshold) != 0.05:
        raise SystemExit(f"distance threshold {base.distance_threshold}")
    if bool(base.block_gripper) is not True:
        raise SystemExit("push training expects a blocked gripper")
    pristine, ids = prepare_base(base)
    return env, base, pristine, ids


def assert_side(base, side: float) -> None:
    import_helpers()
    from round2_common import assert_control_period, set_goal_side

    set_goal_side(base, float(side))
    low = np.asarray(base.goal_range_low, dtype=np.float64)
    high = np.asarray(base.goal_range_high, dtype=np.float64)
    if not np.allclose(high[:2] - low[:2], [side, side], rtol=0.0, atol=1e-8):
        raise SystemExit(f"goal span {high[:2] - low[:2]} for L={side}")
    center = 0.5 * (low + high)
    if not np.allclose(center[:2], [0.6, 0.0], rtol=0.0, atol=1e-8):
        raise SystemExit(f"goal center {center[:2]}")
    if abs(float(base.goal_xy_range) - float(side)) > 0.0:
        raise SystemExit("goal_xy_range attribute was not set to the requested side")
    obj_low = np.asarray(base.obj_range_low, dtype=np.float64)
    obj_high = np.asarray(base.obj_range_high, dtype=np.float64)
    if not np.allclose(obj_high[:2] - obj_low[:2], [0.3, 0.3], rtol=0.0, atol=1e-12):
        raise SystemExit("object box is not 0.30 m")
    assert_control_period(base)


def in_box(point, low, high) -> bool:
    xy = np.asarray(point, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    return bool(low[0] - 1e-9 <= xy[0] <= high[0] + 1e-9 and low[1] - 1e-9 <= xy[1] <= high[1] + 1e-9)


def pose_after_reset(base, obs) -> dict:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
    goal = np.asarray(base.goal, dtype=np.float64)
    if not np.allclose(desired, goal, rtol=0.0, atol=1e-6):
        raise SystemExit("observation desired_goal differs from env.goal")
    return {
        "object": achieved,
        "goal": goal,
        "distance": float(np.linalg.norm(achieved - goal)),
        "goal_low": np.asarray(base.goal_range_low, dtype=np.float64).copy(),
        "goal_high": np.asarray(base.goal_range_high, dtype=np.float64).copy(),
        "obj_low": np.asarray(base.obj_range_low, dtype=np.float64).copy(),
        "obj_high": np.asarray(base.obj_range_high, dtype=np.float64).copy(),
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def dependency_paths() -> dict[str, Path]:
    return {
        "phaseC_mass_implementation.py": ROUND3_SCRIPTS / "phaseC_mass_implementation.py",
        "phaseB_diagnose.py": ROUND3_SCRIPTS / "phaseB_diagnose.py",
        "phaseE_formal.py": ROUND3_SCRIPTS / "phaseE_formal.py",
        "round2_common.py": OLD_SCRIPTS / "round2_common.py",
        "panda_env.py": PANDA_FINAL / "panda_mujoco_gym" / "envs" / "panda_env.py",
        "push.py": PANDA_FINAL / "panda_mujoco_gym" / "envs" / "push.py",
        "push.xml": CANONICAL_PUSH,
        "panda_mocap.xml": PANDA_FINAL / "panda_mujoco_gym" / "assets" / "panda_mocap.xml",
    }


def implementation_paths() -> dict[str, Path]:
    scripts = Path(__file__).resolve().parent
    paths = {
        "controlled_common.py": Path(__file__).resolve(),
        "train_controlled.py": scripts / "train_controlled.py",
        "eval_controlled.py": scripts / "eval_controlled.py",
        "preflight_controlled.py": scripts / "preflight_controlled.py",
        "panda_push.mjb": CACHE_MJB,
    }
    paths.update(dependency_paths())
    return paths


def package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    names = (
        "numpy",
        "torch",
        "mujoco",
        "gymnasium",
        "gymnasium-robotics",
        "stable-baselines3",
        "sb3-contrib",
        "matplotlib",
        "pandas",
        "seaborn",
    )
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "missing"
    return found


def read_protocol() -> dict:
    import json

    if not PROTOCOL_PATH.exists():
        raise SystemExit("training protocol is not locked")
    return json.loads(PROTOCOL_PATH.read_text())


def assert_protocol_code() -> dict:
    protocol = read_protocol()
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit(f"protocol id is {protocol.get('protocol_id')}")
    for name, path in implementation_paths().items():
        current = sha256_file(path)
        recorded = protocol["files"][name]
        if current != recorded:
            raise SystemExit(f"{name} hash {current} does not match the locked protocol")
    versions = package_versions()
    for name, recorded in protocol["packages"].items():
        if versions.get(name) != recorded:
            raise SystemExit(f"{name} version {versions.get(name)} does not match locked {recorded}")
    return protocol
