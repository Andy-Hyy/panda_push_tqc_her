"""Phase B diagnostic rollouts for the frozen selection.csv sample.

Does not retrain, does not edit old CSVs, and does not change XML.
One process should set threads before import when the mode needs it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD_SCRIPTS = ROOT / "data_and_analysis" / "corrective_2026-10-01" / "scripts"
sys.path.insert(0, str(OLD_SCRIPTS))
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import panda_mujoco_gym  # noqa: E402, F401
import torch  # noqa: E402
from round2_common import MODELS, load_policy  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
HORIZON = 100
PROTOCOL_HISTORICAL = "historical_reset_body_mass_after_obs"
PROTOCOL_CLEAN = "clean_reset_v1_default_physics"

# Fixed before any diagnostic result. Outside the 4_000_000 evaluation block.
PRE_A = (
    {"reset_seed": 4_100_001, "delay_steps": 0, "mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0},
    {"reset_seed": 4_100_002, "delay_steps": 3, "mass": 0.1, "friction": 0.2, "damping_scale": 0.05, "gain_scale": 2.0},
)
PRE_B = (
    {"reset_seed": 4_100_003, "delay_steps": 7, "mass": 2.5, "friction": 2.5, "damping_scale": 0.2, "gain_scale": 0.1},
    {"reset_seed": 4_100_004, "delay_steps": 1, "mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0},
)


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

    def queue(self) -> np.ndarray:
        if self.delay_steps == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([np.asarray(item, dtype=np.float32) for item in self.pending])


def make_eval_env():
    env = gym.make(
        "FrankaPushSparse-v0",
        max_episode_steps=HORIZON,
        disable_env_checker=True,
    )
    node = env
    found = False
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            node._max_episode_steps = HORIZON
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit missing")
    base = env.unwrapped
    if float(base.distance_threshold) != 0.05:
        raise SystemExit(f"threshold {base.distance_threshold}")
    return env, base


def snapshot_model(base) -> dict:
    model = base.model
    return {
        "body_mass": model.body_mass.copy(),
        "body_inertia": model.body_inertia.copy(),
        "dof_damping": model.dof_damping.copy(),
        "geom_friction": model.geom_friction.copy(),
        "actuator_gainprm": model.actuator_gainprm.copy(),
    }


def apply_physics(base, originals, mass, friction, damping_scale, gain_scale) -> None:
    base.model.body_mass[originals["obj_body"]] = mass
    base.model.geom_friction[originals["obj_geom"], 0] = friction
    base.model.dof_damping[:] = originals["damping"] * damping_scale
    base.model.actuator_gainprm[:, 0] = originals["gain"] * gain_scale


def capture_state(base, originals) -> dict:
    data = base.data
    model = base.model
    bid = originals["obj_body"]
    gid = originals["obj_geom"]

    def copy(value):
        return np.array(value, copy=True)

    return {
        "time": np.array([data.time], dtype=np.float64),
        "qpos": copy(data.qpos),
        "qvel": copy(data.qvel),
        "act": copy(data.act),
        "ctrl": copy(data.ctrl),
        "qacc": copy(data.qacc),
        "qacc_warmstart": copy(data.qacc_warmstart),
        "qfrc_applied": copy(data.qfrc_applied),
        "mocap_pos": copy(data.mocap_pos),
        "mocap_quat": copy(data.mocap_quat),
        "ncon": np.array([data.ncon], dtype=np.int32),
        "warning_number": copy(data.warning.number),
        "body_mass": np.array([model.body_mass[bid]], dtype=np.float64),
        "body_inertia": copy(model.body_inertia[bid]),
        "body_invweight0": copy(model.body_invweight0[bid]),
        "dof_damping": copy(model.dof_damping),
        "gain": copy(model.actuator_gainprm[:, 0]),
        "friction0": np.array([model.geom_friction[gid, 0]], dtype=np.float64),
    }


def capture_obs(obs) -> dict:
    return {
        "observation": np.asarray(obs["observation"], dtype=np.float64).copy(),
        "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float64).copy(),
        "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float64).copy(),
    }


def install_clean_reset(base, pristine: dict) -> None:
    """Full data clear plus the env's own ctrl and posture. Default physics only."""
    snapshot = pristine
    mujoco = base._mujoco

    def _reset_sim() -> bool:
        model = base.model
        data = base.data
        model.body_mass[:] = snapshot["body_mass"]
        model.body_inertia[:] = snapshot["body_inertia"]
        model.dof_damping[:] = snapshot["dof_damping"]
        model.geom_friction[:] = snapshot["geom_friction"]
        model.actuator_gainprm[:] = snapshot["actuator_gainprm"]
        mujoco.mj_setConst(model, data)
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0
        data.ctrl[0:7] = base.neutral_joint_values[0:7]
        base.set_joint_neutral()
        data.qvel[:] = np.copy(base.initial_qvel)
        base.set_mocap_pose(base.initial_mocap_position, base.grasp_site_pose)
        base._sample_object()
        mujoco.mj_forward(model, data)
        return True

    base._reset_sim = _reset_sim


def run_episode(env, base, model, originals, reset_seed, delay_steps, physics, protocol: str, horizon: int | None = None):
    if horizon is None:
        horizon = HORIZON
    obs, _ = env.reset(seed=int(reset_seed))
    after_reset = capture_state(base, originals)
    obs_after_reset = capture_obs(obs)
    if protocol == PROTOCOL_HISTORICAL:
        apply_physics(
            base,
            originals,
            physics["mass"],
            physics["friction"],
            physics["damping_scale"],
            physics["gain_scale"],
        )
    before_action = capture_state(base, originals)
    desired = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
    delay = Delay(delay_steps, int(np.asarray(env.action_space.shape)[0]))
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    policy_actions = []
    executed_actions = []
    clipped_actions = []
    queues = []
    qpos = [after_reset["qpos"]]
    qvel = [after_reset["qvel"]]
    qacc_warmstart = [after_reset["qacc_warmstart"]]
    ctrl = [after_reset["ctrl"]]
    obj_pos = [obs_after_reset["achieved_goal"]]
    terminated = False
    truncated = False
    info = {}
    steps = 0
    first_policy = None
    first_executed = None
    first_clipped = None
    after_first_step = None
    obs_after_first_step = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).copy()
        queue_before = delay.queue()
        executed = delay.apply(action)
        clipped = np.clip(executed, low, high)
        if first_policy is None:
            first_policy = action.copy()
            first_executed = executed.copy()
            first_clipped = clipped.copy()
        policy_actions.append(action)
        executed_actions.append(executed)
        clipped_actions.append(clipped)
        queues.append(queue_before)
        obs, _, terminated, truncated, info = env.step(executed)
        steps += 1
        state = capture_state(base, originals)
        qpos.append(state["qpos"])
        qvel.append(state["qvel"])
        qacc_warmstart.append(state["qacc_warmstart"])
        ctrl.append(state["ctrl"])
        obj_pos.append(np.asarray(obs["achieved_goal"], dtype=np.float64).copy())
        if after_first_step is None:
            after_first_step = state
            obs_after_first_step = capture_obs(obs)
        if steps > horizon:
            raise SystemExit("horizon overrun")
    final_obs = capture_obs(obs)
    final_distance = float(np.linalg.norm(final_obs["achieved_goal"] - final_obs["desired_goal"]))
    success = int(final_distance < 0.05)
    reported = int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)
    if success != reported:
        raise SystemExit(f"success flag {reported} disagrees with distance {final_distance}")
    if not success and steps != horizon:
        raise SystemExit(f"failure stopped at {steps}")
    if not np.allclose(final_obs["desired_goal"], desired):
        raise SystemExit("goal changed during the episode")
    return {
        "success": np.array([success], dtype=np.int32),
        "steps": np.array([steps], dtype=np.int32),
        "final_distance": np.array([final_distance], dtype=np.float64),
        "obs_after_reset": obs_after_reset["observation"],
        "achieved_after_reset": obs_after_reset["achieved_goal"],
        "desired_after_reset": obs_after_reset["desired_goal"],
        "state_after_reset_full": after_reset,
        "state_before_action_full": before_action,
        "state_after_first_step_full": after_first_step,
        "first_policy": first_policy,
        "first_executed": first_executed,
        "first_clipped": first_clipped,
        "obs_after_first_step": obs_after_first_step["observation"],
        "policy_actions": np.stack(policy_actions),
        "executed_actions": np.stack(executed_actions),
        "clipped_actions": np.stack(clipped_actions),
        "qpos": np.stack(qpos),
        "qvel": np.stack(qvel),
        "qacc_warmstart": np.stack(qacc_warmstart),
        "ctrl": np.stack(ctrl),
        "object_pos": np.stack(obj_pos),
        "queue0": queues[0],
    }


def save_trace(path: Path, bundle: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "success": bundle["success"],
        "steps": bundle["steps"],
        "final_distance": bundle["final_distance"],
        "obs_after_reset": bundle["obs_after_reset"],
        "achieved_after_reset": bundle["achieved_after_reset"],
        "desired_after_reset": bundle["desired_after_reset"],
        "first_policy": bundle["first_policy"],
        "first_executed": bundle["first_executed"],
        "first_clipped": bundle["first_clipped"],
        "obs_after_first_step": bundle["obs_after_first_step"],
        "policy_actions": bundle["policy_actions"],
        "executed_actions": bundle["executed_actions"],
        "clipped_actions": bundle["clipped_actions"],
        "qpos": bundle["qpos"],
        "qvel": bundle["qvel"],
        "qacc_warmstart": bundle["qacc_warmstart"],
        "ctrl": bundle["ctrl"],
        "object_pos": bundle["object_pos"],
        "queue0": bundle["queue0"],
    }
    for prefix in ("after_reset", "before_action", "after_first_step"):
        source = {
            "after_reset": bundle["state_after_reset_full"],
            "before_action": bundle["state_before_action_full"],
            "after_first_step": bundle["state_after_first_step_full"],
        }[prefix]
        for key, value in source.items():
            payload[f"{prefix}__{key}"] = value
    for key, value in meta.items():
        payload[f"meta_{key}"] = np.array(value)
    np.savez_compressed(path, **payload)


def load_selection(seed: int, only: str | None) -> list[dict]:
    rows = []
    with (OUT / "selection.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["policy_seed"]) != seed:
                continue
            if only and row["sample_id"] != only:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no selection rows for seed {seed} only={only}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modes", type=str, default="fresh,reused_a,reused_b,clean,clean_after_a")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--trace-dir", type=Path, default=OUT / "state_action_traces")
    args = parser.parse_args()
    # threads < 0 leaves PyTorch at its process default. Used only for the
    # phase-1-like thread contrast; the clean protocol passes threads=1.
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rows = load_selection(args.seed, args.only or None)
    env, base = make_eval_env()
    originals = {
        "obj_body": int(base._model_names.body_name2id["obj"]),
        "obj_geom": int(base._model_names.geom_name2id["obj_geom"]),
        "mass": float(base.model.body_mass[int(base._model_names.body_name2id["obj"])]),
        "friction": float(base.model.geom_friction[int(base._model_names.geom_name2id["obj_geom"]), 0]),
        "damping": base.model.dof_damping.copy(),
        "gain": base.model.actuator_gainprm[:, 0].copy(),
    }
    pristine = snapshot_model(base)
    model = load_policy(args.seed, env)
    print(f"[seed {args.seed}] loaded threads={torch.get_num_threads()} cases={len(rows)}", flush=True)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    default_physics = {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}
    try:
        for mode in modes:
            for row in rows:
                for repeat in range(args.repeats):
                    path = args.trace_dir / mode / f"seed{args.seed}" / f"{row['sample_id']}_r{repeat}.npz"
                    if path.exists():
                        continue
                    if mode in ("clean", "clean_after_a", "clean_after_b"):
                        run_env, run_base = make_eval_env()
                        protocol = PROTOCOL_CLEAN
                    else:
                        run_env, run_base = make_eval_env()
                        protocol = PROTOCOL_HISTORICAL
                    episode_index = 0
                    if mode in ("reused_a", "clean_after_a"):
                        for pred in PRE_A:
                            run_episode(
                                run_env, run_base, model, originals,
                                pred["reset_seed"], pred["delay_steps"], pred, PROTOCOL_HISTORICAL,
                            )
                            episode_index += 1
                    elif mode in ("reused_b", "clean_after_b"):
                        for pred in PRE_B:
                            run_episode(
                                run_env, run_base, model, originals,
                                pred["reset_seed"], pred["delay_steps"], pred, PROTOCOL_HISTORICAL,
                            )
                            episode_index += 1
                    if protocol == PROTOCOL_CLEAN:
                        install_clean_reset(run_base, pristine)
                    bundle = run_episode(
                        run_env,
                        run_base,
                        model,
                        originals,
                        int(row["reset_seed"]),
                        int(row["delay_steps"]),
                        default_physics,
                        protocol,
                    )
                    save_trace(
                        path,
                        bundle,
                        {
                            "mode": mode,
                            "protocol": protocol,
                            "sample_id": row["sample_id"],
                            "policy_seed": args.seed,
                            "episode_id": int(row["episode_id"]),
                            "delay_steps": int(row["delay_steps"]),
                            "reset_seed": int(row["reset_seed"]),
                            "repeat": repeat,
                            "threads": args.threads,
                            "episode_index_before_case": episode_index,
                            "direction": row["direction"],
                        },
                    )
                    run_env.close()
                    print(
                        f"[seed {args.seed}] {mode} {row['sample_id']} r{repeat} "
                        f"success={int(bundle['success'][0])} steps={int(bundle['steps'][0])}",
                        flush=True,
                    )
    finally:
        env.close()


if __name__ == "__main__":
    main()
