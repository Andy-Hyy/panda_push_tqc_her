"""Phase E formal P0-v3..P3-v3.

Clean reset from the Phase B protocol, mass writer C from the Phase C
XML-checked implementation. Each episode builds a new environment.
Goal-noise columns in the Stage 0 file are read and not applied.

Modes:
  resume-write   new process, P1/P2 episodes 0 and 1, probe files only
  resume-verify  another process, reverse those episodes, then append episode 2
  formal         in-process forward/reverse on episodes 0..7, then 600 x 4

A formal run refuses to append unless the probe markers match the current
checkpoint, code, XML and parameter-list hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import phaseB_diagnose as pb
from phaseC_mass_implementation import (
    MOCAP_XML,
    PUSH_XML,
    install_episode_reset,
    make_named_env,
    object_ids,
    sha256,
    snapshot,
)
from round2_common import MODELS, STAGE0, STAGE0_OTHERS, assert_control_period, load_stage0_rows

OUT = pb.OUT
PROTOCOL_ID = "phaseE_clean_reset_writerC_h100"
HORIZON = 100
N_EPISODES = 600
RESET_BASE = 4_000_000
MODEL_ATOL = 1e-10
MODEL_RTOL = 1e-8
CONDITIONS = ("P0", "P1", "P2", "P3")
PROBE_CONDITIONS = ("P1", "P2")
PROBE_DIR = OUT / "phaseE_resume_probe"
PROTOCOL_PATH = OUT / "phaseE_protocol.json"

EXPECTED = {
    "panda_env.py": "76045e06159706fd6a00077a801158f6d583a794ec62411c36a4c8fde9221a36",
    "push.py": "8ddfb1bdb47d72c80658bd4b434aafcd4a8e1563acebaf7859d10bfe5fcccea0",
    "push.xml": "a5feb6af7c678b11254c05f1a41cf1bddd043b5c2fa93dfbb7c54171bce3f9cb",
    "panda_mocap.xml": "8c1608812335d02b2b0c7178bc1aae91351cf2c890d1633a2748b352e3d7dd09",
    "checkpoint_seed0": "ab38ff9c17dbde348f0d0048858d4753570c04b2e7f486cacf60135717f085bd",
    "checkpoint_seed42": "e7c4840645dae9949b40c2b58045888a354b8193d3d0ffea429ae1a3ba087f74",
    "checkpoint_seed123": "25f64416614872cddcf5dc83075b55fea5f32a81b4ab60caa17454de9f3ea966",
}

FIELDS = [
    "condition",
    "policy_seed",
    "episode_id",
    "reset_seed",
    "delay_steps",
    "latency_s",
    "horizon",
    "goal_xy_range",
    "obj_xy_range",
    "initial_distance",
    "initial_near",
    "initial_x",
    "initial_y",
    "initial_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "success",
    "steps_used",
    "final_distance",
    "mass",
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "invweight_trans",
    "invweight_rot",
    "friction",
    "damping_scale",
    "gain_scale",
    "mass_read",
    "inertia_x_read",
    "inertia_y_read",
    "inertia_z_read",
    "invweight_trans_read",
    "invweight_rot_read",
    "friction_read",
    "damping_scale_read",
    "gain_scale_read",
    "time_after_reset",
    "warmstart_maxabs",
    "ctrl_maxabs_vs_neutral",
    "model_check_ok",
    "writer",
    "protocol_id",
    "protocol_sha256",
]


def fmt(value: float) -> str:
    return format(float(value), ".17g")


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def delay_of(condition: str, row: dict) -> int:
    if condition in ("P0", "P1"):
        return 0
    delay = int(float(row["delay_steps"]))
    if delay < 1 or delay > 9:
        raise SystemExit(f"delay {delay} outside 1..9")
    return delay


def physics_of(condition: str, row: dict) -> dict:
    if condition in ("P0", "P2"):
        return {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}
    return {
        "mass": float(row["mass"]),
        "friction": float(row["friction"]),
        "damping_scale": float(row["damping_scale"]),
        "gain_scale": float(row["gain_scale"]),
    }


def scene_digest(rows: list[dict]) -> str:
    """Lock the Phase 1 row order: 200 episodes for eval seeds 0, 42, 123.

    episode_id is the index in that order, not the 1..200 column inside a block.
    """
    expected = [(seed, episode) for seed in (0, 42, 123) for episode in range(1, 201)]
    if len(rows) != len(expected):
        raise SystemExit(f"stage0 rows {len(rows)}")
    lines = []
    for index, row in enumerate(rows):
        got = (int(float(row["seed"])), int(float(row["episode"])))
        if got != expected[index]:
            raise SystemExit(f"stage0 row {index} is seed {got[0]} episode {got[1]}, expected {expected[index]}")
        lines.append(
            ",".join(
                [
                    str(index),
                    str(RESET_BASE + index),
                    row["mass"],
                    row["friction"],
                    row["damping_scale"],
                    row["gain_scale"],
                    row["delay_steps"],
                    row["goal_noise_x"],
                    row["goal_noise_y"],
                    row["goal_noise_z"],
                ]
            )
        )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def file_table(rows: list[dict]) -> dict:
    code = {
        "panda_env.py": pb.ROOT / "panda_mujoco_gym" / "envs" / "panda_env.py",
        "push.py": pb.ROOT / "panda_mujoco_gym" / "envs" / "push.py",
        "push.xml": PUSH_XML,
        "panda_mocap.xml": MOCAP_XML,
        "round2_common.py": pb.OLD_SCRIPTS / "round2_common.py",
        "phaseB_diagnose.py": Path(__file__).resolve().parent / "phaseB_diagnose.py",
        "phaseC_mass_implementation.py": Path(__file__).resolve().parent / "phaseC_mass_implementation.py",
        "phaseE_formal.py": Path(__file__).resolve(),
        "stage0_seed0": STAGE0,
        "stage0_seed42": STAGE0_OTHERS[0],
        "stage0_seed123": STAGE0_OTHERS[1],
    }
    hashes = {name: sha256(path) for name, path in code.items()}
    for seed, path in MODELS.items():
        hashes[f"checkpoint_seed{seed}"] = sha256(path)
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise SystemExit(f"locked hash mismatch for {name}")
    return hashes


def build_protocol(rows: list[dict]) -> dict:
    body = {
        "protocol_id": PROTOCOL_ID,
        "writer": "C",
        "init": "mj_resetData then neutral ctrl[0:7], joint neutral, initial_qvel, mocap, sample, mj_forward",
        "horizon": HORIZON,
        "n_episodes": N_EPISODES,
        "reset_seed_rule": "4000000+episode_id",
        "goal_xy_range": 0.3,
        "obj_xy_range": 0.3,
        "goal_noise_applied": False,
        "goal_noise_columns_present_but_unused": True,
        "delay_p0_p1": 0,
        "delay_p2_p3": "stage0 delay_steps integer in 1..9",
        "physics_p0_p2": "mass=friction=damping_scale=gain_scale=1",
        "physics_p1_p3": "stage0 mass friction damping_scale gain_scale",
        "success": "object to target distance < 0.05",
        "threads": "torch.set_num_threads(1) and OMP/MKL/OPENBLAS/VECLIB=1",
        "device": "cpu",
        "files": file_table(rows),
        "scene_list_sha256": scene_digest(rows),
    }
    digest = hashlib.sha256(canonical(body).encode()).hexdigest()
    body["protocol_sha256"] = digest
    return body


def lock_protocol(protocol: dict) -> None:
    text = canonical(protocol)
    if PROTOCOL_PATH.exists():
        existing = PROTOCOL_PATH.read_text()
        if existing != text:
            raise SystemExit(f"protocol file differs from this process; refusing to append ({PROTOCOL_PATH})")
        return
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(PROTOCOL_PATH, flags, 0o644)
    except FileExistsError:
        existing = PROTOCOL_PATH.read_text()
        if existing != text:
            raise SystemExit("protocol file differs from this process; refusing to append")
        return
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def assert_ranges(base) -> None:
    if float(base.goal_xy_range) != 0.3 or float(base.obj_xy_range) != 0.3:
        raise SystemExit(f"ranges goal {base.goal_xy_range} object {base.obj_xy_range}")
    goal_span = np.asarray(base.goal_range_high, dtype=np.float64) - np.asarray(base.goal_range_low, dtype=np.float64)
    obj_span = np.asarray(base.obj_range_high, dtype=np.float64) - np.asarray(base.obj_range_low, dtype=np.float64)
    # 0.3 is not a binary fraction, so the derived box edges are not bit-identical to 0.3.
    if not np.allclose(goal_span[:2], [0.3, 0.3], atol=1e-12, rtol=0.0):
        raise SystemExit(f"goal span {goal_span[:2]}")
    if not np.allclose(obj_span[:2], [0.3, 0.3], atol=1e-12, rtol=0.0):
        raise SystemExit(f"object span {obj_span[:2]}")
    goal_center = 0.5 * (np.asarray(base.goal_range_low, dtype=np.float64) + np.asarray(base.goal_range_high, dtype=np.float64))
    obj_center = 0.5 * (np.asarray(base.obj_range_low, dtype=np.float64) + np.asarray(base.obj_range_high, dtype=np.float64))
    if not np.allclose(goal_center[:2], [0.6, 0.0], atol=1e-12, rtol=0.0):
        raise SystemExit(f"goal center {goal_center[:2]}")
    if not np.allclose(obj_center[:2], [0.6, 0.0], atol=1e-12, rtol=0.0):
        raise SystemExit(f"object center {obj_center[:2]}")
    assert_control_period(base)


def in_box_xy(pos: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(low[0] - 1e-9 <= pos[0] <= high[0] + 1e-9 and low[1] - 1e-9 <= pos[1] <= high[1] + 1e-9)


def scale_read(actual: np.ndarray, default: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    default = np.asarray(default, dtype=np.float64)
    mask = np.abs(default) > 1e-15
    if not np.any(mask):
        raise SystemExit("default vector has no nonzero entry")
    return float(np.median(actual[mask] / default[mask]))


def measure(base, bundle, pristine: dict, ids: dict, physics: dict) -> dict:
    state = bundle["state_after_reset_full"]
    body = ids["body"]
    geom = ids["geom"]
    mass0 = float(pristine["body_mass"][body])
    expected_inertia = np.asarray(pristine["body_inertia"][body], dtype=np.float64) * (physics["mass"] / mass0)
    expected_damping = np.asarray(pristine["dof_damping"], dtype=np.float64) * physics["damping_scale"]
    expected_gain = np.asarray(pristine["actuator_gainprm"][:, 0], dtype=np.float64) * physics["gain_scale"]
    mass_read = float(state["body_mass"][0])
    inertia_read = np.asarray(state["body_inertia"], dtype=np.float64).reshape(-1)
    inv_read = np.asarray(state["body_invweight0"], dtype=np.float64).reshape(-1)
    friction_read = float(state["friction0"][0])
    damping_read = np.asarray(state["dof_damping"], dtype=np.float64)
    gain_read = np.asarray(state["gain"], dtype=np.float64)
    friction_tail = np.asarray(base.model.geom_friction[geom, 1:], dtype=np.float64)
    friction_tail_0 = np.asarray(pristine["geom_friction"][geom, 1:], dtype=np.float64)
    neutral = np.asarray(base.neutral_joint_values, dtype=np.float64)
    ctrl = np.asarray(state["ctrl"], dtype=np.float64)
    warm = np.asarray(state["qacc_warmstart"], dtype=np.float64)
    problems = []
    if abs(float(state["time"][0])) != 0.0:
        problems.append(f"time {float(state['time'][0])}")
    if ctrl.shape != neutral.shape or not np.array_equal(ctrl, neutral):
        delta = float(np.max(np.abs(ctrl - neutral))) if ctrl.shape == neutral.shape else float("inf")
        problems.append(f"ctrl maxabs {delta}")
    # mj_forward after mj_resetData fills qacc_warmstart with the neutral-pose
    # acceleration. Phase B's clean traces match that value; it is not residue.
    if not np.all(np.isfinite(warm)):
        problems.append("warmstart is not finite")
    if not np.allclose(mass_read, physics["mass"], atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append(f"mass {mass_read} vs {physics['mass']}")
    if inertia_read.shape != expected_inertia.shape or not np.allclose(inertia_read, expected_inertia, atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append("inertia")
    if not np.allclose(friction_read, physics["friction"], atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append(f"friction {friction_read}")
    if not np.allclose(friction_tail, friction_tail_0, atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append("friction tangential components moved")
    if not np.allclose(damping_read, expected_damping, atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append("damping")
    if not np.allclose(gain_read, expected_gain, atol=MODEL_ATOL, rtol=MODEL_RTOL):
        problems.append("gain")
    if inv_read.shape != (2,):
        problems.append(f"invweight shape {inv_read.shape}")
    achieved = np.asarray(bundle["achieved_after_reset"], dtype=np.float64)
    desired = np.asarray(bundle["desired_after_reset"], dtype=np.float64)
    if not in_box_xy(achieved, base.obj_range_low, base.obj_range_high):
        problems.append(f"object xy {achieved[:2]}")
    if not in_box_xy(desired, base.goal_range_low, base.goal_range_high):
        problems.append(f"goal xy {desired[:2]}")
    if problems:
        raise SystemExit("model check failed: " + "; ".join(problems))
    ctrl_delta = float(np.max(np.abs(ctrl - neutral)))
    warm_delta = float(np.max(np.abs(warm))) if warm.size else 0.0
    return {
        "mass_read": mass_read,
        "inertia": inertia_read,
        "invweight": inv_read,
        "friction_read": friction_read,
        "damping_scale_read": scale_read(damping_read, pristine["dof_damping"]),
        "gain_scale_read": scale_read(gain_read, pristine["actuator_gainprm"][:, 0]),
        "time": float(state["time"][0]),
        "warmstart_maxabs": warm_delta,
        "ctrl_maxabs_vs_neutral": ctrl_delta,
    }


def run_one(model, pristine, ids, reset_seed: int, delay: int, physics: dict, protocol_sha: str, condition: str, episode_id: int, policy_seed: int):
    env, base = make_named_env()
    try:
        assert_ranges(base)
        install_episode_reset(base, pristine, ids, physics, "C")
        bundle = pb.run_episode(
            env,
            base,
            model,
            {"obj_body": ids["body"], "obj_geom": ids["geom"]},
            reset_seed,
            delay,
            physics,
            pb.PROTOCOL_CLEAN,
            HORIZON,
        )
        measured = measure(base, bundle, pristine, ids, physics)
    finally:
        env.close()
    achieved = np.asarray(bundle["achieved_after_reset"], dtype=np.float64)
    desired = np.asarray(bundle["desired_after_reset"], dtype=np.float64)
    initial_distance = float(np.linalg.norm(achieved - desired))
    inertia = measured["inertia"]
    inv = measured["invweight"]
    row = {
        "condition": condition,
        "policy_seed": str(policy_seed),
        "episode_id": str(episode_id),
        "reset_seed": str(reset_seed),
        "delay_steps": str(delay),
        "latency_s": fmt(delay * 0.5),
        "horizon": str(HORIZON),
        "goal_xy_range": "0.3",
        "obj_xy_range": "0.3",
        "initial_distance": fmt(initial_distance),
        "initial_near": str(int(initial_distance < 0.05)),
        "initial_x": fmt(achieved[0]),
        "initial_y": fmt(achieved[1]),
        "initial_z": fmt(achieved[2]),
        "goal_x": fmt(desired[0]),
        "goal_y": fmt(desired[1]),
        "goal_z": fmt(desired[2]),
        "success": str(int(bundle["success"][0])),
        "steps_used": str(int(bundle["steps"][0])),
        "final_distance": fmt(float(bundle["final_distance"][0])),
        "mass": fmt(physics["mass"]),
        "inertia_x": fmt(inertia[0]),
        "inertia_y": fmt(inertia[1]),
        "inertia_z": fmt(inertia[2]),
        "invweight_trans": fmt(inv[0]),
        "invweight_rot": fmt(inv[1]),
        "friction": fmt(physics["friction"]),
        "damping_scale": fmt(physics["damping_scale"]),
        "gain_scale": fmt(physics["gain_scale"]),
        "mass_read": fmt(measured["mass_read"]),
        "inertia_x_read": fmt(inertia[0]),
        "inertia_y_read": fmt(inertia[1]),
        "inertia_z_read": fmt(inertia[2]),
        "invweight_trans_read": fmt(inv[0]),
        "invweight_rot_read": fmt(inv[1]),
        "friction_read": fmt(measured["friction_read"]),
        "damping_scale_read": fmt(measured["damping_scale_read"]),
        "gain_scale_read": fmt(measured["gain_scale_read"]),
        "time_after_reset": fmt(measured["time"]),
        "warmstart_maxabs": fmt(measured["warmstart_maxabs"]),
        "ctrl_maxabs_vs_neutral": fmt(measured["ctrl_maxabs_vs_neutral"]),
        "model_check_ok": "1",
        "writer": "C",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
    }
    return row


def read_table(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"header mismatch in {path}")
        rows = list(reader)
    seen = set()
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id in seen:
            raise SystemExit(f"duplicate episode {episode_id} in {path}")
        seen.add(episode_id)
        if row["protocol_id"] != PROTOCOL_ID:
            raise SystemExit(f"protocol id mismatch in {path}")
    return rows


def append_row(path: Path, row: dict) -> None:
    new_file = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def rows_equal(stored: dict, fresh: dict) -> str:
    for key in FIELDS:
        if stored[key] != fresh[key]:
            return f"{key}: stored {stored[key]} fresh {fresh[key]}"
    return ""


def signature(row: dict) -> tuple:
    return (
        int(row["success"]),
        int(row["steps_used"]),
        float(row["final_distance"]),
        float(row["initial_x"]),
        float(row["initial_y"]),
        float(row["goal_x"]),
        float(row["goal_y"]),
    )


def pretest(model, pristine, ids, rows: list[dict], protocol_sha: str, policy_seed: int) -> None:
    for condition in PROBE_CONDITIONS:
        forward = {}
        for episode_id in range(8):
            row = run_one(
                model, pristine, ids, RESET_BASE + episode_id,
                delay_of(condition, rows[episode_id]), physics_of(condition, rows[episode_id]),
                protocol_sha, condition, episode_id, policy_seed,
            )
            forward[episode_id] = signature(row)
            print(f"[pretest] {condition} forward episode {episode_id} success {row['success']} steps {row['steps_used']}", flush=True)
        for episode_id in range(7, -1, -1):
            row = run_one(
                model, pristine, ids, RESET_BASE + episode_id,
                delay_of(condition, rows[episode_id]), physics_of(condition, rows[episode_id]),
                protocol_sha, condition, episode_id, policy_seed,
            )
            if signature(row) != forward[episode_id]:
                raise SystemExit(f"order check failed {condition} episode {episode_id}")
            print(f"[pretest] {condition} reverse episode {episode_id} matched", flush=True)


def probe_path(condition: str, seed: int) -> Path:
    return PROBE_DIR / f"{condition}_seed{seed}.csv"


def marker_path(seed: int) -> Path:
    return PROBE_DIR / f"verified_seed{seed}.json"


def resume_write(model, pristine, ids, rows, protocol_sha: str, seed: int) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for condition in PROBE_CONDITIONS:
        path = probe_path(condition, seed)
        if path.exists() and path.stat().st_size > 0:
            raise SystemExit(f"probe already exists: {path}")
        for episode_id in (0, 1):
            source = rows[episode_id]
            row = run_one(
                model, pristine, ids, RESET_BASE + episode_id,
                delay_of(condition, source), physics_of(condition, source),
                protocol_sha, condition, episode_id, seed,
            )
            append_row(path, row)
            print(f"[resume-write] {condition} episode {episode_id} success {row['success']} steps {row['steps_used']}", flush=True)


def resume_verify(model, pristine, ids, rows, protocol_sha: str, seed: int) -> None:
    compared = []
    for condition in PROBE_CONDITIONS:
        path = probe_path(condition, seed)
        stored_rows = read_table(path)
        ids_found = [int(row["episode_id"]) for row in stored_rows]
        stored = {int(row["episode_id"]): row for row in stored_rows}
        if ids_found == [0, 1]:
            compare_ids = (1, 0)
            append_id = 2
        elif ids_found == [0, 1, 2]:
            compare_ids = (2, 1, 0)
            append_id = None
        else:
            raise SystemExit(f"unexpected probe contents {path}: {ids_found}")
        for episode_id in compare_ids:
            source = rows[episode_id]
            fresh = run_one(
                model, pristine, ids, RESET_BASE + episode_id,
                delay_of(condition, source), physics_of(condition, source),
                protocol_sha, condition, episode_id, seed,
            )
            mismatch = rows_equal(stored[episode_id], fresh)
            if mismatch:
                raise SystemExit(f"resume mismatch {condition} episode {episode_id}: {mismatch}")
            compared.append(f"{condition}:{episode_id}")
            print(f"[resume-verify] {condition} episode {episode_id} matched in a new process", flush=True)
        if append_id is not None:
            source = rows[append_id]
            fresh = run_one(
                model, pristine, ids, RESET_BASE + append_id,
                delay_of(condition, source), physics_of(condition, source),
                protocol_sha, condition, append_id, seed,
            )
            append_row(path, fresh)
            print(f"[resume-verify] {condition} appended episode {append_id}", flush=True)
    marker = {
        "seed": seed,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "compared": compared,
    }
    marker_path(seed).write_text(canonical(marker))
    print(f"[resume-verify] seed {seed} marker written", flush=True)


def require_probes(protocol_sha: str) -> None:
    for seed in (0, 42, 123):
        path = marker_path(seed)
        if not path.exists():
            raise SystemExit(f"missing resume marker {path}")
        marker = json.loads(path.read_text())
        if marker.get("protocol_sha256") != protocol_sha or marker.get("seed") != seed:
            raise SystemExit(f"resume marker does not match this protocol: {path}")


def formal(model, pristine, ids, rows, protocol_sha: str, seed: int, skip_pretest: bool) -> None:
    require_probes(protocol_sha)
    if not skip_pretest:
        pretest(model, pristine, ids, rows, protocol_sha, seed)
    for condition in CONDITIONS:
        path = OUT / f"{condition}_v3_seed{seed}.csv"
        existing = read_table(path)
        if existing:
            for row in existing:
                if row["protocol_sha256"] != protocol_sha:
                    raise SystemExit(f"existing row protocol hash differs in {path}")
            spot_id = min(int(row["episode_id"]) for row in existing)
            stored = next(row for row in existing if int(row["episode_id"]) == spot_id)
            source = rows[spot_id]
            fresh = run_one(
                model, pristine, ids, RESET_BASE + spot_id,
                delay_of(condition, source), physics_of(condition, source),
                protocol_sha, condition, spot_id, seed,
            )
            mismatch = rows_equal(stored, fresh)
            if mismatch:
                raise SystemExit(f"resume spot check failed {condition} episode {spot_id}: {mismatch}")
            print(f"[resume] {condition} existing episode {spot_id} reproduced", flush=True)
        done = {int(row["episode_id"]) for row in existing}
        for episode_id in range(N_EPISODES):
            if episode_id in done:
                continue
            source = rows[episode_id]
            row = run_one(
                model, pristine, ids, RESET_BASE + episode_id,
                delay_of(condition, source), physics_of(condition, source),
                protocol_sha, condition, episode_id, seed,
            )
            append_row(path, row)
            if (episode_id + 1) % 25 == 0 or episode_id + 1 == N_EPISODES:
                print(f"[seed {seed}] {condition} {episode_id + 1}/{N_EPISODES}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("formal", "resume-write", "resume-verify"), default="formal")
    parser.add_argument("--skip-pretest", action="store_true")
    args = parser.parse_args()
    if args.seed not in (0, 42, 123):
        raise SystemExit(f"seed {args.seed} is not one of the frozen checkpoints")
    import torch

    torch.set_num_threads(1)
    rows = load_stage0_rows()
    if len(rows) != N_EPISODES:
        raise SystemExit(f"stage0 rows {len(rows)}")
    protocol = build_protocol(rows)
    lock_protocol(protocol)
    protocol_sha = protocol["protocol_sha256"]
    template, template_base = make_named_env()
    try:
        assert_ranges(template_base)
        pristine = snapshot(template_base)
        ids = object_ids(template_base)
        model = pb.load_policy(args.seed, template)
        print(
            f"[seed {args.seed}] mode {args.mode} threads {torch.get_num_threads()} protocol {protocol_sha[:12]}",
            flush=True,
        )
        if args.mode == "resume-write":
            resume_write(model, pristine, ids, rows, protocol_sha, args.seed)
        elif args.mode == "resume-verify":
            resume_verify(model, pristine, ids, rows, protocol_sha, args.seed)
        else:
            formal(model, pristine, ids, rows, protocol_sha, args.seed, args.skip_pretest)
    finally:
        template.close()


if __name__ == "__main__":
    main()
