"""Clean single-factor grid aligned with the Phase E reset and writer C.

Mass 1, friction 1 and gain 1 are not run here. They are cited from P0-v3.
Damping has no nominal point in the historical grid. Horizon is 100.
The whole dof_damping array is scaled. Only actuator_gainprm column 0 is scaled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PANDA = Path("/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final")
OUT = PANDA / "data_and_analysis" / "single_factor_and_training_audit_2026-10-02"
ROUND3 = PANDA / "data_and_analysis" / "corrective_round3_2026-10-01" / "scripts"
ROUND2 = PANDA / "data_and_analysis" / "corrective_2026-10-01" / "scripts"
P0_DIR = PANDA / "data_and_analysis" / "corrective_round3_2026-10-01"
SPATIAL_MJB = PANDA / "data_and_analysis" / "spatial_clean_2026-10-02" / "cache" / "panda_push.mjb"
CANONICAL_PUSH = PANDA / "panda_mujoco_gym" / "assets" / "push.xml"
CACHE = OUT / "cache"
MJB = CACHE / "panda_push.mjb"
PROTOCOL_PATH = OUT / "single_factor_protocol.json"
PROTOCOL_ID = "single_factor_clean_reset_writerC_h100"
EXPECTED_MJB = "ff842fd8407c1edde8058bf004774523283bcdc6b81ed33240c945380798fddd"
EXPECTED_PUSH = "a5feb6af7c678b11254c05f1a41cf1bddd043b5c2fa93dfbb7c54171bce3f9cb"
CHECKPOINT = {
    0: "ab38ff9c17dbde348f0d0048858d4753570c04b2e7f486cacf60135717f085bd",
    42: "e7c4840645dae9949b40c2b58045888a354b8193d3d0ffea429ae1a3ba087f74",
    123: "25f64416614872cddcf5dc83075b55fea5f32a81b4ab60caa17454de9f3ea966",
}
N_EPISODES = 600
RESET_BASE = 4_000_000
GATE_EPISODES = (0, 301, 599)
ATOL = 1e-8

for path in (ROUND2, ROUND3, PANDA):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import phaseB_diagnose as pb  # noqa: E402
from phaseC_mass_implementation import (  # noqa: E402
    install_episode_reset,
    make_named_env,
    object_ids,
    sha256,
    snapshot,
)
from phaseE_formal import measure  # noqa: E402
from round2_common import MODELS  # noqa: E402
from sb3_contrib import TQC  # noqa: E402


def physics(mass=1.0, friction=1.0, damping=1.0, gain=1.0) -> dict:
    return {
        "mass": float(mass),
        "friction": float(friction),
        "damping_scale": float(damping),
        "gain_scale": float(gain),
    }


# value tokens are the historical grid spellings. Nominal mass, friction and gain are omitted.
CELLS = [
    ("object_mass", "0.05", physics(mass=0.05)),
    ("object_mass", "0.1", physics(mass=0.1)),
    ("object_mass", "0.5", physics(mass=0.5)),
    ("object_mass", "2", physics(mass=2.0)),
    ("object_mass", "3", physics(mass=3.0)),
    ("object_friction", "0.05", physics(friction=0.05)),
    ("object_friction", "0.2", physics(friction=0.2)),
    ("object_friction", "0.5", physics(friction=0.5)),
    ("object_friction", "0.8", physics(friction=0.8)),
    ("object_friction", "1.2", physics(friction=1.2)),
    ("object_friction", "1.5", physics(friction=1.5)),
    ("object_friction", "2", physics(friction=2.0)),
    ("object_friction", "2.5", physics(friction=2.5)),
    ("object_friction", "3", physics(friction=3.0)),
    ("joint_damping_scale", "0.001", physics(damping=0.001)),
    ("joint_damping_scale", "0.01", physics(damping=0.01)),
    ("joint_damping_scale", "0.02", physics(damping=0.02)),
    ("joint_damping_scale", "0.05", physics(damping=0.05)),
    ("joint_damping_scale", "0.08", physics(damping=0.08)),
    ("joint_damping_scale", "0.1", physics(damping=0.1)),
    ("joint_damping_scale", "0.2", physics(damping=0.2)),
    ("joint_damping_scale", "0.3", physics(damping=0.3)),
    ("actuator_gain_scale", "0", physics(gain=0.0)),
    ("actuator_gain_scale", "2", physics(gain=2.0)),
    ("actuator_gain_scale", "3", physics(gain=3.0)),
    ("actuator_gain_scale", "5", physics(gain=5.0)),
    ("actuator_gain_scale", "6", physics(gain=6.0)),
    ("actuator_gain_scale", "7", physics(gain=7.0)),
    ("actuator_gain_scale", "10", physics(gain=10.0)),
]
assert len(CELLS) == 29

FIELDS = [
    "factor", "value", "policy_seed", "episode_id", "reset_seed",
    "goal_xy_range", "obj_xy_range",
    "goal_low_x", "goal_low_y", "goal_high_x", "goal_high_y",
    "obj_low_x", "obj_low_y", "obj_high_x", "obj_high_y",
    "initial_distance", "initial_near", "initial_x", "initial_y", "initial_z",
    "goal_x", "goal_y", "goal_z",
    "delay_steps", "time_after_reset", "ctrl_maxabs_vs_neutral",
    "mass", "friction", "damping_scale", "gain_scale",
    "mass_read", "inertia_x", "inertia_y", "inertia_z",
    "invweight_trans", "invweight_rot",
    "friction_read", "friction_torsion_read", "friction_roll_read",
    "damping_scale_read", "gain_scale_read", "obj_dof_damping_read", "bias_unchanged",
    "success", "steps_used", "final_distance",
    "checkpoint_sha256", "protocol_id", "protocol_sha256",
    "evaluation_source", "writer", "model_check_ok",
]
COMPARE = [
    "success", "steps_used", "final_distance", "initial_distance",
    "initial_x", "initial_y", "initial_z", "goal_x", "goal_y", "goal_z",
    "time_after_reset", "ctrl_maxabs_vs_neutral",
]


def fmt(value: float) -> str:
    return format(float(value), ".17g")


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def install_binary_loader() -> None:
    import mujoco

    if sha256(CANONICAL_PUSH) != EXPECTED_PUSH:
        raise SystemExit("push.xml hash changed")
    if not SPATIAL_MJB.exists() or sha256(SPATIAL_MJB) != EXPECTED_MJB:
        raise SystemExit("spatial mjb missing or hash changed")
    CACHE.mkdir(parents=True, exist_ok=True)
    if not MJB.exists() or sha256(MJB) != EXPECTED_MJB:
        shutil.copyfile(SPATIAL_MJB, MJB)
    if sha256(MJB) != EXPECTED_MJB:
        raise SystemExit("copied mjb hash mismatch")
    canonical_push = os.path.realpath(CANONICAL_PUSH)
    mjb = str(MJB)

    def from_xml_path(filename, *args, **kwargs):
        if os.path.realpath(filename) == canonical_push:
            return mujoco.MjModel.from_binary_path(mjb)
        raise SystemExit(f"unexpected XML load: {filename}")

    mujoco.MjModel.from_xml_path = from_xml_path


def file_table() -> dict:
    files = {
        "single_factor_formal.py": Path(__file__).resolve(),
        "phaseB_diagnose.py": ROUND3 / "phaseB_diagnose.py",
        "phaseC_mass_implementation.py": ROUND3 / "phaseC_mass_implementation.py",
        "phaseE_formal.py": ROUND3 / "phaseE_formal.py",
        "round2_common.py": ROUND2 / "round2_common.py",
        "push.xml": CANONICAL_PUSH,
        "panda_push.mjb": MJB,
    }
    hashes = {name: sha256(path) for name, path in files.items()}
    for seed, path in MODELS.items():
        digest = sha256(path)
        if digest != CHECKPOINT[seed]:
            raise SystemExit(f"checkpoint {seed} hash {digest}")
        hashes[f"checkpoint_seed{seed}"] = digest
    hashes["mjb_source"] = str(SPATIAL_MJB)
    return hashes


def build_protocol() -> dict:
    body = {
        "protocol_id": PROTOCOL_ID,
        "writer": "C",
        "init": "writer C inside clean reset: restore snapshot, scale inertia by m/m0, mj_setConst, mj_resetData, neutral ctrl",
        "horizon": 100,
        "n_episodes": N_EPISODES,
        "reset_seed_rule": "4000000+episode_id",
        "goal_xy_range": 0.3,
        "obj_xy_range": 0.3,
        "delay_steps": 0,
        "goal_noise_applied": False,
        "cells": [{"factor": factor, "value": value, **spec} for factor, value, spec in CELLS],
        "reused_from_p0": ["object_mass=1", "object_friction=1", "actuator_gain_scale=1"],
        "damping_definition": "entire dof_damping times scale, including the object free joint",
        "gain_definition": "actuator_gainprm column 0 times scale; biasprm unchanged",
        "friction_definition": "obj_geom friction component 0 only",
        "mass_definition": "writer C uniform-density inertia",
        "success": "object to target distance < 0.05 after at least one env.step",
        "threads": "torch.set_num_threads(1) and OMP/MKL/OPENBLAS/VECLIB=1; interop unchanged",
        "device": "cpu",
        "loader": "from_binary_path of copied spatial cache panda_push.mjb",
        "files": file_table(),
    }
    body["protocol_sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return body


def lock_protocol(protocol: dict) -> dict:
    text = canonical(protocol)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text())
        if existing.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise SystemExit("protocol hash differs from the locked file")
        return existing
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(PROTOCOL_PATH, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    return protocol


def load_p0(seed: int) -> dict[int, dict]:
    path = P0_DIR / f"P0_v3_seed{seed}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_EPISODES:
        raise SystemExit(f"P0 seed {seed} has {len(rows)} rows")
    return {int(row["episode_id"]): row for row in rows}


def capture_row(model, pristine, ids, seed: int, episode_id: int, factor: str, value: str, spec: dict, protocol_sha: str) -> tuple[dict, dict]:
    env, base = make_named_env()
    try:
        install_episode_reset(base, pristine, ids, spec, "C")
        bundle = pb.run_episode(
            env, base, model,
            {"obj_body": ids["body"], "obj_geom": ids["geom"]},
            RESET_BASE + episode_id, 0, spec, pb.PROTOCOL_CLEAN, 100,
        )
        measured = measure(base, bundle, pristine, ids, spec)
        bias_delta = float(np.max(np.abs(base.model.actuator_biasprm - pristine["actuator_biasprm"])))
        if bias_delta != 0.0:
            raise SystemExit(f"bias moved by {bias_delta}")
        if not (np.isfinite(bundle["qpos"]).all() and np.isfinite(bundle["qvel"]).all()):
            raise SystemExit("non-finite state")
        friction = np.asarray(base.model.geom_friction[ids["geom"]], dtype=np.float64)
        obj_damp = float(base.model.dof_damping[9])
        achieved = np.asarray(bundle["achieved_after_reset"], dtype=np.float64)
        desired = np.asarray(bundle["desired_after_reset"], dtype=np.float64)
        distance = float(np.linalg.norm(achieved - desired))
        low_g = np.asarray(base.goal_range_low, dtype=np.float64)
        high_g = np.asarray(base.goal_range_high, dtype=np.float64)
        low_o = np.asarray(base.obj_range_low, dtype=np.float64)
        high_o = np.asarray(base.obj_range_high, dtype=np.float64)
        inertia = measured["inertia"]
        inv = measured["invweight"]
        row = {
            "factor": factor,
            "value": value,
            "policy_seed": str(seed),
            "episode_id": str(episode_id),
            "reset_seed": str(RESET_BASE + episode_id),
            "goal_xy_range": fmt(base.goal_xy_range),
            "obj_xy_range": fmt(base.obj_xy_range),
            "goal_low_x": fmt(low_g[0]),
            "goal_low_y": fmt(low_g[1]),
            "goal_high_x": fmt(high_g[0]),
            "goal_high_y": fmt(high_g[1]),
            "obj_low_x": fmt(low_o[0]),
            "obj_low_y": fmt(low_o[1]),
            "obj_high_x": fmt(high_o[0]),
            "obj_high_y": fmt(high_o[1]),
            "initial_distance": fmt(distance),
            "initial_near": str(int(distance < 0.05)),
            "initial_x": fmt(achieved[0]),
            "initial_y": fmt(achieved[1]),
            "initial_z": fmt(achieved[2]),
            "goal_x": fmt(desired[0]),
            "goal_y": fmt(desired[1]),
            "goal_z": fmt(desired[2]),
            "delay_steps": "0",
            "time_after_reset": fmt(measured["time"]),
            "ctrl_maxabs_vs_neutral": fmt(measured["ctrl_maxabs_vs_neutral"]),
            "mass": fmt(spec["mass"]),
            "friction": fmt(spec["friction"]),
            "damping_scale": fmt(spec["damping_scale"]),
            "gain_scale": fmt(spec["gain_scale"]),
            "mass_read": fmt(measured["mass_read"]),
            "inertia_x": fmt(inertia[0]),
            "inertia_y": fmt(inertia[1]),
            "inertia_z": fmt(inertia[2]),
            "invweight_trans": fmt(inv[0]),
            "invweight_rot": fmt(inv[1]),
            "friction_read": fmt(friction[0]),
            "friction_torsion_read": fmt(friction[1]),
            "friction_roll_read": fmt(friction[2]),
            "damping_scale_read": fmt(measured["damping_scale_read"]),
            "gain_scale_read": fmt(measured["gain_scale_read"]),
            "obj_dof_damping_read": fmt(obj_damp),
            "bias_unchanged": "1",
            "success": str(int(bundle["success"][0])),
            "steps_used": str(int(bundle["steps"][0])),
            "final_distance": fmt(float(bundle["final_distance"][0])),
            "checkpoint_sha256": CHECKPOINT[seed],
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha,
            "evaluation_source": "new_run",
            "writer": "C",
            "model_check_ok": "1",
        }
    finally:
        env.close()
    return row, bundle


def differ(left: dict, right: dict) -> list[str]:
    problems = []
    for key in COMPARE:
        if key in ("success", "steps_used"):
            if str(left[key]) != str(right[key]):
                problems.append(f"{key} {left[key]} vs {right[key]}")
            continue
        if abs(float(left[key]) - float(right[key])) > ATOL:
            problems.append(f"{key} {left[key]} vs {right[key]}")
    return problems


def load_policy(seed: int):
    env, base = make_named_env()
    try:
        model = TQC.load(MODELS[seed], env=env, device="cpu")
        pristine = snapshot(base)
        if "actuator_biasprm" not in pristine:
            pristine["actuator_biasprm"] = np.array(base.model.actuator_biasprm, copy=True)
        ids = object_ids(base)
    finally:
        env.close()
    return model, pristine, ids


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"header mismatch in {path}")
        return {(row["factor"], row["value"], row["episode_id"]) for row in reader}


def save_trace(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        success=bundle["success"],
        steps=bundle["steps"],
        final_distance=bundle["final_distance"],
        qpos=bundle["qpos"],
        qvel=bundle["qvel"],
        object_pos=bundle["object_pos"],
        policy_actions=bundle["policy_actions"],
    )


def gate(protocol: dict) -> None:
    nominal = physics()
    problems = []
    for seed in (0, 42, 123):
        print(f"[gate] seed {seed}", flush=True)
        model, pristine, ids = load_policy(seed)
        p0 = load_p0(seed)
        forward = {}
        for episode_id in GATE_EPISODES:
            row, _ = capture_row(model, pristine, ids, seed, episode_id, "nominal", "1", nominal, protocol["protocol_sha256"])
            forward[episode_id] = row
            problems.extend(f"seed {seed} ep {episode_id} {item}" for item in differ(row, p0[episode_id]))
        repeated, _ = capture_row(model, pristine, ids, seed, 0, "nominal", "1", nominal, protocol["protocol_sha256"])
        problems.extend(f"seed {seed} repeat {item}" for item in differ(repeated, forward[0]))
        for episode_id in reversed(GATE_EPISODES):
            row, _ = capture_row(model, pristine, ids, seed, episode_id, "nominal", "1", nominal, protocol["protocol_sha256"])
            problems.extend(f"seed {seed} reverse ep {episode_id} {item}" for item in differ(row, forward[episode_id]))
    marker = OUT / "gate_result.json"
    marker.write_text(json.dumps({"passed": not problems, "problems": problems}, indent=2))
    if problems:
        raise SystemExit("gate failed: " + "; ".join(problems[:12]))
    print("[gate] passed", flush=True)


def readback(protocol: dict) -> None:
    corners = [
        ("object_mass", "0.05", physics(mass=0.05)),
        ("object_friction", "0.05", physics(friction=0.05)),
        ("joint_damping_scale", "0.001", physics(damping=0.001)),
        ("actuator_gain_scale", "0", physics(gain=0.0)),
    ]
    model, pristine, ids = load_policy(0)
    report = []
    for factor, value, spec in corners:
        row, bundle = capture_row(model, pristine, ids, 0, 0, factor, value, spec, protocol["protocol_sha256"])
        expected_obj_damp = 0.01 * spec["damping_scale"]
        item = {
            "factor": factor,
            "value": value,
            "mass_read": row["mass_read"],
            "friction_read": [row["friction_read"], row["friction_torsion_read"], row["friction_roll_read"]],
            "damping_scale_read": row["damping_scale_read"],
            "obj_dof_damping_read": row["obj_dof_damping_read"],
            "expected_obj_dof_damping": fmt(expected_obj_damp),
            "gain_scale_read": row["gain_scale_read"],
            "bias_unchanged": row["bias_unchanged"],
            "success": row["success"],
            "steps_used": row["steps_used"],
            "finite": True,
        }
        if abs(float(row["obj_dof_damping_read"]) - expected_obj_damp) > 1e-12:
            raise SystemExit(f"object damping {row['obj_dof_damping_read']} != {expected_obj_damp}")
        if abs(float(row["damping_scale_read"]) - spec["damping_scale"]) > 1e-8:
            raise SystemExit("damping scale readback mismatch")
        if abs(float(row["gain_scale_read"]) - spec["gain_scale"]) > 1e-8:
            raise SystemExit("gain scale readback mismatch")
        save_trace(OUT / "probes" / f"readback_{factor}_{value}.npz", bundle)
        report.append(item)
        print(f"[readback] {factor}={value} steps {row['steps_used']} success {row['success']}", flush=True)
    (OUT / "probes" / "readback.json").write_text(json.dumps(report, indent=2))
    print("[readback] passed", flush=True)


def formal(protocol: dict, seed: int) -> None:
    path = OUT / f"single_factor_seed{seed}.csv"
    finished = done_keys(path)
    model, pristine, ids = load_policy(seed)
    pending = [(factor, value, spec) for factor, value, spec in CELLS]
    for factor, value, spec in pending:
        for episode_id in range(N_EPISODES):
            if (factor, value, str(episode_id)) in finished:
                continue
            try:
                row, bundle = capture_row(
                    model, pristine, ids, seed, episode_id, factor, value, spec, protocol["protocol_sha256"]
                )
            except SystemExit as exc:
                status = OUT / "logs" / "cell_status.jsonl"
                status.parent.mkdir(parents=True, exist_ok=True)
                with status.open("a") as handle:
                    handle.write(json.dumps({
                        "seed": seed, "factor": factor, "value": value,
                        "episode_id": episode_id, "status": "stopped", "reason": str(exc),
                    }) + "\n")
                print(f"[formal] STOP seed {seed} {factor}={value} ep {episode_id}: {exc}", flush=True)
                return
            append_rows(path, [row])
            if row["success"] != "1":
                save_trace(OUT / "states" / f"fail_seed{seed}_{factor}_{value}_ep{episode_id}.npz", bundle)
            if seed == 0 and episode_id == 0:
                save_trace(OUT / "traces" / f"seed0_{factor}_{value}_ep0.npz", bundle)
            if episode_id % 50 == 0:
                print(f"[formal] seed {seed} {factor}={value} ep {episode_id} success {row['success']}", flush=True)
    count = len(done_keys(path))
    if count != len(CELLS) * N_EPISODES:
        raise SystemExit(f"seed {seed} has {count} rows")
    print(f"[formal] seed {seed} complete {count}", flush=True)


def write_reuse_map(protocol: dict) -> None:
    path = OUT / "reuse_map.csv"
    fields = [
        "factor", "value", "policy_seed", "episode_id", "source_file",
        "source_protocol_id", "source_protocol_sha256", "source_file_sha256", "evaluation_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in (0, 42, 123):
            source = P0_DIR / f"P0_v3_seed{seed}.csv"
            digest = sha256(source)
            p0 = load_p0(seed)
            for episode_id, row in p0.items():
                for factor, value in (("object_mass", "1"), ("object_friction", "1"), ("actuator_gain_scale", "1")):
                    writer.writerow({
                        "factor": factor,
                        "value": value,
                        "policy_seed": seed,
                        "episode_id": episode_id,
                        "source_file": str(source),
                        "source_protocol_id": row["protocol_id"],
                        "source_protocol_sha256": row["protocol_sha256"],
                        "source_file_sha256": digest,
                        "evaluation_source": "reused_P0",
                    })
    print(f"[reuse] {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gate", "readback", "formal", "reuse"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    os.environ.pop("MUJOCO_GL", None)
    import torch
    torch.set_num_threads(1)
    install_binary_loader()
    protocol = lock_protocol(build_protocol())
    if args.mode == "gate":
        gate(protocol)
    elif args.mode == "readback":
        readback(protocol)
    elif args.mode == "reuse":
        write_reuse_map(protocol)
    else:
        if args.seed not in CHECKPOINT:
            raise SystemExit("seed must be 0, 42 or 123")
        marker = json.loads((OUT / "gate_result.json").read_text())
        if not marker.get("passed"):
            raise SystemExit("gate has not passed")
        formal(protocol, args.seed)


if __name__ == "__main__":
    main()
