"""Clean-reset spatial retest.

One new environment per episode. Writer C at the nominal 1 kg model, the
Phase E reset order, and a true delay of 0. Horizon 100 is an outer cap.
H50 is a reading of that same trajectory only after an independent H50 run
matches the H100 prefix. The manuscript and the older CSVs are not modified.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROUND3_SCRIPTS = OUT.parent / "corrective_round3_2026-10-01" / "scripts"
OLD_SPATIAL = OUT.parent / "corrective_2026-10-01"
PANDA_FINAL = OUT.parents[1]
ASSETS = PANDA_FINAL / "panda_mujoco_gym" / "assets"
CANONICAL_PUSH = ASSETS / "push.xml"
CANONICAL_MOCAP = ASSETS / "panda_mocap.xml"
VERIFIED = OUT / "verified_assets"
CACHE = OUT / "cache"
MJB = CACHE / "panda_push.mjb"
CACHE_MANIFEST = CACHE / "asset_cache_manifest.json"
LOCK_PATH = CACHE / "mj_load.lock"
CASES = OUT / "equivalence_cases.csv"
RULE = OUT / "equivalence_case_rule.json"
PROTOCOL_PATH = OUT / "spatial_protocol.json"
H50_OLD = OLD_SPATIAL / "spatial_h50_raw.csv"
H100_OLD = OLD_SPATIAL / "spatial_h100_raw.csv"

PROTOCOL_ID = "spatial_clean_reset_writerC_prefix"
RULE_ID = "smallest_late_success_per_L_plus_L0.30_ep0_early_and_L0.80_ep2_double_fail"
HORIZON_SOURCE = "trajectory_prefix"
RESET_BASE = 6_000_000
N_EPISODES = 600
SIDES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
PHYSICS = {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}
EXPECTED_PACKAGES = {
    "numpy": "2.2.6",
    "torch": "2.14.0",
    "mujoco": "3.1.6",
    "gymnasium": "1.1.1",
    "gymnasium-robotics": "1.3.1",
    "stable-baselines3": "2.6.0",
    "sb3-contrib": "2.6.0",
}
EXPECTED_FILES = {
    "panda_env.py": "76045e06159706fd6a00077a801158f6d583a794ec62411c36a4c8fde9221a36",
    "push.py": "8ddfb1bdb47d72c80658bd4b434aafcd4a8e1563acebaf7859d10bfe5fcccea0",
    "push.xml": "a5feb6af7c678b11254c05f1a41cf1bddd043b5c2fa93dfbb7c54171bce3f9cb",
    "panda_mocap.xml": "8c1608812335d02b2b0c7178bc1aae91351cf2c890d1633a2748b352e3d7dd09",
    "checkpoint_seed0": "ab38ff9c17dbde348f0d0048858d4753570c04b2e7f486cacf60135717f085bd",
    "checkpoint_seed42": "e7c4840645dae9949b40c2b58045888a354b8193d3d0ffea429ae1a3ba087f74",
    "checkpoint_seed123": "25f64416614872cddcf5dc83075b55fea5f32a81b4ab60caa17454de9f3ea966",
}
CASE_FIELDS = [
    "policy_seed",
    "episode_id",
    "reset_seed",
    "L",
    "role",
    "old_h50_success",
    "old_h50_steps",
    "old_h100_success",
    "old_h100_steps",
    "old_initial_distance",
]
FIELDS = [
    "policy_seed",
    "episode_id",
    "reset_seed",
    "L",
    "obj_low_x",
    "obj_low_y",
    "obj_low_z",
    "obj_high_x",
    "obj_high_y",
    "obj_high_z",
    "goal_low_x",
    "goal_low_y",
    "goal_low_z",
    "goal_high_x",
    "goal_high_y",
    "goal_high_z",
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "target_x",
    "target_y",
    "target_z",
    "initial_distance",
    "initial_near",
    "delay_steps",
    "latency_s",
    "mass",
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
    "control_restored",
    "goal_unchanged",
    "success_h50",
    "steps_h50",
    "distance_h50",
    "success_h100",
    "steps_h100",
    "distance_h100",
    "first_success_step",
    "state_h50",
    "state_h100",
    "protocol_id",
    "protocol_sha256",
    "checkpoint_sha256",
    "horizon_source",
    "model_check_ok",
    "writer",
]
TRACE_KEYS = (
    "observation",
    "achieved_in",
    "desired_in",
    "policy_action",
    "executed_action",
    "object_pos",
    "qpos",
    "qvel",
    "qacc",
    "qacc_warmstart",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "distance",
    "warmstart_after_reset",
)


def side_text(side: float) -> str:
    return format(float(side), ".2f")


def reset_seed_for(side: float, episode_id: int) -> int:
    return RESET_BASE + int(round(float(side) * 100)) * 1000 + int(episode_id)


def schedule() -> list[tuple[str, int]]:
    return [(side_text(side), episode_id) for side in SIDES for episode_id in range(N_EPISODES)]


def fmt(value: float) -> str:
    return format(float(value), ".17g")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_old(path: Path) -> dict:
    rows = {}
    for row in read_csv(path):
        key = (int(row["policy_seed"]), side_text(row["L"]), int(row["episode_id"]))
        if key in rows:
            raise SystemExit(f"duplicate old spatial row {key} in {path.name}")
        rows[key] = row
    if len(rows) != len(SIDES) * 3 * N_EPISODES:
        raise SystemExit(f"{path.name} has {len(rows)} rows")
    return rows


def freeze_cases() -> None:
    if CASES.exists() or RULE.exists():
        if not (CASES.exists() and RULE.exists()):
            raise SystemExit("equivalence case files are incomplete; refusing to regenerate")
        rule = json.loads(RULE.read_text())
        if rule.get("rule_id") != RULE_ID or rule.get("cases_sha256") != sha256_file(CASES):
            raise SystemExit("frozen equivalence cases do not match this rule")
        print(f"[freeze] already frozen, {rule['n_cases']} cases", flush=True)
        return
    h50 = index_old(H50_OLD)
    h100 = index_old(H100_OLD)
    chosen = []
    for seed in (0, 42, 123):
        for side in SIDES:
            late = []
            for episode_id in range(N_EPISODES):
                key = (seed, side_text(side), episode_id)
                old50 = h50[key]
                old100 = h100[key]
                expected = reset_seed_for(side, episode_id)
                if int(old50["reset_seed"]) != expected or int(old100["reset_seed"]) != expected:
                    raise SystemExit(f"old reset seed differs at {key}")
                if int(old50["success"]) == 0 and int(old100["success"]) == 1 and int(old100["steps_used"]) > 50:
                    late.append(episode_id)
            if late:
                chosen.append((seed, side, min(late), "late_success"))
        early = h50[(seed, "0.30", 0)]
        early100 = h100[(seed, "0.30", 0)]
        if not (
            int(early["success"]) == 1
            and int(early100["success"]) == 1
            and int(early["steps_used"]) <= 5
            and int(early100["steps_used"]) <= 5
        ):
            raise SystemExit(f"seed {seed} L=0.30 episode 0 is not the early-success anchor")
        chosen.append((seed, 0.30, 0, "early_success"))
        fail50 = h50[(seed, "0.80", 2)]
        fail100 = h100[(seed, "0.80", 2)]
        if int(fail50["success"]) != 0 or int(fail100["success"]) != 0:
            raise SystemExit(f"seed {seed} L=0.80 episode 2 is not the double-failure anchor")
        chosen.append((seed, 0.80, 2, "double_failure"))
    unique = []
    seen = set()
    for seed, side, episode_id, role in chosen:
        key = (seed, side_text(side), episode_id)
        if key in seen:
            raise SystemExit(f"duplicate equivalence case {key}")
        seen.add(key)
        unique.append((seed, side, episode_id, role))
    unique.sort(key=lambda item: (item[0], item[1], item[2]))
    CASES.parent.mkdir(parents=True, exist_ok=True)
    with CASES.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        for seed, side, episode_id, role in unique:
            key = (seed, side_text(side), episode_id)
            old50 = h50[key]
            old100 = h100[key]
            writer.writerow(
                {
                    "policy_seed": str(seed),
                    "episode_id": str(episode_id),
                    "reset_seed": str(reset_seed_for(side, episode_id)),
                    "L": side_text(side),
                    "role": role,
                    "old_h50_success": old50["success"],
                    "old_h50_steps": old50["steps_used"],
                    "old_h100_success": old100["success"],
                    "old_h100_steps": old100["steps_used"],
                    "old_initial_distance": old50["initial_distance"],
                }
            )
    rule = {
        "rule_id": RULE_ID,
        "n_cases": len(unique),
        "cases_sha256": sha256_file(CASES),
        "spatial_h50_raw_sha256": sha256_file(H50_OLD),
        "spatial_h100_raw_sha256": sha256_file(H100_OLD),
        "selection": (
            "For each policy, the smallest episode at each L whose old H50 run failed "
            "and whose old H100 run succeeded after step 50. Plus L=0.30 episode 0 "
            "and L=0.80 episode 2."
        ),
    }
    RULE.write_text(canonical(rule))
    print(f"[freeze] wrote {len(unique)} cases", flush=True)


def readings_from_trace(trace: dict) -> dict:
    if int(trace["horizon"][0]) != 100:
        raise SystemExit("H50/H100 readings require the 100-step trajectory")
    steps = int(trace["steps"][0])
    success_step = int(trace["success_step"][0])
    distance = np.asarray(trace["distance"], dtype=np.float64)
    if distance.shape != (steps,):
        raise SystemExit(f"distance length {distance.shape} does not match {steps} steps")
    if steps < 1:
        raise SystemExit("episode took no environment step")
    if success_step > 0:
        if success_step != steps or not (distance[-1] < 0.05):
            raise SystemExit("success step does not match the final distance")
        if np.any(distance[:-1] < 0.05):
            raise SystemExit("an earlier step was already inside 0.05 m")
    else:
        if steps != 100 or np.any(distance < 0.05):
            raise SystemExit("a failure did not reach step 100 outside 0.05 m")
    if success_step > 0 and success_step <= 50:
        success_h50 = 1
        steps_h50 = success_step
        distance_h50 = float(distance[success_step - 1])
    else:
        if steps < 50 or int(trace["has_state50"][0]) != 1:
            raise SystemExit("missing the step-50 state on a trajectory that was not done by step 50")
        success_h50 = 0
        steps_h50 = 50
        distance_h50 = float(distance[49])
        if distance_h50 < 0.05:
            raise SystemExit("step 50 is inside 0.05 m but the episode did not stop")
    if success_step > 0:
        success_h100 = 1
        steps_h100 = success_step
        distance_h100 = float(distance[success_step - 1])
    else:
        success_h100 = 0
        steps_h100 = 100
        distance_h100 = float(distance[-1])
    if success_h50 and not success_h100:
        raise SystemExit("H50 success without H100 success")
    if success_h50 != int(distance_h50 < 0.05) or success_h100 != int(distance_h100 < 0.05):
        raise SystemExit("success flag does not match its distance")
    return {
        "success_h50": success_h50,
        "steps_h50": steps_h50,
        "distance_h50": distance_h50,
        "success_h100": success_h100,
        "steps_h100": steps_h100,
        "distance_h100": distance_h100,
        "first_success_step": success_step,
    }


def self_check() -> None:
    if reset_seed_for(0.10, 0) != 6_010_000:
        raise SystemExit("reset seed for L=0.10 episode 0")
    if reset_seed_for(0.80, 599) != RESET_BASE + 80_000 + 599:
        raise SystemExit("reset seed for L=0.80 episode 599")
    if len(schedule()) != 4800:
        raise SystemExit("schedule length")

    def trace_from(values: list[float]) -> dict:
        distance = np.asarray(values, dtype=np.float64)
        success = next((index + 1 for index, value in enumerate(distance) if value < 0.05), -1)
        steps = success if success > 0 else len(distance)
        distance = distance[:steps]
        return {
            "horizon": np.array([100], dtype=np.int32),
            "steps": np.array([steps], dtype=np.int32),
            "success_step": np.array([success], dtype=np.int32),
            "distance": distance,
            "has_state50": np.array([1 if steps >= 50 else 0], dtype=np.int32),
        }

    early = readings_from_trace(trace_from([0.2, 0.1, 0.04, 0.01]))
    if (early["success_h50"], early["steps_h50"], early["success_h100"], early["steps_h100"]) != (1, 3, 1, 3):
        raise SystemExit(f"early reading {early}")
    late_values = [0.4] * 59 + [0.04]
    late = readings_from_trace(trace_from(late_values))
    if (late["success_h50"], late["steps_h50"], late["distance_h50"], late["success_h100"], late["steps_h100"]) != (
        0,
        50,
        0.4,
        1,
        60,
    ):
        raise SystemExit(f"late reading {late}")
    failed = readings_from_trace(trace_from([0.4] * 100))
    if (failed["success_h50"], failed["success_h100"], failed["steps_h100"], failed["first_success_step"]) != (
        0,
        0,
        100,
        -1,
    ):
        raise SystemExit(f"failure reading {failed}")
    try:
        readings_from_trace(trace_from([0.4] * 40))
    except SystemExit:
        pass
    else:
        raise SystemExit("a short failure was accepted")
    print("[self-check] reset seeds and horizon readings passed", flush=True)


def require_runtime() -> None:
    if "/tmp/phaseE_py" in os.environ.get("PYTHONPATH", ""):
        raise SystemExit("refusing the unrecorded /tmp/phaseE_py injection")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise SystemExit(f"{name} must be 1 before this process starts")
    import importlib.metadata as metadata

    versions = {}
    for name, expected in EXPECTED_PACKAGES.items():
        got = metadata.version(name)
        if got != expected:
            raise SystemExit(f"{name} {got} does not match the locked pin {expected}")
        versions[name] = got
    import mujoco

    fn = mujoco.MjModel.from_xml_path
    import inspect

    if inspect.isfunction(fn):
        raise SystemExit("MjModel.from_xml_path is already a Python wrapper")
    return versions


def mesh_rows(root: Path) -> list[tuple[str, str]]:
    mesh_root = root / "meshes"
    rows = []
    for path in sorted(mesh_root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rows.append((path.relative_to(root).as_posix(), sha256_file(path)))
    if len(rows) < 60:
        raise SystemExit(f"expected the mesh library under {mesh_root}")
    return rows


def mesh_manifest_sha(rows: list[tuple[str, str]]) -> str:
    text = "".join(f"{name} {digest}\n" for name, digest in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def copy_verified_assets() -> str:
    VERIFIED.mkdir(parents=True, exist_ok=True)
    for source, name in ((CANONICAL_PUSH, "push.xml"), (CANONICAL_MOCAP, "panda_mocap.xml")):
        digest = sha256_file(source)
        if digest != EXPECTED_FILES[name]:
            raise SystemExit(f"canonical {name} hash changed")
        target = VERIFIED / name
        if (not target.exists()) or sha256_file(target) != digest:
            shutil.copy2(source, target)
        if sha256_file(target) != digest:
            raise SystemExit(f"verified {name} hash changed during copy")
    source_rows = mesh_rows(ASSETS)
    dest_mesh = VERIFIED / "meshes"
    if dest_mesh.exists():
        try:
            dest_rows = mesh_rows(VERIFIED)
        except SystemExit:
            dest_rows = []
    else:
        dest_rows = []
    if dest_rows != source_rows:
        if dest_mesh.exists():
            shutil.rmtree(dest_mesh)
        shutil.copytree(ASSETS / "meshes", dest_mesh)
        dest_rows = mesh_rows(VERIFIED)
        if dest_rows != source_rows:
            raise SystemExit("verified mesh copy does not match the canonical meshes")
    return mesh_manifest_sha(source_rows)


def model_arrays(model) -> dict:
    return {
        "body_mass": np.array(model.body_mass, copy=True),
        "body_inertia": np.array(model.body_inertia, copy=True),
        "dof_damping": np.array(model.dof_damping, copy=True),
        "actuator_gainprm": np.array(model.actuator_gainprm, copy=True),
        "geom_friction": np.array(model.geom_friction, copy=True),
        "qpos0": np.array(model.qpos0, copy=True),
        "mesh_vert": np.array(model.mesh_vert, copy=True),
    }


def model_counts(model) -> dict:
    return {name: int(getattr(model, name)) for name in ("nbody", "nq", "nv", "nu", "nmesh")}


def same_model(left, right) -> str:
    if model_counts(left) != model_counts(right):
        return f"counts {model_counts(left)} vs {model_counts(right)}"
    left_arrays = model_arrays(left)
    right_arrays = model_arrays(right)
    for name, value in left_arrays.items():
        other = right_arrays[name]
        if value.shape != other.shape or not np.array_equal(value, other):
            return name
    return ""


def compile_verified_xml():
    import mujoco

    source = VERIFIED / "push.xml"
    try:
        return mujoco.MjModel.from_xml_path(str(source)), str(source)
    except ValueError as exc:
        if "resource not found" not in str(exc):
            raise
    scratch = Path("/tmp/spatial_clean_verified_assets")
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(VERIFIED, scratch)
    compiled = mujoco.MjModel.from_xml_path(str(scratch / "push.xml"))
    return compiled, str(scratch / "push.xml")


def ensure_cache() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        mesh_sha = copy_verified_assets()
        push_sha = sha256_file(VERIFIED / "push.xml")
        mocap_sha = sha256_file(VERIFIED / "panda_mocap.xml")
        if CACHE_MANIFEST.exists() and MJB.exists():
            manifest = json.loads(CACHE_MANIFEST.read_text())
            if (
                manifest.get("push_xml_sha256") == push_sha
                and manifest.get("mocap_sha256") == mocap_sha
                and manifest.get("mesh_manifest_sha256") == mesh_sha
                and manifest.get("mjb_sha256") == sha256_file(MJB)
            ):
                return manifest
        compiled, source = compile_verified_xml()
        import mujoco

        partial = CACHE / "panda_push_partial.mjb"
        if partial.exists():
            partial.unlink()
        mujoco.mj_saveModel(compiled, str(partial), None)
        loaded = mujoco.MjModel.from_binary_path(str(partial))
        mismatch = same_model(compiled, loaded)
        if mismatch:
            raise SystemExit(f"binary cache does not match the compiled XML ({mismatch})")
        partial.replace(MJB)
        manifest = {
            "push_xml_sha256": push_sha,
            "mocap_sha256": mocap_sha,
            "mesh_manifest_sha256": mesh_sha,
            "mjb_sha256": sha256_file(MJB),
            "compile_source": source,
            "field_compare": "np.array_equal on counts, mass, inertia, damping, gain, friction, qpos0, mesh_vert",
            "counts": model_counts(loaded),
        }
        CACHE_MANIFEST.write_text(canonical(manifest))
        return manifest


def install_binary_loader() -> None:
    import mujoco

    canonical = os.path.realpath(CANONICAL_PUSH)
    mjb = str(MJB)

    def from_xml_path(filename, *args, **kwargs):
        if os.path.realpath(filename) == canonical:
            return mujoco.MjModel.from_binary_path(mjb)
        raise SystemExit(f"unexpected XML load outside the verified cache: {filename}")

    mujoco.MjModel.from_xml_path = from_xml_path


def load_cases() -> list[dict]:
    rule = json.loads(RULE.read_text())
    if rule.get("rule_id") != RULE_ID or rule.get("cases_sha256") != sha256_file(CASES):
        raise SystemExit("equivalence cases are not the frozen file")
    rows = read_csv(CASES)
    if len(rows) != int(rule["n_cases"]):
        raise SystemExit("equivalence case count changed")
    return rows


def import_paths() -> None:
    old_scripts = PANDA_FINAL / "data_and_analysis" / "corrective_2026-10-01" / "scripts"
    for path in (old_scripts, ROUND3_SCRIPTS):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def file_table(manifest: dict) -> dict:
    import_paths()
    from round2_common import MODELS

    paths = {
        "spatial_formal.py": Path(__file__).resolve(),
        "phaseB_diagnose.py": ROUND3_SCRIPTS / "phaseB_diagnose.py",
        "phaseC_mass_implementation.py": ROUND3_SCRIPTS / "phaseC_mass_implementation.py",
        "phaseE_formal.py": ROUND3_SCRIPTS / "phaseE_formal.py",
        "round2_common.py": PANDA_FINAL / "data_and_analysis" / "corrective_2026-10-01" / "scripts" / "round2_common.py",
        "panda_env.py": PANDA_FINAL / "panda_mujoco_gym" / "envs" / "panda_env.py",
        "push.py": PANDA_FINAL / "panda_mujoco_gym" / "envs" / "push.py",
        "push.xml": CANONICAL_PUSH,
        "panda_mocap.xml": CANONICAL_MOCAP,
        "verified_push.xml": VERIFIED / "push.xml",
        "verified_panda_mocap.xml": VERIFIED / "panda_mocap.xml",
        "panda_push.mjb": MJB,
        "equivalence_cases.csv": CASES,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes["mesh_manifest_sha256"] = manifest["mesh_manifest_sha256"]
    for seed, path in MODELS.items():
        hashes[f"checkpoint_seed{seed}"] = sha256_file(path)
    for name, expected in EXPECTED_FILES.items():
        if hashes[name] != expected:
            raise SystemExit(f"locked hash mismatch for {name}")
    if hashes["verified_push.xml"] != hashes["push.xml"] or hashes["verified_panda_mocap.xml"] != hashes["panda_mocap.xml"]:
        raise SystemExit("verified XML copy differs from the canonical file")
    phase_e_protocol = PANDA_FINAL / "data_and_analysis" / "corrective_round3_2026-10-01" / "phaseE_protocol.json"
    recorded = json.loads(phase_e_protocol.read_text())["files"]
    for name in ("push.xml", "panda_mocap.xml", "panda_env.py", "push.py", "checkpoint_seed0", "checkpoint_seed42", "checkpoint_seed123"):
        if recorded[name] != hashes[name]:
            raise SystemExit(f"{name} does not match phaseE_protocol.json")
    return hashes


def build_protocol(manifest: dict, versions: dict) -> dict:
    body = {
        "protocol_id": PROTOCOL_ID,
        "writer": "C",
        "init": "snapshot restore, mj_setConst, mj_resetData, neutral ctrl[0:7], joint neutral, initial_qvel, mocap, sample object, mj_forward; goal sampled by reset after that",
        "physics": "mass=friction=damping_scale=gain_scale=1",
        "delay_steps": 0,
        "delay_meaning": "Delay.apply returns the current action; FrankaEnv.step clips it",
        "goal_noise_applied": False,
        "goal_rewrite_after_observation": False,
        "L": [side_text(side) for side in SIDES],
        "goal_center_xy": [0.6, 0.0],
        "obj_xy_range": 0.3,
        "n_episodes": N_EPISODES,
        "reset_seed_rule": "6000000+round(L*100)*1000+episode_id",
        "horizon_cap": 100,
        "horizon_source": HORIZON_SOURCE,
        "horizon_condition": "prefix readings are valid only after equivalence_seed{0,42,123}.json pass",
        "success": "object to target distance < 0.05, after at least one env.step",
        "new_env_every_episode": True,
        "threads": "torch.set_num_threads(1) and OMP/MKL/OPENBLAS/VECLIB=1; interop threads unchanged",
        "device": "cpu",
        "deterministic": True,
        "loader": "from_binary_path of cache/panda_push.mjb, patched inside this script",
        "packages": versions,
        "files": file_table(manifest),
        "mjb_sha256": manifest["mjb_sha256"],
        "cases_sha256": sha256_file(CASES),
    }
    body["protocol_sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return body


def lock_protocol(protocol: dict) -> None:
    text = canonical(protocol)
    if PROTOCOL_PATH.exists():
        if PROTOCOL_PATH.read_text() != text:
            raise SystemExit(f"protocol file differs from this process; refusing to continue ({PROTOCOL_PATH})")
        return
    try:
        fd = os.open(PROTOCOL_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        if PROTOCOL_PATH.read_text() != text:
            raise SystemExit("protocol file differs from this process; refusing to continue")
        return
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def assert_spatial_ranges(base, side: float) -> None:
    if float(base.goal_xy_range) != float(side) or float(base.obj_xy_range) != 0.3:
        raise SystemExit(f"ranges goal {base.goal_xy_range} object {base.obj_xy_range}")
    goal_low = np.asarray(base.goal_range_low, dtype=np.float64)
    goal_high = np.asarray(base.goal_range_high, dtype=np.float64)
    obj_low = np.asarray(base.obj_range_low, dtype=np.float64)
    obj_high = np.asarray(base.obj_range_high, dtype=np.float64)
    goal_span = goal_high - goal_low
    obj_span = obj_high - obj_low
    if not np.allclose(goal_span[:2], [side, side], atol=1e-12, rtol=0.0):
        raise SystemExit(f"goal span {goal_span[:2]} for L={side}")
    if not np.allclose(obj_span[:2], [0.3, 0.3], atol=1e-12, rtol=0.0):
        raise SystemExit(f"object span {obj_span[:2]}")
    goal_center = 0.5 * (goal_low + goal_high)
    obj_center = 0.5 * (obj_low + obj_high)
    if not np.allclose(goal_center[:2], [0.6, 0.0], atol=1e-12, rtol=0.0):
        raise SystemExit(f"goal center {goal_center[:2]}")
    if not np.allclose(obj_center[:2], [0.6, 0.0], atol=1e-12, rtol=0.0):
        raise SystemExit(f"object center {obj_center[:2]}")
    if not np.allclose(obj_low, [0.45, -0.15, 0.0], atol=1e-12, rtol=0.0) or not np.allclose(
        obj_high, [0.75, 0.15, 0.0], atol=1e-12, rtol=0.0
    ):
        raise SystemExit("object box is not the 0.30 m training box")
    from round2_common import assert_control_period

    assert_control_period(base)


def make_env(horizon: int, side: float):
    import gymnasium as gym

    last_error = None
    env = None
    for attempt in range(8):
        try:
            with LOCK_PATH.open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                env = gym.make(
                    "FrankaPushSparse-v0",
                    max_episode_steps=int(horizon),
                    goal_xy_range=float(side),
                    disable_env_checker=True,
                )
            break
        except ValueError as exc:
            last_error = exc
            if "resource not found" not in str(exc) or attempt == 7:
                raise
            time.sleep(0.5 * (attempt + 1))
    if env is None:
        raise last_error
    found = False
    node = env
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            if int(node._max_episode_steps) != int(horizon):
                raise SystemExit(f"TimeLimit is {node._max_episode_steps}, expected {horizon}")
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit missing")
    base = env.unwrapped
    if float(base.distance_threshold) != 0.05:
        raise SystemExit(f"distance threshold {base.distance_threshold}")
    return env, base


def save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + "_partial.npz")
    if partial.exists():
        partial.unlink()
    np.savez_compressed(partial, **payload)
    partial.replace(path)


def rollout(model, pristine, ids, *, side: float, episode_id: int, horizon: int, reset_seed: int) -> dict:
    import phaseB_diagnose as pb
    import phaseE_formal as pe
    from phaseC_mass_implementation import install_episode_reset, object_ids
    from round2_common import set_goal_side

    env, base = make_env(horizon, side)
    try:
        if object_ids(base) != ids:
            raise SystemExit("object body ids differ from the template model")
        install_episode_reset(base, pristine, ids, PHYSICS, "C")
        set_goal_side(base, float(side))
        assert_spatial_ranges(base, float(side))
        obs, _info = env.reset(seed=int(reset_seed))
        originals = {"obj_body": ids["body"], "obj_geom": ids["geom"]}
        after_reset = pb.capture_state(base, originals)
        achieved0 = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
        desired0 = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
        measured = pe.measure(
            base,
            {
                "state_after_reset_full": after_reset,
                "achieved_after_reset": achieved0,
                "desired_after_reset": desired0,
            },
            pristine,
            ids,
            PHYSICS,
        )
        delay = pb.Delay(0, int(np.asarray(env.action_space.shape)[0]))
        observations = []
        achieved_in = []
        desired_in = []
        policy_actions = []
        executed_actions = []
        object_pos = [achieved0]
        qpos = [after_reset["qpos"]]
        qvel = [after_reset["qvel"]]
        qacc = [after_reset["qacc"]]
        warm = [after_reset["qacc_warmstart"]]
        ctrl = [after_reset["ctrl"]]
        mocap_pos = [after_reset["mocap_pos"]]
        mocap_quat = [after_reset["mocap_quat"]]
        distances = []
        state50 = None
        success_step = -1
        steps = 0
        final_state = after_reset
        while steps < horizon:
            seen = np.asarray(obs["observation"], dtype=np.float64).copy()
            seen_achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
            seen_desired = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
            if not np.array_equal(seen_desired, desired0):
                raise SystemExit("goal changed before the action")
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).copy()
            executed = np.asarray(delay.apply(action), dtype=np.float32).copy()
            observations.append(seen)
            achieved_in.append(seen_achieved)
            desired_in.append(seen_desired)
            policy_actions.append(action)
            executed_actions.append(executed)
            obs, _reward, terminated, truncated, info = env.step(executed)
            steps += 1
            if not np.array_equal(np.asarray(obs["desired_goal"], dtype=np.float64), desired0):
                raise SystemExit("goal changed during the episode")
            achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
            distance = float(np.linalg.norm(achieved - desired0))
            reported = int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)
            if reported != int(distance < 0.05):
                raise SystemExit(f"success flag {reported} disagrees with distance {distance}")
            state = pb.capture_state(base, originals)
            object_pos.append(achieved)
            qpos.append(state["qpos"])
            qvel.append(state["qvel"])
            qacc.append(state["qacc"])
            warm.append(state["qacc_warmstart"])
            ctrl.append(state["ctrl"])
            mocap_pos.append(state["mocap_pos"])
            mocap_quat.append(state["mocap_quat"])
            distances.append(distance)
            final_state = state
            if steps == 50:
                state50 = state
            if distance < 0.05:
                success_step = steps
                break
            if steps == horizon:
                if bool(terminated) and not bool(truncated):
                    raise SystemExit("horizon step ended terminated without being inside 0.05 m")
                break
            if bool(terminated) or bool(truncated):
                raise SystemExit(f"episode stopped at step {steps} before the horizon")
        if steps < 1:
            raise SystemExit("no environment step was taken")
        trace = {
            "horizon": np.array([horizon], dtype=np.int32),
            "steps": np.array([steps], dtype=np.int32),
            "success_step": np.array([success_step], dtype=np.int32),
            "distance": np.asarray(distances, dtype=np.float64),
            "has_state50": np.array([0 if state50 is None else 1], dtype=np.int32),
            "observation": np.stack(observations),
            "achieved_in": np.stack(achieved_in),
            "desired_in": np.stack(desired_in),
            "policy_action": np.stack(policy_actions),
            "executed_action": np.stack(executed_actions),
            "object_pos": np.stack(object_pos),
            "qpos": np.stack(qpos),
            "qvel": np.stack(qvel),
            "qacc": np.stack(qacc),
            "qacc_warmstart": np.stack(warm),
            "ctrl": np.stack(ctrl),
            "mocap_pos": np.stack(mocap_pos),
            "mocap_quat": np.stack(mocap_quat),
            "warmstart_after_reset": np.asarray(after_reset["qacc_warmstart"], dtype=np.float64),
            "time_after_reset": np.array([measured["time"]], dtype=np.float64),
            "achieved0": achieved0,
            "desired0": desired0,
            "goal_low": np.asarray(base.goal_range_low, dtype=np.float64).copy(),
            "goal_high": np.asarray(base.goal_range_high, dtype=np.float64).copy(),
            "obj_low": np.asarray(base.obj_range_low, dtype=np.float64).copy(),
            "obj_high": np.asarray(base.obj_range_high, dtype=np.float64).copy(),
            "measured": measured,
            "state50": state50,
            "state_final": final_state,
        }
        return trace
    finally:
        env.close()


def trace_arrays(trace: dict) -> dict:
    payload = {key: np.asarray(trace[key]) for key in TRACE_KEYS}
    payload["horizon"] = trace["horizon"]
    payload["steps"] = trace["steps"]
    payload["success_step"] = trace["success_step"]
    payload["time_after_reset"] = trace["time_after_reset"]
    return payload


def load_trace(path: Path) -> dict:
    with np.load(path) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def compare_traces(short: dict, long: dict, prefix: bool) -> list[str]:
    steps = int(short["steps"][0])
    problems = []
    step_keys = ("observation", "achieved_in", "desired_in", "policy_action", "executed_action", "distance")
    state_keys = ("object_pos", "qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")
    for key in step_keys:
        left = np.asarray(short[key])
        right = np.asarray(long[key])
        if prefix:
            right = right[:steps]
        if not np.array_equal(left, right):
            problems.append(f"{key} maxabs {max_abs(left, right)} shape {left.shape} vs {right.shape}")
    for key in state_keys:
        left = np.asarray(short[key])
        right = np.asarray(long[key])
        if prefix:
            right = right[: steps + 1]
        if not np.array_equal(left, right):
            problems.append(f"{key} maxabs {max_abs(left, right)} shape {left.shape} vs {right.shape}")
    if not np.array_equal(short["warmstart_after_reset"], long["warmstart_after_reset"]):
        problems.append(
            "warmstart_after_reset maxabs "
            f"{max_abs(short['warmstart_after_reset'], long['warmstart_after_reset'])}"
        )
    if not np.array_equal(short["time_after_reset"], long["time_after_reset"]):
        problems.append("time_after_reset differs")
    short_success = int(short["success_step"][0])
    long_success = int(long["success_step"][0])
    if prefix:
        if short_success > 0 and (long_success != short_success or int(long["steps"][0]) != short_success):
            problems.append(f"H100 did not stop at the H50 success step {short_success}; H100 success {long_success}")
        if short_success < 0 and steps != 50:
            problems.append(f"independent H50 failure has {steps} steps")
    elif short_success != long_success or int(short["steps"][0]) != int(long["steps"][0]):
        problems.append("repeat length or success step differs")
    return problems


def trace_path(seed: int, side: str, episode_id: int, tag: str) -> Path:
    return OUT / "equivalence_traces" / f"seed{seed}" / f"L{side}_ep{episode_id:04d}_{tag}.npz"


def marker_path(seed: int) -> Path:
    return OUT / f"equivalence_seed{seed}.json"


def cases_for_seed(seed: int) -> list[dict]:
    rows = [row for row in load_cases() if int(row["policy_seed"]) == seed]
    if len(rows) < 6:
        raise SystemExit(f"seed {seed} has {len(rows)} equivalence cases")
    roles = {row["role"] for row in rows}
    if "late_success" not in roles or "early_success" not in roles or "double_failure" not in roles:
        raise SystemExit(f"seed {seed} equivalence roles are incomplete: {roles}")
    return rows


def run_case(model, pristine, ids, row: dict, horizon: int) -> dict:
    return rollout(
        model,
        pristine,
        ids,
        side=float(row["L"]),
        episode_id=int(row["episode_id"]),
        horizon=horizon,
        reset_seed=int(row["reset_seed"]),
    )


def equivalence(model, pristine, ids, seed: int, protocol_sha: str) -> None:
    rows = cases_for_seed(seed)
    reports = []
    passed = True
    forward = {}
    for row in rows:
        side = row["L"]
        episode_id = int(row["episode_id"])
        h50 = run_case(model, pristine, ids, row, 50)
        h100 = run_case(model, pristine, ids, row, 100)
        prefix_problems = compare_traces(trace_arrays(h50), trace_arrays(h100), prefix=True)
        repeated = run_case(model, pristine, ids, row, 100)
        repeat_problems = compare_traces(trace_arrays(h100), trace_arrays(repeated), prefix=False)
        save_npz(trace_path(seed, side, episode_id, "h50"), trace_arrays(h50))
        save_npz(trace_path(seed, side, episode_id, "h100"), trace_arrays(h100))
        save_npz(trace_path(seed, side, episode_id, "h100_repeat"), trace_arrays(repeated))
        forward[(side, episode_id)] = trace_arrays(h100)
        item = {
            "L": side,
            "episode_id": episode_id,
            "role": row["role"],
            "h50_steps": int(h50["steps"][0]),
            "h100_steps": int(h100["steps"][0]),
            "h50_success_step": int(h50["success_step"][0]),
            "h100_success_step": int(h100["success_step"][0]),
            "prefix_problems": prefix_problems,
            "repeat_problems": repeat_problems,
            "warmstart_maxabs": float(np.max(np.abs(h100["warmstart_after_reset"]))),
            "time_after_reset": float(h100["time_after_reset"][0]),
        }
        if prefix_problems or repeat_problems:
            passed = False
        reports.append(item)
        print(
            f"[equivalence] seed {seed} L={side} ep {episode_id} {row['role']} "
            f"h50_steps {item['h50_steps']} h100_steps {item['h100_steps']} "
            f"warmstart {item['warmstart_maxabs']:.6g}",
            flush=True,
        )
    for row in reversed(rows):
        side = row["L"]
        episode_id = int(row["episode_id"])
        reverse = run_case(model, pristine, ids, row, 100)
        problems = compare_traces(forward[(side, episode_id)], trace_arrays(reverse), prefix=False)
        save_npz(trace_path(seed, side, episode_id, "h100_reverse"), trace_arrays(reverse))
        for item in reports:
            if item["L"] == side and item["episode_id"] == episode_id:
                item["reverse_problems"] = problems
                if problems:
                    passed = False
        print(f"[equivalence] seed {seed} reverse L={side} ep {episode_id} problems {len(problems)}", flush=True)
    marker = {
        "passed": passed,
        "seed": seed,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "horizon_source_if_passed": HORIZON_SOURCE,
        "comparison": "np.array_equal of observations, actions, object trajectory, and recorded state",
        "cases": reports,
    }
    marker_path(seed).write_text(canonical(marker))
    if not passed:
        raise SystemExit(f"equivalence failed for seed {seed}")
    print(f"[equivalence] seed {seed} passed", flush=True)


def require_equivalence(protocol_sha: str) -> None:
    for seed in (0, 42, 123):
        path = marker_path(seed)
        if not path.exists():
            raise SystemExit(f"missing equivalence marker {path}")
        marker = json.loads(path.read_text())
        if not marker.get("passed") or marker.get("protocol_sha256") != protocol_sha or marker.get("seed") != seed:
            raise SystemExit(f"equivalence marker is not a pass for this protocol: {path}")


def state_file(seed: int, side: str, episode_id: int, tag: str) -> Path:
    return OUT / "states" / f"seed{seed}" / f"L{side}" / f"ep{episode_id:04d}_{tag}.npz"


def save_capture(path: Path, state: dict, step: int, distance: float) -> str:
    payload = {key: np.asarray(value) for key, value in state.items()}
    payload["capture_step"] = np.array([step], dtype=np.int32)
    payload["capture_distance"] = np.array([distance], dtype=np.float64)
    save_npz(path, payload)
    return path.relative_to(OUT).as_posix()


def row_from_trace(
    trace: dict,
    *,
    seed: int,
    episode_id: int,
    side: str,
    reset_seed: int,
    protocol_sha: str,
    checkpoint_sha: str,
    persist_states: bool = True,
) -> dict:
    reading = readings_from_trace(trace)
    measured = trace["measured"]
    achieved = trace["achieved0"]
    desired = trace["desired0"]
    initial_distance = float(np.linalg.norm(achieved - desired))
    inertia = measured["inertia"]
    inv = measured["invweight"]
    state_h50 = ""
    state_h100 = ""
    if reading["success_h50"] == 0:
        if trace["state50"] is None:
            raise SystemExit("H50 failure has no step-50 capture")
        target = state_file(seed, side, episode_id, "h50")
        state_h50 = target.relative_to(OUT).as_posix()
        if persist_states:
            save_capture(target, trace["state50"], 50, reading["distance_h50"])
    if reading["success_h100"] == 0:
        target = state_file(seed, side, episode_id, "h100")
        state_h100 = target.relative_to(OUT).as_posix()
        if persist_states:
            save_capture(target, trace["state_final"], 100, reading["distance_h100"])
    if measured["ctrl_maxabs_vs_neutral"] != 0.0 or measured["time"] != 0.0:
        raise SystemExit("reset control or time check failed")
    obj_low = trace["obj_low"]
    obj_high = trace["obj_high"]
    goal_low = trace["goal_low"]
    goal_high = trace["goal_high"]
    row = {
        "policy_seed": str(seed),
        "episode_id": str(episode_id),
        "reset_seed": str(reset_seed),
        "L": side,
        "obj_low_x": fmt(obj_low[0]),
        "obj_low_y": fmt(obj_low[1]),
        "obj_low_z": fmt(obj_low[2]),
        "obj_high_x": fmt(obj_high[0]),
        "obj_high_y": fmt(obj_high[1]),
        "obj_high_z": fmt(obj_high[2]),
        "goal_low_x": fmt(goal_low[0]),
        "goal_low_y": fmt(goal_low[1]),
        "goal_low_z": fmt(goal_low[2]),
        "goal_high_x": fmt(goal_high[0]),
        "goal_high_y": fmt(goal_high[1]),
        "goal_high_z": fmt(goal_high[2]),
        "initial_object_x": fmt(achieved[0]),
        "initial_object_y": fmt(achieved[1]),
        "initial_object_z": fmt(achieved[2]),
        "target_x": fmt(desired[0]),
        "target_y": fmt(desired[1]),
        "target_z": fmt(desired[2]),
        "initial_distance": fmt(initial_distance),
        "initial_near": str(int(initial_distance < 0.05)),
        "delay_steps": "0",
        "latency_s": fmt(0.0),
        "mass": fmt(PHYSICS["mass"]),
        "friction": fmt(PHYSICS["friction"]),
        "damping_scale": fmt(PHYSICS["damping_scale"]),
        "gain_scale": fmt(PHYSICS["gain_scale"]),
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
        "control_restored": "1",
        "goal_unchanged": "1",
        "success_h50": str(reading["success_h50"]),
        "steps_h50": str(reading["steps_h50"]),
        "distance_h50": fmt(reading["distance_h50"]),
        "success_h100": str(reading["success_h100"]),
        "steps_h100": str(reading["steps_h100"]),
        "distance_h100": fmt(reading["distance_h100"]),
        "first_success_step": str(reading["first_success_step"]),
        "state_h50": state_h50,
        "state_h100": state_h100,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
        "horizon_source": HORIZON_SOURCE,
        "model_check_ok": "1",
        "writer": "C",
    }
    missing = [key for key in FIELDS if key not in row]
    if missing:
        raise SystemExit(f"row is missing {missing}")
    return row


def read_formal(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"header mismatch in {path}")
        rows = list(reader)
    expected = schedule()
    for index, row in enumerate(rows):
        if (row["L"], int(row["episode_id"])) != expected[index]:
            raise SystemExit(f"{path.name} row {index} is not schedule position {expected[index]}")
        if int(row["reset_seed"]) != reset_seed_for(float(row["L"]), int(row["episode_id"])):
            raise SystemExit(f"reset seed mismatch at row {index}")
        if row["protocol_id"] != PROTOCOL_ID:
            raise SystemExit("protocol id mismatch in an existing formal row")
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
        if stored.get(key) != fresh.get(key):
            return f"{key}: stored {stored.get(key)} fresh {fresh.get(key)}"
    return ""


def acquire_pid(seed: int, mode: str) -> Path:
    path = OUT / "logs" / f"{mode}_seed{seed}.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = int(path.read_text().strip())
        except ValueError:
            old = -1
        if old > 0:
            try:
                os.kill(old, 0)
            except OSError:
                pass
            else:
                raise SystemExit(f"{mode} seed {seed} is already running as pid {old}")
    path.write_text(str(os.getpid()))
    return path


def formal(model, pristine, ids, seed: int, protocol_sha: str, checkpoint_sha: str) -> None:
    require_equivalence(protocol_sha)
    for row in cases_for_seed(seed):
        fresh = trace_arrays(run_case(model, pristine, ids, row, 100))
        stored = load_trace(trace_path(seed, row["L"], int(row["episode_id"]), "h100"))
        problems = compare_traces(stored, fresh, prefix=False)
        if problems:
            raise SystemExit(
                f"cross-process repeat failed L={row['L']} episode {row['episode_id']}: {problems[0]}"
            )
        print(f"[formal] seed {seed} reproduced equivalence L={row['L']} ep {row['episode_id']}", flush=True)
    path = OUT / f"spatial_seed{seed}.csv"
    existing = read_formal(path)
    if existing:
        if existing[0]["protocol_sha256"] != protocol_sha or existing[0]["checkpoint_sha256"] != checkpoint_sha:
            raise SystemExit("existing formal rows belong to another protocol or checkpoint")
        first = existing[0]
        fresh = rollout(
            model,
            pristine,
            ids,
            side=float(first["L"]),
            episode_id=int(first["episode_id"]),
            horizon=100,
            reset_seed=int(first["reset_seed"]),
        )
        rebuilt = row_from_trace(
            fresh,
            seed=seed,
            episode_id=int(first["episode_id"]),
            side=first["L"],
            reset_seed=int(first["reset_seed"]),
            protocol_sha=protocol_sha,
            checkpoint_sha=checkpoint_sha,
            persist_states=False,
        )
        mismatch = rows_equal(first, rebuilt)
        if mismatch:
            raise SystemExit(f"resume spot check failed: {mismatch}")
        print(f"[formal] seed {seed} resume row L={first['L']} ep {first['episode_id']} matched", flush=True)
    done = {(row["L"], int(row["episode_id"])) for row in existing}
    started = time.time()
    completed = len(existing)
    for side, episode_id in schedule():
        if (side, episode_id) in done:
            continue
        trace = rollout(
            model,
            pristine,
            ids,
            side=float(side),
            episode_id=episode_id,
            horizon=100,
            reset_seed=reset_seed_for(float(side), episode_id),
        )
        row = row_from_trace(
            trace,
            seed=seed,
            episode_id=episode_id,
            side=side,
            reset_seed=reset_seed_for(float(side), episode_id),
            protocol_sha=protocol_sha,
            checkpoint_sha=checkpoint_sha,
        )
        append_row(path, row)
        completed += 1
        if episode_id % 25 == 0 or episode_id == 599:
            elapsed = time.time() - started
            rate = (completed - len(existing)) / elapsed if elapsed else 0.0
            print(
                f"[seed {seed}] L={side} episode {episode_id} done {completed}/4800 rate {rate:.2f}/s",
                flush=True,
            )


def prepare_model(seed: int, versions: dict):
    import torch

    torch.set_num_threads(1)
    import_paths()
    import phaseB_diagnose as pb
    from phaseC_mass_implementation import object_ids, snapshot

    manifest = ensure_cache()
    install_binary_loader()
    protocol = build_protocol(manifest, versions)
    lock_protocol(protocol)
    template, template_base = make_env(100, 0.30)
    pristine = snapshot(template_base)
    ids = object_ids(template_base)
    assert_spatial_ranges(template_base, 0.30)
    model = pb.load_policy(seed, template)
    print(
        f"[seed {seed}] threads {torch.get_num_threads()} interop {torch.get_num_interop_threads()} "
        f"protocol {protocol['protocol_sha256'][:12]}",
        flush=True,
    )
    return model, pristine, ids, protocol, template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-check", "freeze", "equivalence", "formal"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "self-check":
        self_check()
        return
    if args.mode == "freeze":
        freeze_cases()
        return
    if args.seed not in (0, 42, 123):
        raise SystemExit("seed must be one of 0, 42, 123")
    versions = require_runtime()
    pid_path = acquire_pid(args.seed, args.mode)
    template = None
    try:
        model, pristine, ids, protocol, template = prepare_model(args.seed, versions)
        protocol_sha = protocol["protocol_sha256"]
        checkpoint_sha = protocol["files"][f"checkpoint_seed{args.seed}"]
        if args.mode == "equivalence":
            equivalence(model, pristine, ids, args.seed, protocol_sha)
        else:
            formal(model, pristine, ids, args.seed, protocol_sha, checkpoint_sha)
    finally:
        if template is not None:
            template.close()
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
