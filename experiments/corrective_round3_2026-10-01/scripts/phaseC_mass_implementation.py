"""Phase C mass-implementation check.

Writers, fixed before any Phase C outcome:

A. Set body_mass only. Do not scale inertia. Do not call mj_setConst.
B. Set body_mass, leave the compiled inertia, call mj_setConst.
C. Set body_mass and replace body_inertia with the pristine inertia times m/m0,
   then call mj_setConst. The multiplier is always the pristine 1 kg inertia,
   never the value left by a previous episode.
D. Rewrite obj_geom mass in a temporary copy of the XML and compile that.
   The original XML files are only read.

C is the same-shape uniform-density implementation. D is its compile reference.
This script does not retrain, does not edit old CSVs, and does not edit XML assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phaseB_diagnose as pb  # noqa: E402

OUT = pb.OUT
ROOT = pb.ROOT
ASSETS = ROOT / "panda_mujoco_gym" / "assets"
PUSH_XML = ASSETS / "push.xml"
MOCAP_XML = ASSETS / "panda_mocap.xml"
SCENES = OUT / "phaseC_scenes.csv"
MASSES = (0.05, 0.1, 0.5, 1.0)
SHORT_STEPS = 20
MODEL_ATOL = 1e-10
MODEL_RTOL = 1e-8
MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_invweight0",
    "dof_damping",
    "dof_armature",
    "dof_invweight0",
    "geom_friction",
    "actuator_gainprm",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scenes(role: str) -> list[dict]:
    with SCENES.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == role]
    if role == "static" and len(rows) != 10:
        raise SystemExit(f"expected 10 static scenes, found {len(rows)}")
    if role == "pilot" and len(rows) != 50:
        raise SystemExit(f"expected 50 pilot scenes, found {len(rows)}")
    return rows


def snapshot(base) -> dict:
    model = base.model
    saved = {}
    for name in MODEL_FIELDS:
        if hasattr(model, name):
            saved[name] = np.array(getattr(model, name), copy=True)
    return saved


def restore_model(base, saved: dict) -> None:
    model = base.model
    for name, value in saved.items():
        getattr(model, name)[:] = value


def object_ids(base) -> dict:
    return {
        "body": int(base._model_names.body_name2id["obj"]),
        "geom": int(base._model_names.geom_name2id["obj_geom"]),
    }


def apply_writer(base, saved: dict, ids: dict, mass: float, writer: str, friction: float, damping_scale: float, gain_scale: float) -> None:
    """Restore the pristine model, then apply one writer. Inertia is never multiplied twice."""
    restore_model(base, saved)
    model = base.model
    body = ids["body"]
    geom = ids["geom"]
    mass0 = float(saved["body_mass"][body])
    if not np.isclose(mass0, 1.0):
        raise SystemExit(f"pristine object mass is {mass0}")
    if writer == "A":
        model.body_mass[body] = mass
    elif writer == "B":
        model.body_mass[body] = mass
        base._mujoco.mj_setConst(model, base.data)
    elif writer == "C":
        model.body_mass[body] = mass
        model.body_inertia[body] = np.array(saved["body_inertia"][body], copy=True) * (mass / mass0)
        base._mujoco.mj_setConst(model, base.data)
    else:
        raise SystemExit(writer)
    model.geom_friction[geom, 0] = friction
    model.dof_damping[:] = saved["dof_damping"] * damping_scale
    model.actuator_gainprm[:, 0] = saved["actuator_gainprm"][:, 0] * gain_scale
    # A must keep the compiled invweight. B/C already refreshed constants from
    # the inertia written above; friction, damping and gain are not setConst inputs.
    if writer == "A":
        model.body_invweight0[:] = saved["body_invweight0"]
        if "dof_invweight0" in saved:
            model.dof_invweight0[:] = saved["dof_invweight0"]


def install_episode_reset(base, saved: dict, ids: dict, physics: dict, writer: str) -> None:
    mujoco = base._mujoco

    def _reset_sim() -> bool:
        apply_writer(
            base,
            saved,
            ids,
            physics["mass"],
            writer,
            physics["friction"],
            physics["damping_scale"],
            physics["gain_scale"],
        )
        model = base.model
        data = base.data
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


def field_maxabs(left, right, name: str) -> float:
    if not hasattr(left, name) or not hasattr(right, name):
        return float("nan")
    a = np.asarray(getattr(left, name))
    b = np.asarray(getattr(right, name))
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        return 0.0 if np.array_equal(a, b) else float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def object_record(base, ids: dict) -> dict:
    model = base.model
    body = ids["body"]
    return {
        "body_mass": float(model.body_mass[body]),
        "body_inertia": np.array(model.body_inertia[body], copy=True),
        "body_invweight0": np.array(model.body_invweight0[body], copy=True),
        "body_ipos": np.array(model.body_ipos[body], copy=True),
    }


def make_temp_xml(mass: float, directory: Path) -> Path:
    mesh_source = ASSETS / "meshes"
    if mesh_source.exists():
        shutil.copytree(mesh_source, directory / "meshes")
    push_root = ET.parse(PUSH_XML).getroot()
    mocap_root = ET.parse(MOCAP_XML).getroot()
    geom = push_root.find(".//geom[@name='obj_geom']")
    if geom is None:
        raise SystemExit("obj_geom missing")
    geom.set("mass", f"{mass:.12g}")
    include = push_root.find(".//include")
    if include is None:
        raise SystemExit("push.xml include missing")
    include.set("file", "panda_mocap_temp.xml")
    push_path = directory / "push_temp.xml"
    ET.ElementTree(push_root).write(push_path, encoding="utf-8")
    ET.ElementTree(mocap_root).write(directory / "panda_mocap_temp.xml", encoding="utf-8")
    return push_path


def make_named_env(model_path: str | None = None):
    import gymnasium as gym

    kwargs = {"max_episode_steps": pb.HORIZON, "disable_env_checker": True}
    if model_path is not None:
        kwargs["model_path"] = model_path
    env = None
    last_error = None
    for attempt in range(3):
        try:
            env = gym.make("FrankaPushSparse-v0", **kwargs)
            break
        except ValueError as exc:
            last_error = exc
            if "resource not found" not in str(exc) or attempt == 2:
                raise
            time.sleep(1.0)
    if env is None:
        raise last_error
    node = env
    found = False
    while node is not None:
        if hasattr(node, "_max_episode_steps"):
            node._max_episode_steps = pb.HORIZON
            found = True
        node = getattr(node, "env", None)
    if not found:
        raise SystemExit("TimeLimit missing")
    return env, env.unwrapped


def formatted_vector(value: np.ndarray) -> str:
    return " ".join(f"{float(item):.16g}" for item in np.asarray(value).reshape(-1))


def static_rows() -> list[dict]:
    env, base = make_named_env()
    saved = snapshot(base)
    ids = object_ids(base)
    rows = []
    try:
        for mass in MASSES:
            with tempfile.TemporaryDirectory(prefix="phaseC_xml_") as temp_name:
                xml_path = make_temp_xml(mass, Path(temp_name))
                denv, dbase = make_named_env(str(xml_path))
                try:
                    drec = object_record(dbase, object_ids(dbase))
                    for writer in ("A", "B", "C"):
                        apply_writer(base, saved, ids, mass, writer, 1.0, 1.0, 1.0)
                        rec = object_record(base, ids)
                        diffs = {
                            name: field_maxabs(base.model, dbase.model, name)
                            for name in MODEL_FIELDS
                            if hasattr(base.model, name) and hasattr(dbase.model, name)
                        }
                        worst = max(diffs, key=diffs.get)
                        rows.append(
                            {
                                "check": "static_model",
                                "writer": writer,
                                "mass_requested": f"{mass:.12g}",
                                "scene_index": "",
                                "reset_seed": "",
                                "body_mass": f"{rec['body_mass']:.16g}",
                                "body_inertia": formatted_vector(rec["body_inertia"]),
                                "body_invweight0": formatted_vector(rec["body_invweight0"]),
                                "body_ipos": formatted_vector(rec["body_ipos"]),
                                "reference_mass": f"{drec['body_mass']:.16g}",
                                "reference_inertia": formatted_vector(drec["body_inertia"]),
                                "reference_invweight0": formatted_vector(drec["body_invweight0"]),
                                "reference_ipos": formatted_vector(drec["body_ipos"]),
                                "worst_model_field": worst,
                                "worst_model_maxabs": f"{diffs[worst]:.3e}",
                                "model_fields_within_tol": int(
                                    all(
                                        np.isfinite(value)
                                        and np.allclose(value, 0.0, atol=MODEL_ATOL, rtol=MODEL_RTOL)
                                        for value in diffs.values()
                                    )
                                ),
                            }
                        )
                finally:
                    denv.close()
                    restore_model(base, saved)
    finally:
        env.close()
    return rows


def capture_reference(base) -> dict:
    state = pb.capture_state(base, {"obj_body": object_ids(base)["body"], "obj_geom": object_ids(base)["geom"]})
    state["goal"] = np.array(base.goal, copy=True)
    return state


def impose_state(base, reference: dict) -> None:
    data = base.data
    data.time = float(reference["time"][0])
    data.qpos[:] = reference["qpos"]
    data.qvel[:] = reference["qvel"]
    if reference["act"].size:
        data.act[:] = reference["act"]
    data.ctrl[:] = reference["ctrl"]
    data.qacc[:] = reference["qacc"]
    data.qacc_warmstart[:] = reference["qacc_warmstart"]
    data.mocap_pos[:] = reference["mocap_pos"]
    data.mocap_quat[:] = reference["mocap_quat"]
    base.goal[:] = reference["goal"]
    base._mujoco.mj_forward(base.model, base.data)


def open_loop_action(base) -> np.ndarray:
    """Fixed from the imposed geometry. The same vector is replayed for every step."""
    ee = np.asarray(base.get_ee_position(), dtype=np.float64).copy()
    obj = np.asarray(base._utils.get_site_xpos(base.model, base.data, "obj_site"), dtype=np.float64).copy()
    delta = obj[:2] - ee[:2]
    norm = float(np.linalg.norm(delta))
    action = np.zeros(3, dtype=np.float32)
    if norm < 1e-6:
        action[0] = 1.0
    else:
        # _set_action multiplies xy by 0.05 m, so 1.0 commands a 5 cm end-effector step.
        scale = min(1.0, norm / 0.05)
        action[:2] = (delta / norm * scale).astype(np.float32)
    return action


def rollout_imposed(env, base, reference: dict, action: np.ndarray) -> dict:
    impose_state(base, reference)
    obs0 = pb.capture_obs(base._get_obs())
    qpos = [np.array(base.data.qpos, copy=True)]
    achieved = [obs0["achieved_goal"]]
    for _ in range(SHORT_STEPS):
        obs, _, _, _, _ = env.step(action)
        qpos.append(np.array(base.data.qpos, copy=True))
        achieved.append(np.asarray(obs["achieved_goal"], dtype=np.float64).copy())
        if not np.allclose(np.asarray(obs["desired_goal"]), reference["goal"]):
            raise SystemExit("goal changed during the forced trajectory")
    return {
        "qpos": np.stack(qpos),
        "achieved": np.stack(achieved),
        "obs0": obs0["observation"],
        "action": action.copy(),
    }


def native_reset_record(env, base, reset_seed: int) -> dict:
    obs, _ = env.reset(seed=int(reset_seed))
    return {
        "qpos": np.array(base.data.qpos, copy=True),
        "qvel": np.array(base.data.qvel, copy=True),
        "ctrl": np.array(base.data.ctrl, copy=True),
        "time": float(base.data.time),
        "goal": np.array(base.goal, copy=True),
        "achieved": np.asarray(obs["achieved_goal"], dtype=np.float64).copy(),
        "initial_qvel_attr": np.array(base.initial_qvel, copy=True),
    }


def trajectory_check() -> tuple[list[dict], bool]:
    scenes = read_scenes("static")
    rows = []
    passed = True
    ref_dir = OUT / "mass_reference_traces"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        reset_seed = int(scene["reset_seed"])
        ref_env, ref_base = make_named_env()
        ref_saved = snapshot(ref_base)
        pb.install_clean_reset(ref_base, {"body_mass": ref_saved["body_mass"], "body_inertia": ref_saved["body_inertia"], "dof_damping": ref_saved["dof_damping"], "geom_friction": ref_saved["geom_friction"], "actuator_gainprm": ref_saved["actuator_gainprm"]})
        ref_env.reset(seed=reset_seed)
        reference = capture_reference(ref_base)
        action = open_loop_action(ref_base)
        ref_env.close()
        for mass in MASSES:
            cenv, cbase = make_named_env()
            csaved = snapshot(cbase)
            cids = object_ids(cbase)
            install_episode_reset(
                cbase,
                csaved,
                cids,
                {"mass": mass, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0},
                "C",
            )
            native_c = native_reset_record(cenv, cbase, reset_seed)
            with tempfile.TemporaryDirectory(prefix="phaseC_xml_") as temp_name:
                xml_path = make_temp_xml(mass, Path(temp_name))
                denv, dbase = make_named_env(str(xml_path))
                native_d = native_reset_record(denv, dbase, reset_seed)
                # Re-apply C on the same process state, then overwrite data with the reference.
                apply_writer(cbase, csaved, cids, mass, "C", 1.0, 1.0, 1.0)
                forced_c = rollout_imposed(cenv, cbase, reference, action)
                forced_d = rollout_imposed(denv, dbase, reference, action)
                qpos_diff = np.max(np.abs(forced_c["qpos"] - forced_d["qpos"]))
                obs_diff = float(np.max(np.abs(forced_c["obs0"] - forced_d["obs0"])))
                achieved_diff = float(np.max(np.abs(forced_c["achieved"] - forced_d["achieved"])))
                model_diffs = {
                    name: field_maxabs(cbase.model, dbase.model, name)
                    for name in ("body_mass", "body_inertia", "body_invweight0", "body_ipos", "dof_invweight0")
                    if hasattr(cbase.model, name)
                }
                within = bool(
                    qpos_diff <= MODEL_ATOL
                    or np.allclose(forced_c["qpos"], forced_d["qpos"], atol=MODEL_ATOL, rtol=MODEL_RTOL)
                )
                within = within and np.allclose(forced_c["obs0"], forced_d["obs0"], atol=MODEL_ATOL, rtol=MODEL_RTOL)
                within = within and all(np.allclose(value, 0.0, atol=MODEL_ATOL, rtol=MODEL_RTOL) for value in model_diffs.values())
                passed = passed and within
                native_qvel = float(np.max(np.abs(native_c["qvel"] - native_d["qvel"])))
                native_qpos = float(np.max(np.abs(native_c["qpos"] - native_d["qpos"])))
                rows.append(
                    {
                        "check": "forced_short_trajectory",
                        "writer": "C_vs_D",
                        "mass_requested": f"{mass:.12g}",
                        "scene_index": scene["index"],
                        "reset_seed": str(reset_seed),
                        "qpos_maxabs": f"{float(qpos_diff):.3e}",
                        "obs0_maxabs": f"{obs_diff:.3e}",
                        "achieved_maxabs": f"{achieved_diff:.3e}",
                        "native_qpos_maxabs": f"{native_qpos:.3e}",
                        "native_qvel_maxabs": f"{native_qvel:.3e}",
                        "native_time_c": f"{native_c['time']:.3e}",
                        "native_time_d": f"{native_d['time']:.3e}",
                        "model_mass_maxabs": f"{model_diffs.get('body_mass', float('nan')):.3e}",
                        "model_inertia_maxabs": f"{model_diffs.get('body_inertia', float('nan')):.3e}",
                        "model_invweight_maxabs": f"{model_diffs.get('body_invweight0', float('nan')):.3e}",
                        "within_tol": int(within),
                        "action": formatted_vector(action),
                    }
                )
                np.savez_compressed(
                    ref_dir / f"mass{mass:.2f}_seed{reset_seed}.npz",
                    qpos_c=forced_c["qpos"],
                    qpos_d=forced_d["qpos"],
                    achieved_c=forced_c["achieved"],
                    achieved_d=forced_d["achieved"],
                    obs0_c=forced_c["obs0"],
                    obs0_d=forced_d["obs0"],
                    action=action,
                    native_qpos_c=native_c["qpos"],
                    native_qpos_d=native_d["qpos"],
                    native_qvel_c=native_c["qvel"],
                    native_qvel_d=native_d["qvel"],
                    goal=reference["goal"],
                )
                denv.close()
            cenv.close()
            print(f"[trajectory] mass {mass} seed {reset_seed} within_tol={int(within)} qpos_maxabs={qpos_diff:.3e}", flush=True)
    return rows, passed


def run_pilot(seed: int) -> None:
    import torch

    torch.set_num_threads(1)
    scenes = read_scenes("pilot")
    dest = OUT / f"mass_pilot_seed{seed}.csv"
    done = set()
    if dest.exists():
        with dest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((row["writer"], row["condition"], row["episode_id"]))
    fields = [
        "check",
        "policy_seed",
        "writer",
        "condition",
        "episode_id",
        "reset_seed",
        "mass",
        "friction",
        "damping_scale",
        "gain_scale",
        "success",
        "steps",
        "final_distance",
        "body_mass",
        "body_inertia",
        "body_invweight0",
    ]
    env, base = make_named_env()
    saved = snapshot(base)
    ids = object_ids(base)
    model = pb.load_policy(seed, env)
    new_file = not dest.exists()
    with dest.open("a", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer_csv.writeheader()
        for scene in scenes:
            physics_rows = {
                "mass_only": {
                    "mass": float(scene["mass"]),
                    "friction": 1.0,
                    "damping_scale": 1.0,
                    "gain_scale": 1.0,
                },
                "joint_physical": {
                    "mass": float(scene["mass"]),
                    "friction": float(scene["friction"]),
                    "damping_scale": float(scene["damping_scale"]),
                    "gain_scale": float(scene["gain_scale"]),
                },
            }
            for condition, physics in physics_rows.items():
                for writer_name in ("A", "B", "C"):
                    key = (writer_name, condition, str(scene["episode_id"]))
                    if key in done:
                        continue
                    run_env, run_base = make_named_env()
                    install_episode_reset(run_base, saved, ids, physics, writer_name)
                    bundle = pb.run_episode(
                        run_env,
                        run_base,
                        model,
                        {"obj_body": ids["body"], "obj_geom": ids["geom"]},
                        int(scene["reset_seed"]),
                        0,
                        physics,
                        pb.PROTOCOL_CLEAN,
                    )
                    # run_episode with PROTOCOL_CLEAN does not call apply_physics.
                    # The reset installed above is the only parameter write.
                    rec = object_record(run_base, ids)
                    expected_inertia = np.array(saved["body_inertia"][ids["body"]], copy=True)
                    if writer_name == "C":
                        expected_inertia = expected_inertia * (physics["mass"] / float(saved["body_mass"][ids["body"]]))
                    if abs(rec["body_mass"] - physics["mass"]) > 1e-12:
                        raise SystemExit(f"mass drifted to {rec['body_mass']}")
                    if float(np.max(np.abs(rec["body_inertia"] - expected_inertia))) > 1e-12:
                        raise SystemExit(f"inertia drifted for writer {writer_name}")
                    writer_csv.writerow(
                        {
                            "check": "pilot",
                            "policy_seed": seed,
                            "writer": writer_name,
                            "condition": condition,
                            "episode_id": scene["episode_id"],
                            "reset_seed": scene["reset_seed"],
                            "mass": f"{physics['mass']:.16g}",
                            "friction": f"{physics['friction']:.16g}",
                            "damping_scale": f"{physics['damping_scale']:.16g}",
                            "gain_scale": f"{physics['gain_scale']:.16g}",
                            "success": int(bundle["success"][0]),
                            "steps": int(bundle["steps"][0]),
                            "final_distance": f"{float(bundle['final_distance'][0]):.16g}",
                            "body_mass": f"{rec['body_mass']:.16g}",
                            "body_inertia": formatted_vector(rec["body_inertia"]),
                            "body_invweight0": formatted_vector(rec["body_invweight0"]),
                        }
                    )
                    handle.flush()
                    run_env.close()
                    print(
                        f"[pilot seed {seed}] {writer_name} {condition} ep {scene['episode_id']} "
                        f"success={int(bundle['success'][0])} steps={int(bundle['steps'][0])}",
                        flush=True,
                    )
    env.close()


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("static", "trajectory", "pilot"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    before = {"push": sha256(PUSH_XML), "mocap": sha256(MOCAP_XML)}
    if args.stage == "static":
        rows = static_rows()
        write_table(OUT / "mass_static_check.csv", rows)
        failed = [row for row in rows if row["writer"] == "C" and row["model_fields_within_tol"] != 1]
        print(f"[static] rows {len(rows)} C_failures {len(failed)}", flush=True)
        if failed:
            raise SystemExit(2)
    elif args.stage == "trajectory":
        rows, passed = trajectory_check()
        write_table(OUT / "mass_trajectory_check.csv", rows)
        print(f"[trajectory] rows {len(rows)} passed={passed}", flush=True)
        if not passed:
            raise SystemExit(2)
    else:
        run_pilot(args.seed)
    after = {"push": sha256(PUSH_XML), "mocap": sha256(MOCAP_XML)}
    if before != after:
        raise SystemExit(f"XML hash changed {before} -> {after}")


if __name__ == "__main__":
    main()
