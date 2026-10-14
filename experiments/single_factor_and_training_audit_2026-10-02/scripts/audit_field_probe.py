"""Current-tree field probe for the 2026-10-02 single-factor audit.

Compiles the XML that is on disk now. It does not modify those files and
does not claim the compiled arrays are the June 2025 binary.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PANDA = Path("/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final")
ASSETS = PANDA / "panda_mujoco_gym" / "assets"
PUSH = ASSETS / "push.xml"
MOCAP = ASSETS / "panda_mocap.xml"
OUT = PANDA / "data_and_analysis" / "single_factor_and_training_audit_2026-10-02" / "probes"


def compile_old(param_name: str, value: float) -> mujoco.MjModel:
    """Replay worker_single_physics.modify_xml_and_copy_assets without writing back."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        shutil.copytree(ASSETS / "meshes", temp / "meshes")
        root_push = ET.parse(PUSH).getroot()
        root_mocap = ET.parse(MOCAP).getroot()
        if param_name == "object_mass":
            root_push.find(".//geom[@name='obj_geom']").set("mass", str(value))
        elif param_name == "object_friction":
            root_push.find(".//geom[@name='obj_geom']").set("friction", str(value))
        elif param_name == "joint_damping_scale":
            for joint in root_mocap.findall(".//joint"):
                if joint.get("damping"):
                    joint.set("damping", str(1.0 * float(value)))
        elif param_name == "actuator_gain_scale":
            original_gains = {
                "actuator1": 4500,
                "actuator2": 4500,
                "actuator3": 3500,
                "actuator4": 3500,
                "actuator5": 2000,
                "actuator6": 2000,
                "actuator7": 2000,
                "r_gripper_finger_joint": 500,
                "l_gripper_finger_joint": 500,
            }
            for actuator in root_mocap.findall(".//actuator/general"):
                name = actuator.get("name")
                if name in original_gains:
                    actuator.set("gainprm", str(original_gains[name] * float(value)))
        include = root_push.find(".//include")
        include.set("file", "panda_mocap_temp.xml")
        ET.ElementTree(root_push).write(temp / "push_temp.xml", encoding="utf-8")
        ET.ElementTree(root_mocap).write(temp / "panda_mocap_temp.xml", encoding="utf-8")
        return mujoco.MjModel.from_xml_path(str(temp / "push_temp.xml"))


def dof_rows(model: mujoco.MjModel) -> list[dict]:
    rows = []
    for dof in range(model.nv):
        joint = int(model.dof_jntid[dof])
        rows.append(
            {
                "dof": dof,
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                "damping": float(model.dof_damping[dof]),
            }
        )
    return rows


def geom_friction(model: mujoco.MjModel) -> list[float]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
    return [float(x) for x in model.geom_friction[gid]]


def body_mass_inertia(model: mujoco.MjModel) -> dict:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    return {
        "body": float(model.body_mass[bid]),
        "inertia": [float(x) for x in model.body_inertia[bid]],
        "invweight0": [float(x) for x in model.body_invweight0[bid]],
    }


def actuators(model: mujoco.MjModel) -> list[dict]:
    rows = []
    for i in range(model.nu):
        rows.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                "gainprm": [float(x) for x in model.actuator_gainprm[i, :3]],
                "biasprm": [float(x) for x in model.actuator_biasprm[i, :3]],
                "biastype": int(model.actuator_biastype[i]),
                "gaintype": int(model.actuator_gaintype[i]),
            }
        )
    return rows


def explicit_damping_joints() -> list[str]:
    root = ET.parse(MOCAP).getroot()
    found = []
    for joint in root.findall(".//joint"):
        if joint.get("damping"):
            parent = ""
            found.append(
                {
                    "name": joint.get("name"),
                    "class": joint.get("class"),
                    "damping": joint.get("damping"),
                    "parent_tag": joint.getparent().tag if hasattr(joint, "getparent") else "",
                }
            )
    # ElementTree has no getparent. Record the path via a second walk.
    found = []
    for element in root.iter():
        if element.tag == "joint" and element.get("damping"):
            found.append(
                {
                    "name": element.get("name"),
                    "class": element.get("class"),
                    "damping": element.get("damping"),
                }
            )
    return found


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    pristine = mujoco.MjModel.from_xml_path(str(PUSH))
    old_mass = compile_old("object_mass", 0.05)
    old_friction = compile_old("object_friction", 0.05)
    old_damping = compile_old("joint_damping_scale", 0.001)
    old_gain0 = compile_old("actuator_gain_scale", 0)
    old_gain2 = compile_old("actuator_gain_scale", 2)
    report = {
        "label": "current_tree_compile",
        "push_xml": str(PUSH),
        "explicit_damping_joints_in_mocap": explicit_damping_joints(),
        "pristine": {
            "dof": dof_rows(pristine),
            "obj_friction": geom_friction(pristine),
            "obj": body_mass_inertia(pristine),
            "actuators": actuators(pristine),
        },
        "old_xml_mass_0.05": body_mass_inertia(old_mass),
        "old_xml_friction_0.05": geom_friction(old_friction),
        "old_xml_damping_0.001": dof_rows(old_damping),
        "old_xml_gain_0": actuators(old_gain0),
        "old_xml_gain_2": actuators(old_gain2),
    }
    # New P0-P3 writer on a fresh compile: whole dof_damping and gain column.
    fresh = mujoco.MjModel.from_xml_path(str(PUSH))
    bid = mujoco.mj_name2id(fresh, mujoco.mjtObj.mjOBJ_BODY, "obj")
    gid = mujoco.mj_name2id(fresh, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
    mass0 = float(fresh.body_mass[bid])
    inertia0 = np.array(fresh.body_inertia[bid], copy=True)
    fresh.body_mass[bid] = 0.05
    fresh.body_inertia[bid] = inertia0 * (0.05 / mass0)
    mujoco.mj_setConst(fresh, mujoco.MjData(fresh))
    report["writer_C_mass_0.05"] = body_mass_inertia(fresh)
    report["writer_C_vs_old_xml_inertia_maxabs"] = float(
        np.max(np.abs(np.array(report["writer_C_mass_0.05"]["inertia"]) - np.array(report["old_xml_mass_0.05"]["inertia"])))
    )
    report["writer_C_vs_old_xml_mass_abs"] = abs(
        report["writer_C_mass_0.05"]["body"] - report["old_xml_mass_0.05"]["body"]
    )
    fresh_f = mujoco.MjModel.from_xml_path(str(PUSH))
    gid = mujoco.mj_name2id(fresh_f, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
    fresh_f.geom_friction[gid, 0] = 0.05
    report["new_friction_0.05"] = geom_friction(fresh_f)
    report["friction_tail_unchanged"] = report["new_friction_0.05"][1:] == report["pristine"]["obj_friction"][1:]
    report["old_friction_equals_new"] = report["old_xml_friction_0.05"] == report["new_friction_0.05"]
    damp0 = np.array(pristine.dof_damping, copy=True)
    report["new_damping_0.001"] = (damp0 * 0.001).tolist()
    report["old_damping_0.001"] = [row["damping"] for row in report["old_xml_damping_0.001"]]
    report["damping_maxabs_old_vs_new"] = float(
        np.max(np.abs(np.array(report["old_damping_0.001"]) - np.array(report["new_damping_0.001"])))
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "field_probe.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "field_probe.json")
    print("explicit damping joints", report["explicit_damping_joints_in_mocap"])
    print("pristine dof", report["pristine"]["dof"])
    print("old damping", report["old_damping_0.001"])
    print("new damping", report["new_damping_0.001"])
    print("damping maxabs", report["damping_maxabs_old_vs_new"])
    print("friction pristine", report["pristine"]["obj_friction"])
    print("friction old", report["old_xml_friction_0.05"], "new", report["new_friction_0.05"])
    print("mass old", report["old_xml_mass_0.05"])
    print("mass writerC", report["writer_C_mass_0.05"])
    print("inertia maxabs", report["writer_C_vs_old_xml_inertia_maxabs"])
    print("gain0", report["old_xml_gain_0"])


if __name__ == "__main__":
    main()
