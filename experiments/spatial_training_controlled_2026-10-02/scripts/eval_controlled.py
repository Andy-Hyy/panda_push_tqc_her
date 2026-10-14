"""Shared-distribution evaluation for the six controlled models.

Validation uses the 7_000_000 seed series and the six predetermined checkpoints.
The final test uses the 6_000_000 series and only the predetermined-budget
checkpoint. Standard-policy checkpoints are not loaded.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import controlled_common as common

REPRESENTATIVE_RULE = (
    "On the final test only, save the full trajectory of episode_id 0 for every "
    "model and every L. The index is fixed before any outcome is known."
)
FAILURE_RULE = (
    "On the final test only, walk episode_id from 0 upward. For each model and "
    "each L, save the failure state of the first 10 episodes whose success is 0. "
    "Keep fewer when fewer than 10 fail. Do not reorder by distance."
)

EVAL_FIELDS = (
    "phase",
    "method",
    "train_seed",
    "checkpoint_steps",
    "checkpoint_sha256",
    "L",
    "episode_id",
    "reset_seed",
    "goal_low_x",
    "goal_low_y",
    "goal_low_z",
    "goal_high_x",
    "goal_high_y",
    "goal_high_z",
    "obj_low_x",
    "obj_low_y",
    "obj_low_z",
    "obj_high_x",
    "obj_high_y",
    "obj_high_z",
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "target_x",
    "target_y",
    "target_z",
    "initial_distance",
    "initial_near",
    "mass_read",
    "friction_read",
    "damping_scale_read",
    "gain_scale_read",
    "time_after_reset",
    "warmstart_maxabs",
    "ctrl_maxabs_vs_neutral",
    "goal_unchanged",
    "success",
    "steps_used",
    "final_distance",
    "termination_reason",
    "protocol_id",
    "protocol_sha256",
)


def save_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + "_partial.npz")
    if partial.exists():
        partial.unlink()
    np.savez_compressed(partial, **payload)
    partial.replace(path)


def load_model(path: Path):
    import torch
    from sb3_contrib import TQC

    from train_controlled import Float32Wrapper

    torch.set_num_threads(1)
    # The saved policy was trained on float32 observations. Loading checks that
    # space. Rollouts below still use the float64 evaluation environment.
    holder, _base, _pristine, _ids = common.make_push_env(0.30)
    holder = Float32Wrapper(holder)
    model = TQC.load(str(path), env=holder, device="cpu")
    return model, holder


def rollout_episode(model, env, base, pristine, ids, *, side, episode_id, reset_seed, meta: dict) -> tuple[dict, dict]:
    common.import_helpers()
    import phaseB_diagnose as phase_b
    import phaseE_formal as phase_e

    common.assert_side(base, float(side))
    obs, _info = env.reset(seed=int(reset_seed))
    originals = {"obj_body": ids["body"], "obj_geom": ids["geom"]}
    after_reset = phase_b.capture_state(base, originals)
    pose = common.pose_after_reset(base, obs)
    if not common.in_box(pose["goal"], pose["goal_low"], pose["goal_high"]):
        raise SystemExit("eval goal is outside the requested box")
    if not common.in_box(pose["object"], pose["obj_low"], pose["obj_high"]):
        raise SystemExit("eval object is outside the 0.30 m box")
    measured = phase_e.measure(
        base,
        {
            "state_after_reset_full": after_reset,
            "achieved_after_reset": pose["object"],
            "desired_after_reset": pose["goal"],
        },
        pristine,
        ids,
        common.PHYSICS,
    )
    desired0 = pose["goal"].copy()
    delay = phase_b.Delay(0, int(np.asarray(env.action_space.shape)[0]))
    object_pos = [pose["object"].copy()]
    qpos = [np.asarray(after_reset["qpos"], dtype=np.float64).copy()]
    qvel = [np.asarray(after_reset["qvel"], dtype=np.float64).copy()]
    ctrl = [np.asarray(after_reset["ctrl"], dtype=np.float64).copy()]
    actions = []
    distances = []
    steps = 0
    final_distance = pose["distance"]
    while steps < common.HORIZON:
        seen_desired = np.asarray(obs["desired_goal"], dtype=np.float64)
        if not np.allclose(seen_desired, desired0, rtol=0.0, atol=1e-8):
            raise SystemExit("goal changed before the action")
        action, _state = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        executed = np.asarray(delay.apply(action), dtype=np.float32).reshape(-1).copy()
        if not np.array_equal(executed, action):
            raise SystemExit("zero delay changed the action")
        actions.append(executed)
        obs, _reward, terminated, truncated, info = env.step(executed)
        steps += 1
        if not np.allclose(np.asarray(obs["desired_goal"], dtype=np.float64), desired0, rtol=0.0, atol=1e-8):
            raise SystemExit("goal changed during the episode")
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
        final_distance = float(np.linalg.norm(achieved - desired0))
        reported = int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)
        if reported != int(final_distance < 0.05):
            raise SystemExit(f"success flag {reported} disagrees with distance {final_distance}")
        state = phase_b.capture_state(base, originals)
        object_pos.append(achieved)
        qpos.append(np.asarray(state["qpos"], dtype=np.float64).copy())
        qvel.append(np.asarray(state["qvel"], dtype=np.float64).copy())
        ctrl.append(np.asarray(state["ctrl"], dtype=np.float64).copy())
        distances.append(final_distance)
        if final_distance < 0.05:
            break
        if steps == common.HORIZON:
            if bool(terminated) and not bool(truncated):
                raise SystemExit("horizon step ended terminated without being inside 0.05 m")
            break
        if bool(terminated) or bool(truncated):
            raise SystemExit(f"episode stopped at step {steps} before the horizon")
    if steps < 1:
        raise SystemExit("no environment step was taken")
    success = int(final_distance < 0.05)
    reason = "success" if success else "time_limit"
    if not success and steps != common.HORIZON:
        raise SystemExit(f"failure stopped at step {steps}")
    low = pose["goal_low"]
    high = pose["goal_high"]
    obj_low = pose["obj_low"]
    obj_high = pose["obj_high"]
    obj = pose["object"]
    row = {
        "phase": meta["phase"],
        "method": meta["method"],
        "train_seed": int(meta["train_seed"]),
        "checkpoint_steps": int(meta["checkpoint_steps"]),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "L": float(side),
        "episode_id": int(episode_id),
        "reset_seed": int(reset_seed),
        "goal_low_x": float(low[0]),
        "goal_low_y": float(low[1]),
        "goal_low_z": float(low[2]),
        "goal_high_x": float(high[0]),
        "goal_high_y": float(high[1]),
        "goal_high_z": float(high[2]),
        "obj_low_x": float(obj_low[0]),
        "obj_low_y": float(obj_low[1]),
        "obj_low_z": float(obj_low[2]),
        "obj_high_x": float(obj_high[0]),
        "obj_high_y": float(obj_high[1]),
        "obj_high_z": float(obj_high[2]),
        "initial_object_x": float(obj[0]),
        "initial_object_y": float(obj[1]),
        "initial_object_z": float(obj[2]),
        "target_x": float(desired0[0]),
        "target_y": float(desired0[1]),
        "target_z": float(desired0[2]),
        "initial_distance": float(pose["distance"]),
        "initial_near": int(pose["distance"] < 0.05),
        "mass_read": float(measured["mass_read"]),
        "friction_read": float(measured["friction_read"]),
        "damping_scale_read": float(measured["damping_scale_read"]),
        "gain_scale_read": float(measured["gain_scale_read"]),
        "time_after_reset": float(measured["time"]),
        "warmstart_maxabs": float(measured["warmstart_maxabs"]),
        "ctrl_maxabs_vs_neutral": float(measured["ctrl_maxabs_vs_neutral"]),
        "goal_unchanged": 1,
        "success": success,
        "steps_used": int(steps),
        "final_distance": float(final_distance),
        "termination_reason": reason,
        "protocol_id": meta["protocol_id"],
        "protocol_sha256": meta["protocol_sha256"],
    }
    arrays = {
        "episode_id": np.array([episode_id], dtype=np.int32),
        "reset_seed": np.array([reset_seed], dtype=np.int64),
        "L": np.array([float(side)], dtype=np.float64),
        "success": np.array([success], dtype=np.int32),
        "steps": np.array([steps], dtype=np.int32),
        "distance": np.asarray(distances, dtype=np.float64),
        "object_pos": np.stack(object_pos),
        "goal": desired0,
        "executed_action": np.stack(actions),
        "qpos": np.stack(qpos),
        "qvel": np.stack(qvel),
        "ctrl": np.stack(ctrl),
    }
    return row, arrays


def probe_checkpoint(path: Path, n: int = 2) -> list[dict]:
    model, holder = load_model(path)
    env, base, pristine, ids = common.make_push_env(0.30)
    rows = []
    try:
        digest = common.sha256_file(path)
        for episode_id in range(n):
            row, _arrays = rollout_episode(
                model,
                env,
                base,
                pristine,
                ids,
                side=0.30,
                episode_id=episode_id,
                reset_seed=8_200_000 + episode_id,
                meta={
                    "phase": "smoke_probe",
                    "method": "probe",
                    "train_seed": -1,
                    "checkpoint_steps": common.SMOKE_EXPECTED_STEPS,
                    "checkpoint_sha256": digest,
                    "protocol_id": "preflight",
                    "protocol_sha256": "",
                },
            )
            if int(row["steps_used"]) < 1 or not math.isfinite(float(row["final_distance"])):
                raise SystemExit("probe rollout was not a finished finite episode")
            rows.append(row)
    finally:
        env.close()
        holder.close()
    return rows


def record_path(phase: str, method: str, train_seed: int) -> Path:
    return common.OUT / "records" / f"{phase}_{method}_seed{train_seed}.csv"


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoints_for(method: str, train_seed: int, phase: str) -> list[dict]:
    manifest_path = common.OUT / "runs" / f"{method}_seed{train_seed}" / "checkpoints_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing checkpoint manifest for {method} seed {train_seed}")
    import json

    manifest = json.loads(manifest_path.read_text())
    by_step = {int(item["steps"]): item for item in manifest["checkpoints"]}
    wanted = common.CHECKPOINT_STEPS if phase == "validation" else (common.CHECKPOINT_STEPS[-1],)
    chosen = []
    for step in wanted:
        if step not in by_step:
            raise SystemExit(f"{method} seed {train_seed} has no checkpoint at {step}")
        item = by_step[step]
        path = common.OUT / item["path"]
        digest = common.sha256_file(path)
        if digest != item["sha256"]:
            raise SystemExit(f"checkpoint hash changed for {path}")
        chosen.append({"steps": step, "sha256": digest, "path": path})
    if phase == "final" and chosen[0]["steps"] != common.CHECKPOINT_STEPS[-1]:
        raise SystemExit("final test refused a checkpoint other than the predetermined budget")
    return chosen


def run_phase(method: str, train_seed: int, phase: str) -> dict:
    if phase not in ("validation", "final"):
        raise SystemExit(phase)
    protocol = common.assert_protocol_code()
    n_episodes = 100 if phase == "validation" else 600
    seed_of = common.validation_reset_seed if phase == "validation" else common.final_reset_seed
    destination = record_path(phase, method, train_seed)
    existing = read_rows(destination)

    def scene_key(steps: int, side: float, episode_id: int) -> tuple[int, int, int]:
        return (int(steps), int(round(float(side) * 100)), int(episode_id))

    done = {
        scene_key(int(row["checkpoint_steps"]), float(row["L"]), int(row["episode_id"])): row
        for row in existing
    }
    written = 0
    for checkpoint in checkpoints_for(method, train_seed, phase):
        model, holder = load_model(checkpoint["path"])
        env, base, pristine, ids = common.make_push_env(0.30)
        try:
            meta = {
                "phase": phase,
                "method": method,
                "train_seed": int(train_seed),
                "checkpoint_steps": checkpoint["steps"],
                "checkpoint_sha256": checkpoint["sha256"],
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol["protocol_sha256"],
            }
            for side in common.SIDES:
                seen_failures = 0
                for episode_id in range(n_episodes):
                    key = scene_key(checkpoint["steps"], float(side), episode_id)
                    prior = done.get(key)
                    if prior is not None:
                        if int(prior["success"]) == 0:
                            seen_failures += 1
                        continue
                    row, arrays = rollout_episode(
                        model,
                        env,
                        base,
                        pristine,
                        ids,
                        side=float(side),
                        episode_id=episode_id,
                        reset_seed=seed_of(float(side), episode_id),
                        meta=meta,
                    )
                    if phase == "final" and episode_id == 0:
                        trace_path = (
                            common.OUT / "traces" / "final" / f"{method}_seed{train_seed}" / f"L{side:.2f}" / "episode0.npz"
                        )
                        save_npz(trace_path, arrays)
                    if phase == "final" and int(row["success"]) == 0 and seen_failures < 10:
                        failure_path = (
                            common.OUT
                            / "failures"
                            / "final"
                            / f"{method}_seed{train_seed}"
                            / f"L{side:.2f}"
                            / f"ep{episode_id:03d}.npz"
                        )
                        save_npz(
                            failure_path,
                            {
                                "episode_id": arrays["episode_id"],
                                "reset_seed": arrays["reset_seed"],
                                "L": arrays["L"],
                                "steps": arrays["steps"],
                                "final_distance": np.array([row["final_distance"]], dtype=np.float64),
                                "object_final": arrays["object_pos"][-1],
                                "goal": arrays["goal"],
                                "qpos_final": arrays["qpos"][-1],
                                "qvel_final": arrays["qvel"][-1],
                                "ctrl_final": arrays["ctrl"][-1],
                            },
                        )
                    if int(row["success"]) == 0:
                        seen_failures += 1
                    ordered = {field: row[field] for field in EVAL_FIELDS}
                    common.append_csv(destination, ordered)
                    done[key] = ordered
                    written += 1
        finally:
            env.close()
            holder.close()
    return {"phase": phase, "method": method, "train_seed": int(train_seed), "written": written, "rows": len(done)}


def _pose(row: dict) -> np.ndarray:
    return np.array(
        [
            float(row["initial_object_x"]),
            float(row["initial_object_y"]),
            float(row["initial_object_z"]),
            float(row["target_x"]),
            float(row["target_y"]),
            float(row["target_z"]),
        ],
        dtype=np.float64,
    )


def audit() -> dict:
    problems = []
    groups: dict[tuple, list[dict]] = {}
    counts = {"validation": 0, "final": 0}
    for phase, per_model in (("validation", 6 * 8 * 100), ("final", 8 * 600)):
        for method in common.METHODS:
            for train_seed in common.TRAIN_SEEDS:
                rows = read_rows(record_path(phase, method, train_seed))
                counts[phase] += len(rows)
                if len(rows) != per_model:
                    problems.append(f"{phase} {method} seed {train_seed} has {len(rows)} rows, expected {per_model}")
                seen = set()
                for row in rows:
                    if int(row["steps_used"]) < 1:
                        problems.append(f"{phase} {method} seed {train_seed} episode {row['episode_id']} took no step")
                    if row["termination_reason"] not in ("success", "time_limit"):
                        problems.append(f"unexpected termination {row['termination_reason']}")
                    if int(row["success"]) not in (0, 1):
                        problems.append("success outside 0/1")
                    key = (int(row["checkpoint_steps"]), row["L"], int(row["episode_id"]))
                    if key in seen:
                        problems.append(f"duplicate key {phase} {method} {train_seed} {key}")
                    seen.add(key)
                    scene = (phase, int(row["reset_seed"]))
                    groups.setdefault(scene, []).append(row)
    if counts["validation"] != 28800:
        problems.append(f"validation rows {counts['validation']}")
    if counts["final"] != 28800:
        problems.append(f"final rows {counts['final']}")
    for scene, rows in groups.items():
        reference = _pose(rows[0])
        for row in rows[1:]:
            if np.max(np.abs(_pose(row) - reference)) > 1e-8:
                problems.append(f"scene {scene} pose differs between models")
                break
            if int(row["initial_near"]) != int(rows[0]["initial_near"]):
                problems.append(f"scene {scene} initial_near differs")
                break
    for method in common.METHODS:
        for train_seed in common.TRAIN_SEEDS:
            rows = read_rows(record_path("final", method, train_seed))
            by_side: dict[str, list[dict]] = {}
            for row in rows:
                by_side.setdefault(f"{float(row['L']):.2f}", []).append(row)
            for side_text, side_rows in by_side.items():
                ordered = sorted(side_rows, key=lambda item: int(item["episode_id"]))
                expected_failures = [int(row["episode_id"]) for row in ordered if int(row["success"]) == 0][:10]
                folder = common.OUT / "failures" / "final" / f"{method}_seed{train_seed}" / f"L{side_text}"
                found = sorted(int(path.stem[2:]) for path in folder.glob("ep*.npz")) if folder.exists() else []
                if found != expected_failures:
                    problems.append(f"failure files {method} seed {train_seed} L={side_text} are {found}, expected {expected_failures}")
                trace = common.OUT / "traces" / "final" / f"{method}_seed{train_seed}" / f"L{side_text}" / "episode0.npz"
                if not trace.exists():
                    problems.append(f"missing representative trace {trace}")
    if problems:
        raise SystemExit("audit failed: " + "; ".join(problems[:12]))
    return {"validation_rows": counts["validation"], "final_rows": counts["final"], "scenes": len(groups), "problems": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("validation", "final", "audit"))
    parser.add_argument("--method", choices=common.METHODS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "audit":
        print(common.canonical(audit()))
        return
    if args.method is None or args.seed is None:
        raise SystemExit("method and seed are required")
    print(common.canonical(run_phase(args.method, args.seed, args.phase)))


if __name__ == "__main__":
    main()
