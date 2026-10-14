"""Phase 0 + 0A. Read the code and the saved Stage 0 CSV. Do not infer ranges.

Writes audit_config_manifest.json and delay_semantics_trace.csv.
Exits 0 only if delay semantics for the corrective wrapper are unambiguous
and match DomainRandomizationWrapper at delay_steps 1, 2 and 3.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import panda_mujoco_gym  # noqa: E402, F401

MODEL_ROOT = ROOT.parent / "panda_project_physics_generalization" / "logs"
TRAIN_ROOT = ROOT.parent / "panda_project" / "logs" / "FrankaPushSparse_v0_TQC"
MODELS = {
    0: (
        "batch512_her4_seed0_2025-06-17_13-50-58",
        MODEL_ROOT / "batch512_her4_seed0_2025-06-17_13-50-58" / "final_model.zip",
        TRAIN_ROOT / "batch512_her4_seed0_2025-06-17_13-50-58" / "final_model.zip",
    ),
    42: (
        "batch512_her4_seed42_2025-06-17_23-01-53",
        MODEL_ROOT / "batch512_her4_seed42_2025-06-17_23-01-53" / "final_model.zip",
        TRAIN_ROOT / "batch512_her4_seed42_2025-06-17_23-01-53" / "final_model.zip",
    ),
    123: (
        "batch512_her4_seed123_2025-06-18_23-21-02",
        MODEL_ROOT / "batch512_her4_seed123_2025-06-18_23-21-02" / "final_model.zip",
        TRAIN_ROOT / "batch512_her4_seed123_2025-06-18_23-21-02" / "final_model.zip",
    ),
}
STAGE0_DIR = ROOT / "data_and_analysis" / "physical_element"
STAGE0_FILES = {
    seed: STAGE0_DIR / f"eval_multi_param_seed{seed}.csv" for seed in (0, 42, 123)
}
SOURCE_FILES = [
    ROOT / "utils" / "domain_rand.py",
    ROOT / "worker_multi_random.py",
    ROOT / "worker_single_physics.py",
    ROOT / "test_push_phycial_element_generalization.py",
    ROOT / "train_push_basic.py",
    ROOT / "panda_mujoco_gym" / "__init__.py",
    ROOT / "panda_mujoco_gym" / "envs" / "push.py",
    ROOT / "panda_mujoco_gym" / "envs" / "panda_env.py",
    ROOT / "panda_mujoco_gym" / "assets" / "push.xml",
    ROOT / "panda_mujoco_gym" / "assets" / "panda_mocap.xml",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_domain_rand():
    path = ROOT / "utils" / "domain_rand.py"
    spec = importlib.util.spec_from_file_location("stage0_domain_rand", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DomainRandomizationWrapper


class CorrectiveDelay:
    """The delay used by the corrective 2x2.

    delay_steps=0 applies the current action.
    delay_steps=d applies the action commanded d steps earlier.
    The buffer starts filled with zeros, so the first d steps apply a zero action.
    """

    def __init__(self, delay_steps: int, action_dim: int):
        self.delay_steps = int(delay_steps)
        zero = np.zeros(action_dim, dtype=np.float64)
        self.pending = deque(zero.copy() for _ in range(self.delay_steps))

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        if self.delay_steps == 0:
            return action
        executed = self.pending.popleft()
        self.pending.append(action)
        return executed


def commands(n: int) -> list[np.ndarray]:
    # Nonzero and inside [-1, 1], so a logged zero is the buffer fill, not A0.
    return [np.array([0.1 * (t + 1), 0.0, 0.0], dtype=np.float64) for t in range(n)]


def trace_corrective(delay: int, n_steps: int) -> list[dict]:
    rows = []
    wrapper = CorrectiveDelay(delay, 3)
    for t, action in enumerate(commands(n_steps)):
        applied = wrapper.apply(action)
        rows.append(row("corrective_deque", delay, t, action, applied))
    return rows


def trace_original(wrapper_cls, delay: int, n_steps: int) -> list[dict]:
    env = wrapper_cls(
        reward_type="sparse",
        goal_range={"low": [-0.15, -0.15, 0.0], "high": [0.15, 0.15, 0.0]},
        mass_range=(0.05, 3.0),
        friction_range=(0.05, 3.0),
        damping_scale_range=(0.001, 0.3),
        gain_scale_range=(0.0, 10.0),
        delay_range=(1, 10),
        max_episode_steps=100,
    )
    try:
        env.reset(seed=0)
        env.last_params["delay_steps"] = delay
        rows = []
        for t, action in enumerate(commands(n_steps)):
            # Mirror DomainRandomizationWrapper.step without advancing a second copy.
            current_delay = env.last_params.get("delay_steps", 1)
            applied = np.asarray(env.action_buffer[-current_delay], dtype=np.float64).copy()
            env.action_buffer.append(action.copy())
            env.action_buffer.pop(0)
            rows.append(row("stage0_DomainRandomizationWrapper", delay, t, action, applied))
            if t + 1 >= n_steps:
                break
        return rows
    finally:
        env.close()


def row(implementation: str, delay: int, t: int, commanded, applied) -> dict:
    return {
        "implementation": implementation,
        "configured_delay_steps": delay,
        "timestep": t,
        "commanded_0": f"{commanded[0]:.1f}",
        "commanded_1": f"{commanded[1]:.1f}",
        "commanded_2": f"{commanded[2]:.1f}",
        "applied_0": f"{applied[0]:.1f}",
        "applied_1": f"{applied[1]:.1f}",
        "applied_2": f"{applied[2]:.1f}",
    }


def same_applied(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ("applied_0", "applied_1", "applied_2"))


def read_stage0(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stage0_summary(rows: list[dict]) -> dict:
    delays = [int(float(r["delay_steps"])) for r in rows]
    successes = [int(float(r["success"])) for r in rows]
    by_delay = {}
    for delay in sorted(set(delays)):
        idx = [i for i, d in enumerate(delays) if d == delay]
        k = sum(successes[i] for i in idx)
        by_delay[str(delay)] = {"n": len(idx), "successes": k, "success_rate": k / len(idx)}
    return {
        "n": len(rows),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(rows),
        "delay_values_present": sorted(set(delays)),
        "by_delay": by_delay,
    }


def model_record(seed: int) -> dict:
    name, used, trained = MODELS[seed]
    with zipfile.ZipFile(used) as archive:
        data = json.loads(archive.read("data"))
        system = archive.read("system_info.txt").decode()
        version = archive.read("_stable_baselines3_version").decode().strip()
    keep = [
        "learning_rate",
        "gamma",
        "batch_size",
        "buffer_size",
        "tau",
        "ent_coef",
        "top_quantiles_to_drop_per_net",
        "policy_kwargs",
        "replay_buffer_kwargs",
        "seed",
        "num_timesteps",
        "learning_starts",
        "gradient_steps",
        "train_freq",
        "target_update_interval",
    ]
    return {
        "training_seed": seed,
        "run_name": name,
        "checkpoint_used_for_2026_09_29_delay_eval": str(used),
        "training_tree_copy": str(trained),
        "sha256": sha256(used),
        "training_copy_same_bytes": trained.exists() and sha256(trained) == sha256(used),
        "sb3_version_in_zip": version,
        "system_info_in_zip": system,
        "hyperparameters": {key: data.get(key) for key in keep},
        "replay_buffer_class_module": data["replay_buffer_class"].get("__module__"),
        "replay_buffer_class_name": "HerReplayBuffer",
    }


def live_defaults() -> dict:
    env = gym.make("FrankaPushSparse-v0", disable_env_checker=True)
    try:
        base = env.unwrapped
        obs, _ = env.reset(seed=0)
        obj_body = int(base._model_names.body_name2id["obj"]) if hasattr(base, "_model_names") else None
        # panda_env stores names on self._model_names
        obj_body = int(base._model_names.body_name2id["obj"])
        obj_geom = int(base._model_names.geom_name2id["obj_geom"])
        joint_names = list(base._model_names.joint_names)
        return {
            "registered_max_episode_steps": 50,
            "live_spec_max_episode_steps": int(env.spec.max_episode_steps) if env.spec is not None else None,
            "n_substeps": int(base.n_substeps),
            "mujoco_timestep_s": float(base.model.opt.timestep),
            "physics_steps_per_policy_decision": 10 * int(base.n_substeps),
            "control_period_s": 10 * int(base.n_substeps) * float(base.model.opt.timestep),
            "distance_threshold_m": float(base.distance_threshold),
            "goal_xy_range_m": float(base.goal_xy_range),
            "obj_xy_range_m": float(base.obj_xy_range),
            "goal_range_low": base.goal_range_low.tolist(),
            "goal_range_high": base.goal_range_high.tolist(),
            "obj_range_low": base.obj_range_low.tolist(),
            "obj_range_high": base.obj_range_high.tolist(),
            "object_body_mass_kg": float(base.model.body_mass[obj_body]),
            "object_geom_friction": base.model.geom_friction[obj_geom].tolist(),
            "object_sliding_friction": float(base.model.geom_friction[obj_geom, 0]),
            "dof_damping": base.model.dof_damping.tolist(),
            "joint_names": joint_names,
            "actuator_names": list(base._model_names.actuator_names),
            "actuator_gainprm_col0": base.model.actuator_gainprm[:, 0].tolist(),
            "action_space_low": env.action_space.low.tolist(),
            "action_space_high": env.action_space.high.tolist(),
            "reset_seed_0_achieved_goal": np.asarray(obs["achieved_goal"]).tolist(),
            "reset_seed_0_desired_goal": np.asarray(obs["desired_goal"]).tolist(),
            "block_gripper": bool(base.block_gripper),
            "reward_type": base.reward_type,
        }
    finally:
        env.close()


def package_versions() -> dict:
    import gymnasium
    import mujoco
    import numpy
    import pandas
    import stable_baselines3
    import torch
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "gymnasium_robotics": version("gymnasium-robotics"),
        "mujoco": mujoco.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": version("sb3-contrib"),
        "torch": torch.__version__,
        "pandas": pandas.__version__,
    }


def main() -> None:
    n_steps = 12
    corrective = {d: trace_corrective(d, n_steps) for d in (0, 1, 2, 3)}
    wrapper_cls = load_domain_rand()
    original = {d: trace_original(wrapper_cls, d, n_steps) for d in (0, 1, 2, 3)}

    mismatches = []
    for delay in (1, 2, 3):
        for left, right in zip(corrective[delay], original[delay]):
            if not same_applied(left, right):
                mismatches.append({"delay": delay, "timestep": left["timestep"], "corrective": left, "stage0": right})

    zero_is_identity = all(
        r["commanded_0"] == r["applied_0"] and r["applied_1"] == "0.0" and r["applied_2"] == "0.0"
        for r in corrective[0]
    )
    original_zero_is_identity = all(r["commanded_0"] == r["applied_0"] for r in original[0])

    # Positive-delay rule: first d applied actions are 0; later applied_0 equals the command from d steps earlier.
    rule_ok = zero_is_identity
    for delay in (1, 2, 3):
        for record in corrective[delay]:
            t = int(record["timestep"])
            if t < delay:
                if record["applied_0"] != "0.0":
                    rule_ok = False
            else:
                expected = f"{0.1 * ((t - delay) + 1):.1f}"
                if record["applied_0"] != expected:
                    rule_ok = False

    gate_pass = rule_ok and not mismatches and not original_zero_is_identity

    rows = []
    for delay in (0, 1, 2, 3):
        rows.extend(corrective[delay])
        rows.extend(original[delay])
    trace_path = OUT / "delay_semantics_trace.csv"
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stage0 = {seed: read_stage0(path) for seed, path in STAGE0_FILES.items()}
    param_cols = [
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
    params_identical = all(
        [tuple(r[c] for c in param_cols) for r in stage0[0]]
        == [tuple(r[c] for c in param_cols) for r in stage0[other]]
        for other in (42, 123)
    )

    manifest = {
        "phase": "0",
        "paper_not_modified": True,
        "git_commit": None,
        "git_note": "panda_final is not a git checkout. No commit hash is available for the code that generated Stage 0. The separate GitHub repo panda_mujoco_gym at e9881d9 is a different tree and was not used as the source of these ranges.",
        "source_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in SOURCE_FILES
        ],
        "checkpoints": [model_record(seed) for seed in (0, 42, 123)],
        "checkpoint_identification": {
            "stored_in_stage0_csv": False,
            "decision": "Use the three batch512_her4 final_model.zip files dated 2025-06-17/18. Their bytes match the training-tree copies under panda_project/logs. The 2026-09-29 delay script hardcoded these paths. The Stage 0 CSV names are eval_multi_param_seed{0,42,123}.csv and the three files have different success columns, so three policies were evaluated. The CSV itself does not store a path.",
            "not_used": str(ROOT / "logs" / "FrankaPushSparse_v0_TQC" / "batch256_her8_seed0_2025-06-18_07-49-51" / "final_model.zip"),
        },
        "tqc_her_configuration_from_checkpoint": {
            "policy": "MultiInputPolicy",
            "learning_rate": 0.001,
            "buffer_size": 1000000,
            "batch_size": 512,
            "tau": 0.05,
            "gamma": 0.95,
            "top_quantiles_to_drop_per_net": 2,
            "ent_coef": "auto",
            "n_quantiles": 25,
            "net_arch": {"pi": [256, 256, 256], "qf": [256, 256, 256]},
            "replay_buffer": "HerReplayBuffer",
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
            "source": "read from final_model.zip data, seed 0. train_push_basic.py contains the same literals.",
        },
        "environment_defaults_measured": live_defaults(),
        "success_criterion": {
            "code": "FrankaEnv._is_success: goal_distance(achieved_goal, desired_goal) < distance_threshold",
            "distance_threshold_m": 0.05,
            "episode_terminates_on_success": True,
            "source": "panda_mujoco_gym/envs/panda_env.py step() and _is_success()",
        },
        "goal_range_in_physical_task": {
            "constructor": "FrankaPushEnv goal_xy_range=0.3, obj_xy_range=0.3, goal_x_offset=0.0, goal_z_range=0.0, then panda_env adds 0.6 to both x bounds",
            "source": "panda_mujoco_gym/envs/push.py and panda_env.py",
        },
        "max_episode_steps": {
            "gym_registration": 50,
            "registration_source": "panda_mujoco_gym/__init__.py register(..., max_episode_steps=50)",
            "single_factor_worker": "worker_single_physics.py calls gym.make, so the registered TimeLimit of 50 applies. Its own MAX_EPISODE_STEPS=100 cannot extend an episode the TimeLimit already ended.",
            "stage0_wrapper": 100,
            "stage0_source": "worker_multi_random.py MAX_EPISODE_STEPS=100 passed to DomainRandomizationWrapper. That class subclasses FrankaPushEnv directly and is not wrapped by gymnasium TimeLimit. compute_truncated returns False, so the wrapper's counter is the horizon.",
            "delay_sweep_2026_09_29": 50,
            "delay_sweep_source": "test_push_delay_generalization.py uses gym.make('FrankaPushSparse-v0')",
        },
        "xml_defaults_as_written": {
            "object_geom_mass": 1,
            "object_geom_friction_attribute": "1",
            "object_free_joint_damping": 0.01,
            "arm_joint_damping": 1,
            "actuator_gainprm": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500],
            "source": "panda_mujoco_gym/assets/push.xml and panda_mocap.xml",
        },
        "stage0_ranges_from_code": {
            "source_files": ["worker_multi_random.py", "utils/domain_rand.py"],
            "mass": {"distribution": "uniform", "low": 0.05, "high": 3.0, "inclusive": "numpy uniform continuous"},
            "friction": {"distribution": "uniform", "low": 0.05, "high": 3.0, "applied_to": "geom_friction[obj_geom, 0]"},
            "damping_scale": {
                "distribution": "uniform",
                "low": 0.001,
                "high": 0.3,
                "applied_as": "model.dof_damping[:] = original_dof_damping * scale",
            },
            "actuator_gain_scale": {
                "distribution": "uniform",
                "low": 0.0,
                "high": 10.0,
                "applied_as": "model.actuator_gainprm[:, 0] = original_gainprm[:, 0] * scale",
            },
            "delay_steps": {
                "distribution": "numpy randint",
                "low_inclusive": 1,
                "high_exclusive": 10,
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "code": "DELAY_RANGE = (1, 10); np.random.randint(*delay_range)",
            },
            "goal_noise": {
                "distribution": "uniform per axis",
                "low": [-0.15, -0.15, 0.0],
                "high": [0.15, 0.15, 0.0],
                "applied_as": "added to self.goal after super()._sample_goal(), inside reset, after the observation has already been built",
            },
            "episodes": "200 per numpy seed in [0, 42, 123], so 600 episodes per model file",
            "numpy_seeds_are_not_training_seeds": True,
        },
        "saved_stage0_csv": {
            "files": {str(seed): str(path) for seed, path in STAGE0_FILES.items()},
            "parameter_columns_identical_across_model_files": params_identical,
            "per_model": {str(seed): stage0_summary(rows) for seed, rows in stage0.items()},
            "manuscript_22_56_and_76_50": "Neither 0.2256 nor 0.7650 is the success rate stored in these CSV files. They are not used as a target.",
        },
        "single_factor_ranges_from_code": {
            "source": "test_push_phycial_element_generalization.py",
            "object_mass": [0.05, 0.1, 0.5, 1.0, 2.0, 3.0],
            "object_friction": [0.05, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
            "joint_damping_scale": [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.3],
            "actuator_gain_scale": [0, 1, 2, 3, 5, 6, 7, 10],
            "episodes": "worker_single_physics.py: 200 episodes x seeds [0, 42, 123] per grid point, on one model",
            "does_not_include_delay": True,
        },
        "delay_semantics_gate": {
            "trace": str(trace_path),
            "command_sequence": "A_t = [0.1*(t+1), 0, 0] for t=0..11. Zero is reserved for the initial buffer.",
            "corrective_delay_0_applies_current_action": zero_is_identity,
            "stage0_wrapper_delay_0_applies_current_action": original_zero_is_identity,
            "stage0_wrapper_matches_corrective_at_delay_1_2_3": not mismatches,
            "mismatches": mismatches,
            "gate_pass": gate_pass,
            "meaning": {
                "corrective_delay_0": "no delay; the commanded action is applied on that timestep",
                "corrective_and_stage0_delay_d_for_d_ge_1": "the action commanded d timesteps earlier is applied; the first d timesteps apply a zero action",
                "stage0_delay_0": "not a no-delay condition. DomainRandomizationWrapper indexes action_buffer[-delay]. Python treats [-0] as [0], the oldest slot of a length-10 buffer. Stage 0 never samples this value.",
                "stage0_sampled_delay_1": "one full decision of latency, not the training condition",
            },
        },
        "phase1_locked_decisions": {
            "conditions": ["P0", "P1", "P2", "P3"],
            "episodes_per_model_per_condition": 600,
            "models": [0, 42, 123],
            "horizon": 100,
            "horizon_reason": "Match Stage 0's wrapper horizon. All four cells use it. The registered 50-step TimeLimit is not used in this 2x2.",
            "true_zero_delay": "corrective delay_steps=0, current action applied",
            "delay_distribution_for_P2_and_P3": "the delay_steps column of the saved Stage 0 CSV, values 1..9, not redrawn",
            "physical_draws_for_P1_and_P3": "mass, friction, damping_scale, gain_scale columns of eval_multi_param_seed0.csv, shared across the three models because the three files contain the same parameter rows",
            "goal_noise": "not applied in any cell. The locked 2x2 does not include it. This is a deviation from bitwise Stage 0.",
            "scene_seed": "reset_seed = 4000000 + episode_id, shared across conditions and models. Stage 0 did not seed the environment.",
            "policy": "deterministic predict, no retraining",
            "phase3_not_run": True,
        },
        "eval_environment_versions": package_versions(),
    }

    manifest_path = OUT / "audit_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {trace_path}")
    print(f"wrote {manifest_path}")
    print(f"GATE_PASS {gate_pass}")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
