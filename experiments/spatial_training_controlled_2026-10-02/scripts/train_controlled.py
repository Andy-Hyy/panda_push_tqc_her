"""One trainer for SpatialDR and Fixed Large.

The two methods differ only in the goal-box side chosen before each reset.
Checkpoints exclude the replay buffer, so an interrupted formal run is renamed
and started over. It is not a continuation.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import controlled_common as common

EPISODE_FIELDS = (
    "method",
    "train_seed",
    "env_index",
    "episode_id",
    "L",
    "goal_low_x",
    "goal_low_y",
    "goal_low_z",
    "goal_high_x",
    "goal_high_y",
    "goal_high_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "object_x",
    "object_y",
    "object_z",
    "initial_distance",
    "success",
    "steps_used",
    "termination_reason",
)


def _float(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0]) if np.ndim(value) else float(value)


class Float32Wrapper:
    """Match the verified trainer: observations seen by TQC are float32."""

    def __new__(cls, env):
        import gymnasium as gym
        from gymnasium import spaces

        class _Wrapper(gym.ObservationWrapper):
            def __init__(self, inner):
                super().__init__(inner)
                updated = {}
                for key, space in inner.observation_space.spaces.items():
                    if isinstance(space, spaces.Box) and space.dtype == np.float64:
                        updated[key] = spaces.Box(
                            low=space.low.astype(np.float32),
                            high=space.high.astype(np.float32),
                            shape=space.shape,
                            dtype=np.float32,
                        )
                    else:
                        updated[key] = space
                self.observation_space = spaces.Dict(updated)

            def observation(self, observation):
                return {
                    key: value.astype(np.float32) if isinstance(value, np.ndarray) and value.dtype == np.float64 else value
                    for key, value in observation.items()
                }

        return _Wrapper(env)


class GoalDistribution:
    """Draw or assign L, write the sampling bounds, then reset."""

    def __new__(cls, env, *, method: str, train_seed: int, env_index: int, log_path: Path):
        import gymnasium as gym

        class _Wrapper(gym.Wrapper):
            def __init__(self, inner):
                super().__init__(inner)
                if method not in common.METHODS:
                    raise SystemExit(f"unknown method {method}")
                self.method = method
                self.train_seed = int(train_seed)
                self.env_index = int(env_index)
                self.log_path = Path(log_path)
                self.rng = np.random.Generator(np.random.PCG64(common.l_seed(self.train_seed, self.env_index)))
                self.episode_id = -1
                self.steps = 0
                self.open_episode = None
                self.desired = None

            def reset(self, *, seed=None, options=None):
                if self.method == "spatial_dr":
                    side = float(self.rng.uniform(common.DR_LOW, common.DR_HIGH))
                    if not (common.DR_LOW <= side < common.DR_HIGH):
                        raise SystemExit(f"drawn L {side} is outside [0.30, 0.80)")
                else:
                    side = float(common.FIXED_SIDE)
                base = self.env.unwrapped
                common.assert_side(base, side)
                obs, info = self.env.reset(seed=seed, options=options)
                if np.asarray(obs["desired_goal"]).dtype != np.float32:
                    raise SystemExit("training observation desired_goal is not float32")
                pose = common.pose_after_reset(base, obs)
                if not common.in_box(pose["goal"], pose["goal_low"], pose["goal_high"]):
                    raise SystemExit("goal fell outside the bounds written before reset")
                if not common.in_box(pose["object"], pose["obj_low"], pose["obj_high"]):
                    raise SystemExit("object fell outside the 0.30 m box")
                # The observation is already built. Do not write base.goal after this.
                self.episode_id += 1
                self.steps = 0
                self.desired = pose["goal"].copy()
                low = pose["goal_low"]
                high = pose["goal_high"]
                obj = pose["object"]
                goal = pose["goal"]
                self.open_episode = {
                    "method": self.method,
                    "train_seed": self.train_seed,
                    "env_index": self.env_index,
                    "episode_id": self.episode_id,
                    "L": side,
                    "goal_low_x": float(low[0]),
                    "goal_low_y": float(low[1]),
                    "goal_low_z": float(low[2]),
                    "goal_high_x": float(high[0]),
                    "goal_high_y": float(high[1]),
                    "goal_high_z": float(high[2]),
                    "goal_x": float(goal[0]),
                    "goal_y": float(goal[1]),
                    "goal_z": float(goal[2]),
                    "object_x": float(obj[0]),
                    "object_y": float(obj[1]),
                    "object_z": float(obj[2]),
                    "initial_distance": float(pose["distance"]),
                    "success": -1,
                    "steps_used": 0,
                    "termination_reason": "open",
                }
                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.steps += 1
                desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                if self.desired is None or not np.allclose(desired, self.desired, rtol=0.0, atol=1e-5):
                    raise SystemExit("goal changed during a training episode")
                if terminated or truncated:
                    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                    distance = float(np.linalg.norm(achieved - self.desired))
                    success = int(distance < 0.05)
                    reported = int(float(np.asarray(info.get("is_success", 0)).item()) > 0.5)
                    if reported != success:
                        raise SystemExit(f"success flag {reported} disagrees with distance {distance}")
                    if success:
                        reason = "success"
                    elif truncated:
                        reason = "time_limit"
                    else:
                        raise SystemExit(f"episode terminated without success at step {self.steps}")
                    if reason == "time_limit" and self.steps != common.HORIZON:
                        raise SystemExit(f"time limit at step {self.steps}")
                    if self.steps < 1:
                        raise SystemExit("episode ended before a step")
                    self._write(success, self.steps, reason)
                return obs, reward, terminated, truncated, info

            def close(self):
                if self.open_episode is not None:
                    self._write(-1, self.steps, "training_stopped")
                return self.env.close()

            def _write(self, success: int, steps: int, reason: str) -> None:
                row = dict(self.open_episode)
                row["success"] = int(success)
                row["steps_used"] = int(steps)
                row["termination_reason"] = reason
                ordered = {key: row[key] for key in EPISODE_FIELDS}
                common.append_csv(self.log_path, ordered)
                self.open_episode = None

        return _Wrapper(env)


def training_env_fn(method: str, train_seed: int, env_index: int, log_path: str):
    def _init():
        import torch
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        env, _base, _pristine, _ids = common.make_push_env(0.30)
        env = Float32Wrapper(env)
        env = GoalDistribution(
            env,
            method=method,
            train_seed=train_seed,
            env_index=env_index,
            log_path=Path(log_path),
        )
        return Monitor(env)

    return _init


def structural_config(model) -> dict:
    strategy = model.replay_buffer.goal_selection_strategy
    strategy_text = strategy.value if hasattr(strategy, "value") else str(strategy)
    unit = model.train_freq.unit
    unit_text = unit.value if hasattr(unit, "value") else str(unit)
    return {
        "learning_rate": float(model.lr_schedule(1.0)),
        "buffer_size": int(model.buffer_size),
        "batch_size": int(model.batch_size),
        "tau": float(model.tau),
        "gamma": float(model.gamma),
        "learning_starts": int(model.learning_starts),
        "train_freq": [int(model.train_freq.frequency), str(unit_text)],
        "gradient_steps": int(model.gradient_steps),
        "top_quantiles_to_drop_per_net": int(model.top_quantiles_to_drop_per_net),
        "use_sde": bool(model.use_sde),
        "optimize_memory_usage": bool(model.optimize_memory_usage),
        "n_quantiles": int(model.critic.n_quantiles),
        "n_critics": int(model.critic.n_critics),
        "actor_net_arch": [int(value) for value in model.actor.net_arch],
        "critic_net_arch": [int(value) for value in model.policy.critic_kwargs["net_arch"]],
        "activation": model.actor.activation_fn.__name__,
        "device": str(model.device),
        "target_entropy": float(np.asarray(model.target_entropy).reshape(-1)[0]),
        "n_sampled_goal": int(model.replay_buffer.n_sampled_goal),
        "goal_selection_strategy": str(strategy_text),
        "optimizer": type(model.actor.optimizer).__name__,
        "ent_coef_setting": "auto",
    }


def save_model(model, path: Path) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    zip_path = path if path.suffix == ".zip" else Path(str(path) + ".zip")
    if not zip_path.exists():
        raise SystemExit(f"model save did not create {zip_path}")
    size = zip_path.stat().st_size
    if size > 50_000_000:
        raise SystemExit(f"checkpoint {zip_path.name} is {size} bytes; refusing a file that may contain the replay buffer")
    return zip_path, common.sha256_file(zip_path)


class StepCheckpoint:
    def __new__(cls, targets: set[int], directory: Path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.targets = set(int(step) for step in targets)
                self.directory = directory
                self.saved: dict[int, str] = {}

            def _on_step(self) -> bool:
                steps = int(self.model.num_timesteps)
                if steps in self.targets and steps not in self.saved:
                    _path, digest = save_model(self.model, self.directory / f"tqc_{steps}_steps")
                    self.saved[steps] = digest
                return True

        return _Callback()


def prepare_run_dir(run_dir: Path, *, smoke: bool, expected_steps: int) -> str:
    final_path = run_dir / "final_model.zip"
    meta_path = run_dir / "run_meta.json"
    if not smoke and final_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("num_timesteps", -1)) == expected_steps and meta.get("final_sha256") == common.sha256_file(final_path):
            return "complete"
    if run_dir.exists() and any(run_dir.iterdir()):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        parked = run_dir.parent / f"interrupted_{run_dir.name}_{stamp}"
        run_dir.rename(parked)
    run_dir.mkdir(parents=True, exist_ok=True)
    return "fresh"


def run_training(method: str, train_seed: int, *, smoke: bool = False) -> dict:
    import multiprocessing as mp

    import torch
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    if method not in common.METHODS:
        raise SystemExit(method)
    if int(train_seed) not in common.TRAIN_SEEDS:
        raise SystemExit(f"seed {train_seed} is outside 0, 42, 123")
    torch.set_num_threads(1)
    expected = common.SMOKE_EXPECTED_STEPS if smoke else common.CHECKPOINT_STEPS[-1]
    requested = common.SMOKE_TIMESTEPS if smoke else common.TOTAL_TIMESTEPS
    root = common.OUT / ("smoke" if smoke else "runs")
    run_dir = root / f"{method}_seed{train_seed}"
    state = prepare_run_dir(run_dir, smoke=smoke, expected_steps=expected)
    if state == "complete":
        meta = json.loads((run_dir / "run_meta.json").read_text())
        meta["skipped"] = True
        return meta

    protocol = None
    if not smoke:
        protocol = common.assert_protocol_code()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log_dir = run_dir / "episodes"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_fns = [
        training_env_fn(method, int(train_seed), index, str(log_dir / f"env{index}.csv"))
        for index in range(common.N_ENVS)
    ]
    start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    vec = SubprocVecEnv(env_fns, start_method=start_method)
    try:
        cfg = common.TRAIN_CONFIG
        model = TQC(
            policy=cfg["policy"],
            env=vec,
            seed=int(train_seed),
            learning_rate=cfg["learning_rate"],
            buffer_size=cfg["buffer_size"],
            learning_starts=cfg["learning_starts"],
            batch_size=cfg["batch_size"],
            tau=cfg["tau"],
            gamma=cfg["gamma"],
            train_freq=cfg["train_freq"],
            gradient_steps=cfg["gradient_steps"],
            ent_coef=cfg["ent_coef"],
            target_update_interval=cfg["target_update_interval"],
            target_entropy=cfg["target_entropy"],
            top_quantiles_to_drop_per_net=cfg["top_quantiles_to_drop_per_net"],
            use_sde=cfg["use_sde"],
            sde_sample_freq=cfg["sde_sample_freq"],
            use_sde_at_warmup=cfg["use_sde_at_warmup"],
            optimize_memory_usage=cfg["optimize_memory_usage"],
            stats_window_size=cfg["stats_window_size"],
            action_noise=None,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": cfg["n_sampled_goal"],
                "goal_selection_strategy": cfg["goal_selection_strategy"],
            },
            policy_kwargs={
                "net_arch": {"pi": list(cfg["net_arch_pi"]), "qf": list(cfg["net_arch_qf"])},
                "n_quantiles": cfg["n_quantiles"],
            },
            device=cfg["device"],
            tensorboard_log=None,
            verbose=1,
        )
        resolved = structural_config(model)
        if protocol is not None and resolved != protocol["observed_defaults"]:
            raise SystemExit(f"resolved defaults differ from the locked protocol: {resolved}")
        (run_dir / "config_resolved.json").write_text(common.canonical(resolved))
        targets = set() if smoke else set(common.CHECKPOINT_STEPS)
        callback = StepCheckpoint(targets, run_dir / "checkpoints")
        model.learn(total_timesteps=requested, callback=callback, log_interval=50, progress_bar=False)
        steps = int(model.num_timesteps)
        if steps != expected:
            raise SystemExit(f"num_timesteps {steps} != expected {expected}")
        if steps not in callback.saved:
            _saved_path, saved_sha = save_model(model, run_dir / "checkpoints" / f"tqc_{steps}_steps")
            callback.saved[steps] = saved_sha
        if not smoke:
            missing = [step for step in common.CHECKPOINT_STEPS if step not in callback.saved]
            if missing:
                raise SystemExit(f"missing checkpoints {missing}")
        checkpoint_zip = run_dir / "checkpoints" / f"tqc_{steps}_steps.zip"
        final_zip = run_dir / "final_model.zip"
        shutil.copyfile(checkpoint_zip, final_zip)
        if common.sha256_file(final_zip) != common.sha256_file(checkpoint_zip):
            raise SystemExit("final model hash differs from the predetermined-budget checkpoint")
        manifest = {
            "num_timesteps": steps,
            "checkpoints": [
                {
                    "steps": step,
                    "sha256": callback.saved[step],
                    "path": str((run_dir / "checkpoints" / f"tqc_{step}_steps.zip").relative_to(common.OUT)),
                }
                for step in sorted(callback.saved)
            ],
        }
        ended = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta = {
            "method": method,
            "train_seed": int(train_seed),
            "smoke": smoke,
            "skipped": False,
            "requested_timesteps": requested,
            "num_timesteps": steps,
            "final_sha256": common.sha256_file(final_zip),
            "started": started,
            "ended": ended,
            "argv": sys.argv,
            "hostname": platform.node(),
            "start_method": start_method,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "device": "cpu",
            "packages": common.package_versions(),
            "resolved": resolved,
            "checkpoints": manifest["checkpoints"],
            "sampling": common.SAMPLING,
            "resume": "not used; saved checkpoints exclude the replay buffer",
        }
        (run_dir / "checkpoints_manifest.json").write_text(common.canonical(manifest))
        (run_dir / "run_meta.json").write_text(common.canonical(meta))
        return meta
    finally:
        vec.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=common.METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    meta = run_training(args.method, args.seed, smoke=args.smoke)
    print(common.canonical({"method": meta["method"], "seed": meta["train_seed"], "steps": meta["num_timesteps"], "skipped": meta["skipped"]}))


if __name__ == "__main__":
    main()
