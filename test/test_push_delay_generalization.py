"""Single-factor action-delay evaluation for the three trained push baselines.

Everything except action delay stays at the training default. delay_steps=0
executes the current action (the training condition). delay_steps=d executes
the action issued d decisions earlier. The buffer starts full of zeros, so the
first d steps apply a zero action.

Episode horizon is the registered TimeLimit of 50. Each reported cell is
3 eval seeds x 200 episodes, matching worker_single_physics.py. The same
reset seeds are reused at every delay, so initial object and goal positions
are paired across the delay sweep.
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC

import panda_mujoco_gym  # noqa: F401  registers FrankaPushSparse-v0

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
MODELS = {
    0: MODEL_ROOT / "batch512_her4_seed0_2025-06-17_13-50-58" / "final_model.zip",
    42: MODEL_ROOT / "batch512_her4_seed42_2025-06-17_23-01-53" / "final_model.zip",
    123: MODEL_ROOT / "batch512_her4_seed123_2025-06-18_23-21-02" / "final_model.zip",
}
DELAYS = list(range(0, 10))
EVAL_SEEDS = [0, 42, 123]
EPISODES_PER_SEED = 200
OUT_DIR = ROOT / "experiments" / "physical_element"
DETAIL_FIELDS = [
    "model_seed",
    "delay_steps",
    "eval_seed",
    "episodes",
    "successes",
    "success_rate",
    "seconds",
]


class ActionDelayWrapper(gym.Wrapper):
    """Delay the action actually sent to the robot by a fixed number of steps."""

    def __init__(self, env, delay_steps: int):
        super().__init__(env)
        self.delay_steps = int(delay_steps)
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self._pending = deque()

    def reset(self, **kwargs):
        zero = np.zeros(self.action_space.shape, dtype=np.float32)
        self._pending = deque(zero.copy() for _ in range(self.delay_steps))
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.delay_steps == 0:
            executed = action
        else:
            executed = self._pending.popleft()
            self._pending.append(action)
        return self.env.step(executed)


def make_env(delay_steps: int):
    env = gym.make("FrankaPushSparse-v0", disable_env_checker=True)
    return ActionDelayWrapper(env, delay_steps)


def detail_path(model_seed: int) -> Path:
    return OUT_DIR / f"eval_phy_delay_steps_detail_seed{model_seed}.csv"


def load_done(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add((int(row["delay_steps"]), int(row["eval_seed"])))
    return done


def append_row(path: Path, row: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def is_success(info) -> bool:
    flag = info.get("is_success", False)
    return bool(np.asarray(flag).item())


def run_episodes(model, env, eval_seed: int, n_episodes: int):
    successes = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=eval_seed + i)
        terminated = False
        truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        if is_success(info):
            successes += 1
    return successes


def check_delay_zero_matches_training(model_path: Path, n_episodes: int = 10):
    """delay=0 must reproduce the unmodified environment on the same seeds."""
    raw = gym.make("FrankaPushSparse-v0", disable_env_checker=True)
    delayed = make_env(0)
    model = TQC.load(str(model_path), env=raw, device="cpu")
    try:
        for i in range(n_episodes):
            raw_success, raw_steps = _rollout(model, raw, seed=i)
            delay_success, delay_steps = _rollout(model, delayed, seed=i)
            if raw_success != delay_success or raw_steps != delay_steps:
                raise RuntimeError(
                    f"delay=0 diverged from the training env at seed {i}: "
                    f"raw success={raw_success} steps={raw_steps}, "
                    f"delay0 success={delay_success} steps={delay_steps}"
                )
    finally:
        raw.close()
        delayed.close()
    print(f"[check] delay=0 matches the unmodified env on {n_episodes} episodes", flush=True)


def _rollout(model, env, seed: int):
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    steps = 0
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
    return is_success(info), steps


def evaluate_model(model_seed: int, delays, n_episodes: int):
    model_path = MODELS[model_seed]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    path = detail_path(model_seed)
    done = load_done(path)
    print(f"[model {model_seed}] {model_path}", flush=True)
    check_delay_zero_matches_training(model_path)

    env = None
    model = None
    current_delay = None
    try:
        for delay in delays:
            if all((delay, seed) in done for seed in EVAL_SEEDS):
                print(f"[model {model_seed}] delay={delay} already done", flush=True)
                continue
            if current_delay != delay:
                if env is not None:
                    env.close()
                env = make_env(delay)
                model = TQC.load(str(model_path), env=env, device="cpu")
                current_delay = delay
            for eval_seed in EVAL_SEEDS:
                if (delay, eval_seed) in done:
                    continue
                t0 = time.perf_counter()
                successes = run_episodes(model, env, eval_seed, n_episodes)
                elapsed = time.perf_counter() - t0
                rate = successes / n_episodes
                append_row(
                    path,
                    {
                        "model_seed": model_seed,
                        "delay_steps": delay,
                        "eval_seed": eval_seed,
                        "episodes": n_episodes,
                        "successes": successes,
                        "success_rate": f"{rate:.6f}",
                        "seconds": f"{elapsed:.1f}",
                    },
                )
                print(
                    f"[model {model_seed}] delay={delay} eval_seed={eval_seed} "
                    f"{successes}/{n_episodes} = {rate:.3f} in {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()


def summarize():
    import pandas as pd

    frames = []
    for model_seed in MODELS:
        path = detail_path(model_seed)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("no delay detail csv found")
    df = pd.concat(frames, ignore_index=True)
    pooled = (
        df.groupby(["model_seed", "delay_steps"], as_index=False)
        .agg(episodes=("episodes", "sum"), successes=("successes", "sum"))
    )
    pooled["success_rate"] = pooled["successes"] / pooled["episodes"]
    pooled_path = OUT_DIR / "eval_phy_delay_steps.csv"
    with pooled_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model seed", "param_name", "value", "success_rate", "successes", "episodes"]
        )
        writer.writeheader()
        for _, row in pooled.sort_values(["delay_steps", "model_seed"]).iterrows():
            writer.writerow(
                {
                    "model seed": int(row["model_seed"]),
                    "param_name": "delay_steps",
                    "value": int(row["delay_steps"]),
                    "success_rate": f"{row['success_rate']:.6f}",
                    "successes": int(row["successes"]),
                    "episodes": int(row["episodes"]),
                }
            )
    across = pooled.groupby("delay_steps")["success_rate"].agg(["mean", "std", "count"])
    summary_path = OUT_DIR / "eval_phy_delay_steps_summary.csv"
    across.to_csv(summary_path)
    print(across.to_string())
    print(f"wrote {pooled_path}")
    print(f"wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, choices=sorted(MODELS), action="append")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_SEED)
    parser.add_argument("--delays", type=int, nargs="*", default=DELAYS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
        return
    seeds = args.model_seed or sorted(MODELS)
    for model_seed in seeds:
        evaluate_model(model_seed, args.delays, args.episodes)


if __name__ == "__main__":
    main()
