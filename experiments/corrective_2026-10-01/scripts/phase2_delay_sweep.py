"""Phase 2. Horizon-100 delay sweep at the training physical defaults.

delay_steps=0 is true zero latency. Scenes use reset_seed = 4_000_000 + episode_id,
the same pairing as the locked P0–P3 run. Target region stays 0.30 m.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from round2_common import (
    DELAYS,
    DELAY_RESET_BASE,
    MODELS,
    N_EPISODES,
    PARTIAL,
    apply_physics,
    assert_default_ranges,
    assert_horizon_binds,
    completed_episodes,
    fmt,
    load_policy,
    make_env,
    open_writer,
    originals_from,
    run_episode,
)

FIELDS = [
    "policy_seed",
    "episode_id",
    "reset_seed",
    "delay_steps",
    "latency_s",
    "success",
    "steps_used",
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "target_x",
    "target_y",
    "target_z",
    "initial_distance",
    "final_distance",
    "horizon",
]


def partial_path(partial_dir: Path, delay_steps: int, model_seed: int) -> Path:
    return partial_dir / f"delay_h100_d{delay_steps}_seed{model_seed}.csv"


def run_model(model_seed: int, n_episodes: int, partial_dir: Path, delays: tuple[int, ...]) -> None:
    torch.set_num_threads(1)
    env, base = make_env(100, 0.3)
    assert_default_ranges(base)
    period = 0.5
    assert_horizon_binds(env, 100)
    originals = originals_from(base)
    model = load_policy(model_seed, env)
    print(f"[model {model_seed}] delay sweep loaded {MODELS[model_seed].name}", flush=True)
    try:
        for delay_steps in delays:
            path = partial_path(partial_dir, delay_steps, model_seed)
            start = completed_episodes(path)
            if start >= n_episodes:
                print(f"[model {model_seed}] delay {delay_steps} already complete", flush=True)
                continue
            handle, writer = open_writer(path, FIELDS, start)
            print(f"[model {model_seed}] delay {delay_steps} from episode {start}", flush=True)
            t0 = time.perf_counter()
            try:
                for episode_id in range(start, n_episodes):
                    reset_seed = DELAY_RESET_BASE + episode_id
                    obs, _ = env.reset(seed=reset_seed)
                    apply_physics(base, originals, originals["mass"], originals["friction"], 1.0, 1.0)
                    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                    success, steps, final_distance = run_episode(env, model, obs, desired, delay_steps, 100)
                    writer.writerow(
                        {
                            "policy_seed": model_seed,
                            "episode_id": episode_id,
                            "reset_seed": reset_seed,
                            "delay_steps": delay_steps,
                            "latency_s": f"{delay_steps * period:.1f}",
                            "success": success,
                            "steps_used": steps,
                            "initial_object_x": fmt(float(achieved[0])),
                            "initial_object_y": fmt(float(achieved[1])),
                            "initial_object_z": fmt(float(achieved[2])),
                            "target_x": fmt(float(desired[0])),
                            "target_y": fmt(float(desired[1])),
                            "target_z": fmt(float(desired[2])),
                            "initial_distance": fmt(float(np.linalg.norm(achieved - desired))),
                            "final_distance": fmt(final_distance),
                            "horizon": 100,
                        }
                    )
                    handle.flush()
                    if (episode_id + 1) % 20 == 0 or episode_id + 1 == n_episodes:
                        elapsed = time.perf_counter() - t0
                        print(
                            f"[model {model_seed}] delay {delay_steps} "
                            f"{episode_id + 1}/{n_episodes} in {elapsed:.0f}s",
                            flush=True,
                        )
            finally:
                handle.close()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seed", type=int, choices=sorted(MODELS), required=True)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--partial-dir", type=Path, default=PARTIAL)
    parser.add_argument("--delays", type=int, nargs="+", default=list(DELAYS))
    args = parser.parse_args()
    run_model(args.model_seed, args.episodes, args.partial_dir, tuple(args.delays))


if __name__ == "__main__":
    main()
