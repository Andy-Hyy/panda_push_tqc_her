"""Phase 4. Paired spatial sweeps at horizons 50 and 100.

No action delay. Physical parameters stay at the training defaults.
Object spawn stays at 0.30 m. L is the target-region side length.
reset_seed = 6_000_000 + round(L * 100) * 1000 + episode_id, shared by both horizons.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from round2_common import (
    HORIZONS,
    LENGTHS,
    MODELS,
    N_EPISODES,
    PARTIAL,
    SPATIAL_RESET_BASE,
    apply_physics,
    assert_horizon_binds,
    completed_episodes,
    fmt,
    load_policy,
    make_env,
    open_writer,
    originals_from,
    run_episode,
    set_goal_side,
    set_horizon,
)

FIELDS = [
    "policy_seed",
    "episode_id",
    "reset_seed",
    "horizon",
    "L",
    "initial_object_x",
    "initial_object_y",
    "initial_object_z",
    "target_x",
    "target_y",
    "target_z",
    "initial_distance",
    "success",
    "steps_used",
    "final_distance",
]


def spatial_reset_seed(side: float, episode_id: int) -> int:
    return SPATIAL_RESET_BASE + int(round(side * 100)) * 1000 + episode_id


def partial_path(partial_dir: Path, horizon: int, side: float, model_seed: int) -> Path:
    tenths = int(round(side * 100))
    return partial_dir / f"spatial_h{horizon}_L{tenths:03d}_seed{model_seed}.csv"


def run_model(
    model_seed: int,
    n_episodes: int,
    partial_dir: Path,
    horizons: tuple[int, ...],
    lengths: tuple[float, ...],
) -> None:
    torch.set_num_threads(1)
    env, base = make_env(max(horizons), 0.3)
    for horizon in horizons:
        assert_horizon_binds(env, horizon)
    originals = originals_from(base)
    model = load_policy(model_seed, env)
    print(f"[model {model_seed}] spatial sweep loaded {MODELS[model_seed].name}", flush=True)
    try:
        for side in lengths:
            set_goal_side(base, side)
            for horizon in horizons:
                set_horizon(env, horizon)
                path = partial_path(partial_dir, horizon, side, model_seed)
                start = completed_episodes(path)
                if start >= n_episodes:
                    print(f"[model {model_seed}] H{horizon} L{side:.2f} already complete", flush=True)
                    continue
                handle, writer = open_writer(path, FIELDS, start)
                print(f"[model {model_seed}] H{horizon} L{side:.2f} from episode {start}", flush=True)
                t0 = time.perf_counter()
                try:
                    for episode_id in range(start, n_episodes):
                        reset_seed = spatial_reset_seed(side, episode_id)
                        obs, _ = env.reset(seed=reset_seed)
                        apply_physics(base, originals, originals["mass"], originals["friction"], 1.0, 1.0)
                        achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                        desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                        # Sampled target must lie inside the intended side length.
                        centre = np.array([0.6, 0.0, float(desired[2])])
                        half = side / 2
                        if abs(float(desired[0]) - 0.6) > half + 1e-8 or abs(float(desired[1])) > half + 1e-8:
                            raise SystemExit(f"target outside L={side}: {desired}")
                        if abs(float(achieved[0]) - 0.6) > 0.15 + 1e-8 or abs(float(achieved[1])) > 0.15 + 1e-8:
                            raise SystemExit(f"object outside the 0.30 m spawn: {achieved}")
                        del centre
                        success, steps, final_distance = run_episode(env, model, obs, desired, 0, horizon)
                        writer.writerow(
                            {
                                "policy_seed": model_seed,
                                "episode_id": episode_id,
                                "reset_seed": reset_seed,
                                "horizon": horizon,
                                "L": f"{side:.2f}",
                                "initial_object_x": fmt(float(achieved[0])),
                                "initial_object_y": fmt(float(achieved[1])),
                                "initial_object_z": fmt(float(achieved[2])),
                                "target_x": fmt(float(desired[0])),
                                "target_y": fmt(float(desired[1])),
                                "target_z": fmt(float(desired[2])),
                                "initial_distance": fmt(float(np.linalg.norm(achieved - desired))),
                                "success": success,
                                "steps_used": steps,
                                "final_distance": fmt(final_distance),
                            }
                        )
                        handle.flush()
                        if (episode_id + 1) % 20 == 0 or episode_id + 1 == n_episodes:
                            elapsed = time.perf_counter() - t0
                            print(
                                f"[model {model_seed}] H{horizon} L{side:.2f} "
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
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--lengths", type=float, nargs="+", default=list(LENGTHS))
    args = parser.parse_args()
    run_model(args.model_seed, args.episodes, args.partial_dir, tuple(args.horizons), tuple(args.lengths))


if __name__ == "__main__":
    main()
