"""Phase 3. Leave-one-out and single-factor attribution at horizon 100.

True zero delay. Same Stage 0 rows and the same reset seeds as P1.
Ranges are replayed, not redrawn and not narrowed.
P1-all is not rolled out here; the summarizer copies the locked P1 file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from round2_common import (
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
    load_stage0_rows,
    make_env,
    open_writer,
    originals_from,
    run_episode,
)

# factor order: mass, friction, damping, gain. True means replay the Stage 0 draw.
CONDITIONS = {
    "P1-no-mass": (False, True, True, True),
    "P1-no-friction": (True, False, True, True),
    "P1-no-damping": (True, True, False, True),
    "P1-no-gain": (True, True, True, False),
    "mass-only": (True, False, False, False),
    "friction-only": (False, True, False, False),
    "damping-only": (False, False, True, False),
    "gain-only": (False, False, False, True),
}

FIELDS = [
    "condition",
    "source",
    "policy_seed",
    "episode_id",
    "reset_seed",
    "delay_steps",
    "horizon",
    "mass",
    "friction",
    "damping_scale",
    "actuator_gain_scale",
    "mass_randomized",
    "friction_randomized",
    "damping_randomized",
    "gain_randomized",
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
]


def partial_path(partial_dir: Path, condition: str, model_seed: int) -> Path:
    return partial_dir / f"ablation_{condition}_seed{model_seed}.csv"


def draw_values(flags, row, originals) -> tuple[float, float, float, float]:
    mass = float(row["mass"]) if flags[0] else originals["mass"]
    friction = float(row["friction"]) if flags[1] else originals["friction"]
    damping = float(row["damping_scale"]) if flags[2] else 1.0
    gain = float(row["gain_scale"]) if flags[3] else 1.0
    return mass, friction, damping, gain


def run_model(model_seed: int, n_episodes: int, partial_dir: Path, conditions: tuple[str, ...]) -> None:
    torch.set_num_threads(1)
    rows = load_stage0_rows()[:n_episodes]
    env, base = make_env(100, 0.3)
    assert_default_ranges(base)
    assert_horizon_binds(env, 100)
    originals = originals_from(base)
    model = load_policy(model_seed, env)
    print(f"[model {model_seed}] physical ablation loaded", flush=True)
    try:
        for condition in conditions:
            flags = CONDITIONS[condition]
            path = partial_path(partial_dir, condition, model_seed)
            start = completed_episodes(path)
            if start >= n_episodes:
                print(f"[model {model_seed}] {condition} already complete", flush=True)
                continue
            handle, writer = open_writer(path, FIELDS, start)
            print(f"[model {model_seed}] {condition} from episode {start}", flush=True)
            t0 = time.perf_counter()
            try:
                for episode_id in range(start, n_episodes):
                    source = rows[episode_id]
                    mass, friction, damping, gain = draw_values(flags, source, originals)
                    reset_seed = DELAY_RESET_BASE + episode_id
                    obs, _ = env.reset(seed=reset_seed)
                    apply_physics(base, originals, mass, friction, damping, gain)
                    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
                    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
                    success, steps, final_distance = run_episode(env, model, obs, desired, 0, 100)
                    writer.writerow(
                        {
                            "condition": condition,
                            "source": "rolled_out",
                            "policy_seed": model_seed,
                            "episode_id": episode_id,
                            "reset_seed": reset_seed,
                            "delay_steps": 0,
                            "horizon": 100,
                            "mass": fmt(mass),
                            "friction": fmt(friction),
                            "damping_scale": fmt(damping),
                            "actuator_gain_scale": fmt(gain),
                            "mass_randomized": int(flags[0]),
                            "friction_randomized": int(flags[1]),
                            "damping_randomized": int(flags[2]),
                            "gain_randomized": int(flags[3]),
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
                        }
                    )
                    handle.flush()
                    if (episode_id + 1) % 20 == 0 or episode_id + 1 == n_episodes:
                        elapsed = time.perf_counter() - t0
                        print(
                            f"[model {model_seed}] {condition} {episode_id + 1}/{n_episodes} "
                            f"in {elapsed:.0f}s",
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
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    unknown = [name for name in args.conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    run_model(args.model_seed, args.episodes, args.partial_dir, tuple(args.conditions))


if __name__ == "__main__":
    main()
