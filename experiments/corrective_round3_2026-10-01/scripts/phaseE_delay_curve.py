"""Full delay curve under the Phase E clean reset and writer C.

Phase D showed that delays 1 and 2 change success on the clean reset, so the
old per-delay table cannot be cited as this protocol. This reruns delays 0..9
on the same 600 scenes and three frozen policies. It does not overwrite the
round-2 CSV. Do not start it while phaseE.status is running.
"""

from __future__ import annotations

import argparse

import torch

import phaseE_formal as pe

DELAYS = tuple(range(10))
DEFAULT_PHYSICS = {"mass": 1.0, "friction": 1.0, "damping_scale": 1.0, "gain_scale": 1.0}


def done_pairs(path) -> set[tuple[int, int]]:
    found = set()
    for row in pe.read_table(path):
        found.add((int(row["delay_steps"]), int(row["episode_id"])))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in (0, 42, 123):
        raise SystemExit(f"seed {args.seed} is not one of the frozen checkpoints")
    torch.set_num_threads(1)
    rows = pe.load_stage0_rows()
    if len(rows) != pe.N_EPISODES:
        raise SystemExit(f"stage0 rows {len(rows)}")
    protocol = pe.build_protocol(rows)
    pe.lock_protocol(protocol)
    protocol_sha = protocol["protocol_sha256"]
    template, template_base = pe.make_named_env()
    try:
        pe.assert_ranges(template_base)
        pristine = pe.snapshot(template_base)
        ids = pe.object_ids(template_base)
        model = pe.pb.load_policy(args.seed, template)
        path = pe.OUT / f"delay_curve_v3_seed{args.seed}.csv"
        done = done_pairs(path)
        print(f"[seed {args.seed}] delay curve done {len(done)} protocol {protocol_sha[:12]}", flush=True)
        for delay in DELAYS:
            for episode_id in range(pe.N_EPISODES):
                if (delay, episode_id) in done:
                    continue
                row = pe.run_one(
                    model,
                    pristine,
                    ids,
                    pe.RESET_BASE + episode_id,
                    delay,
                    DEFAULT_PHYSICS,
                    protocol_sha,
                    "delay_curve",
                    episode_id,
                    args.seed,
                )
                if int(row["delay_steps"]) != delay:
                    raise SystemExit("delay column drifted")
                pe.append_row(path, row)
                done.add((delay, episode_id))
                finished = episode_id + 1
                if finished % 25 == 0 or finished == pe.N_EPISODES:
                    print(f"[seed {args.seed}] delay {delay} {finished}/{pe.N_EPISODES}", flush=True)
    finally:
        template.close()


if __name__ == "__main__":
    main()
