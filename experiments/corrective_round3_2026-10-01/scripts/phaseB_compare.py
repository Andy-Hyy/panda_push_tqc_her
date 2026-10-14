"""Compare Phase B traces. Tolerances are diagnostic thresholds, not a claim that a difference is negligible."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
TRACE = OUT / "state_action_traces"
MODEL_ATOL = 1e-10
MODEL_RTOL = 1e-8
POLICY_ATOL = 1e-7
POLICY_RTOL = 1e-6

STATE_KEYS = (
    "after_reset__qpos",
    "after_reset__qvel",
    "after_reset__act",
    "after_reset__ctrl",
    "after_reset__qacc_warmstart",
    "after_reset__body_mass",
    "after_reset__body_inertia",
    "after_reset__body_invweight0",
    "achieved_after_reset",
    "desired_after_reset",
    "first_policy",
    "first_executed",
    "qpos",
    "policy_actions",
    "object_pos",
)


def close(a, b, kind: str) -> tuple[bool, bool, float]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, False, float("inf")
    if a.size == 0:
        return True, True, 0.0
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    exact = bool(np.array_equal(a, b))
    if kind == "policy":
        tol = bool(np.allclose(a, b, atol=POLICY_ATOL, rtol=POLICY_RTOL, equal_nan=True))
    else:
        tol = bool(np.allclose(a, b, atol=MODEL_ATOL, rtol=MODEL_RTOL, equal_nan=True))
    return exact, tol, float(diff)


def load_pair_csv() -> dict:
    rows = {}
    path = OUT / "paired_P2_sweep_comparison.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["policy_seed"]), int(row["episode_id"]))] = row
    return rows


def main() -> None:
    paired = load_pair_csv()
    selection = list(csv.DictReader((OUT / "selection.csv").open(newline="")))
    modes = sorted(p.name for p in TRACE.iterdir() if p.is_dir()) if TRACE.exists() else []
    fields = [
        "sample_id",
        "policy_seed",
        "episode_id",
        "delay_steps",
        "sample_role",
        "mode",
        "n_repeats",
        "successes",
        "steps",
        "final_distances",
        "success_stable",
        "steps_stable",
        "initial_qpos_exact",
        "initial_qpos_tol",
        "initial_qpos_maxabs",
        "warmstart_exact",
        "warmstart_tol",
        "warmstart_maxabs",
        "first_policy_exact",
        "first_policy_tol",
        "first_policy_maxabs",
        "qpos_trace_exact",
        "qpos_trace_tol",
        "qpos_trace_maxabs",
        "policy_trace_exact",
        "policy_trace_tol",
        "policy_trace_maxabs",
        "p2_success",
        "sweep_success",
        "p2_steps",
        "sweep_steps",
        "matches_p2_outcome",
        "matches_sweep_outcome",
    ]
    out_rows = []
    for row in selection:
        seed = int(row["policy_seed"])
        episode = int(row["episode_id"])
        ref = paired[(seed, episode)]
        for mode in modes:
            paths = sorted((TRACE / mode / f"seed{seed}").glob(f"{row['sample_id']}_r*.npz"))
            if not paths:
                continue
            traces = [np.load(path) for path in paths]
            successes = [int(item["success"][0]) for item in traces]
            steps = [int(item["steps"][0]) for item in traces]
            finals = [float(item["final_distance"][0]) for item in traces]
            base = traces[0]

            def across(key, kind):
                exact = True
                tol = True
                worst = 0.0
                for other in traces[1:]:
                    ex, tl, diff = close(base[key], other[key], kind)
                    exact = exact and ex
                    tol = tol and tl
                    worst = max(worst, diff)
                if len(traces) == 1:
                    return True, True, 0.0
                return exact, tol, worst

            qpos_ex, qpos_tl, qpos_d = across("after_reset__qpos", "model")
            warm_ex, warm_tl, warm_d = across("after_reset__qacc_warmstart", "model")
            pol_ex, pol_tl, pol_d = across("first_policy", "policy")
            tr_ex, tr_tl, tr_d = across("qpos", "model")
            act_ex, act_tl, act_d = across("policy_actions", "policy")
            out_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "policy_seed": seed,
                    "episode_id": episode,
                    "delay_steps": int(row["delay_steps"]),
                    "sample_role": row["sample_role"],
                    "mode": mode,
                    "n_repeats": len(traces),
                    "successes": " ".join(str(v) for v in successes),
                    "steps": " ".join(str(v) for v in steps),
                    "final_distances": " ".join(f"{v:.8f}" for v in finals),
                    "success_stable": int(len(set(successes)) == 1),
                    "steps_stable": int(len(set(steps)) == 1),
                    "initial_qpos_exact": int(qpos_ex),
                    "initial_qpos_tol": int(qpos_tl),
                    "initial_qpos_maxabs": f"{qpos_d:.3e}",
                    "warmstart_exact": int(warm_ex),
                    "warmstart_tol": int(warm_tl),
                    "warmstart_maxabs": f"{warm_d:.3e}",
                    "first_policy_exact": int(pol_ex),
                    "first_policy_tol": int(pol_tl),
                    "first_policy_maxabs": f"{pol_d:.3e}",
                    "qpos_trace_exact": int(tr_ex),
                    "qpos_trace_tol": int(tr_tl),
                    "qpos_trace_maxabs": f"{tr_d:.3e}",
                    "policy_trace_exact": int(act_ex),
                    "policy_trace_tol": int(act_tl),
                    "policy_trace_maxabs": f"{act_d:.3e}",
                    "p2_success": int(ref["success_p2"]),
                    "sweep_success": int(ref["success_sweep"]),
                    "p2_steps": int(ref["steps_p2"]),
                    "sweep_steps": int(ref["steps_sweep"]),
                    "matches_p2_outcome": int(len(set(successes)) == 1 and successes[0] == int(ref["success_p2"]) and len(set(steps)) == 1 and steps[0] == int(ref["steps_p2"])),
                    "matches_sweep_outcome": int(len(set(successes)) == 1 and successes[0] == int(ref["success_sweep"]) and len(set(steps)) == 1 and steps[0] == int(ref["steps_sweep"])),
                }
            )
            for item in traces:
                item.close()
    dest = OUT / "repeatability_diagnostic.csv"
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"rows {len(out_rows)} -> {dest}")


if __name__ == "__main__":
    main()
