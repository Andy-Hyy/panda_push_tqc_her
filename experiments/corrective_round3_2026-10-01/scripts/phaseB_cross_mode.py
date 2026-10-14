"""Cross-mode numeric comparison for the frozen Phase B sample.

Tolerances are the predeclared diagnostic thresholds:
model atol=1e-10 rtol=1e-8, float32 policy atol=1e-7 rtol=1e-6.
Integers and configuration fields are exact. A difference inside tolerance
is still reported as a nonzero max absolute difference.
"""

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
POLICY_KEYS = {
    "first_policy",
    "first_executed",
    "first_clipped",
    "policy_actions",
    "executed_actions",
    "clipped_actions",
}


def selection_rows() -> list[dict]:
    with (OUT / "selection.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_trace(mode: str, seed: int, sample_id: str, repeat: int) -> np.lib.npyio.NpzFile:
    path = TRACE / mode / f"seed{seed}" / f"{sample_id}_r{repeat}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return np.load(path)


def kind_of(key: str) -> str:
    return "policy" if key in POLICY_KEYS else "model"


def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return 0.0 if np.array_equal(a, b) else float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def within_tol(a: np.ndarray, b: np.ndarray, key: str) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO" or a.dtype.kind in "iu" or b.dtype.kind in "iu":
        return bool(np.array_equal(a, b))
    atol, rtol = (POLICY_ATOL, POLICY_RTOL) if kind_of(key) == "policy" else (MODEL_ATOL, MODEL_RTOL)
    return bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))


def compare_files(left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile) -> dict[str, float]:
    keys = sorted(set(left.files) | set(right.files))
    out = {}
    for key in keys:
        if key not in left.files or key not in right.files:
            out[key] = float("inf")
            continue
        out[key] = maxabs(left[key], right[key])
    return out


def first_qpos_divergence(left: np.ndarray, right: np.ndarray) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if not np.array_equal(left[index], right[index]):
            return index
    if len(left) != len(right):
        return limit
    return None


def main() -> None:
    rows = selection_rows()
    modes = ["fresh", "reused_a", "reused_b", "clean", "clean_after_a", "clean_after_b"]
    summary_lines = []

    def line(text: str) -> None:
        summary_lines.append(text)
        print(text, flush=True)

    # Within-mode bitwise check on the arrays that define the pass condition.
    watch = (
        "success",
        "steps",
        "after_reset__qpos",
        "after_reset__qvel",
        "after_reset__ctrl",
        "after_reset__qacc",
        "after_reset__qacc_warmstart",
        "before_action__qacc",
        "before_action__qacc_warmstart",
        "first_policy",
        "qpos",
        "policy_actions",
        "object_pos",
    )
    for mode in modes:
        worst = {key: 0.0 for key in watch}
        unstable = 0
        for row in rows:
            seed = int(row["policy_seed"])
            traces = [load_trace(mode, seed, row["sample_id"], repeat) for repeat in range(3)]
            if len({int(item["success"][0]) for item in traces}) != 1 or len({int(item["steps"][0]) for item in traces}) != 1:
                unstable += 1
            base = traces[0]
            for other in traces[1:]:
                for key in watch:
                    worst[key] = max(worst[key], maxabs(base[key], other[key]))
            for item in traces:
                item.close()
        nonzero = {key: value for key, value in worst.items() if value != 0.0}
        line(f"within {mode}: unstable_outcomes={unstable} nonzero={nonzero or 'none'}")

    # Clean versus each clean-after predecessor, repeat by repeat.
    detail_path = OUT / "clean_predecessor_comparison.csv"
    detail_fields = [
        "sample_id",
        "policy_seed",
        "repeat",
        "other_mode",
        "success_clean",
        "success_other",
        "steps_clean",
        "steps_other",
        "outcome_agree",
        "all_arrays_exact",
        "worst_key",
        "worst_maxabs",
        "qpos_first_divergence",
    ]
    detail_rows = []
    for other_mode in ("clean_after_a", "clean_after_b"):
        exact_cases = 0
        outcome_cases = 0
        worst_global = ("", 0.0)
        for row in rows:
            seed = int(row["policy_seed"])
            case_exact = True
            case_outcome = True
            for repeat in range(3):
                left = load_trace("clean", seed, row["sample_id"], repeat)
                right = load_trace(other_mode, seed, row["sample_id"], repeat)
                diffs = compare_files(left, right)
                worst_key = max(diffs, key=diffs.get)
                worst_value = diffs[worst_key]
                if worst_value > worst_global[1]:
                    worst_global = (f"{row['sample_id']} r{repeat} {worst_key}", worst_value)
                arrays_exact = worst_value == 0.0
                outcome = int(left["success"][0]) == int(right["success"][0]) and int(left["steps"][0]) == int(right["steps"][0])
                diverge = first_qpos_divergence(left["qpos"], right["qpos"])
                case_exact = case_exact and arrays_exact
                case_outcome = case_outcome and outcome
                detail_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "policy_seed": seed,
                        "repeat": repeat,
                        "other_mode": other_mode,
                        "success_clean": int(left["success"][0]),
                        "success_other": int(right["success"][0]),
                        "steps_clean": int(left["steps"][0]),
                        "steps_other": int(right["steps"][0]),
                        "outcome_agree": int(outcome),
                        "all_arrays_exact": int(arrays_exact),
                        "worst_key": worst_key,
                        "worst_maxabs": repr(worst_value),
                        "qpos_first_divergence": "" if diverge is None else str(diverge),
                    }
                )
                left.close()
                right.close()
            exact_cases += int(case_exact)
            outcome_cases += int(case_outcome)
        line(
            f"{other_mode} vs clean: outcome {outcome_cases}/36 exact_arrays {exact_cases}/36 "
            f"worst={worst_global[0]} maxabs={worst_global[1]!r}"
        )

    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    line(f"wrote {detail_path} rows={len(detail_rows)}")

    # Historical sensitivity: fresh versus reused_a / clean, using repeat 0.
    # Report only keys that differ, grouped by checkpoint.
    checkpoints = (
        "after_reset__qpos",
        "after_reset__qvel",
        "after_reset__ctrl",
        "after_reset__qacc",
        "after_reset__qacc_warmstart",
        "after_reset__body_mass",
        "after_reset__body_inertia",
        "after_reset__body_invweight0",
        "after_reset__dof_damping",
        "after_reset__gain",
        "after_reset__friction0",
        "before_action__qpos",
        "before_action__qacc",
        "before_action__qacc_warmstart",
        "before_action__body_mass",
        "before_action__dof_damping",
        "before_action__gain",
        "before_action__friction0",
        "after_first_step__qpos",
        "after_first_step__qvel",
        "after_first_step__qacc",
        "after_first_step__qacc_warmstart",
        "first_policy",
        "obs_after_reset",
    )
    for other_mode in ("reused_a", "reused_b", "clean"):
        outcome_flip = []
        key_worst = {key: 0.0 for key in checkpoints}
        for row in rows:
            seed = int(row["policy_seed"])
            left = load_trace("fresh", seed, row["sample_id"], 0)
            right = load_trace(other_mode, seed, row["sample_id"], 0)
            if int(left["success"][0]) != int(right["success"][0]) or int(left["steps"][0]) != int(right["steps"][0]):
                outcome_flip.append(
                    f"{row['sample_id']} fresh {int(left['success'][0])}/{int(left['steps'][0])} "
                    f"{other_mode} {int(right['success'][0])}/{int(right['steps'][0])}"
                )
            for key in checkpoints:
                key_worst[key] = max(key_worst[key], maxabs(left[key], right[key]))
            left.close()
            right.close()
        nonzero = {key: value for key, value in key_worst.items() if value != 0.0}
        line(f"fresh vs {other_mode} repeat0 outcome_flips={len(outcome_flip)}")
        for item in outcome_flip:
            line(f"  flip {item}")
        for key, value in nonzero.items():
            line(f"  maxabs {key} {value!r}")

    dest = OUT / "phaseB_cross_mode_summary.txt"
    dest.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
