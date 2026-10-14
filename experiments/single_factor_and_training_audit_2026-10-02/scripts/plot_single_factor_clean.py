"""Four single-factor figures for the 2026-10-02 clean-reset retest.

Adaptations relative to original_physical_plot.py, an unchanged copy of the
paper script data-map-single-phy-element-disturbance.py:

1. Reads single_factor_rates.csv. It does not open the June eval_phy_*.csv
   files or the historical summary tables.
2. Mass 1 kg, sliding friction 1, and gain scale 1 are the reused P0-v3
   rates already stored in that file. The axis text says so. Damping scale
   1 is P0 and is not a point on the damping figure.
3. Axis text names the quantity that was actually scaled. Damping is the
   full dof_damping array. Gain is actuator_gainprm column 0. Friction is
   geom component 0.
4. The error bar is the sample standard deviation (ddof=1) of the three
   policy rates, in percentage points, matching DataFrame.std in the
   original script. It is not a confidence interval.
5. Each figure is a 300 dpi PNG and an SVG. plt.show() is not called.
6. Policy seeds stay integers, so the legend reads "Seed 0".
7. The y-axis remains 0 to 110, the original Figure 1 range.
8. After drawing, the plotted means are checked against
   single_factor_summary.csv. A mismatch raises SystemExit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PANELS = (
    (
        "object_mass",
        "object_mass",
        "Object mass (kg), same-shape inertia\nMass = 1 kg is P0-v3, not rerun",
        "Success Rate under Object Mass",
    ),
    (
        "object_friction",
        "object_friction",
        "Sliding friction, geom component 0\nFriction = 1 is P0-v3, not rerun",
        "Success Rate under Sliding Friction",
    ),
    (
        "joint_damping_scale",
        "dof_damping_scale",
        "dof_damping scale, all 15 DoFs\nNominal scale 1 is P0 and is outside this grid",
        "Success Rate under dof_damping Scale",
    ),
    (
        "actuator_gain_scale",
        "actuator_gainprm0_scale",
        "Actuator gainprm[:,0] scale\nGain scale = 1 is P0-v3, not rerun",
        "Success Rate under Gainprm Column 0",
    ),
)


def draw(rates: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set(style="darkgrid", font_scale=1.25)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotted = []
    for factor, stem, xlabel, title in PANELS:
        part = rates[rates["factor"] == factor].copy()
        if part.empty:
            raise SystemExit(f"no rows for {factor}")
        plt.figure(figsize=(8, 5))
        for seed in sorted(part["policy_seed"].unique()):
            seed_part = part[part["policy_seed"] == seed].sort_values("value_float")
            plt.plot(
                seed_part["value_float"],
                seed_part["success_rate"] * 100,
                marker="o",
                label=f"Seed {int(seed)}",
                alpha=0.7,
            )
        grouped = part.groupby("value_float")["success_rate"]
        mean = grouped.mean() * 100
        std = grouped.std(ddof=1) * 100
        plt.errorbar(
            mean.index,
            mean,
            yerr=std,
            fmt="o--",
            color="black",
            capsize=4,
            label="Mean ± Std",
        )
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=13)
        plt.title(title, fontsize=15)
        plt.ylim(0, 110)
        plt.legend()
        plt.tight_layout()
        target = out_dir / stem
        plt.savefig(target.with_suffix(".png"), dpi=300)
        plt.savefig(target.with_suffix(".svg"))
        plt.close()
        reference = summary[summary["factor"] == factor].set_index("value_float")
        for value, mean_pct in mean.items():
            recorded = float(reference.loc[value, "mean_pct"])
            recorded_sd = float(reference.loc[value, "sd_pct"])
            if abs(recorded - float(mean_pct)) > 1e-9 or abs(recorded_sd - float(std.loc[value])) > 1e-9:
                raise SystemExit(f"{factor}={value} plot {mean_pct} != summary {recorded}")
            plotted.append(
                {
                    "factor": factor,
                    "value_float": value,
                    "mean_pct": float(mean_pct),
                    "sd_pct": float(std.loc[value]),
                }
            )
    pd.DataFrame(plotted).to_csv(out_dir / "figure_values.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rates = pd.read_csv(args.root / "single_factor_rates.csv")
    summary = pd.read_csv(args.root / "single_factor_summary.csv")
    draw(rates, summary, args.root / "figures")


if __name__ == "__main__":
    main()
