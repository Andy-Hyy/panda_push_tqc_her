"""Figures for the 2026-10-02 clean-reset spatial retest.

Adaptations relative to data-map-spatial_generalization_original.py,
which is an unchanged copy of the paper script:

1. Reads spatial_fraction_h50.csv and spatial_fraction_h100.csv. It does
   not open the paper file spatial_generalization.csv.
2. The values are success fractions in [0, 1]. The original *100
   conversion still produces a percentage. They are not percent strings.
3. H50 and H100 are two figures because the retest stores two readings.
   They come from one 100-step trajectory per scene, not from 28800
   independent runs.
4. The band is labeled SD. DataFrame.std uses sample standard deviation
   (ddof=1) across the three policies.
5. Each figure is written as a 300 dpi PNG and an SVG. plt.show() is not
   called, so the script can run without a display.
6. The CSV index is the seed number, read by pandas as an integer.
   Look the row up with that integer. f"Seed {seed}" then reads
   "Seed 0". The paper file's index was the text "seed 0".
7. The title names the horizon so the two files can be told apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw(csv_path: Path, horizon: str, out_dir: Path) -> None:
    sns.set(style="darkgrid", font_scale=1.25)
    df = pd.read_csv(csv_path, index_col=0)
    if df.shape != (3, 8):
        raise SystemExit(f"{csv_path.name} shape is {df.shape}, expected 3 policies by 8 lengths")
    goal_ranges = [float(col) for col in df.columns]
    seeds = [int(seed) for seed in df.index]
    plt.figure(figsize=(8, 5))
    palette = sns.color_palette("Set2", n_colors=len(seeds))
    for i, seed in enumerate(seeds):
        plt.plot(
            goal_ranges,
            df.loc[seed].astype(float) * 100,
            marker="o",
            label=f"Seed {seed}",
            color=palette[i],
        )
    mean_curve = df.astype(float).mean(axis=0) * 100
    std_curve = df.astype(float).std(axis=0) * 100
    plt.plot(goal_ranges, mean_curve, color="k", linestyle="--", label="Mean", linewidth=2)
    plt.fill_between(
        goal_ranges,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="gray",
        alpha=0.17,
        label="SD",
    )
    plt.xlabel("Goal XY Range", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title(f"Spatial Generalization Success Rate ({horizon})", fontsize=16)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"spatial_{horizon.lower()}_clean"
    plt.savefig(stem.with_suffix(".png"), dpi=300)
    plt.savefig(stem.with_suffix(".svg"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    draw(args.out / "spatial_fraction_h50.csv", "H50", args.out / "figures")
    draw(args.out / "spatial_fraction_h100.csv", "H100", args.out / "figures")


if __name__ == "__main__":
    main()
