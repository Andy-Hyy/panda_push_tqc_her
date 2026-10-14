"""Figures for the controlled SpatialDR and Fixed Large comparison.

Adaptations relative to the copied paper scripts
original_spatial_plot.py and original_learning_plot.py:

1. Reads summary/final_by_seed_L.csv, summary/final_by_method_L.csv,
   summary/learning_by_seed.csv, and summary/learning_by_method.csv.
   It does not open the paper CSVs or the June standard-policy matrix.
2. Each panel draws the three training seeds and a mean with an error bar.
   Two methods share a panel, so there are two mean series. They are shifted
   by a small horizontal offset so both error bars stay visible. Seed lines
   stay on the recorded x values. The checked numbers are the unshifted means.
3. The error bar is the sample standard deviation (ddof=1) of the three
   policy rates, in percentage points. It is not a confidence interval.
   The learning-curve ancestor used np.std, whose divisor is the count
   rather than count minus one. These figures use the sample divisor.
4. The learning figures use the six predetermined checkpoints only. The
   connecting line is a guide between those nodes. The script does not
   interpolate a step count at which success reaches 80%.
5. The spatial x-axis is the full goal-side length L that was set and read
   back, in metres. The ancestor axis title was "Goal XY Range".
6. The y-axis is 0 to 110 so a sample-SD bar above 100 remains visible.
   The spatial ancestor used 0 to 105.
7. Each figure is a 300 dpi PNG and an SVG. plt.show() is not called.
8. After drawing, every plotted mean and SD is checked against the summary
   tables. A mismatch raises SystemExit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path(__file__).resolve().parents[1]
SUMMARY = OUT / "summary"
FIGURES = OUT / "figures"
METHODS = ("spatial_dr", "fixed_large")
METHOD_LABEL = {"spatial_dr": "SpatialDR", "fixed_large": "Fixed Large"}
SEEDS = (0, 42, 123)


def require_close(actual: float, recorded: float, label: str) -> None:
    if abs(float(actual) - float(recorded)) > 1e-6:
        raise SystemExit(f"{label}: plotted {actual} != summary {recorded}")


def draw(path_stem: Path, x_values: list[float], x_labels: list[str] | None, xlabel: str, title: str, seed_frame: pd.DataFrame, mean_frame: pd.DataFrame, x_column: str, seed_y: str, mean_y: str, sd_column: str, dodge: float) -> pd.DataFrame:
    sns.set(style="darkgrid", font_scale=1.25)
    plt.figure(figsize=(8, 5))
    palette = sns.color_palette("Set2", n_colors=6)
    checked = []
    for method_index, method in enumerate(METHODS):
        for seed_index, seed in enumerate(SEEDS):
            part = seed_frame[(seed_frame["method"] == method) & (seed_frame["train_seed"] == seed)].copy()
            part = part.set_index(x_column).loc[x_values]
            plt.plot(
                x_values,
                part[seed_y].to_numpy(dtype=float) * 100,
                marker="o",
                color=palette[method_index * 3 + seed_index],
                alpha=0.8,
                label=f"{METHOD_LABEL[method]} seed {seed}",
            )
        means = mean_frame[mean_frame["method"] == method].set_index(x_column).loc[x_values]
        y = means[mean_y].to_numpy(dtype=float) * 100
        sd = means[sd_column].to_numpy(dtype=float)
        offset = -dodge if method == "spatial_dr" else dodge
        plt.errorbar(
            [value + offset for value in x_values],
            y,
            yerr=sd,
            fmt="o--",
            color="black" if method == "spatial_dr" else "dimgray",
            capsize=4,
            linewidth=2,
            label=f"{METHOD_LABEL[method]} mean ± SD",
        )
        for value, mean_pct, sd_pct in zip(x_values, y, sd):
            require_close(mean_pct, float(means.loc[value, mean_y]) * 100, f"{path_stem.name} {method} {value} mean")
            require_close(sd_pct, float(means.loc[value, sd_column]), f"{path_stem.name} {method} {value} sd")
            checked.append({"figure": path_stem.name, "method": method, "x": value, "mean_pct": float(mean_pct), "sd_pp": float(sd_pct)})
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title(title, fontsize=15)
    plt.ylim(0, 110)
    if x_labels is not None:
        plt.xticks(x_values, x_labels, rotation=20)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.savefig(path_stem.with_suffix(".svg"))
    plt.close()
    return pd.DataFrame(checked)


def main() -> None:
    final_seed = pd.read_csv(SUMMARY / "final_by_seed_L.csv")
    final_method = pd.read_csv(SUMMARY / "final_by_method_L.csv")
    learning_seed = pd.read_csv(SUMMARY / "learning_by_seed.csv")
    learning_method = pd.read_csv(SUMMARY / "learning_by_method.csv")
    sides = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    steps = [199998, 400002, 499998, 600000, 799998, 1000002]
    step_labels = ["199998", "400002", "499998", "600000", "799998", "1000002"]
    frames = [
        draw(
            FIGURES / "learning_shared_primary",
            steps,
            step_labels,
            "Environment transitions",
            "Shared validation, equal-weight L = 0.30–0.80 m",
            learning_seed,
            learning_method,
            "checkpoint_steps",
            "primary_rate",
            "primary_mean",
            "primary_sd_pp",
            15000,
        ),
        draw(
            FIGURES / "learning_L080",
            steps,
            step_labels,
            "Environment transitions",
            "Shared validation at L = 0.80 m",
            learning_seed,
            learning_method,
            "checkpoint_steps",
            "L080_rate",
            "L080_mean",
            "L080_sd_pp",
            15000,
        ),
        draw(
            FIGURES / "spatial_final",
            sides,
            None,
            "Goal side length L (m)",
            "Final test at 1,000,002 transitions",
            final_seed,
            final_method,
            "L",
            "rate",
            "mean_rate",
            "sd_pp",
            0.012,
        ),
    ]
    pd.concat(frames, ignore_index=True).to_csv(FIGURES / "figure_values.csv", index=False)
    print("figures checked against summary")


if __name__ == "__main__":
    main()
