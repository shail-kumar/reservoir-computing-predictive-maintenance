"""Per-condition normalization demo chart referenced in WRITEUP.md's Dataset
section, showing the effect on real FD004 data. Not part of the main
pipeline - a one-off visualization aid.

Run from inside src/:  python plot_normalization_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

from data import add_rul, load_cmapss
from normalization import fit_operating_conditions, assign_operating_condition

plt.rcParams["text.usetex"] = True
# Base matplotlib font size (10pt) reads smaller than surrounding body text
# once embedded in the writeup; bump labels/legend/ticks to stay legible.
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14

CONDITION_COLORS = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:purple",
    "tab:brown",
]


def _plot_before_after(
    cycle,
    raw,
    condition,
    condition_labels,
    before_ylabel,
    after_ylabel,
    before_title,
    after_title,
    out_path,
):
    z = np.empty_like(raw, dtype=float)
    for c in np.unique(condition):
        mask = condition == c
        z[mask] = (raw[mask] - raw[mask].mean()) / raw[mask].std()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    for ax, y, ylabel, title in [
        (axes[0], raw, before_ylabel, before_title),
        (axes[1], z, after_ylabel, after_title),
    ]:
        ax.plot(cycle, y, color="lightgray", zorder=1, linewidth=1)
        for i, c in enumerate(np.unique(condition)):
            mask = condition == c
            ax.scatter(
                cycle[mask],
                y[mask],
                color=CONDITION_COLORS[i % len(CONDITION_COLORS)],
                label=condition_labels[i],
                zorder=2,
                s=30,
            )
        ax.set_xlabel("cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    # Capped at 3 columns (wraps to more rows instead) so a many-condition
    # legend never grows wider than the figure itself - past that point,
    # bbox_inches="tight" would expand the saved canvas to fit the legend,
    # making this chart wider than the others despite the same figsize.
    ncol = min(3, len(condition_labels))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_real_data_demo(
    out_path="../results/conditions_normalization_real_data.png",
    subset="FD004",
    unit=1,
    sensor="sensor_2",
):
    train, _, _ = load_cmapss(subset)
    train = add_rul(train)
    kmeans = fit_operating_conditions(train)
    train = assign_operating_condition(train, kmeans)

    unit_df = train[train["unit"] == unit].sort_values("cycle")
    cycle = unit_df["cycle"].to_numpy()
    raw = unit_df[sensor].to_numpy()
    condition = unit_df["condition"].to_numpy()
    condition_labels = [f"Condition {c}" for c in np.unique(condition)]

    sensor_escaped = sensor.replace("_", r"\_")
    _plot_before_after(
        cycle,
        raw,
        condition,
        condition_labels,
        before_ylabel=sensor_escaped,
        after_ylabel=f"{sensor_escaped} (z-score per condition)",
        before_title=f"Before: raw {sensor_escaped}, engine unit {unit} ({subset}, real data)",
        after_title=f"After: {sensor_escaped} normalized within each of the {len(condition_labels)} operating conditions",
        out_path=out_path,
    )


if __name__ == "__main__":
    plot_real_data_demo()
