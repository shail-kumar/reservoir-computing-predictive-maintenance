"""Comparison figures shared by the experiment scripts.

Call `use_writeup_style()` before plotting to match WRITEUP.md. It enables
LaTeX text rendering and so requires a working LaTeX installation.
"""

import matplotlib.pyplot as plt
import numpy as np


def use_writeup_style() -> None:
    """Apply the figure style used in WRITEUP.md."""
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14


def plot_comparison(results, out_path):
    """Scatter each model's C-MAPSS score against its training time."""
    # Same 8-inch width as plot_predictions() and the normalization demo
    # figure, so a fixed point-size font reads consistently across all
    # three once embedded in the writeup at a similar display width.
    fig, ax = plt.subplots(figsize=(8, 6))
    slowest = max(r["train_time"] for r in results)
    for r in results:
        ax.scatter(r["train_time"], r["score"], s=80)
        # The slowest point sits at the right edge of the log-x axis and
        # near the top of the log-y axis (worst score); any right or up
        # offset pushes its label past the plot boundary, so it goes to
        # the left instead, vertically centered on the marker.
        if r["train_time"] == slowest:
            ax.annotate(
                r["name"],
                (r["train_time"], r["score"]),
                textcoords="offset points",
                xytext=(-8, 0),
                ha="right",
                va="center",
            )
        else:
            ax.annotate(
                r["name"],
                (r["train_time"], r["score"]),
                textcoords="offset points",
                xytext=(6, 6),
            )
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("C-MAPSS score (lower is better)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Accuracy vs. training cost")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_predictions(results, out_path):
    """Plot predicted vs. actual RUL, one subplot per model."""
    # All three models predict the same test units in the same order, so one
    # sort order is valid for every subplot. Sharing the y-axis keeps
    # prediction spread comparable model to model.
    order = np.argsort(results[0]["y_test"])
    y_test = results[0]["y_test"][order]
    x = np.arange(len(y_test))

    fig, axes = plt.subplots(
        len(results), 1, figsize=(8, 3.5 * len(results)), sharex=True, sharey=True
    )
    for ax, result in zip(axes, results):
        y_pred = result["y_pred"][order]
        margin = result.get("margin")

        ax.plot(x, y_test, label="Actual RUL", color="black", linewidth=1)
        ax.plot(
            x,
            y_pred,
            label=f"{result['name']} prediction",
            color="tab:blue",
            linewidth=1,
        )
        if margin is not None:
            # Baseline has no conformal calibration, so no interval to draw.
            ax.fill_between(
                x,
                y_pred - margin,
                y_pred + margin,
                alpha=0.2,
                label=f"90\\% conformal interval ($\\pm${margin:.1f})",
            )
        ax.set_ylabel("RUL (cycles)")
        ax.legend()

    axes[-1].set_xlabel("Test engine (sorted by actual RUL)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
