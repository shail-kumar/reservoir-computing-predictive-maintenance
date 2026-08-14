"""EDA: sensor trends, informative vs. near-constant channels."""

import matplotlib.pyplot as plt

from rcpm.data import add_rul, load_fd001
from rcpm.paths import result_path

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def informative_sensors(train, std_threshold: float = 1e-8):
    """Drop sensors with near-zero variance - weak signal and mostly noise.
    Computed from the data, so it adapts across subsets automatically.

    std_threshold: A tiny threshold that runs on data at
    different scales depending on subset (raw vs. normalized), so it needs
    to sit safely below real signal at either scale, not just "small".
    """
    stds = train[SENSOR_COLS].std()
    kept = stds[stds > std_threshold].index.tolist()
    dropped = stds[stds <= std_threshold].index.tolist()
    print(f"Kept {len(kept)} sensors, dropped as near-constant: {dropped}")
    return kept


def plot_unit_trajectory(train, unit: int, sensors, out_path):
    unit_df = train[train["unit"] == unit]
    fig, axes = plt.subplots(
        len(sensors), 1, figsize=(8, 2 * len(sensors)), sharex=True
    )
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        ax.plot(unit_df["cycle"], unit_df[sensor])
        ax.set_ylabel(sensor)
    axes[-1].set_xlabel("cycle")
    fig.suptitle(f"Unit {unit} sensor trajectories")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    train, test, test_rul = load_fd001()
    train = add_rul(train)
    kept = informative_sensors(train)
    plot_unit_trajectory(
        train, unit=1, sensors=kept[:6], out_path=result_path("unit1_sensors.png")
    )
