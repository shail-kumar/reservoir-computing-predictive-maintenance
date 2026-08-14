"""Redraw figures from a saved compare_models run, without refitting models.

Run compare_models at least once first to produce the results_*.pkl.

    python -m rcpm.experiments.plot_results
"""

import pickle

from rcpm.config import SUBSET
from rcpm.paths import result_path
from rcpm.plotting import plot_comparison, plot_predictions, use_writeup_style


def main(subset=SUBSET):
    with open(result_path(f"results_{subset}_nlags-19.pkl"), "rb") as f:
        results = pickle.load(f)

    use_writeup_style()
    plot_comparison(
        results, result_path(f"accuracy_vs_train_time_{subset}_nlags-19.png")
    )
    plot_predictions(results, result_path(f"predictions_{subset}_nlags-19.png"))


if __name__ == "__main__":
    main()
