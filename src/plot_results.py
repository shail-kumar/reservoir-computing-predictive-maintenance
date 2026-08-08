"""Regenerate charts from a saved compare_models.py run, without refitting
any models. Lets chart appearance (labels, layout, colors) be tweaked
quickly - the baseline alone takes ~80s to fit, so redrawing from cached
results instead of rerunning compare_models.py every time saves real time.

Run compare_models.py at least once first to produce the results_*.pkl
this script reads.

Run from inside src/:  python plot_results.py
"""

import pickle

from compare_models import SUBSET, plot_comparison, plot_predictions


def main(subset=SUBSET):
    with open(f"../results/results_{subset}_nlags-19.pkl", "rb") as f:
        results = pickle.load(f)

    plot_comparison(results, f"../results/accuracy_vs_train_time_{subset}_nlags-19.png")
    plot_predictions(results, f"../results/predictions_{subset}_nlags-19.png")


if __name__ == "__main__":
    main()
