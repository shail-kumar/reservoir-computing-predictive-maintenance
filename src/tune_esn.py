"""Hyperparameter tuning for the classical ESN.

Same validation-split methodology as tune_ngrc.py - tunes on a split of the
TRAINING units only, real test set untouched. Fixed seed=0, so the grid
search picks structural hyperparameters independent of which random
reservoir draw got lucky; seed-sensitivity of the winning config is checked
separately afterward (see seed_sensitivity_esn.py).

Run from inside src/:  python tune_esn.py
"""

import itertools
import time

import numpy as np

from compare_models import (
    split_calibration,
    truncate_randomly,
    SEED,
    SUBSET,
    MULTI_CONDITION_SUBSETS,
    SENSOR_COLS,
)
from normalization import normalize_by_condition
from data import load_cmapss, add_rul
from esn import EchoStateNetwork
from metrics import rmse, cmapss_score
from sequences import build_sequences, scale_sequences

# spectral_radius/sparsity/leak_rate/ridge_alpha are fixed at their own best
# values from separate sweeps over each; this grid only re-sweeps n_reservoir.
N_RESERVOIR_GRID = [800, 1200, 1600, 2000]
SPECTRAL_RADIUS_GRID = [1.05]
SPARSITY_GRID = [0.9]
LEAK_RATE_GRID = [0.05]
RIDGE_ALPHA_GRID = [1.0]


def evaluate_config(kwargs, n_inputs, fit_seq, fit_tgt, val_seq, val_tgt):
    model = EchoStateNetwork(n_inputs=n_inputs, seed=SEED, **kwargs)
    model.fit(fit_seq, fit_tgt)

    y_true = np.array([t[-1] for t in val_tgt])
    y_pred = model.predict_last(val_seq)
    return rmse(y_true, y_pred), cmapss_score(y_true, y_pred)


def main(subset=SUBSET):
    train, test, _ = load_cmapss(subset)
    train = add_rul(train)

    if subset in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    train_sequences, train_targets, test_sequences, sensors = build_sequences(
        train, test
    )
    if subset not in MULTI_CONDITION_SUBSETS:
        train_sequences, test_sequences, _ = scale_sequences(
            train_sequences, test_sequences
        )
    n_inputs = len(sensors)

    fit_seq, fit_tgt, val_seq, val_tgt = split_calibration(
        train_sequences, train_targets
    )
    val_seq, val_tgt = truncate_randomly(val_seq, val_tgt, seed=SEED + 2)

    grid = itertools.product(
        N_RESERVOIR_GRID,
        SPECTRAL_RADIUS_GRID,
        SPARSITY_GRID,
        LEAK_RATE_GRID,
        RIDGE_ALPHA_GRID,
    )
    results = []
    search_start = time.perf_counter()
    for n_reservoir, spectral_radius, sparsity, leak_rate, ridge_alpha in grid:
        kwargs = dict(
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            leak_rate=leak_rate,
            ridge_alpha=ridge_alpha,
        )
        val_rmse, val_score = evaluate_config(
            kwargs, n_inputs, fit_seq, fit_tgt, val_seq, val_tgt
        )
        results.append((val_score, val_rmse, kwargs))
    search_elapsed = time.perf_counter() - search_start
    print(f"\nSearch done in {search_elapsed:.1f}s", flush=True)

    results.sort(key=lambda r: r[0])

    print(f"Subset: {subset}")
    print(f"{'score':>12}{'rmse':>10}  config")
    for score, val_rmse, kwargs in results[:10]:
        print(f"{score:>12.1f}{val_rmse:>10.2f}  {kwargs}")

    best_score, best_rmse, best_kwargs = results[0]
    print(
        f"\nBest by validation C-MAPSS score: {best_kwargs} "
        f"(val RMSE={best_rmse:.2f}, val score={best_score:.1f})"
    )
    return best_score, best_rmse, best_kwargs


if __name__ == "__main__":
    main()
