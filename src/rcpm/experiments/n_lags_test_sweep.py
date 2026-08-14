"""NG-RC's real-test-set score across n_lags (degree=1/ridge_alpha=100) - the
actual held-out benchmark, not a validation-split or k-fold proxy. Test
engines shorter than the largest n_lags tested can't produce a prediction at
all, so the test subset is fixed up front to units long enough for the
grid's max n_lags, keeping every row's population identical.

python -m rcpm.experiments.n_lags_test_sweep
"""

import time

import numpy as np

from rcpm.config import MULTI_CONDITION_SUBSETS, SENSOR_COLS, SUBSET
from rcpm.data import add_rul, load_cmapss
from rcpm.metrics import cmapss_score, rmse
from rcpm.ngrc import NGRC
from rcpm.normalization import normalize_by_condition
from rcpm.sequences import build_sequences

N_LAGS_GRID = list(range(2, 41))
DEGREE = 1
RIDGE_ALPHA = 100.0


def main(subset=SUBSET):
    train, test, test_rul = load_cmapss(subset)
    train = add_rul(train)
    y_test_full = test_rul["RUL"].clip(upper=125).to_numpy()

    if subset in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    train_sequences, train_targets, test_sequences, sensors = build_sequences(
        train, test
    )

    max_n_lags = max(N_LAGS_GRID)
    long_enough = [len(s) >= max_n_lags for s in test_sequences]
    n_dropped = len(test_sequences) - sum(long_enough)
    test_sequences = [s for s, keep in zip(test_sequences, long_enough) if keep]
    y_test = y_test_full[np.array(long_enough)]
    print(
        f"Fixed test subset: {len(test_sequences)}/{len(long_enough)} units "
        f"(>= {max_n_lags} cycles long; {n_dropped} dropped for this whole sweep)"
    )

    print(
        f"Subset: {subset}, degree={DEGREE}, ridge_alpha={RIDGE_ALPHA}, real test set"
    )
    results = {}
    search_start = time.perf_counter()
    for n_lags in N_LAGS_GRID:
        model = NGRC(n_lags=n_lags, degree=DEGREE, ridge_alpha=RIDGE_ALPHA)
        start = time.perf_counter()
        model.fit(train_sequences, train_targets)
        train_time = time.perf_counter() - start

        y_pred = model.predict_last(test_sequences)
        test_rmse = rmse(y_test, y_pred)
        test_score = cmapss_score(y_test, y_pred)
        results[n_lags] = {
            "rmse": test_rmse,
            "score": test_score,
            "train_time": train_time,
        }
        print(
            f"  n_lags={n_lags:3d}: rmse={test_rmse:.2f}  score={test_score:.1f}  "
            f"train_time={train_time:.2f}s",
            flush=True,
        )
    search_elapsed = time.perf_counter() - search_start
    print(f"\nSearch done in {search_elapsed:.1f}s", flush=True)

    best = min(results, key=lambda k: results[k]["score"])
    print(f"\nBest by real test-set C-MAPSS score: n_lags={best}")
    return results


if __name__ == "__main__":
    main()
