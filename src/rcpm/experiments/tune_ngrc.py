"""Hyperparameter tuning for NG-RC.

Tunes on a validation split carved out of the TRAINING units only - the real
test set is never touched here. Touching it during tuning would leak test
information into model selection and make the final comparison numbers
optimistic/invalid.

python -m rcpm.experiments.tune_ngrc
"""

import itertools
import math
import time

import numpy as np

from rcpm.config import MULTI_CONDITION_SUBSETS, SEED, SENSOR_COLS, SUBSET
from rcpm.data import add_rul, load_cmapss
from rcpm.metrics import cmapss_score, rmse
from rcpm.ngrc import NGRC
from rcpm.normalization import normalize_by_condition
from rcpm.sequences import build_sequences, scale_sequences
from rcpm.splits import split_calibration, truncate_randomly

# Predates the eventual final n_lags choice (a later, separate follow-up).
N_LAGS_GRID = [8, 10, 12, 16]
DEGREE_GRID = [1, 2]
RIDGE_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# To reproduce WRITEUP.md's Table 4 (n_lags=2-16, degree=1/ridge_alpha=100
# fixed) instead, comment out the grid above and uncomment this:
# N_LAGS_GRID = list(range(2, 17))
# DEGREE_GRID = [1]
# RIDGE_ALPHA_GRID = [100.0]

# This grid's own max n_lags sets the truncation floor (not
# TUNED_NGRC_KWARGS's), or NGRC's delay embedding returns empty rows for
# the shortest truncations.
MIN_TRUNCATE_LEN = max(N_LAGS_GRID) + 4

# Cap on expanded polynomial features. PolynomialFeatures with degree d on
# k linear inputs produces C(k+d, d) - 1 features, growing combinatorially
# (degree=3 at n_lags=8 alone is ~300,000 features). A dense array that
# size is tens of GB, and a true OOM gets SIGKILL'd before Python can catch
# it - skip these analytically before ever calling fit_transform.
#
# NGRC's Ridge(solver='lsqr') avoids the (features x features)
# normal-equations matrix, so the ceiling is set by the design matrix alone
# (~13.5GB for n_lags=16/degree=2's 37,400 features, within this machine's
# ~23GB free). degree=3 stays excluded regardless (187k-438k features even
# at n_lags=6 - the design matrix alone would exceed available memory).
MAX_POLY_FEATURES = 40000


def poly_feature_count(n_linear: int, degree: int) -> int:
    return math.comb(n_linear + degree, degree) - 1


def evaluate_config(n_lags, degree, ridge_alpha, fit_seq, fit_tgt, val_seq, val_tgt):
    model = NGRC(n_lags=n_lags, degree=degree, ridge_alpha=ridge_alpha)
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

    # Reuse the same train/calibration split machinery as compare_models.py,
    # but here the "calibration" split plays the role of a tuning validation
    # set - a separate, later conformal calibration still happens fresh
    # inside compare_models.py using the chosen config.
    fit_seq, fit_tgt, val_seq, val_tgt = split_calibration(
        train_sequences, train_targets
    )
    val_seq, val_tgt = truncate_randomly(
        val_seq, val_tgt, seed=SEED + 2, min_len=MIN_TRUNCATE_LEN
    )

    n_sensors = len(sensors)
    grid = list(itertools.product(N_LAGS_GRID, DEGREE_GRID, RIDGE_ALPHA_GRID))
    print(
        f"Sweeping {len(grid)} configs (some may be skipped by the feature cap)",
        flush=True,
    )

    results = []
    search_start = time.perf_counter()
    for i, (n_lags, degree, ridge_alpha) in enumerate(grid, 1):
        n_linear = n_sensors * n_lags
        n_features = poly_feature_count(n_linear, degree)
        if n_features > MAX_POLY_FEATURES:
            print(
                f"  [{i}/{len(grid)}] skipped n_lags={n_lags} degree={degree} alpha={ridge_alpha}: "
                f"{n_features:,} polynomial features exceeds cap ({MAX_POLY_FEATURES:,})",
                flush=True,
            )
            continue

        start = time.perf_counter()
        val_rmse, val_score = evaluate_config(
            n_lags, degree, ridge_alpha, fit_seq, fit_tgt, val_seq, val_tgt
        )
        elapsed = time.perf_counter() - start
        print(
            f"  [{i}/{len(grid)}] n_lags={n_lags} degree={degree} alpha={ridge_alpha} "
            f"({n_features:,} features): rmse={val_rmse:.2f} score={val_score:.1f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        results.append((val_score, val_rmse, n_lags, degree, ridge_alpha))
    search_elapsed = time.perf_counter() - search_start
    print(f"\nSearch done in {search_elapsed:.1f}s", flush=True)

    results.sort(key=lambda r: r[0])

    print(f"Subset: {subset}")
    print(f"{'score':>12}{'rmse':>10}{'n_lags':>8}{'degree':>8}{'ridge_alpha':>12}")
    for score, val_rmse, n_lags, degree, ridge_alpha in results[:10]:
        print(f"{score:>12.1f}{val_rmse:>10.2f}{n_lags:>8}{degree:>8}{ridge_alpha:>12}")

    best = results[0]
    print(
        f"\nBest by validation C-MAPSS score: n_lags={best[2]}, degree={best[3]}, "
        f"ridge_alpha={best[4]} (val RMSE={best[1]:.2f}, val score={best[0]:.1f})"
    )
    return best


if __name__ == "__main__":
    main()
