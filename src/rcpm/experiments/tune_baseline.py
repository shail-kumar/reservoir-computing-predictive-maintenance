"""Hyperparameter tuning for the GBM baseline via randomized search.

Uses the same single train/validation split methodology as tune_esn.py/
tune_ngrc.py (unit-level split, seed=0, fraction=0.2) -- NOT k-fold, for
consistency with the rest of the project's tuning. Parallelized across all
cores via RandomizedSearchCV's n_jobs=-1, since GBM candidates are
independent of each other even though a single GBM fit itself is sequential
(no internal parallelism).

Search ranges are literature-informed (common well-supported GBM ranges),
not copied from any specific paper's exact hyperparameters: different
libraries (LightGBM/CatBoost vs sklearn's GBM) and different feature
pipelines mean literature hyperparameters don't reliably transfer, only the
general ranges are useful.

python -m rcpm.experiments.tune_baseline
"""

import time

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from rcpm.config import MULTI_CONDITION_SUBSETS, SENSOR_COLS, SUBSET
from rcpm.data import add_rul, last_cycle_per_unit, load_cmapss
from rcpm.features import build_features
from rcpm.metrics import cmapss_score, rmse
from rcpm.normalization import normalize_by_condition

SEED = 0
CALIBRATION_FRACTION = 0.2
N_ITER = 30

PARAM_DISTRIBUTIONS = {
    "learning_rate": uniform(0.01, 0.09),  # 0.01-0.10
    "max_depth": randint(3, 9),  # 3-8
    "n_estimators": randint(100, 501),  # 100-500
    "subsample": uniform(0.7, 0.3),  # 0.7-1.0
}


def split_units(train, fraction=CALIBRATION_FRACTION, seed=SEED):
    """Same unit-level split methodology as compare_models.py's
    split_calibration, adapted for a row/unit dataframe instead of a
    sequences list."""
    units = train["unit"].unique()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(units))
    n_cal = max(1, int(len(units) * fraction))
    cal_units = set(units[idx[:n_cal]])
    fit_units = set(units[idx[n_cal:]])
    return fit_units, cal_units


def main():
    train, test, test_rul = load_cmapss(SUBSET)
    train = add_rul(train)
    if SUBSET in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    fit_units, cal_units = split_units(train)

    train_feat, test_feat, feature_cols = build_features(train, test)

    is_cal = train_feat["unit"].isin(cal_units)
    X = train_feat[feature_cols]
    y = train_feat["RUL"]
    # PredefinedSplit: -1 marks a row as always-train, 0 marks it as the
    # single validation fold -- this makes RandomizedSearchCV do exactly
    # ONE train/validation split per candidate, not k-fold.
    test_fold = np.where(is_cal, 0, -1)
    ps = PredefinedSplit(test_fold)

    scorer = make_scorer(cmapss_score, greater_is_better=False)

    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=ps,
        scoring=scorer,
        n_jobs=-1,
        random_state=SEED,
        verbose=2,
    )

    print(
        f"Starting randomized search: {N_ITER} configs, single train/val split, all cores",
        flush=True,
    )
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    print(f"\nSearch done in {elapsed:.1f}s", flush=True)
    print(f"Best params: {search.best_params_}")
    print(f"Best val C-MAPSS score: {-search.best_score_:.1f}")

    # Final: refit on ALL training data (fit+cal) with the best params, then
    # evaluate on the real, held-out test set.
    final_model = GradientBoostingRegressor(random_state=SEED, **search.best_params_)
    start = time.perf_counter()
    final_model.fit(X, y)
    train_time = time.perf_counter() - start

    X_test = last_cycle_per_unit(test_feat)[feature_cols]
    y_test = test_rul["RUL"].clip(upper=125)
    y_pred = final_model.predict(X_test)

    print("\nFinal tuned baseline (train on all data, real test set):")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  RMSE:          {rmse(y_test, y_pred):.2f}")
    print(f"  C-MAPSS score: {cmapss_score(y_test, y_pred):.1f}")
    print(
        "\nFor reference, current untuned baseline (n_estimators=100, "
        "learning_rate=0.1, max_depth=3): RMSE=59.56, score=1,100,330.0"
    )


if __name__ == "__main__":
    main()
