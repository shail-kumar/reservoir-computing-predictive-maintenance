"""One-off check: does swapping the baseline's model family to LightGBM
(defaults) change anything on the identical feature set? See "Tuning the
baseline" in WRITEUP.md for why this was tested and what it means.

python -m rcpm.experiments.lightgbm_baseline
"""

import time

from lightgbm import LGBMRegressor

from rcpm.config import MULTI_CONDITION_SUBSETS, SENSOR_COLS, SUBSET
from rcpm.data import add_rul, last_cycle_per_unit, load_cmapss
from rcpm.features import build_features
from rcpm.metrics import cmapss_score, rmse
from rcpm.normalization import normalize_by_condition


def main():
    train, test, test_rul = load_cmapss(SUBSET)
    train = add_rul(train)
    if SUBSET in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    train_feat, test_feat, feature_cols = build_features(train, test)
    X_train = train_feat[feature_cols]
    y_train = train_feat["RUL"]
    X_test = last_cycle_per_unit(test_feat)[feature_cols]
    y_test = test_rul["RUL"].clip(upper=125)

    model = LGBMRegressor(random_state=0, verbose=-1)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)

    print(f"Subset: {SUBSET}")
    print("LightGBM baseline (defaults, identical feature set as baseline.py)")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  RMSE:          {rmse(y_test, y_pred):.2f}")
    print(f"  C-MAPSS score: {cmapss_score(y_test, y_pred):.1f}")


if __name__ == "__main__":
    main()
