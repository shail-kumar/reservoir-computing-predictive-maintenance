"""One-off check: does swapping the baseline's model family to LightGBM
(defaults) change anything on the identical feature set? See "Tuning the
baseline" in WRITEUP.md for why this was tested and what it means.

Run from inside src/:  python lightgbm_baseline.py
"""

import time

from lightgbm import LGBMRegressor

from compare_models import SUBSET, MULTI_CONDITION_SUBSETS, SENSOR_COLS
from normalization import normalize_by_condition
from data import load_cmapss, add_rul, last_cycle_per_unit
from features import build_features
from metrics import rmse, cmapss_score


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
