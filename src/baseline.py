"""Gradient boosting regression on engineered features - the conventional
feature-engineering + tree-ensemble reference point against which ESN and
NG-RC are compared in compare_models.py, on both accuracy and training time.

Run from inside src/:  python baseline.py
"""

import time

from sklearn.ensemble import GradientBoostingRegressor

from data import load_fd001, add_rul, last_cycle_per_unit
from features import build_features
from metrics import rmse, cmapss_score


def run_baseline(train, test, test_rul, model_kwargs=None):
    """Fit the GradientBoostingRegressor baseline and return a result dict
    shared with the ESN/NG-RC comparison in compare_models.py."""
    train_feat, test_feat, feature_cols = build_features(train, test)

    X_train = train_feat[feature_cols]
    y_train = train_feat["RUL"]

    X_test = last_cycle_per_unit(test_feat)[feature_cols]
    y_test = test_rul["RUL"].clip(upper=125)

    model = GradientBoostingRegressor(random_state=0, **(model_kwargs or {}))

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)

    return {
        "name": "Baseline (GBM)" if model_kwargs else "Baseline (GBM, untuned)",
        "train_time": train_time,
        "rmse": rmse(y_test, y_pred),
        "score": cmapss_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test.to_numpy(),
    }


def main():
    train, test, test_rul = load_fd001()
    train = add_rul(train)
    result = run_baseline(train, test, test_rul)

    print("Baseline (GradientBoostingRegressor)")
    print(f"  Training time: {result['train_time']:.2f}s")
    print(f"  RMSE:          {result['rmse']:.2f}")
    print(f"  C-MAPSS score: {result['score']:.1f}")


if __name__ == "__main__":
    main()
