"""Lightweight smoke testsfor quick sanity checks. No pytest dependency: plain asserts, run
directly.

Run from inside src/:  python smoke_test.py
"""

import numpy as np

from conformal import conformal_margin
from data import DATA_DIR, add_rul, last_cycle_per_unit, load_cmapss
from esn import EchoStateNetwork
from metrics import cmapss_score, rmse
from ngrc import NGRC

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except Exception as e:
        FAILURES.append(name)
        print(f"  FAIL  {name}: {e}")


def check_metrics():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == 3.5355339059327378

    # d = pred - true. d=-13 -> exp(-d/13)-1 = exp(1)-1; d=10 -> exp(d/10)-1 = exp(1)-1.
    score = cmapss_score([113, 90], [100, 100])
    assert abs(score - 2 * (np.exp(1) - 1)) < 1e-9


def check_conformal_margin():
    errors = np.concatenate([np.full(9, 1.0), [10.0]])  # one outlier in ten
    margin = conformal_margin(errors, coverage=0.9)
    assert margin > 0
    assert margin >= 1.0


def check_ngrc_fit_predict():
    rng = np.random.default_rng(0)
    sequences = [rng.normal(size=(20, 3)) for _ in range(5)]
    targets = [np.arange(20)[::-1].astype(float) for _ in range(5)]

    model = NGRC(n_lags=4, degree=1, ridge_alpha=1.0)
    model.fit(sequences, targets)
    preds = model.predict_last(sequences)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def check_esn_fit_predict():
    rng = np.random.default_rng(0)
    sequences = [rng.normal(size=(20, 3)) for _ in range(5)]
    targets = [np.arange(20)[::-1].astype(float) for _ in range(5)]

    model = EchoStateNetwork(n_inputs=3, n_reservoir=20, seed=0)
    model.fit(sequences, targets, washout=2)
    preds = model.predict_last(sequences)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def check_data_pipeline():
    assert hasattr(DATA_DIR, "__truediv__"), (
        "DATA_DIR must be a Path (supports '/'), not a plain string - "
        "this exact regression broke every default-arg caller of load_cmapss()"
    )

    train, test, test_rul = load_cmapss("FD004")
    assert len(train) > 0 and len(test) > 0 and len(test_rul) > 0

    train = add_rul(train)
    assert train["RUL"].max() <= 125
    assert train["RUL"].min() >= 0

    last = last_cycle_per_unit(train)
    assert len(last) == train["unit"].nunique()


def main():
    print("Metrics and conformal (no data dependency):")
    check("rmse/cmapss_score formulas", check_metrics)
    check("conformal_margin", check_conformal_margin)

    print("Model fit/predict on synthetic sequences:")
    check("NGRC fit/predict", check_ngrc_fit_predict)
    check("EchoStateNetwork fit/predict", check_esn_fit_predict)

    print("Data pipeline (requires data/ downloaded, see README):")
    try:
        load_cmapss("FD004")
        check("load_cmapss/add_rul/last_cycle_per_unit", check_data_pipeline)
    except FileNotFoundError:
        print("  skip  data/ not found - download it first (see README.md)")

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
