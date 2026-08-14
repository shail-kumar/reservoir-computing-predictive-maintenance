"""Sanity checks on metrics, model fit/predict, and data loading.

Tests touching the dataset skip when data/ is absent (see README).
"""

import numpy as np
import pytest

from rcpm.conformal import conformal_margin
from rcpm.data import DATA_DIR, add_rul, last_cycle_per_unit, load_cmapss
from rcpm.esn import EchoStateNetwork
from rcpm.metrics import cmapss_score, rmse
from rcpm.ngrc import NGRC

requires_data = pytest.mark.skipif(
    not (DATA_DIR / "train_FD004.txt").is_file(),
    reason="C-MAPSS data not downloaded - see README.md",
)


@pytest.fixture
def synthetic_sequences():
    """Five 20-cycle, 3-sensor sequences with linearly decreasing RUL."""
    rng = np.random.default_rng(0)
    sequences = [rng.normal(size=(20, 3)) for _ in range(5)]
    targets = [np.arange(20)[::-1].astype(float) for _ in range(5)]
    return sequences, targets


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(3.5355339059327378)


def test_cmapss_score_is_asymmetric():
    # d = pred - true. d=-13 -> exp(-d/13)-1 = exp(1)-1; d=10 -> exp(d/10)-1 = exp(1)-1.
    assert cmapss_score([113, 90], [100, 100]) == pytest.approx(2 * (np.exp(1) - 1))

    # Late predictions (overestimating remaining life) must cost more than
    # early ones of the same magnitude.
    assert cmapss_score([100], [110]) > cmapss_score([100], [90])


def test_conformal_margin_covers_outlier():
    errors = np.concatenate([np.full(9, 1.0), [10.0]])  # one outlier in ten
    assert conformal_margin(errors, coverage=0.9) >= 1.0


def test_ngrc_fit_predict(synthetic_sequences):
    sequences, targets = synthetic_sequences
    model = NGRC(n_lags=4, degree=1, ridge_alpha=1.0)
    model.fit(sequences, targets)
    preds = model.predict_last(sequences)

    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_esn_fit_predict(synthetic_sequences):
    sequences, targets = synthetic_sequences
    model = EchoStateNetwork(n_inputs=3, n_reservoir=20, seed=0)
    model.fit(sequences, targets, washout=2)
    preds = model.predict_last(sequences)

    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_esn_is_deterministic_for_a_fixed_seed(synthetic_sequences):
    sequences, targets = synthetic_sequences
    preds = []
    for _ in range(2):
        model = EchoStateNetwork(n_inputs=3, n_reservoir=20, seed=0)
        model.fit(sequences, targets, washout=2)
        preds.append(model.predict_last(sequences))

    np.testing.assert_allclose(preds[0], preds[1])


def test_data_dir_is_a_path():
    # A plain string here silently breaks every default-arg caller of
    # load_cmapss(), which builds paths with '/'.
    assert hasattr(DATA_DIR, "__truediv__")


@requires_data
def test_load_cmapss():
    train, test, test_rul = load_cmapss("FD004")
    assert len(train) > 0 and len(test) > 0 and len(test_rul) > 0


@requires_data
def test_add_rul_clips_target():
    train, _, _ = load_cmapss("FD004")
    train = add_rul(train)
    assert train["RUL"].max() <= 125
    assert train["RUL"].min() >= 0


@requires_data
def test_last_cycle_per_unit_returns_one_row_per_unit():
    train, _, _ = load_cmapss("FD004")
    last = last_cycle_per_unit(train)
    assert len(last) == train["unit"].nunique()
