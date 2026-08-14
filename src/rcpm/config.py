"""Shared experiment configuration: subset choice, column names, and the
final tuned hyperparameters for all three models.

WRITEUP.md documents how these values were arrived at; the scripts in
`rcpm.experiments` reproduce the searches.
"""

CALIBRATION_FRACTION = 0.2
SEED = 0
SUBSET = "FD004"

# FD002/FD004 have 6 operating conditions with substantially different raw
# sensor levels per condition - normalize per-condition (normalization.py)
# instead of the single global scaler that's sufficient for FD001/FD003.
MULTI_CONDITION_SUBSETS = {"FD002", "FD004"}

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# Winning RandomizedSearchCV config from tune_baseline.py (30-candidate search).
TUNED_BASELINE_KWARGS = {
    "learning_rate": 0.022169635606490684,
    "max_depth": 7,
    "n_estimators": 153,
    "subsample": 0.7449024601551049,
}

TUNED_ESN_KWARGS = {
    "n_reservoir": 800,
    "spectral_radius": 1.05,
    "sparsity": 0.9,
    "leak_rate": 0.05,
    "ridge_alpha": 1.0,
}

# degree=1 (no polynomial expansion); n_lags=19, the largest value that
# still produces a prediction for every real FD004 test engine (shortest
# test trajectory: 19 cycles).
TUNED_NGRC_KWARGS = {"n_lags": 19, "degree": 1, "ridge_alpha": 100.0}

# Truncated calibration/validation sequences must stay >= n_lags, or NGRC's
# delay embedding returns an empty array.
MIN_TRUNCATE_LEN = TUNED_NGRC_KWARGS["n_lags"] + 4
