"""Duplicated from ../src/metrics.py -- see data_utils.py's note on why."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cmapss_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred) - np.asarray(y_true)
    penalty = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(penalty))
