"""Evaluation metrics: RMSE and the C-MAPSS asymmetric prognostics score."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cmapss_score(y_true, y_pred) -> float:
    """Asymmetric scoring function from the original C-MAPSS benchmark paper
    (Saxena et al., 2008). Penalizes late predictions (overestimating
    remaining life) more heavily than early ones, since underestimating
    failure risk is the worse business outcome - report this alongside RMSE,
    not instead of it, since RMSE alone hides that asymmetry.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    penalty = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(penalty))
