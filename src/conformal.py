"""Split-conformal prediction intervals.

A simple, well-founded way to attach an uncertainty band to a point
prediction: hold out a calibration set the model never trained on, measure
its residuals, and use an empirical quantile of |residual| as a symmetric
margin. Under exchangeability of calibration and test errors this gives a
distribution-free coverage guarantee (Vovk et al., 2005; Lei et al., 2018 for the
split-conformal variant used here, see WRITEUP.md References).
"""

import numpy as np


def conformal_margin(calibration_errors, coverage=0.9):
    """calibration_errors: array of (y_pred - y_true) on a held-out
    calibration set. Returns the symmetric margin m such that
    [pred - m, pred + m] targets the given coverage level.
    """
    abs_errors = np.abs(calibration_errors)
    n = len(abs_errors)
    # finite-sample-corrected quantile level, standard split-conformal formula
    q_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(abs_errors, q_level))
