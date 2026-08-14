"""Unit-level splitting for conformal calibration."""

import numpy as np

from rcpm.config import CALIBRATION_FRACTION, MIN_TRUNCATE_LEN, SEED


def split_calibration(sequences, targets, fraction=CALIBRATION_FRACTION, seed=SEED):
    """Hold out a subset of training units (not timesteps) for conformal
    calibration - the model must never have trained on these."""
    rng = np.random.default_rng(seed)
    n = len(sequences)
    idx = rng.permutation(n)
    n_cal = max(1, int(n * fraction))
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    fit_seq = [sequences[i] for i in fit_idx]
    fit_tgt = [targets[i] for i in fit_idx]
    cal_seq = [sequences[i] for i in cal_idx]
    cal_tgt = [targets[i] for i in cal_idx]
    return fit_seq, fit_tgt, cal_seq, cal_tgt


def truncate_randomly(sequences, targets, seed, min_len=MIN_TRUNCATE_LEN):
    """Truncate each sequence/target pair at a random cycle, matching how test
    trajectories are cut off before failure. Keeps calibration residuals
    exchangeable with the test-time task.

    min_len must stay >= the model's n_lags, or the delay embedding comes
    back empty.
    """
    rng = np.random.default_rng(seed)
    trunc_seq, trunc_tgt = [], []
    for seq, tgt in zip(sequences, targets):
        cutoff = (
            len(seq) if len(seq) <= min_len else rng.integers(min_len, len(seq) + 1)
        )
        trunc_seq.append(seq[:cutoff])
        trunc_tgt.append(tgt[:cutoff])
    return trunc_seq, trunc_tgt
