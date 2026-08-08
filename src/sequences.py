"""Per-engine sequence construction shared by the ESN and NG-RC models.

Unlike the baseline's windowed/aggregated feature table, these models
use the raw per-cycle sensor sequence directly - the ESN's reservoir
state and NG-RC's delay embedding are what supply the temporal memory.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from eda import informative_sensors


def build_sequences(train, test):
    """Return (train_sequences, train_targets, test_sequences, sensor_cols).

    train_sequences: list of (T_i, n_sensors) arrays, one per training unit
    train_targets:   list of (T_i,) RUL arrays, aligned with train_sequences
    test_sequences:  list of (T_i, n_sensors) arrays, one per test unit,
                      each truncated before failure as in the real dataset
    """
    sensors = informative_sensors(train)

    train_sequences, train_targets = [], []
    for _, df in train.sort_values(["unit", "cycle"]).groupby("unit"):
        train_sequences.append(df[sensors].to_numpy())
        train_targets.append(df["RUL"].to_numpy())

    test_sequences = []
    for _, df in test.sort_values(["unit", "cycle"]).groupby("unit"):
        test_sequences.append(df[sensors].to_numpy())

    return train_sequences, train_targets, test_sequences, sensors


def scale_sequences(train_sequences, test_sequences):
    """Fit a StandardScaler on all training rows pooled together, then apply
    it consistently to both train and test sequences. Matters a lot here:
    raw sensor scales differ by orders of magnitude, which would saturate
    the ESN's tanh nonlinearity and let large-scale sensors dominate NG-RC's
    polynomial features.
    """
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_sequences))
    train_scaled = [scaler.transform(seq) for seq in train_sequences]
    test_scaled = [scaler.transform(seq) for seq in test_sequences]
    return train_scaled, test_scaled, scaler
