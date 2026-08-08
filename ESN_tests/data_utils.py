"""Data loading/preprocessing for ESN_tests. Duplicated from ../src/, not
imported, so this directory stays fully independent."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
N_OPERATING_CONDITIONS = 6  # FD004-specific, see ../src/normalization.py's note on this constant


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS
    return df


def load_fd004():
    train = _read_cmapss_file(DATA_DIR / "train_FD004.txt")
    test = _read_cmapss_file(DATA_DIR / "test_FD004.txt")
    test_rul = pd.read_csv(DATA_DIR / "RUL_FD004.txt", header=None, names=["RUL"])
    return train, test, test_rul


def add_rul(train: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=clip)
    return train


def informative_sensors(train, std_threshold: float = 1e-3):
    stds = train[SENSOR_COLS].std()
    return stds[stds > std_threshold].index.tolist()


def normalize_by_condition(train, test, sensor_cols):
    kmeans = KMeans(n_clusters=N_OPERATING_CONDITIONS, n_init=10, random_state=0)
    kmeans.fit(train[OP_SETTING_COLS])

    def assign(df):
        df = df.copy()
        df["condition"] = kmeans.predict(df[OP_SETTING_COLS])
        return df

    train, test = assign(train), assign(test)
    stats = train.groupby("condition")[sensor_cols].agg(["mean", "std"])

    def apply_norm(df):
        df = df.copy()
        df[sensor_cols] = df[sensor_cols].astype(float)
        for cond, group in df.groupby("condition"):
            idx = group.index
            for col in sensor_cols:
                mean = stats.loc[cond, (col, "mean")]
                std = stats.loc[cond, (col, "std")]
                std = std if std > 1e-8 else 1.0
                df.loc[idx, col] = (df.loc[idx, col] - mean) / std
        return df

    return apply_norm(train), apply_norm(test)


def build_sequences(train, test, sensors):
    train_sequences, train_targets = [], []
    for _, df in train.sort_values(["unit", "cycle"]).groupby("unit"):
        train_sequences.append(df[sensors].to_numpy())
        train_targets.append(df["RUL"].to_numpy())

    test_sequences = []
    for _, df in test.sort_values(["unit", "cycle"]).groupby("unit"):
        test_sequences.append(df[sensors].to_numpy())

    return train_sequences, train_targets, test_sequences


def split_by_unit(sequences, targets, fraction=0.2, seed=0):
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


def truncate_to_length(sequences, targets, length):
    """Truncate every sequence to exactly `length` cycles, dropping shorter
    ones. Fixed length, not the main pipeline's random truncation, so
    variants are compared at the same length."""
    trunc_seq, trunc_tgt = [], []
    for seq, tgt in zip(sequences, targets):
        if len(seq) >= length:
            trunc_seq.append(seq[:length])
            trunc_tgt.append(tgt[:length])
    return trunc_seq, trunc_tgt
