"""Load and prepare NASA C-MAPSS RUL benchmark data."""

from pathlib import Path

import pandas as pd

from rcpm.paths import DATA_DIR

COLUMNS = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read a raw C-MAPSS space-separated file.

    Known quirk: these files often have trailing whitespace per line, which
    makes pandas parse one or two extra all-NaN columns at the end. Drop any
    fully-empty columns before assigning names.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns after dropping empty ones, "
            f"got {df.shape[1]}. Check the file wasn't corrupted/truncated during extraction."
        )
    df.columns = COLUMNS
    return df


def load_cmapss(subset: str = "FD001", data_dir: Path = DATA_DIR):
    """Return (train_df, test_df, test_rul) for the given C-MAPSS subset
    (one of FD001-FD004).

    Expects train_{subset}.txt, test_{subset}.txt, RUL_{subset}.txt in
    data_dir, as extracted from CMAPSSData.zip (NASA Open Data Portal).
    """
    train = _read_cmapss_file(data_dir / f"train_{subset}.txt")
    test = _read_cmapss_file(data_dir / f"test_{subset}.txt")
    test_rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None, names=["RUL"])
    return train, test, test_rul


def load_fd001(data_dir: Path = DATA_DIR):
    """Back-compat wrapper: the FD001 subset specifically."""
    return load_cmapss("FD001", data_dir)


def add_rul(train: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    """Attach a clipped RUL target to each row of the training set.

    Clipping at 125 cycles is standard practice on this benchmark: early-life
    degradation is effectively unpredictable/nonlinear, so the target is flat
    until an engine enters its actual degradation phase.
    """
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=clip)
    return train


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """The final observed row per unit - the only data available at
    prediction time, since test trajectories are truncated before failure
    (RUL_{subset}.txt gives the true remaining life at that cutoff)."""
    return df.sort_values("cycle").groupby("unit").tail(1)
