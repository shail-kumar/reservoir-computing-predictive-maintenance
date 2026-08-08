"""Operating-condition clustering and per-condition sensor normalization.

FD002/FD004 cycle through 6 discrete operating regimes, and raw sensor
levels shift substantially between them - a single global scaler (fine for
FD001's one condition) would blur the actual degradation signal under
condition-driven swings. This mirrors Eq. 1 in Rigamonti et al.: normalize
each signal using the mean/std computed *within its own operating
condition*, fit from training data only.
"""

import pandas as pd
from sklearn.cluster import KMeans

OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
# Source: NASA C-MAPSS docs (Saxena et al., 2008) and Rigamonti et al. -
# both state FD002/FD004 have 6 operating regimes. Not derived by this
# code; won't generalize to a dataset with an unknown condition count.
N_OPERATING_CONDITIONS = 6


def fit_operating_conditions(
    train: pd.DataFrame, n_conditions: int = N_OPERATING_CONDITIONS
) -> KMeans:
    """Cluster the 3 operating-condition settings into their discrete
    regimes. Clustering (rather than rounding the raw values) is needed
    because the nominal setpoints carry floating-point noise, e.g. one
    setpoint appears as both 10.00/0.25 and 10.01/0.25 in the raw data,
    and setting_2=0.25 sits exactly on a rounding boundary that naive
    rounding misassigns. Fit on training data only.
    """
    kmeans = KMeans(n_clusters=n_conditions, n_init=10, random_state=0)
    kmeans.fit(train[OP_SETTING_COLS])
    return kmeans


def assign_operating_condition(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = kmeans.predict(df[OP_SETTING_COLS])
    return df


def normalize_by_condition(train: pd.DataFrame, test: pd.DataFrame, sensor_cols):
    """Cluster operating conditions (fit on train) and z-score each sensor
    within its own condition, using train-only mean/std. Overwrites
    sensor_cols in place so downstream code needs no changes.
    """
    kmeans = fit_operating_conditions(train)
    train = assign_operating_condition(train, kmeans)
    test = assign_operating_condition(test, kmeans)

    stats = train.groupby("condition")[sensor_cols].agg(["mean", "std"])

    def apply_norm(df):
        df = df.copy()
        # Some sensor columns load as int64 (all-integer readings in the raw
        # file); normalizing produces floats, which pandas 2.x refuses to
        # write back into an int column in place - cast up front.
        df[sensor_cols] = df[sensor_cols].astype(float)
        for cond, group in df.groupby("condition"):
            idx = group.index
            for col in sensor_cols:
                mean = stats.loc[cond, (col, "mean")]
                std = stats.loc[cond, (col, "std")]
                std = (
                    std if std > 1e-8 else 1.0
                )  # guard near-constant sensors within a condition
                df.loc[idx, col] = (df.loc[idx, col] - mean) / std
        return df

    return apply_norm(train), apply_norm(test)
