"""Feature engineering for the gradient-boosting baseline only - ESN and
NG-RC consume raw sequences directly via sequences.py instead."""

from eda import informative_sensors


def build_features(train, test, window: int = 5):
    """Return (train_features, test_features, feature_cols).

    Uses each informative sensor's raw reading plus a rolling mean over the
    last `window` cycles, so the model sees short-term trend, not just an
    instant snapshot.
    """
    sensors = informative_sensors(train)

    def add_rolling(df):
        df = df.sort_values(["unit", "cycle"]).copy()
        for s in sensors:
            df[f"{s}_roll_mean"] = df.groupby("unit")[s].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        return df

    train_feat = add_rolling(train)
    test_feat = add_rolling(test)

    feature_cols = sensors + [f"{s}_roll_mean" for s in sensors]
    return train_feat, test_feat, feature_cols
