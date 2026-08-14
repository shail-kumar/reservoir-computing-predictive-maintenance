"""ESN seed-sensitivity check on the final tuned FD004 config.

Motivation: NG-RC's real engineering advantage over ESN (beyond raw
accuracy/speed) is having no randomness at all - deterministic given the
data, versus ESN's random reservoir draw. This only matters as an argument
if ESN's performance actually varies meaningfully across seeds. Rather than
assert that qualitatively, measure it: refit the tuned ESN config across
many seeds and report the spread.

python -m rcpm.experiments.seed_sensitivity_esn
"""

import time

import numpy as np

from rcpm.config import (
    MULTI_CONDITION_SUBSETS,
    SENSOR_COLS,
    SUBSET,
    TUNED_ESN_KWARGS,
)
from rcpm.data import add_rul, load_cmapss
from rcpm.esn import EchoStateNetwork
from rcpm.metrics import cmapss_score, rmse
from rcpm.normalization import normalize_by_condition
from rcpm.sequences import build_sequences

N_SEEDS = 100


def main(subset=SUBSET):
    train, test, test_rul = load_cmapss(subset)
    train = add_rul(train)
    y_test = test_rul["RUL"].clip(upper=125).to_numpy()

    if subset in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    train_seq, train_tgt, test_seq, sensors = build_sequences(train, test)
    n_inputs = len(sensors)

    kwargs = {k: v for k, v in TUNED_ESN_KWARGS.items()}

    rmses, scores, train_times = [], [], []
    search_start = time.perf_counter()
    for seed in range(N_SEEDS):
        model = EchoStateNetwork(n_inputs=n_inputs, seed=seed, **kwargs)
        start = time.perf_counter()
        model.fit(train_seq, train_tgt)
        train_time = time.perf_counter() - start
        y_pred = model.predict_last(test_seq)
        r, s = rmse(y_test, y_pred), cmapss_score(y_test, y_pred)
        rmses.append(r)
        scores.append(s)
        train_times.append(train_time)
        print(
            f"  seed={seed:2d}  rmse={r:.2f}  score={s:.1f}  train_time={train_time:.2f}s",
            flush=True,
        )
    search_elapsed = time.perf_counter() - search_start

    rmses, scores, train_times = (
        np.array(rmses),
        np.array(scores),
        np.array(train_times),
    )
    print(f"\nOver {N_SEEDS} seeds ({search_elapsed:.1f}s total):")
    print(
        f"  RMSE:  mean={rmses.mean():.2f}  std={rmses.std():.2f}  "
        f"min={rmses.min():.2f}  max={rmses.max():.2f}"
    )
    print(
        f"  score: mean={scores.mean():.1f}  std={scores.std():.1f}  "
        f"min={scores.min():.1f}  max={scores.max():.1f}"
    )
    print(f"  train_time: mean={train_times.mean():.2f}s  std={train_times.std():.2f}s")


if __name__ == "__main__":
    main()
