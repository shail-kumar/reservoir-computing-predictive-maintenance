"""Compares ESN washout/warm-start variants on short sequences, where
washout=5 should actually matter.

Run from inside ESN_tests/:  python run_experiment.py
"""
import time

import numpy as np

from data_utils import (
    load_fd004, add_rul, informative_sensors, normalize_by_condition,
    build_sequences, split_by_unit, truncate_to_length,
)
from esn_v2 import EchoStateNetworkV2
from metrics import rmse, cmapss_score

TUNED_KWARGS = dict(n_reservoir=800, spectral_radius=1.05, sparsity=0.9,
                     leak_rate=0.05, ridge_alpha=1.0)
SHORT_LENGTHS = [8, 15, 20, 30]


def evaluate(model, cal_seq, cal_tgt, length):
    seq_t, tgt_t = truncate_to_length(cal_seq, cal_tgt, length)
    if not seq_t:
        return None
    y_true = np.array([t[-1] for t in tgt_t])
    y_pred = model.predict_last(seq_t)
    return rmse(y_true, y_pred), cmapss_score(y_true, y_pred), len(seq_t)


def main():
    train, test, _ = load_fd004()
    train = add_rul(train)
    train, test = normalize_by_condition(train, test, [f"sensor_{i}" for i in range(1, 22)])
    sensors = informative_sensors(train)
    print(f"informative sensors: {len(sensors)}")

    train_seq, train_tgt, _ = build_sequences(train, test, sensors)
    fit_seq, fit_tgt, cal_seq, cal_tgt = split_by_unit(train_seq, train_tgt)
    n_inputs = len(sensors)

    variants = {
        "A: zero-start, washout=5 (current production)": dict(warm_start=False, washout=5),
        "B: zero-start, washout=90":                       dict(warm_start=False, washout=90),
        "C: warm-start(100), washout=5":                    dict(warm_start=True, washout=5),
        "D: zero-start, washout=2":                         dict(warm_start=False, washout=2),
        "E: zero-start, washout=0":                         dict(warm_start=False, washout=0),
    }

    results = {}
    for name, cfg in variants.items():
        model = EchoStateNetworkV2(n_inputs=n_inputs, seed=0, warm_start=cfg["warm_start"],
                                    **TUNED_KWARGS)
        start = time.perf_counter()
        model.fit(fit_seq, fit_tgt, washout=cfg["washout"])
        elapsed = time.perf_counter() - start
        print(f"\n{name}  (fit: {elapsed:.2f}s)")

        # full-length calibration set, as a sanity check this doesn't regress
        y_true_full = np.array([t[-1] for t in cal_tgt])
        y_pred_full = model.predict_last(cal_seq)
        r, s = rmse(y_true_full, y_pred_full), cmapss_score(y_true_full, y_pred_full)
        print(f"  full-length calibration set: rmse={r:.2f} score={s:.1f}")

        results[name] = {}
        for length in SHORT_LENGTHS:
            out = evaluate(model, cal_seq, cal_tgt, length)
            if out is None:
                continue
            r, s, n = out
            print(f"  length={length:3d} (n={n:3d}): rmse={r:.2f} score={s:.1f}")
            results[name][length] = (r, s)

    print("\n=== Summary: RMSE at each short length ===")
    header = f"{'length':>8}" + "".join(f"{name.split(':')[0]:>10}" for name in variants)
    print(header)
    for length in SHORT_LENGTHS:
        row = f"{length:>8}"
        for name in variants:
            r = results[name].get(length, (None, None))[0]
            row += f"{r:>10.2f}" if r is not None else f"{'--':>10}"
        print(row)

    print("\n=== Summary: C-MAPSS score at each short length ===")
    print(header)
    for length in SHORT_LENGTHS:
        row = f"{length:>8}"
        for name in variants:
            s = results[name].get(length, (None, None))[1]
            row += f"{s:>10.1f}" if s is not None else f"{'--':>10}"
        print(row)


if __name__ == "__main__":
    main()
