"""Three-way comparison: gradient-boosting baseline vs. classical ESN vs. NG-RC.

    python -m rcpm.experiments.compare_models
"""

import pickle
import time

import numpy as np

from rcpm.baseline import run_baseline
from rcpm.config import (
    MULTI_CONDITION_SUBSETS,
    SEED,
    SENSOR_COLS,
    SUBSET,
    TUNED_BASELINE_KWARGS,
    TUNED_ESN_KWARGS,
    TUNED_NGRC_KWARGS,
)
from rcpm.conformal import conformal_margin
from rcpm.data import add_rul, load_cmapss
from rcpm.esn import EchoStateNetwork
from rcpm.metrics import cmapss_score, rmse
from rcpm.ngrc import NGRC
from rcpm.normalization import normalize_by_condition
from rcpm.paths import result_path
from rcpm.plotting import plot_comparison, plot_predictions, use_writeup_style
from rcpm.sequences import build_sequences, scale_sequences
from rcpm.splits import split_calibration, truncate_randomly


def run_sequence_model(
    model_cls,
    model_kwargs,
    train_sequences,
    train_targets,
    test_sequences,
    y_test,
    name,
):
    """Shared train/calibrate/evaluate flow for ESN and NG-RC: both expose
    fit(sequences, targets) and predict_last(sequences)."""
    fit_seq, fit_tgt, cal_seq, cal_tgt = split_calibration(
        train_sequences, train_targets
    )
    cal_seq, cal_tgt = truncate_randomly(cal_seq, cal_tgt, seed=SEED + 1)

    # Calibration pass: train on the fit split only, measure residuals on
    # the held-out calibration split.
    cal_model = model_cls(**model_kwargs)
    cal_model.fit(fit_seq, fit_tgt)
    cal_true = np.array([t[-1] for t in cal_tgt])
    cal_pred = cal_model.predict_last(cal_seq)
    margin = conformal_margin(cal_pred - cal_true)

    # Final pass: train on all available training units for the real
    # test-set predictions.
    final_model = model_cls(**model_kwargs)
    start = time.perf_counter()
    final_model.fit(train_sequences, train_targets)
    train_time = time.perf_counter() - start

    y_pred = final_model.predict_last(test_sequences)

    return {
        "name": name,
        "train_time": train_time,
        "rmse": rmse(y_test, y_pred),
        "score": cmapss_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test,
        "margin": margin,
    }


def main(
    subset=SUBSET,
    ngrc_kwargs=TUNED_NGRC_KWARGS,
    esn_kwargs=TUNED_ESN_KWARGS,
    baseline_kwargs=TUNED_BASELINE_KWARGS,
):
    """ngrc_kwargs/esn_kwargs/baseline_kwargs default to the final tuned FD004
    configs. Pass {} explicitly for untuned (constructor defaults) - e.g. the
    first checkpoint on a new subset before any tuning has happened."""
    train, test, test_rul = load_cmapss(subset)
    train = add_rul(train)
    y_test = test_rul["RUL"].clip(upper=125).to_numpy()

    if subset in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    results = [run_baseline(train, test, test_rul, model_kwargs=baseline_kwargs)]

    train_sequences, train_targets, test_sequences, sensors = build_sequences(
        train, test
    )
    if subset not in MULTI_CONDITION_SUBSETS:
        # FD001/FD003: single condition, a global scaler is sufficient.
        # FD002/FD004 are already per-condition normalized above, so an
        # extra global scaler would be redundant on top of that.
        train_sequences, test_sequences, _ = scale_sequences(
            train_sequences, test_sequences
        )
    n_inputs = len(sensors)

    esn_label = "Classical ESN" if esn_kwargs else "Classical ESN (untuned)"
    ngrc_label = "NG-RC" if ngrc_kwargs else "NG-RC (untuned)"

    results.append(
        run_sequence_model(
            EchoStateNetwork,
            {"n_inputs": n_inputs, "seed": SEED, **(esn_kwargs or {})},
            train_sequences,
            train_targets,
            test_sequences,
            y_test,
            esn_label,
        )
    )

    results.append(
        run_sequence_model(
            NGRC,
            {**(ngrc_kwargs or {})},
            train_sequences,
            train_targets,
            test_sequences,
            y_test,
            ngrc_label,
        )
    )

    print(f"Subset: {subset}")
    print(f"{'Model':<22}{'Train time (s)':>16}{'RMSE':>10}{'C-MAPSS score':>16}")
    for r in results:
        print(
            f"{r['name']:<22}{r['train_time']:>16.2f}{r['rmse']:>10.2f}{r['score']:>16.1f}"
        )

    with open(result_path(f"results_{subset}_nlags-19.pkl"), "wb") as f:
        pickle.dump(results, f)

    use_writeup_style()
    plot_comparison(results, result_path(f"accuracy_vs_train_time_{subset}_nlags-19.png"))
    plot_predictions(results, result_path(f"predictions_{subset}_nlags-19.png"))
    return results


if __name__ == "__main__":
    main()
