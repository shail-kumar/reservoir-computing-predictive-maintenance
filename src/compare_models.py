"""Three-way comparison - baseline vs. classical ESN vs. NG-RC.

Run from inside src/:  python compare_models.py
"""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from baseline import run_baseline
from normalization import normalize_by_condition
from conformal import conformal_margin
from data import load_cmapss, add_rul
from esn import EchoStateNetwork
from metrics import rmse, cmapss_score
from ngrc import NGRC
from sequences import build_sequences, scale_sequences

plt.rcParams["text.usetex"] = True
# Base matplotlib font size (10pt) reads smaller than surrounding body text
# once embedded in the writeup; bump labels/legend/ticks to stay legible.
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14

CALIBRATION_FRACTION = 0.2
SEED = 0
SUBSET = "FD004"
# FD002/FD004 have 6 operating conditions with substantially different raw
# sensor levels per condition - normalize per-condition (normalization.py)
# instead of the single global scaler that's sufficient for FD001/FD003.
MULTI_CONDITION_SUBSETS = {"FD002", "FD004"}
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# Final tuned configs on FD004 (see WRITEUP.md / tune_baseline.py / tune_esn.py /
# tune_ngrc.py / kfold_ngrc_check.py / n_lags_test_sweep.py for the tuning
# history behind these choices).
# Winning RandomizedSearchCV config from tune_baseline.py (30-candidate search).
TUNED_BASELINE_KWARGS = {
    "learning_rate": 0.022169635606490684,
    "max_depth": 7,
    "n_estimators": 153,
    "subsample": 0.7449024601551049,
}
TUNED_ESN_KWARGS = {
    "n_reservoir": 800,
    "spectral_radius": 1.05,
    "sparsity": 0.9,
    "leak_rate": 0.05,
    "ridge_alpha": 1.0,
}
# degree=1 (no polynomial expansion); n_lags=19, the largest value that
# still produces a prediction for every real FD004 test engine (shortest
# test trajectory: 19 cycles).
TUNED_NGRC_KWARGS = {"n_lags": 19, "degree": 1, "ridge_alpha": 100.0}
# Truncated calibration/validation sequences must stay >= n_lags, or NGRC's
# delay embedding returns an empty array - computed dynamically since a
# previous hardcoded floor caused exactly this when n_lags grew past it.
MIN_TRUNCATE_LEN = TUNED_NGRC_KWARGS["n_lags"] + 4


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
    """Truncate each sequence/target pair at a random cycle, mimicking how
    real test trajectories are cut off before failure at an arbitrary point.
    Needed so calibration residuals are exchangeable with the actual
    test-time task - predicting from a mid-life truncation, not from a full
    run-to-failure trajectory (which would make every calibration target
    RUL=0 and badly understate real prediction uncertainty).

    min_len must stay >= whatever model's n_lags/washout is in use, or the
    delay embedding returns an empty array on the shortest truncations.
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


def plot_comparison(results, out_path):
    # Same 8-inch width as plot_predictions() and the normalization demo
    # chart, so a fixed point-size font reads consistently across all
    # three once embedded in the writeup at a similar display width.
    fig, ax = plt.subplots(figsize=(8, 6))
    slowest = max(r["train_time"] for r in results)
    for r in results:
        ax.scatter(r["train_time"], r["score"], s=80)
        # The slowest point sits at the right edge of the log-x axis and
        # near the top of the log-y axis (worst score); any right or up
        # offset pushes its label past the plot boundary, so it goes to
        # the left instead, vertically centered on the marker.
        if r["train_time"] == slowest:
            ax.annotate(
                r["name"],
                (r["train_time"], r["score"]),
                textcoords="offset points",
                xytext=(-8, 0),
                ha="right",
                va="center",
            )
        else:
            ax.annotate(
                r["name"],
                (r["train_time"], r["score"]),
                textcoords="offset points",
                xytext=(6, 6),
            )
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("C-MAPSS score (lower is better)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Accuracy vs. training cost")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_predictions(results, out_path):
    """One figure, one subplot per model, sharing both axes - all three
    models' y_test arrays are identical and same-order (same test units),
    verified directly, so a single sort order is valid for all of them.
    Sharing the y-axis too keeps the vertical scale identical across
    subplots, so prediction spread is directly comparable model to model.
    """
    order = np.argsort(results[0]["y_test"])
    y_test = results[0]["y_test"][order]
    x = np.arange(len(y_test))

    fig, axes = plt.subplots(
        len(results), 1, figsize=(8, 3.5 * len(results)), sharex=True, sharey=True
    )
    for ax, result in zip(axes, results):
        y_pred = result["y_pred"][order]
        margin = result.get("margin")

        ax.plot(x, y_test, label="Actual RUL", color="black", linewidth=1)
        ax.plot(
            x,
            y_pred,
            label=f"{result['name']} prediction",
            color="tab:blue",
            linewidth=1,
        )
        if margin is not None:
            # Baseline has no conformal calibration (run_baseline() doesn't
            # compute one), so there's no interval band to draw for it.
            ax.fill_between(
                x,
                y_pred - margin,
                y_pred + margin,
                alpha=0.2,
                label=f"90\\% conformal interval ($\\pm${margin:.1f})",
            )
        ax.set_ylabel("RUL (cycles)")
        ax.legend()

    axes[-1].set_xlabel("Test engine (sorted by actual RUL)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


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

    with open(f"../results/results_{subset}_nlags-19.pkl", "wb") as f:
        pickle.dump(results, f)

    plot_comparison(results, f"../results/accuracy_vs_train_time_{subset}_nlags-19.png")
    plot_predictions(results, f"../results/predictions_{subset}_nlags-19.png")
    return results


if __name__ == "__main__":
    main()
