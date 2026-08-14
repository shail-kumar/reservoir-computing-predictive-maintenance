"""K-fold verification of NG-RC's n_lags sweep.

Re-tests every n_lags from 2 to 16 under 5-fold CV, checking whether the
single validation split's flat, tied-looking cluster among small n_lags
holds up or was an artifact of that split. Scoped to degree=1 only (degree=2
already lost robustly elsewhere, not re-verified here); ridge_alpha fixed at
its established flat-plateau value, not re-swept.

python -m rcpm.experiments.kfold_ngrc_check
"""

import time

import numpy as np

from rcpm.config import MULTI_CONDITION_SUBSETS, SEED, SENSOR_COLS, SUBSET
from rcpm.data import add_rul, load_cmapss
from rcpm.metrics import cmapss_score, rmse
from rcpm.ngrc import NGRC
from rcpm.normalization import normalize_by_condition
from rcpm.sequences import build_sequences
from rcpm.splits import truncate_randomly

N_FOLDS = 5
N_LAGS_CANDIDATES = list(range(2, 17))
DEGREE = 1
RIDGE_ALPHA = 100.0
MIN_TRUNCATE_LEN = 20  # >= max n_lags candidate + margin


def make_folds(n_units, n_folds=N_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_units)
    return np.array_split(idx, n_folds)


def evaluate_fold(n_lags, train_seq, train_tgt, val_seq, val_tgt):
    model = NGRC(n_lags=n_lags, degree=DEGREE, ridge_alpha=RIDGE_ALPHA)
    start = time.perf_counter()
    model.fit(train_seq, train_tgt)
    train_time = time.perf_counter() - start
    y_true = np.array([t[-1] for t in val_tgt])
    y_pred = model.predict_last(val_seq)
    return rmse(y_true, y_pred), cmapss_score(y_true, y_pred), train_time


def main():
    train, test, _ = load_cmapss(SUBSET)
    train = add_rul(train)
    if SUBSET in MULTI_CONDITION_SUBSETS:
        train, test = normalize_by_condition(train, test, SENSOR_COLS)

    sequences, targets, _, sensors = build_sequences(train, test)
    folds = make_folds(len(sequences))

    print(
        f"Subset: {SUBSET}, {N_FOLDS}-fold CV, degree={DEGREE}, ridge_alpha={RIDGE_ALPHA}"
    )
    results = {}
    search_start = time.perf_counter()
    for n_lags in N_LAGS_CANDIDATES:
        fold_rmses, fold_scores, fold_times = [], [], []
        for k in range(N_FOLDS):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])

            fold_train_seq = [sequences[i] for i in train_idx]
            fold_train_tgt = [targets[i] for i in train_idx]
            fold_val_seq = [sequences[i] for i in val_idx]
            fold_val_tgt = [targets[i] for i in val_idx]

            # Truncate the validation fold to mimic short-sequence test-time
            # realism -- same rationale as compare_models.py's truncate_randomly.
            fold_val_seq, fold_val_tgt = truncate_randomly(
                fold_val_seq, fold_val_tgt, seed=SEED + k, min_len=MIN_TRUNCATE_LEN
            )

            r, s, t = evaluate_fold(
                n_lags, fold_train_seq, fold_train_tgt, fold_val_seq, fold_val_tgt
            )
            fold_rmses.append(r)
            fold_scores.append(s)
            fold_times.append(t)

        results[n_lags] = {
            "rmse_mean": np.mean(fold_rmses),
            "rmse_std": np.std(fold_rmses),
            "score_mean": np.mean(fold_scores),
            "score_std": np.std(fold_scores),
            "train_time_mean": np.mean(fold_times),
            "train_time_std": np.std(fold_times),
        }
        print(
            f"  n_lags={n_lags:3d}: rmse={np.mean(fold_rmses):.2f}+-{np.std(fold_rmses):.2f}  "
            f"score={np.mean(fold_scores):.1f}+-{np.std(fold_scores):.1f}  "
            f"train_time={np.mean(fold_times):.2f}s+-{np.std(fold_times):.2f}s"
        )
    search_elapsed = time.perf_counter() - search_start
    print(f"\nSearch done in {search_elapsed:.1f}s", flush=True)

    best = min(results, key=lambda k: results[k]["score_mean"])
    print(f"\nBest by mean k-fold C-MAPSS score: n_lags={best}")


if __name__ == "__main__":
    main()
