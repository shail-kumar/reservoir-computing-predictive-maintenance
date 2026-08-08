# Predictive Maintenance via Reservoir Computing

Compares a gradient-boosting baseline, a classical Echo State Network, and NG-RC
(next-generation reservoir computing) on NASA's C-MAPSS FD004 turbofan degradation benchmark —
the hardest of the four standard subsets (6 operating conditions, 2 simultaneous fault modes).

**Start here**: [WRITEUP.md](WRITEUP.md) for the full write-up — dataset, models, evaluation,
results, and the tuning/debugging investigations. [SUMMARY.md](SUMMARY.md) for a short version.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Download `CMAPSSData.zip` from the NASA Open Data Portal:
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Extract `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` into `data/`. (The zip contains
all four subsets — FD001-FD004; this project's default pipeline uses FD004 only. `data.py`
supports any subset via `load_cmapss(subset)` if you want to point it elsewhere.)

## Running things

All commands run from inside `src/`.

**Full three-way comparison** (baseline, tuned ESN, tuned NG-RC — the numbers in `WRITEUP.md`):
```
python compare_models.py
```
Saves figures to `../results/`. `plot_results.py` redraws those same charts from the saved
`results_*.pkl` without refitting anything, and `maintenance_alert_demo.py` reads that same pkl
to produce `WRITEUP.md`'s Table 3 worked example — run `compare_models.py` at least once first.

**Individual pieces:**
```
python eda.py               # informative-sensor selection, one engine's trajectory plot
python plot_normalization_demo.py  # per-condition normalization demo chart (real FD004 data)
python baseline.py          # untuned gradient-boosting baseline alone
python tune_baseline.py     # 30-config randomized search for the baseline (~9 min)
python lightgbm_baseline.py # LightGBM (defaults) on the same features, as a model-family check
python tune_esn.py          # ESN hyperparameter grid search
python tune_ngrc.py         # NG-RC hyperparameter grid search
python seed_sensitivity_esn.py  # ESN accuracy spread across 100 random seeds (~7 min)
python kfold_ngrc_check.py  # 5-fold CV check of NG-RC's n_lags choice (~15s)
python n_lags_test_sweep.py # NG-RC's real-test-set score across n_lags 2-40 (~20s)
python plot_results.py      # redraw compare_models.py's charts from its saved pkl, no refitting
python maintenance_alert_demo.py  # Table 3 worked example, needs compare_models.py's pkl first
python smoke_test.py        # quick sanity checks on metrics, models, data loading (~1s)
```

`kfold_ngrc_check.py` re-tests NG-RC's `n_lags` candidates (`degree=1`, `ridge_alpha` fixed at
its established plateau value) under 5-fold, unit-level cross-validation, to check whether the
single-split tuning tie in `WRITEUP.md` holds up under a more robust split — see the "K-fold
check" subsection of `WRITEUP.md` for the result.

## `ESN_tests/`

A self-contained, isolated investigation (no imports from `src/`, no modification of anything
outside its own directory) into an ESN hyperparameter question. See `ESN_tests/README.md`.

## Other docs

- [PLAN.md](PLAN.md) — the original project plan (written for an earlier, smaller scope; kept
  for history, superseded by `WRITEUP.md` for what actually happened)

## Layout

```
data/               # raw C-MAPSS files (gitignored — see "Get the data" above)
src/                 # pipeline code (see "Running things" above)
ESN_tests/           # isolated diagnostic, self-contained
results/             # generated charts
```
