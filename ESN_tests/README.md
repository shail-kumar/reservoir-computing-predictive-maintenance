# ESN_tests: washout and reservoir warm-start experiment

Self-contained. No imports from `../src/`, no changes outside this directory. Reads `../data/`
read-only; all processing code duplicated here.

## Why

`src/esn.py`'s `washout=5` was never tuned. A diagnostic suggested ~80-100 steps needed. Test
sequences truncate to 8 cycles, too short for more washout to help. Tests two fixes: longer
washout, reservoir warm-start.

## Variants

Tuned config: `n_reservoir=800, spectral_radius=1.05, sparsity=0.9, leak_rate=0.05,
ridge_alpha=1.0`. Evaluated on 8-30 cycle sequences.

- A: washout=5 (current)
- B: washout=90
- C: warm-start (100 steps), washout=5
- D: washout=2
- E: washout=0

## Files

- `data_utils.py`: data loading, normalization, sequences
- `esn_v2.py`: ESN with configurable washout/warm-start
- `metrics.py`: rmse, cmapss_score
- `run_experiment.py`: the comparison

## Results

RMSE by length (C-MAPSS score: same pattern, sharper; variant C at length=8 scored 40,875,625):

| length | A | B | C | D | E |
|---|---|---|---|---|---|
| 8  | 6.42  | 22.14 | 50.62 | 6.43  | 6.49 |
| 15 | 9.03  | 30.13 | 42.14 | 9.00  | 9.00 |
| 20 | 9.09  | 31.03 | 36.90 | 9.08  | 9.08 |
| 30 | 10.50 | 29.82 | 27.56 | 10.44 | 10.40 |

## Conclusion

Washout in `[0, 5]` is inert: A, D, E match. B and C are both worse, not better. Mechanism:
large washout/warm-start trains the readout only on mature reservoir states, so it never sees
the immature state a short sequence actually produces at its last cycle - a train/test
mismatch. `washout=5` is fine as is; no change to `src/esn.py`.
