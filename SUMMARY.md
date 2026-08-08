# Predictive Maintenance via Reservoir Computing

**What.** I compare three approaches for predicting how much useful life remains
in an aircraft engine before failure, on NASA's hardest C-MAPSS benchmark, FD004
(6 operating conditions, 2 simultaneous fault modes): a gradient-boosting
baseline, a classical Echo State Network (ESN), and NG-RC (next-generation
reservoir computing) — a newer, fully deterministic reservoir technique with no
prior published application to this problem that I could find.

**Why.** Manufacturers need to know how much useful life remains in a machine
before it fails.  Schedule maintenance too early and you waste good parts; too
late and you risk unplanned downtime. Prediction accuracy and retraining cost
both matter, and NG-RC's pitch — reservoir computing without the randomness — is
worth testing directly rather than taking on faith.

**How.** I prepare NASA's C-MAPSS FD004 sensor telemetry using the train/test
split NASA provides: auto-selecting informative sensors by variance, normalizing
per operating condition (six regimes), and clipping the Remaining Useful Life
(RUL) target at 125 cycles, standard practice on this benchmark since early-life
degradation is effectively unpredictable. All three models then run on that same
pipeline — the gradient-boosting baseline on windowed sensor features, ESN with
a fixed random reservoir and ridge readout, NG-RC with delay-embedded history
and a ridge readout but no reservoir at all — so the only variable across them
is the algorithm.  Each is hyperparameter-tuned on a validation split carved
from the training units only, with the real test set untouched throughout
tuning; NG-RC's `n_lags` additionally gets a 5-fold cross-validation check
rather than being trusted from a single split. Every tuned model is then refit
on the full training set and scored once on the real test set: RMSE, the
domain-specific C-MAPSS asymmetric score, training time, and a 90%-coverage
conformal prediction interval around every RUL estimate. All three run through
the identical pipeline, calibration, and test set, so the head-to-head numbers
below are apples-to-apples. (NG-RC's tuning history, including an
early false lead, is documented in depth; the baseline and ESN are tuned the
same way.)

**Result.** No single winner — a genuine tradeoff. ESN is more accurate: RMSE
16.38 vs. NG-RC's 21.08, and the C-MAPSS score (the metric that actually maps
onto maintenance risk) 1,298.9 vs.  NG-RC's 2,315.3. NG-RC trains faster (~6x)
and is fully deterministic, giving the same answer every run. ESN's own score
swings by more than 2.5x across random seeds. That's not a deployment problem —
a trained, seed-fixed ESN is just as reproducible as NG-RC once shipped.  It's
an extra training-time cost: a single blind draw isn't guaranteed to give the
best prediction. Finding a good seed to deploy takes trying several. Both
dramatically outperform the tuned baseline (RMSE 58, C-MAPSS score 953,273.7,
three orders of magnitude worse).

Which one to use depends on what's being optimized for: accuracy favors ESN;
frequent retraining, edge deployment, or skipping the seed-search step favors
NG-RC.

See [WRITEUP.md](WRITEUP.md) for the full write-up, tuning history, and
literature comparison.
