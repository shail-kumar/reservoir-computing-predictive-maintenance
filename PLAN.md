# NG-RC for Remaining Useful Life Prediction — The Plan

## Business framing

Manufacturers need to know how much useful life remains in a machine before it fails, so
maintenance can be scheduled without wasting parts or risking unplanned downtime — that's the
angle to build this around. Prediction accuracy trades off against that cost directly, and
training/retraining cost matters for real-time edge deployment on factory-floor sensors.

## Dataset

Start with NASA C-MAPSS's FD001 subset (single operating condition, single fault mode) for the
initial build. Treat FD002/FD004 (multiple operating conditions/fault modes) as an optional
stretch goal once the pipeline works — they're genuinely harder and much less commonly tackled
in public tutorials, so getting there would be a nice differentiator.

Clip the RUL target at 125 cycles (standard practice on this benchmark, since early-life
degradation is nonlinear/unpredictable anyway) — remember to explain this choice in the writeup.

## Models (three-way comparison, same train/test split)

1. **Baseline** — regression/gradient boosting on engineered/windowed sensor features. My
   "obvious industry-standard" reference point.
2. **Classical ESN** — match the setup style of the existing PHM Society paper ("Echo State
   Network for the Remaining Useful Life Prediction of a Turbofan Engine"). My
   literature-anchored comparison, not the novel contribution.
3. **NG-RC (primary, novel)** — delay-embedded sensor history → polynomial feature expansion →
   linear/ridge regression readout. Couldn't find any prior published work combining NG-RC with
   this dataset — not a systematic
   literature review. Frame this as "no prior work found" in the writeup, rather than
   "first ever."

## Evaluation

Track:
- RMSE
- The C-MAPSS asymmetric scoring function from the original benchmark literature (penalizes late
  predictions — overestimating remaining life — more heavily than early ones)
- Training time / compute cost across all three models — my real business argument for NG-RC/ESN
  over deep learning: fast retraining matters for edge deployment
- An uncertainty/prediction interval on RUL, rather than a single point estimate

## Weekend schedule

**Day 1**
1. Get the FD001 data, quick EDA (plot sensor trends, identify informative vs. flat channels)
2. Define my RUL target (clipped at 125 cycles)
3. Build the baseline model
4. Build the classical ESN (reference to literature setup)

**Day 2**
1. Build the NG-RC model (numpy: delay embedding + polynomial features + ridge readout)
2. Run all three models, produce the accuracy + training-time comparison chart
3. Add uncertainty quantification to my RUL predictions
4. Write it up: business framing → approach → comparison chart → business implication →
   limitations (FD001 is single-condition; real deployment needs FD002-FD004) → link to code

## Sources checked for the novelty claim

- Echo State Network for the Remaining Useful Life Prediction of a Turbofan Engine (PHM Society)
- Performance and Explainability of Reservoir Computing Models for Industrial Prognosis (Springer)
- Predicting remaining useful life of turbofan engines using degradation signal based ESN
- Next generation reservoir computing (Nature Communications, Gauthier et al.)
- Adaptive Nonlinear Vector Autoregression: Robust Forecasting for Noisy Chaotic Time Series (arXiv)

Didn't find any NG-RC + C-MAPSS / RUL / predictive-maintenance combination in these searches. Do
a  Google Scholar / IEEE Xplore check before finalizing the "no prior work" claim in the
writeup.
