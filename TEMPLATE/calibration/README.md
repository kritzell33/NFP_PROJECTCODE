# Calibration workbench

The Bayesian half of the suite. Everything here runs on your dev machine and
is **never** shipped inside the .exe - PyMC compiles model code at runtime,
which does not survive PyInstaller freezing (and does not need to: the
Assessor only consumes the *posterior draws* this workbench produces).

## Setup (separate venv on purpose)

    python -m venv .venv-calib
    .venv-calib\Scripts\activate
    pip install -r requirements-calibration.txt

## Workflow

1. Fit measurement models over battery item data -> latent trait posteriors
   per candidate (start from `fit_measurement_demo.py`).
2. Save results as ArviZ InferenceData: `data/posteriors/measurement_v1.nc`.
3. The Assessor's `nfp.scoring.traits.sample_trait_draws` then swaps from
   (mean, sd) summaries to joint draws from that file - preserving
   between-trait correlations.

Validation habits to keep from Statistical Rethinking: prior predictive
checks before fitting, posterior predictive checks after, and
parameter-recovery tests on synthetic candidates before touching real data.
