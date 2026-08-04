"""Minimal PyMC measurement-model demo (calibration workbench).

Fits a toy graded item battery for a handful of synthetic candidates and
writes latent-trait posteriors to data/posteriors/. This is the pattern the
real 1,104-item battery calibration will follow (as a proper IRT / graded
response model); the point here is the *pipeline shape*:

    raw item responses -> PyMC model -> InferenceData (.nc) -> Assessor

Run inside the calibration venv:  python calibration/fit_measurement_demo.py
"""

from pathlib import Path

import numpy as np

try:
    import arviz as az
    import pymc as pm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMC/ArviZ not installed. Activate the calibration venv first:\n"
        "  python -m venv .venv-calib && .venv-calib\\Scripts\\activate\n"
        "  pip install -r requirements-calibration.txt"
    ) from exc

rng = np.random.default_rng(11)

# --- synthetic ground truth: 6 candidates, 12 items measuring one trait ----
n_cand, n_items = 6, 12
true_theta = rng.normal(0, 1, size=n_cand)          # latent trait
item_difficulty = rng.normal(0, 0.7, size=n_items)
item_loading = rng.uniform(0.6, 1.4, size=n_items)

# continuous item responses with noise (swap for ordered-logistic later)
y = (item_loading[None, :] * true_theta[:, None]
     - item_difficulty[None, :]
     + rng.normal(0, 0.5, size=(n_cand, n_items)))

# --- model ------------------------------------------------------------------
with pm.Model() as model:
    theta = pm.Normal("theta", 0.0, 1.0, shape=n_cand)
    diff = pm.Normal("difficulty", 0.0, 1.0, shape=n_items)
    load = pm.LogNormal("loading", 0.0, 0.3, shape=n_items)
    sigma = pm.Exponential("sigma", 2.0)

    mu = load[None, :] * theta[:, None] - diff[None, :]
    pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                      random_seed=11)

print(az.summary(idata, var_names=["theta"]))
print("\nParameter recovery check (posterior mean vs truth):")
post_mean = idata.posterior["theta"].mean(dim=("chain", "draw")).values
for i, (est, truth) in enumerate(zip(post_mean, true_theta)):
    print(f"  candidate {i}: est {est:+.2f}  truth {truth:+.2f}")

out = Path("data/posteriors/measurement_demo.nc")
out.parent.mkdir(parents=True, exist_ok=True)
idata.to_netcdf(out)
print(f"\nSaved posterior draws -> {out}")
