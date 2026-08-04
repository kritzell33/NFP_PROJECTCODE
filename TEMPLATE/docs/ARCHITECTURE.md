# NFP Suite Architecture

## The two-program split

The suite is deliberately two programs sharing one data contract:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  CALIBRATION WORKBENCH      │         │  NFP ASSESSOR (the .exe)     │
│  (dev machine only)         │  .nc    │  (shippable)                 │
│                             │ files   │                              │
│  PyMC 5 + ArviZ             │ ──────► │  numpy / scipy / SimPy       │
│  · measurement models over  │         │  · draw traits from stored   │
│    the item battery (IRT)   │ posterior│    posteriors per replication│
│  · structural models per    │  draws  │  · utility-AI crew agents    │
│    DAG annotations          │         │  · mission Monte Carlo       │
│  · multilevel pooling       │         │  · archetype match probs     │
│  · PPC + param recovery     │         │  · HTML report               │
└─────────────────────────────┘         └──────────────────────────────┘
```

Why: PyMC (like Stan and NumPyro) compiles model code at runtime, which makes
PyInstaller freezing miserable — and unnecessary. Scoring a *new* profile
against calibrated parameters needs only grid/quadratic approximation or
stored-draw resampling (Statistical Rethinking ch. 2–4 applied at runtime),
which is milliseconds of numpy. So MCMC stays on the workbench; the exe
consumes its outputs.

## Uncertainty propagation (the core idea)

Each Monte Carlo replication draws one trait vector per candidate from that
candidate's *posterior*, then runs a full mission with agents parameterized by
those draws. Uncertainty about who someone is therefore propagates into
uncertainty about what they'll do — the report's outcome distributions reflect
scenario randomness **and** measurement uncertainty jointly. Point estimates
never enter the pipeline.

## Game-AI layer (how the sim thinks)

Built the way colony/strategy sims actually do it — no ML at runtime:

- **Utility AI** (`sim/agents.py`): agents score actions each tick and pick
  via softmax. Utility weights are functions of latent traits, which is where
  Bayesian decision theory and game AI turn out to be the same machinery.
- **FSM** (`transitions`): nominal / stressed / exhausted modes with
  hysteresis; state gates capability (exhausted agents can't work).
- **Event director** (`sim/events.py`): RimWorld-style storyteller sampling
  daily adversity conditioned on sim state (wear → failures; stress + low
  cohesion → friction). It's just a generative model, so its rates are
  parameters you can sweep or calibrate.
- **Relationship graph** (`sim/social.py`, networkx): weighted crew edges;
  cohesion summarized from edge weights; feeds back into friction hazard.
- **SimPy backbone** (`sim/mission.py`): hour-resolution discrete-event loop,
  shared resources (workstations, gym), task backlog with repair SLAs,
  forced sleep window.

Phase-2 slots already reserved: py_trees for structured routines (daily
schedules, emergency protocols), Mesa when spatial habitat layout matters,
HTN planning (GTPyhop) if agents should decompose goals into task plans.

## Registry-driven design

`data/registry/*.csv` is the single source of truth for variables. The
loader (`registry/loader.py`) validates it; the trait sampler, sim, and
report all key off `var_code`s. Swapping the 10-row sample for the real
271-row registry is a data change, not a code change. Archetypes
(`data/archetypes/*.json`) are likewise config, not code.

## Reproducibility contract

A batch is fully determined by (crew file, scenario file, n_runs, master
seed); child seeds come from numpy `SeedSequence.spawn`. Reports print the
exact reproduce command plus package versions. This matters the moment the
tool informs real selection decisions.

## Repo layout

```
src/nfp/          the installable package (registry, scoring, sim, report, gui, cli)
calibration/      PyMC workbench — separate venv, never frozen
data/             registry, archetypes, crew files, posterior store
scenarios/        mission scenario JSONs
scripts/          Windows .bat helpers + synthetic crew generator
packaging/        PyInstaller spec + build script
tests/            pytest suite (determinism, FSM, utility shaping, batch shapes)
```
