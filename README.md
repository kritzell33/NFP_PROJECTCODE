# NFP — New Frontier Project

Bayesian candidate assessment and mission simulation suite for space-settlement
crew selection. Two programs, one data contract:

- **Calibration workbench** (`calibration/`) — PyMC Bayesian models fit latent
  trait posteriors from psychometric battery data. Dev machine only.
- **NFP Assessor** (`src/nfp/`) — draws candidate traits from those posteriors,
  runs crews through Monte Carlo mission simulations (utility-AI agents, event
  director, relationship graphs), and renders comprehensive HTML reports.
  Ships as a Windows desktop app.

See `docs/ARCHITECTURE.md` for the why, `docs/ROADMAP.md` for the plan.

---

## First-time setup (Windows)

You only do this once. Everything assumes **native Windows Python** (not
WSL2) because the GUI and the .exe build are Windows targets.

### 0. Prerequisites

1. **Python 3.12** — [python.org/downloads](https://www.python.org/downloads/).
   During install, **check "Add python.exe to PATH."**
2. **Git** — [git-scm.com](https://git-scm.com/downloads). Defaults are fine.

Verify in a fresh PowerShell window:

    python --version
    git --version

### 1. Get the repository onto your machine

    cd %USERPROFILE%\Documents
    git clone https://github.com/kritzell33/NFP_PROJECTCODE.git
    cd NFP_PROJECTCODE

### 2. Clear out the template leftovers (one time only)

The repo was created from a generic template. Delete everything **except**
`LICENSE` and the hidden `.git` folder, then copy this scaffold's contents in
(so this README, `pyproject.toml`, `src/`, etc. sit at the repo root next to
`LICENSE`).

### 3. Create the virtual environment and install

A *virtual environment* (venv) is a private folder of Python packages just
for this project, so nothing you install here touches the rest of your
system. The easy way:

    scripts\setup.bat

Or manually, which is worth seeing once:

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -e ".[gui,dev]"

`-e` is an "editable" install: Python runs the code straight from `src/`, so
your edits take effect immediately — no reinstalling.

> **If PowerShell refuses to run `activate`** with an execution-policy error,
> run this once and retry:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Whenever you open a new terminal to work on the project:

    cd path\to\NFP_PROJECTCODE
    .venv\Scripts\activate

You'll see `(.venv)` in your prompt while it's active.

### 4. Prove it works

    scripts\run_demo.bat        (or:  nfp run --runs 100 --seed 42 --open)

That runs 100 seeded replications of the bundled 14-day "Habitat Shakedown"
scenario with 4 sample candidates and opens the HTML report. Then:

    scripts\run_tests.bat       (or:  pytest -v)
    scripts\run_gui.bat         (or:  nfp gui)

### 5. Commit and push

    git add -A
    git commit -m "Replace template with NFP suite scaffold"
    git push origin master

That's the whole loop you'll repeat forever: edit -> test -> commit -> push.
Commit small and often; each commit is a save point you can return to.

---

## The calibration environment (separate, later)

PyMC lives in its **own** venv so the Assessor's environment stays lean
enough to freeze into an .exe. When you reach Phase 1:

    python -m venv .venv-calib
    .venv-calib\Scripts\activate
    pip install -r requirements-calibration.txt
    python calibration\fit_measurement_demo.py

Details in `calibration/README.md`.

## Building the .exe

    packaging\build_exe.bat

Output: `dist\NFP_Assessor\NFP_Assessor.exe`. See `packaging/README.md`.

---

## Project layout

    src/nfp/            the installable package
      registry/         variable-registry loader (single source of truth)
      scoring/          posterior trait sampling + archetype matching
      sim/              agents (utility AI + FSM), events, social graph,
                        SimPy mission engine, Monte Carlo runner
      report/           HTML report builder + Jinja2 template
      gui/              PySide6 desktop shell
      cli.py            `nfp run | demo | gui`
    calibration/        PyMC workbench (separate venv, never shipped)
    data/               registry, archetypes, crew files, posterior store
    scenarios/          mission scenario definitions (JSON)
    scripts/            Windows helpers + synthetic crew generator
    packaging/          PyInstaller spec for the .exe
    tests/              pytest suite

## Everyday commands

    nfp run --runs 500 --seed 7 --open      bigger batch, open report
    nfp run --crew data\crew\synth_crew.json ...   your own crew file
    python scripts\gen_synthetic_crew.py --n 8     make test candidates
    pytest -v                                run the test suite
    ruff check src tests                     lint

## Swapping in real NFP data

1. Export the master variable registry to CSV with the columns in
   `data/registry/registry_sample.csv` and point the tools at it.
2. Replace `data/archetypes/archetypes_sample.json` with the real archetype
   definitions (weights + thresholds).
3. Crew files follow `data/crew/sample_crew.json`: per-candidate posterior
   mean/sd per trait now; joint draws from `data/posteriors/*.nc` after
   Phase 1 calibration.
