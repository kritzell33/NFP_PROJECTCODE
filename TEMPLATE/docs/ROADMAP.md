# Roadmap

Vertical slice first: every phase keeps the full pipeline runnable
end-to-end (calibrate -> score -> simulate -> report -> exe) rather than
perfecting one layer at a time.

## Phase 0 — Workspace (this scaffold)  ✅
Repo structure, dependency management, sample data, toy sim wired
end-to-end, tests, report, GUI shell, PyInstaller spec.

## Phase 1 — Calibration engine
- Port real battery data into the workbench; graded-response / 2PL
  measurement model for 1–2 trait blocks (start small).
- Prior + posterior predictive checks; parameter recovery on synthetic
  candidates before real data.
- Export InferenceData -> data/posteriors/; switch
  `scoring.traits.sample_trait_draws` to joint posterior draws (.nc).
- Maps to Statistical Rethinking: early lectures power the runtime scorer,
  DAG/confound lectures justify structural choices, multilevel +
  measurement-error lectures are this phase.

## Phase 2 — Sim fidelity
- Replace toy utility functions with registry-grounded mappings
  (skill-pipeline matrix -> action competencies).
- py_trees daily routines + emergency protocols; richer event catalog
  (EVA, comms blackout, resupply slip); Mesa if habitat space matters.
- Validate qualitative behavior against ICE/Antarctic isolation findings.

## Phase 3 — Decision layer
- Utility functions over posterior predictive outcomes; candidate + unit
  rankings with uncertainty; threshold policies (P(outcome) >= target).
- Crew-composition search offline (evolutionary / RL in the workbench only),
  shipping results not the optimizer.

## Phase 4 — Product polish
- Registry-driven profile editor in the GUI; live pyqtgraph dashboard
  during batches; PDF export (fpdf2) with NFP branding; scenario editor.
- Inno Setup installer around the PyInstaller build.
