"""Monte Carlo runner: many seeded mission replications -> outcome distributions.

Reproducibility contract: a batch is fully determined by (crew file, scenario
file, n_runs, master seed). Child seeds are spawned with numpy's
``SeedSequence`` so replications are independent but replayable - which
matters if this is ever used as part of a real selection instrument.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..scoring.traits import (
    Archetype,
    Candidate,
    archetype_match_probabilities,
    load_archetypes,
    load_crew,
    sample_trait_draws,
)
from .agents import CrewAgent
from .mission import Mission, Scenario


@dataclass
class BatchResult:
    scenario: Scenario
    crew: list[Candidate]
    archetypes: list[Archetype]
    n_runs: int
    master_seed: int
    per_run: pd.DataFrame
    trait_draws: list[dict[str, dict[str, float]]]
    timeseries: list[dict] = field(default_factory=list)

    # -- aggregates ---------------------------------------------------------

    def summary(self) -> dict:
        num = self.per_run.select_dtypes(include="number")
        agg = {
            col: {
                "mean": float(num[col].mean()),
                "sd": float(num[col].std(ddof=1)) if len(num) > 1 else 0.0,
                "p05": float(num[col].quantile(0.05)),
                "p95": float(num[col].quantile(0.95)),
            }
            for col in num.columns
        }
        agg["p_cascade_free"] = float((~self.per_run["critical_cascade"]).mean())
        return agg

    def archetype_probs(self) -> dict[str, dict[str, float]]:
        return archetype_match_probabilities(self.trait_draws, self.crew, self.archetypes)

    def provenance(self) -> dict:
        import matplotlib
        import networkx
        import simpy

        return {
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "scenario": self.scenario.name,
            "n_runs": self.n_runs,
            "master_seed": self.master_seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "simpy": simpy.__version__,
            "networkx": networkx.__version__,
            "matplotlib": matplotlib.__version__,
        }


def run_single(scenario: Scenario, crew: list[Candidate], seed_seq: np.random.SeedSequence
               ) -> tuple[dict, dict[str, dict[str, float]], dict]:
    """One replication: draw traits from posteriors, run the mission."""
    rng = np.random.default_rng(seed_seq)
    draws = sample_trait_draws(crew, rng)
    agents = [CrewAgent(c.id, c.name, draws[c.id], rng) for c in crew]
    mission = Mission(scenario, agents, rng)
    metrics = mission.run()
    ts = {
        "day": mission.ts_day,
        "cohesion": mission.ts_cohesion,
        "backlog": mission.ts_backlog,
        "stress": mission.ts_stress,
    }
    return metrics, draws, ts


def run_batch(crew_path: str | Path, scenario_path: str | Path,
              archetypes_path: str | Path, n_runs: int = 200,
              seed: int = 42) -> BatchResult:
    scenario = Scenario.from_json(scenario_path)
    crew = load_crew(crew_path)
    archetypes = load_archetypes(archetypes_path)

    master = np.random.SeedSequence(seed)
    children = master.spawn(n_runs)

    rows, draws_all, ts_all = [], [], []
    for i, child in enumerate(children):
        metrics, draws, ts = run_single(scenario, crew, child)
        metrics["run"] = i
        rows.append(metrics)
        draws_all.append(draws)
        ts_all.append(ts)

    per_run = pd.DataFrame(rows)
    return BatchResult(
        scenario=scenario, crew=crew, archetypes=archetypes,
        n_runs=n_runs, master_seed=seed, per_run=per_run,
        trait_draws=draws_all, timeseries=ts_all,
    )


def save_batch(result: BatchResult, out_dir: str | Path) -> Path:
    """Write results.csv + metrics.json into a timestamped run folder."""
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    result.per_run.to_csv(run_dir / "results.csv", index=False)
    payload = {
        "provenance": result.provenance(),
        "summary": result.summary(),
        "archetype_match_probabilities": result.archetype_probs(),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir
