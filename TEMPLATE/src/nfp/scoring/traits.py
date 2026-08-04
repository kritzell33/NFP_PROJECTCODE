"""Candidate trait sampling and archetype matching.

This module is the bridge between the Bayesian half of the suite and the
game/simulation half:

* Calibration (PyMC, offline) produces a *posterior* over each candidate's
  latent traits - not a point estimate.
* Every Monte Carlo replication of the mission draws one trait vector per
  candidate from that posterior, so uncertainty about *who someone is*
  propagates into uncertainty about *what they will do* in the sim.

v0.1 stores posteriors as (mean, sd) summaries in the crew JSON and samples
independent normals. Once the calibration workbench exists, swap
``sample_trait_draws`` to pull joint draws straight from the ArviZ
InferenceData file (``data/posteriors/*.nc``) so between-trait correlations
are preserved. The function signature is designed to make that a drop-in
change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """One crew candidate with posterior summaries over latent traits."""

    id: str
    name: str
    # trait_code -> {"mean": float, "sd": float}, traits on a z-score scale
    traits: dict[str, dict[str, float]] = field(default_factory=dict)

    def trait_codes(self) -> list[str]:
        return list(self.traits.keys())


@dataclass(frozen=True)
class Archetype:
    """A compound-trait archetype: weighted trait profile + match threshold."""

    code: str
    name: str
    weights: dict[str, float]
    threshold: float


def load_crew(path: str | Path) -> list[Candidate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    crew = [
        Candidate(id=c["id"], name=c["name"], traits=c["traits"]) for c in payload["crew"]
    ]
    if not crew:
        raise ValueError(f"Crew file {path} contains no candidates")
    return crew


def load_archetypes(path: str | Path) -> list[Archetype]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Archetype(
            code=a["code"],
            name=a["name"],
            weights=a["weights"],
            threshold=float(a["threshold"]),
        )
        for a in payload["archetypes"]
    ]


def sample_trait_draws(
    crew: list[Candidate], rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    """Draw one trait vector per candidate from their posterior.

    Returns ``{candidate_id: {trait_code: value}}`` for a single Monte Carlo
    replication. Values are clipped to the conventional z-score range so a
    wild tail draw cannot push utility functions into absurd territory.

    TODO(calibration): replace the independent-normal draw with a row sampled
    from the joint posterior stored in InferenceData (.nc) once available.
    """
    draws: dict[str, dict[str, float]] = {}
    for cand in crew:
        vec = {}
        for code, post in cand.traits.items():
            value = rng.normal(float(post["mean"]), float(post["sd"]))
            vec[code] = float(np.clip(value, -3.0, 3.0))
        draws[cand.id] = vec
    return draws


def archetype_match_score(trait_vec: dict[str, float], archetype: Archetype) -> float:
    """Weighted-sum match score for one trait draw against one archetype.

    Weights are normalized by their absolute sum, so scores stay on roughly
    the same z-score scale regardless of how many traits an archetype uses.
    Traits the archetype does not mention contribute nothing.
    """
    total_abs = sum(abs(w) for w in archetype.weights.values())
    if total_abs == 0:
        return 0.0
    score = 0.0
    for code, weight in archetype.weights.items():
        score += (weight / total_abs) * trait_vec.get(code, 0.0)
    return score


def archetype_match_probabilities(
    trait_draws_per_run: list[dict[str, dict[str, float]]],
    crew: list[Candidate],
    archetypes: list[Archetype],
) -> dict[str, dict[str, float]]:
    """P(match score >= threshold) per candidate x archetype, across all draws.

    ``trait_draws_per_run`` is the list of per-replication draws that the
    Monte Carlo runner already produced - reusing them keeps the archetype
    probabilities consistent with the exact trait values the sim ran on.
    """
    out: dict[str, dict[str, float]] = {c.id: {} for c in crew}
    n = len(trait_draws_per_run)
    if n == 0:
        return out
    for cand in crew:
        for arch in archetypes:
            hits = sum(
                1
                for draws in trait_draws_per_run
                if archetype_match_score(draws[cand.id], arch) >= arch.threshold
            )
            out[cand.id][arch.code] = hits / n
    return out
