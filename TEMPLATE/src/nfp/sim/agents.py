"""Crew agents: utility-AI action selection + stress state machine.

This is the "game opponent AI" core, built the way strategy/colony sims
actually do it:

* **Utility AI** - each tick, the agent scores every available action with a
  utility function and picks via softmax. The crucial NFP twist: the utility
  weights are functions of the agent's *latent traits*, and each Monte Carlo
  replication draws those traits from the candidate's Bayesian posterior.
* **Finite state machine** (`transitions` library) - agents move between
  nominal / stressed / exhausted modes with hysteresis, which changes what
  they are capable of (exhausted agents cannot work).

Trait codes used here match the sample registry. When you swap in the real
271-variable registry, extend ``ACTIONS`` utility functions to use whichever
codes you care about - unknown codes default to 0.0 (population average).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from transitions import Machine

# ---------------------------------------------------------------------------
# Stress state machine
# ---------------------------------------------------------------------------

STATES = ["nominal", "stressed", "exhausted"]

# Hysteresis thresholds: it takes more recovery to come back down a level
# than it took strain to go up, which mirrors the ICE-expedition literature
# on cumulative fatigue.
STRESS_UP_STRESSED = 55.0
STRESS_UP_EXHAUSTED = 85.0
STRESS_DOWN_NOMINAL = 45.0
STRESS_DOWN_STRESSED = 70.0

TRANSITIONS = [
    {"trigger": "pressure", "source": "nominal", "dest": "stressed",
     "conditions": "is_over_stressed_threshold"},
    {"trigger": "pressure", "source": "stressed", "dest": "exhausted",
     "conditions": "is_over_exhausted_threshold"},
    {"trigger": "recover", "source": "exhausted", "dest": "stressed",
     "conditions": "is_under_stressed_recovery"},
    {"trigger": "recover", "source": "stressed", "dest": "nominal",
     "conditions": "is_under_nominal_recovery"},
]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    name: str
    duration_h: int


ACTIONS = {
    "work": Action("work", 2),        # pull a task from the backlog (needs workstation)
    "exercise": Action("exercise", 1),  # gym slot, lowers stress, builds endurance buffer
    "socialize": Action("socialize", 1),  # common area, builds cohesion edges
    "rest": Action("rest", 1),        # personal time, recovers stress
}


class CrewAgent:
    """One crew member inside a mission replication."""

    def __init__(self, cand_id: str, name: str, traits: dict[str, float],
                 rng: np.random.Generator):
        self.id = cand_id
        self.name = name
        self.traits = traits
        self.rng = rng

        self.stress = 20.0 + 5.0 * -self.trait("PSY_RES")  # resilient crews start calmer
        self.stress = float(np.clip(self.stress, 0.0, 100.0))
        self.fatigue = 0.0            # short-horizon fatigue, reset nightly
        self.hours_worked = 0.0
        self.tasks_completed = 0
        self.conflicts = 0
        self.exhausted_hours = 0.0

        self.machine = Machine(
            model=self,
            states=STATES,
            transitions=TRANSITIONS,
            initial="nominal",
            ignore_invalid_triggers=True,
        )

    # -- trait access -------------------------------------------------------

    def trait(self, code: str) -> float:
        """Latent trait value on a z-score scale; unknown codes -> 0.0."""
        return float(self.traits.get(code, 0.0))

    # -- FSM condition callbacks -------------------------------------------

    def is_over_stressed_threshold(self) -> bool:
        return self.stress >= STRESS_UP_STRESSED

    def is_over_exhausted_threshold(self) -> bool:
        return self.stress >= STRESS_UP_EXHAUSTED

    def is_under_stressed_recovery(self) -> bool:
        return self.stress <= STRESS_DOWN_STRESSED

    def is_under_nominal_recovery(self) -> bool:
        return self.stress <= STRESS_DOWN_NOMINAL

    def adjust_stress(self, delta: float) -> None:
        """Apply a stress change, damped by resilience, then update the FSM."""
        if delta > 0:
            delta *= float(np.clip(1.0 - 0.25 * self.trait("PSY_RES"), 0.4, 1.6))
        else:
            delta *= float(np.clip(1.0 + 0.15 * self.trait("PSY_RES"), 0.6, 1.5))
        self.stress = float(np.clip(self.stress + delta, 0.0, 100.0))
        # Fire only the triggers valid from the current state; conditions on
        # the transitions decide whether anything actually changes.
        if self.state in ("nominal", "stressed"):
            self.pressure()
        if self.state in ("stressed", "exhausted"):
            self.recover()

    # -- Utility AI ---------------------------------------------------------

    def action_utilities(self, context: dict) -> dict[str, float]:
        """Score each action given current mission context.

        ``context`` keys:
            backlog_urgency  - 0..1, how much work is piling up / overdue
            social_need      - 0..1, decays when the agent has not socialized
            gym_available    - bool
            workstation_available - bool
        """
        u: dict[str, float] = {}

        stress_frac = self.stress / 100.0
        urgency = float(context.get("backlog_urgency", 0.0))

        # Work: driven by conscientiousness + initiative + backlog pressure,
        # suppressed by stress/fatigue; impossible when exhausted or no station.
        if self.state != "exhausted" and context.get("workstation_available", True):
            u["work"] = (
                1.0
                + 0.55 * self.trait("PSY_CON")
                + 0.35 * self.trait("LDR_INI")
                + 1.6 * urgency
                - 0.9 * stress_frac
                - 0.4 * self.fatigue
            )

        # Exercise: endurance-inclined agents use it as a pressure valve.
        if context.get("gym_available", True):
            u["exercise"] = (
                0.4
                + 0.45 * self.trait("PHY_END")
                + 0.5 * stress_frac
                - 0.3 * self.fatigue
            )

        # Socialize: sociable agents seek it; rising stress makes agreeable
        # agents seek company and disagreeable agents avoid it.
        u["socialize"] = (
            0.5
            + 0.5 * self.trait("SOC_COH")
            + 0.6 * float(context.get("social_need", 0.5))
            + 0.25 * stress_frac * self.trait("SOC_AGR")
        )

        # Rest: the higher stress and fatigue climb, the louder it calls.
        u["rest"] = 0.3 + 1.4 * stress_frac + 0.8 * self.fatigue

        return u

    def choose_action(self, context: dict, temperature: float = 0.6) -> str:
        """Softmax selection over utilities - stochastic but trait-shaped.

        Lower temperature -> more deterministic (argmax-like) behavior.
        """
        utilities = self.action_utilities(context)
        names = list(utilities.keys())
        values = np.array([utilities[n] for n in names], dtype=float)
        z = (values - values.max()) / max(temperature, 1e-6)
        probs = np.exp(z)
        probs /= probs.sum()
        return str(self.rng.choice(names, p=probs))

    # -- effects of performing actions -------------------------------------

    def work_speed_multiplier(self) -> float:
        """Effort applied per hour of work, shaped by aptitude and state."""
        base = 1.0 + 0.30 * self.trait("COG_TEC") + 0.15 * self.trait("PSY_CON")
        if self.state == "stressed":
            base *= 1.0 - 0.25 * (1.0 - 0.3 * self.trait("COG_ADP"))
        return float(np.clip(base, 0.25, 2.0))

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return f"<CrewAgent {self.id} {self.state} stress={self.stress:.0f}>"
