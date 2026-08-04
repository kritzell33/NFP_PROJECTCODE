"""Event director: a probabilistic "storyteller" for mission scenarios.

Modeled on how colony sims schedule adversity: each day the director samples
scenario events from rates defined in the scenario JSON, *conditioned on the
current simulation state* (crew stress, cohesion, equipment wear). Because it
is just a generative model, it slots naturally into the Bayesian framing -
scenario event rates are parameters you can later calibrate or sweep.

Event types in v0.1:
    equipment_failure   -> injects an urgent repair task with an SLA deadline
    medical_minor       -> sidelines a random agent for a few hours + stress
    friction            -> interpersonal conflict between a sampled pair
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MissionEvent:
    day: int
    kind: str
    detail: str
    agents: tuple[str, ...] = ()


class EventDirector:
    def __init__(self, event_rates: dict[str, float], rng: np.random.Generator):
        self.rates = event_rates
        self.rng = rng
        self.log: list[MissionEvent] = []

    def sample_day(self, day: int, mean_stress: float, cohesion: float,
                   wear: float) -> list[MissionEvent]:
        """Sample the day's events given current mission state.

        * Equipment failure hazard grows with accumulated wear.
        * Friction hazard grows with mean crew stress and drops with cohesion,
          which is exactly the feedback loop the ICE literature describes:
          strained crews generate more friction, which strains them further.
        """
        events: list[MissionEvent] = []

        fail_rate = self.rates.get("equipment_failure", 0.0) * (1.0 + 0.8 * wear)
        n_failures = self.rng.poisson(fail_rate)
        for i in range(n_failures):
            events.append(MissionEvent(day, "equipment_failure",
                                       f"Subsystem fault #{i + 1} on day {day}"))

        if self.rng.random() < self.rates.get("medical_minor", 0.0):
            events.append(MissionEvent(day, "medical_minor",
                                       "Minor medical issue - crew member on light duty"))

        stress_frac = mean_stress / 100.0
        friction_rate = self.rates.get("friction_base", 0.0) * (
            0.5 + 1.5 * stress_frac
        ) * float(np.clip(1.2 - 0.6 * cohesion, 0.3, 1.6))
        if self.rng.random() < friction_rate:
            events.append(MissionEvent(day, "friction",
                                       "Interpersonal friction incident"))

        self.log.extend(events)
        return events
