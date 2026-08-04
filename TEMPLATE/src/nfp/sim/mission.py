"""Mission engine: a SimPy discrete-event simulation of one habitat mission.

One ``Mission`` = one replication of the scenario with one sampled trait
vector per crew member. The Monte Carlo runner constructs many missions with
different seeds/draws and aggregates the outcomes.

Time unit is one hour. Days run 24h with a forced sleep window 23:00-07:00.
Agents wake, look at the mission context (backlog pressure, free equipment,
their own stress), pick actions via softmax utility, and act. A daily event
director injects failures, medical issues, and interpersonal friction whose
hazards depend on the evolving state of the crew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy

from .agents import ACTIONS, CrewAgent
from .events import EventDirector
from .social import RelationshipGraph

SLEEP_START = 23  # 23:00
SLEEP_END = 7     # 07:00


@dataclass
class Scenario:
    name: str
    duration_days: int
    resources: dict[str, int]
    tasks_per_day: dict[str, int]        # e.g. {"maintenance": 3, "science": 2}
    event_rates: dict[str, float]
    task_effort_h: dict[str, float] = field(
        default_factory=lambda: {"maintenance": 2.0, "science": 3.0, "repair": 3.0}
    )
    repair_sla_h: float = 12.0

    @classmethod
    def from_json(cls, path: str | Path) -> "Scenario":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            duration_days=int(raw["duration_days"]),
            resources=raw.get("resources", {"workstations": 2, "gym": 1}),
            tasks_per_day=raw.get("tasks_per_day", {"maintenance": 3, "science": 2}),
            event_rates=raw.get("event_rates", {}),
            task_effort_h=raw.get(
                "task_effort_h", {"maintenance": 2.0, "science": 3.0, "repair": 3.0}
            ),
            repair_sla_h=float(raw.get("repair_sla_h", 12.0)),
        )


@dataclass
class Task:
    kind: str
    effort_remaining: float
    created_h: float
    deadline_h: float | None = None   # only critical repairs carry deadlines
    critical: bool = False
    completed_h: float | None = None
    claimed_by: str | None = None     # prevents two agents working one task

    @property
    def done(self) -> bool:
        return self.effort_remaining <= 0.0


class Mission:
    def __init__(self, scenario: Scenario, agents: list[CrewAgent],
                 rng: np.random.Generator):
        self.scenario = scenario
        self.agents = agents
        self.rng = rng

        self.env = simpy.Environment()
        self.workstations = simpy.Resource(
            self.env, capacity=scenario.resources.get("workstations", 2))
        self.gym = simpy.Resource(self.env, capacity=scenario.resources.get("gym", 1))

        self.backlog: list[Task] = []
        self.completed: list[Task] = []
        self.graph = RelationshipGraph([a.id for a in agents])
        self.director = EventDirector(scenario.event_rates, rng)

        self.wear = 0.1                 # equipment wear in [0, 1]
        self.overdue_critical = 0       # critical repairs that blew their SLA
        self.social_need = {a.id: 0.5 for a in agents}
        self.medical_hold_until = {a.id: 0.0 for a in agents}

        # daily time series for the report
        self.ts_day: list[int] = []
        self.ts_cohesion: list[float] = []
        self.ts_stress: dict[str, list[float]] = {a.id: [] for a in agents}
        self.ts_backlog: list[int] = []

    # ------------------------------------------------------------------ util

    def hour_of_day(self) -> int:
        return int(self.env.now) % 24

    def mean_stress(self) -> float:
        return float(np.mean([a.stress for a in self.agents]))

    def backlog_urgency(self) -> float:
        """0..1 pressure signal for the utility functions."""
        if not self.backlog:
            return 0.0
        overdue = sum(
            1 for t in self.backlog
            if t.deadline_h is not None and self.env.now > t.deadline_h
        )
        load = min(len(self.backlog) / (3.0 * len(self.agents)), 1.0)
        return float(np.clip(0.6 * load + 0.4 * min(overdue, 3) / 3.0, 0.0, 1.0))

    def pick_task(self) -> Task | None:
        """Most urgent unclaimed task: critical-by-deadline first, then oldest."""
        open_tasks = [t for t in self.backlog if not t.done and t.claimed_by is None]
        if not open_tasks:
            return None
        open_tasks.sort(
            key=lambda t: (
                not t.critical,
                t.deadline_h if t.deadline_h is not None else float("inf"),
                t.created_h,
            )
        )
        return open_tasks[0]

    # ------------------------------------------------------------- processes

    def daily_process(self):
        """Runs each morning: generate routine tasks, sample the day's events."""
        day = 0
        while True:
            day += 1
            # routine task generation
            for kind, count in self.scenario.tasks_per_day.items():
                for _ in range(count):
                    self.backlog.append(Task(
                        kind=kind,
                        effort_remaining=self.scenario.task_effort_h.get(kind, 2.0),
                        created_h=self.env.now,
                    ))
            self.wear = float(np.clip(self.wear + 0.05, 0.0, 1.0))
            self.graph.decay(0.97)

            # event director samples adversity conditioned on crew state
            events = self.director.sample_day(
                day, self.mean_stress(), self.graph.cohesion(), self.wear)
            for ev in events:
                self.apply_event(ev)

            yield self.env.timeout(24.0)

    def apply_event(self, ev) -> None:
        if ev.kind == "equipment_failure":
            self.backlog.append(Task(
                kind="repair",
                effort_remaining=self.scenario.task_effort_h.get("repair", 3.0),
                created_h=self.env.now,
                deadline_h=self.env.now + self.scenario.repair_sla_h,
                critical=True,
            ))
        elif ev.kind == "medical_minor":
            agent = self.agents[int(self.rng.integers(len(self.agents)))]
            self.medical_hold_until[agent.id] = self.env.now + 4.0
            agent.adjust_stress(+10.0)
        elif ev.kind == "friction":
            a_id, b_id = self.graph.most_strained_pair(self.rng)
            a = next(x for x in self.agents if x.id == a_id)
            b = next(x for x in self.agents if x.id == b_id)
            self.graph.adjust(a_id, b_id, -0.25)
            a.adjust_stress(+12.0 * (1.0 - 0.2 * a.trait("SOC_AGR")))
            b.adjust_stress(+9.0 * (1.0 - 0.2 * b.trait("SOC_AGR")))
            a.conflicts += 1
            b.conflicts += 1

    def monitor_process(self):
        """Nightly snapshot at 22:00 for the report time series."""
        yield self.env.timeout(22.0)  # first snapshot on day 1 at 22:00
        day = 1
        while True:
            self.ts_day.append(day)
            self.ts_cohesion.append(self.graph.cohesion())
            self.ts_backlog.append(len([t for t in self.backlog if not t.done]))
            for a in self.agents:
                self.ts_stress[a.id].append(a.stress)
            day += 1
            yield self.env.timeout(24.0)

    def agent_process(self, agent: CrewAgent):
        while True:
            hour = self.hour_of_day()

            # forced sleep window with overnight recovery
            if hour >= SLEEP_START or hour < SLEEP_END:
                hours_to_wake = ((SLEEP_END - hour) % 24) or 24
                for _ in range(int(hours_to_wake)):
                    agent.adjust_stress(-1.4)
                    self._tick_bookkeeping(agent, 1.0, awake=False)
                    yield self.env.timeout(1.0)
                agent.fatigue = 0.0
                continue

            # medical hold: light duty, treated as rest
            if self.env.now < self.medical_hold_until[agent.id]:
                agent.adjust_stress(-1.0)
                self._tick_bookkeeping(agent, 1.0)
                yield self.env.timeout(1.0)
                continue

            context = {
                "backlog_urgency": self.backlog_urgency(),
                "social_need": self.social_need[agent.id],
                "workstation_available": self.workstations.count < self.workstations.capacity,
                "gym_available": self.gym.count < self.gym.capacity,
            }
            choice = agent.choose_action(context)

            if choice == "work":
                yield from self._do_work(agent)
            elif choice == "exercise":
                yield from self._do_exercise(agent)
            elif choice == "socialize":
                yield from self._do_socialize(agent)
            else:
                yield from self._do_rest(agent)

    # ----------------------------------------------------------- action impl

    def _tick_bookkeeping(self, agent: CrewAgent, hours: float,
                          awake: bool = True) -> None:
        self.social_need[agent.id] = float(
            np.clip(self.social_need[agent.id] + 0.03 * hours, 0.0, 1.0))
        if awake:
            # background confinement strain: isolation itself costs something,
            # every waking hour, before any task stress is added
            agent.adjust_stress(+0.30 * hours)
        if agent.state == "exhausted":
            agent.exhausted_hours += hours

    def _do_work(self, agent: CrewAgent):
        duration = ACTIONS["work"].duration_h
        with self.workstations.request() as req:
            result = yield req | self.env.timeout(1.0)
            if req not in result:
                # queued for an hour and gave up - mild frustration
                agent.adjust_stress(+1.5)
                self._tick_bookkeeping(agent, 1.0)
                return
            task = self.pick_task()
            if task is None:
                # nothing to do after all; convert to rest
                yield self.env.timeout(1.0)
                agent.adjust_stress(-1.0)
                self._tick_bookkeeping(agent, 1.0)
                return
            task.claimed_by = agent.id
            yield self.env.timeout(float(duration))
            task.claimed_by = None
            effort = duration * agent.work_speed_multiplier()
            task.effort_remaining -= effort
            agent.hours_worked += duration
            agent.fatigue = float(np.clip(agent.fatigue + 0.08 * duration, 0.0, 1.0))
            agent.adjust_stress(+2.8 * duration)
            self._tick_bookkeeping(agent, float(duration))

            if task.done and task in self.backlog:
                task.completed_h = self.env.now
                self.backlog.remove(task)
                self.completed.append(task)
                agent.tasks_completed += 1
                if task.critical:
                    if task.deadline_h is not None and self.env.now > task.deadline_h:
                        self.overdue_critical += 1
                    self.wear = float(np.clip(self.wear - 0.15, 0.0, 1.0))
                # cooperative glow: finishing work slightly bonds the crew
                others = [a for a in self.agents if a.id != agent.id]
                if others:
                    other = others[int(self.rng.integers(len(others)))]
                    self.graph.adjust(agent.id, other.id, +0.01)

    def _do_exercise(self, agent: CrewAgent):
        with self.gym.request() as req:
            result = yield req | self.env.timeout(1.0)
            if req not in result:
                self._tick_bookkeeping(agent, 1.0)
                return
            yield self.env.timeout(1.0)
            agent.adjust_stress(-3.0 * (1.0 + 0.2 * agent.trait("PHY_END")))
            agent.fatigue = float(np.clip(agent.fatigue + 0.02, 0.0, 1.0))
            self._tick_bookkeeping(agent, 1.0)

    def _do_socialize(self, agent: CrewAgent):
        yield self.env.timeout(1.0)
        others = [a for a in self.agents if a.id != agent.id]
        if others:
            other = others[int(self.rng.integers(len(others)))]
            warmth = 0.035 * (
                1.0 + 0.3 * (agent.trait("SOC_COH") + other.trait("SOC_COH")) / 2.0)
            self.graph.adjust(agent.id, other.id, float(np.clip(warmth, 0.0, 0.08)))
            agent.adjust_stress(-1.5)
            other.adjust_stress(-0.8)
        self.social_need[agent.id] = 0.0
        self._tick_bookkeeping(agent, 1.0)
        self.social_need[agent.id] = 0.0  # bookkeeping re-adds a sliver; re-zero

    def _do_rest(self, agent: CrewAgent):
        yield self.env.timeout(1.0)
        agent.adjust_stress(-2.5)
        agent.fatigue = float(np.clip(agent.fatigue - 0.15, 0.0, 1.0))
        self._tick_bookkeeping(agent, 1.0)

    # ---------------------------------------------------------------- runner

    def run(self) -> dict:
        self.env.process(self.daily_process())
        self.env.process(self.monitor_process())
        for agent in self.agents:
            self.env.process(self.agent_process(agent))
        self.env.run(until=self.scenario.duration_days * 24.0)

        # final SLA sweep: unfinished critical tasks past deadline count too
        for t in self.backlog:
            if t.critical and t.deadline_h is not None and self.env.now > t.deadline_h:
                self.overdue_critical += 1

        critical_done = [t for t in self.completed if t.critical]
        repairs_within_sla = sum(
            1 for t in critical_done
            if t.deadline_h is None or (t.completed_h or 0.0) <= t.deadline_h
        )
        n_critical = len(critical_done) + sum(
            1 for t in self.backlog if t.critical and not t.done)

        tasks_generated = len(self.completed) + len(self.backlog)
        metrics = {
            "tasks_generated": tasks_generated,
            "tasks_completed": len(self.completed),
            "completion_rate": (
                len(self.completed) / tasks_generated if tasks_generated else 1.0),
            "critical_tasks": n_critical,
            "repairs_within_sla": repairs_within_sla,
            "overdue_critical": self.overdue_critical,
            # >= 2 blown repair SLAs in one mission = a critical cascade
            "critical_cascade": self.overdue_critical >= 2,
            "conflict_events": int(sum(a.conflicts for a in self.agents) / 2),
            "final_cohesion": self.graph.cohesion(),
            "mean_stress_final": self.mean_stress(),
            "exhausted_agent_hours": float(
                sum(a.exhausted_hours for a in self.agents)),
        }
        return metrics
