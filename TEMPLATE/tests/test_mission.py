import numpy as np

from nfp.scoring.traits import load_crew, sample_trait_draws
from nfp.sim.agents import CrewAgent
from nfp.sim.mission import Mission, Scenario
from nfp.sim.montecarlo import run_batch

CREW = "data/crew/sample_crew.json"
SCENARIO = "scenarios/shakedown_14d.json"
ARCHETYPES = "data/archetypes/archetypes_sample.json"


def run_once(seed):
    rng = np.random.default_rng(seed)
    crew = load_crew(CREW)
    draws = sample_trait_draws(crew, rng)
    agents = [CrewAgent(c.id, c.name, draws[c.id], rng) for c in crew]
    scenario = Scenario.from_json(SCENARIO)
    return Mission(scenario, agents, rng).run()


def test_mission_completes_with_sane_metrics():
    m = run_once(seed=1)
    assert 0.0 <= m["completion_rate"] <= 1.0
    assert m["tasks_generated"] >= 14 * 5          # 5 routine tasks/day minimum
    assert isinstance(m["critical_cascade"], bool)
    assert -1.0 <= m["final_cohesion"] <= 1.0


def test_mission_is_deterministic_given_seed():
    assert run_once(seed=123) == run_once(seed=123)


def test_batch_shapes_and_archetype_probs():
    result = run_batch(CREW, SCENARIO, ARCHETYPES, n_runs=5, seed=9)
    assert len(result.per_run) == 5
    probs = result.archetype_probs()
    assert set(probs.keys()) == {"C001", "C002", "C003", "C004"}
    for cand_probs in probs.values():
        for p in cand_probs.values():
            assert 0.0 <= p <= 1.0
