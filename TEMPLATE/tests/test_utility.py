import numpy as np

from nfp.sim.agents import CrewAgent


def make_agent(traits, seed=0):
    return CrewAgent("T001", "Test", traits, np.random.default_rng(seed))


def test_utilities_shift_with_traits():
    """A conscientious agent should value work more than a lax one, all else equal."""
    ctx = {"backlog_urgency": 0.5, "social_need": 0.2,
           "workstation_available": True, "gym_available": True}
    diligent = make_agent({"PSY_CON": 1.5})
    lax = make_agent({"PSY_CON": -1.5})
    assert diligent.action_utilities(ctx)["work"] > lax.action_utilities(ctx)["work"]


def test_softmax_prefers_higher_utility():
    """Over many draws the top-utility action must dominate the choice counts."""
    ctx = {"backlog_urgency": 1.0, "social_need": 0.0,
           "workstation_available": True, "gym_available": True}
    agent = make_agent({"PSY_CON": 2.0, "LDR_INI": 2.0}, seed=7)
    counts = {}
    for _ in range(1000):
        choice = agent.choose_action(ctx)
        counts[choice] = counts.get(choice, 0) + 1
    assert counts.get("work", 0) > 600


def test_stress_fsm_hysteresis():
    agent = make_agent({})
    agent.stress = 20.0
    agent.adjust_stress(0.0)
    assert agent.state == "nominal"
    agent.adjust_stress(+70.0)          # push well past both thresholds
    agent.adjust_stress(+70.0)
    assert agent.state == "exhausted"
    agent.adjust_stress(-15.0)          # small dip: still above recovery line
    assert agent.state == "exhausted"
    agent.stress = 60.0                 # force below exhausted-recovery threshold
    agent.adjust_stress(0.0)
    assert agent.state == "stressed"
