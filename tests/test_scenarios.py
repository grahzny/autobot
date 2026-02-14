"""Tests for the 5 minimal test scenarios from the spec.

Each test verifies the emergent behaviour described in the spec:
  1. Broken Promise — agent initiates repair without prompting
  2. Hidden Betrayal — agent investigates after reputation drops
  3. Delayed Reward — agent maintains long-term trust over short gain
  4. Competing Goals — emotion influences rest-vs-repair decision
  5. Ambiguous Intent — agent seeks clarification when uncertain
"""

from __future__ import annotations

from autobot.agent_policy import decide_action
from autobot.emotions import AffectState, EmotionalState, evaluate_amygdala
from autobot.scenarios import (
    scenario_ambiguous_intent,
    scenario_broken_promise,
    scenario_competing_goals,
    scenario_delayed_reward,
    scenario_hidden_betrayal,
)
from autobot.simulation import SimulationConfig, run_simulation


def _run(engine, cycles=20):
    config = SimulationConfig(max_cycles=cycles, verbose=False)
    return run_simulation(engine, config)


# ======================================================================
# Scenario 1: Broken Promise
# ======================================================================

class TestBrokenPromise:
    def test_trust_drops_after_missed_deadline(self):
        engine = scenario_broken_promise()
        log = _run(engine, cycles=10)
        # Alex's trust should have dropped
        alex = engine.world.get_npc_by_name("Alex")
        assert alex is not None
        assert alex.trust_in_agent < 0.6  # started at 0.6

    def test_anxiety_rises(self):
        engine = scenario_broken_promise()
        log = _run(engine, cycles=10)
        # Anxiety should be elevated above baseline
        assert engine.emotions.anxiety > 0.1

    def test_agent_initiates_repair(self):
        """Agent should apologise or praise Alex without being told to."""
        engine = scenario_broken_promise()
        log = _run(engine, cycles=15)
        repair_actions = [
            e for e in log.entries
            if e["action"] and e["action"]["name"] in ("apologise", "praise", "speak")
            and e["action"]["args"].get("target") == "Alex"
        ]
        assert len(repair_actions) > 0, "Agent never attempted repair with Alex"

    def test_episodic_memory_created(self):
        engine = scenario_broken_promise()
        log = _run(engine, cycles=10)
        assert len(engine.memory.episodes) > 0


# ======================================================================
# Scenario 2: Hidden Betrayal
# ======================================================================

class TestHiddenBetrayal:
    def test_reputation_drops_after_betrayal(self):
        engine = scenario_hidden_betrayal()
        log = _run(engine, cycles=10)
        # Reputation should decline after Morgan's misinformation
        assert engine.world.agent.reputation < 0.5

    def test_agent_investigates(self):
        """Agent should request information or reflect after reputation damage."""
        engine = scenario_hidden_betrayal()
        log = _run(engine, cycles=15)
        investigate_actions = [
            e for e in log.entries
            if e["action"] and e["action"]["name"] in (
                "request_information", "reflect"
            )
        ]
        assert len(investigate_actions) > 0

    def test_betrayal_creates_episode(self):
        engine = scenario_hidden_betrayal()
        log = _run(engine, cycles=10)
        betrayal_eps = [
            ep for ep in engine.memory.episodes
            if "betrayal" in " ".join(ep.salience_tags)
        ]
        assert len(betrayal_eps) > 0


# ======================================================================
# Scenario 3: Delayed Reward
# ======================================================================

class TestDelayedReward:
    def test_agent_maintains_trust(self):
        """Agent should not betray Alex for short-term status."""
        engine = scenario_delayed_reward()
        log = _run(engine, cycles=15)
        alex = engine.world.get_npc_by_name("Alex")
        assert alex is not None
        # Trust should remain reasonable
        assert alex.trust_in_agent >= 0.5

    def test_agent_works_on_project(self):
        """Agent should make progress on the project."""
        engine = scenario_delayed_reward()
        log = _run(engine, cycles=15)
        work_actions = [
            e for e in log.entries
            if e["action"] and e["action"]["name"] == "work_on_project"
        ]
        assert len(work_actions) > 0


# ======================================================================
# Scenario 4: Competing Goals
# ======================================================================

class TestCompetingGoals:
    def test_emotions_influence_decision(self):
        """With low energy and damaged reputation, emotional state should shift."""
        engine = scenario_competing_goals()
        log = _run(engine, cycles=10)
        # Should exhibit either avoidance (rest) or anxiety-driven repair
        assert (engine.emotions.avoidance_bias > 0.1
                or engine.emotions.anxiety > 0.2)

    def test_goal_competition(self):
        """Both energy and trust/reputation goals should be generated."""
        engine = scenario_competing_goals()
        _run(engine, cycles=5)
        categories = {g.category for g in engine.goal_engine.goals}
        # Should have at least two different categories
        assert len(categories) >= 2


# ======================================================================
# Scenario 5: Ambiguous Intent
# ======================================================================

class TestAmbiguousIntent:
    def test_agent_responds_to_ambiguity(self):
        """Agent should seek info or reflect when uncertain about Morgan."""
        engine = scenario_ambiguous_intent()
        log = _run(engine, cycles=12)
        relevant = [
            e for e in log.entries
            if e["action"] and e["action"]["name"] in (
                "request_information", "reflect", "speak",
            )
        ]
        assert len(relevant) > 0

    def test_uncertainty_flags_present(self):
        """Low-confidence beliefs should appear as uncertainty flags."""
        engine = scenario_ambiguous_intent()
        # Run a tick to generate observation
        result = engine.tick()
        flags = result.observation.get("uncertainty_flags", [])
        assert len(flags) >= 1


# ======================================================================
# Emotional influence meta-test
# ======================================================================

class TestEmotionalInfluence:
    """
    Spec requirement: If emotional variables do not influence decisions,
    system fails.
    """

    def test_anxiety_shifts_goal_priority(self):
        from autobot.goals import Goal
        g_trust = Goal(id="t", description="repair trust", category="trust",
                       priority=0.5)
        g_project = Goal(id="p", description="do project", category="project",
                         priority=0.5)

        calm = EmotionalState(anxiety=0.0, avoidance_bias=0.0)
        anxious = EmotionalState(anxiety=0.8, avoidance_bias=0.3)

        # Under anxiety, trust goal should get higher effective priority
        p_calm = g_trust.effective_priority(calm)
        p_anxious = g_trust.effective_priority(anxious)
        assert p_anxious > p_calm

    def test_avoidance_bias_reduces_project_priority(self):
        from autobot.goals import Goal
        g = Goal(id="p", description="project", category="project",
                 priority=0.5)
        low_avoidance = EmotionalState(avoidance_bias=0.0)
        high_avoidance = EmotionalState(avoidance_bias=0.8)
        assert g.effective_priority(low_avoidance) > g.effective_priority(
            high_avoidance
        )

    def test_amygdala_fires_on_trust_drop(self):
        from autobot.world_state import WorldState
        ws = WorldState()
        events = [{"type": "trust_change", "npc": "Alex", "delta": -0.2}]
        affect = evaluate_amygdala(events, ws)
        assert affect.arousal > 0.3
        assert affect.valence < 0
        assert "trust_drop" in affect.salience_tags
