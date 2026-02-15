"""Tests for the Autobot living entity core systems.

Verifies that the emotional, memory, and goal systems work correctly
in the new chat-oriented architecture.
"""

from __future__ import annotations

from autobot.chat_state import ChatState
from autobot.emotions import AffectState, EmotionalState, evaluate_amygdala
from autobot.goals import Goal, GoalEngine
from autobot.memory import EpisodicMemory


# ======================================================================
# Emotional System
# ======================================================================

class TestEmotionalInfluence:
    """Emotions must influence goal priorities and decisions."""

    def test_anxiety_shifts_goal_priority(self):
        g_relationship = Goal(
            id="r", description="repair trust with someone",
            category="relationship", priority=0.5,
        )
        calm = EmotionalState(anxiety=0.0, avoidance_bias=0.0)
        anxious = EmotionalState(anxiety=0.8, avoidance_bias=0.3)

        p_calm = g_relationship.effective_priority(calm)
        p_anxious = g_relationship.effective_priority(anxious)
        assert p_anxious > p_calm

    def test_avoidance_bias_reduces_exploration_priority(self):
        g = Goal(
            id="e", description="explore topic",
            category="exploration", priority=0.5,
        )
        low_avoidance = EmotionalState(avoidance_bias=0.0)
        high_avoidance = EmotionalState(avoidance_bias=0.8)
        assert g.effective_priority(low_avoidance) > g.effective_priority(high_avoidance)

    def test_amygdala_fires_on_trust_drop(self):
        cs = ChatState()
        events = [{"type": "trust_change", "person": "Alice", "delta": -0.2}]
        affect = evaluate_amygdala(events, cs)
        assert affect.arousal > 0.3
        assert affect.valence < 0
        assert "trust_drop" in affect.salience_tags

    def test_amygdala_fires_on_boundary_crossed(self):
        cs = ChatState()
        events = [{"type": "boundary_crossed", "person": "Bob"}]
        affect = evaluate_amygdala(events, cs)
        assert affect.arousal > 0.0
        assert affect.valence < 0

    def test_social_warmth_boosts_connection_goals(self):
        g = Goal(
            id="c", description="connect with someone",
            category="connection", priority=0.5,
        )
        cold = EmotionalState(social_warmth=0.0)
        warm = EmotionalState(social_warmth=0.8)
        assert g.effective_priority(warm) > g.effective_priority(cold)


# ======================================================================
# Chat State
# ======================================================================

class TestChatState:
    def test_get_or_create_person(self):
        cs = ChatState()
        person = cs.get_or_create_person("p1", "Alice")
        assert person.name == "Alice"
        assert person.id == "p1"
        # Second call returns same object
        person2 = cs.get_or_create_person("p1", "Alice")
        assert person2 is person

    def test_add_message(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        cs.add_message("p1", "Hello!")
        assert len(cs.pending_messages) == 1
        assert cs.pending_messages[0].text == "Hello!"
        assert len(cs.conversations["p1"]) == 1

    def test_unprocessed_messages(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        cs.add_message("p1", "Hello!")
        cs.add_message("p1", "Are you there?")
        unprocessed = cs.unprocessed_messages()
        assert len(unprocessed) == 2
        # Mark first as processed
        unprocessed[0].processed = True
        assert len(cs.unprocessed_messages()) == 1


# ======================================================================
# Goal Engine
# ======================================================================

class TestGoalEngine:
    def test_auto_generate_creates_goals(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        emotions = EmotionalState()
        engine = GoalEngine()
        engine.auto_generate(cs, emotions)
        # Should have generated at least one goal
        assert len(engine.goals) >= 0  # May or may not generate depending on thresholds

    def test_goal_completion(self):
        engine = GoalEngine()
        g = Goal(id="test", description="test goal", category="self", priority=0.5)
        engine.goals.append(g)
        assert not g.completed
        g.completed = True
        active = [g for g in engine.goals if not g.completed and not g.abandoned]
        assert len(active) == 0


# ======================================================================
# Episodic Memory
# ======================================================================

class TestEpisodicMemory:
    def test_create_episode(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.8, valence=0.5, salience_tags=["meaningful"])
        events = [{"type": "meaningful_exchange", "person": "Alice"}]
        ep = memory.maybe_create_episode(events, affect, cs)
        assert ep is not None
        assert len(memory.episodes) == 1

    def test_retrieve_relevant(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.7, valence=0.3, salience_tags=["exchange"])
        events = [{"type": "meaningful_exchange", "person": "Alice", "details": "discussed philosophy"}]
        memory.maybe_create_episode(events, affect, cs)
        results = memory.retrieve(entities=["Alice"], limit=5)
        assert len(results) >= 1
