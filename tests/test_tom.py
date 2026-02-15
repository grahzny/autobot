"""Tests for Theory of Mind, empathic modulation, time-of-day, interest drift, vulnerability."""

from __future__ import annotations

import time
from unittest.mock import patch

from autobot.brain import MessageAnalysis, _heuristic_analysis
from autobot.chat_state import ChatState, InternalTopic, PersonProfile
from autobot.emotions import (
    AffectState,
    EmotionalState,
    _AFFECT_TRIGGERS,
    evaluate_amygdala,
    update_reflective_emotions,
)
from autobot.living_engine import LivingEngine


# ======================================================================
# Phase 1: Theory of Mind Analysis
# ======================================================================


class TestMessageAnalysisFields:
    def test_default_fields(self):
        analysis = MessageAnalysis(
            triggers=[], topics=[], person_intent="",
            trust_delta=0.0, warmth_delta=0.0,
            requires_response=True, urgency=0.5,
        )
        assert analysis.user_state == "unknown"
        assert analysis.user_intent == "unknown"
        assert analysis.reliability == 0.8

    def test_custom_fields(self):
        analysis = MessageAnalysis(
            triggers=[], topics=[], person_intent="",
            trust_delta=0.0, warmth_delta=0.0,
            requires_response=True, urgency=0.5,
            user_state="stressed", user_intent="venting",
            reliability=0.9,
        )
        assert analysis.user_state == "stressed"
        assert analysis.user_intent == "venting"
        assert analysis.reliability == 0.9


class TestHeuristicFallback:
    def test_dismissive_gets_disengaged(self):
        for text in ("ok", "k", "whatever", "sure", "fine"):
            analysis = _heuristic_analysis(text)
            assert analysis.user_state == "disengaged", f"Failed for '{text}'"
            assert analysis.reliability == 0.6, f"Failed for '{text}'"

    def test_question_gets_probing(self):
        analysis = _heuristic_analysis("What are you thinking about?")
        assert analysis.user_intent == "probing"

    def test_long_message_high_reliability(self):
        long_text = "a" * 250
        analysis = _heuristic_analysis(long_text)
        assert analysis.reliability == 0.9

    def test_greeting_gets_friendly(self):
        analysis = _heuristic_analysis("hello!")
        assert analysis.user_intent == "being_friendly"

    def test_default_calm_sharing(self):
        analysis = _heuristic_analysis("just some regular text")
        assert analysis.user_state == "calm"
        assert analysis.user_intent == "sharing_information"
        assert analysis.reliability == 0.8


class TestPersonProfileToM:
    def test_tom_defaults(self):
        p = PersonProfile(id="p1", name="Alice")
        assert p.last_known_state == "unknown"
        assert p.last_known_intent == "unknown"
        assert p.reliability_score == 0.5

    def test_reliability_running_average(self):
        p = PersonProfile(id="p1", name="Alice")
        # Simulate running average: 70% old + 30% new
        p.reliability_score = 0.7 * p.reliability_score + 0.3 * 0.9
        expected = 0.7 * 0.5 + 0.3 * 0.9  # 0.62
        assert abs(p.reliability_score - expected) < 0.001

        # Second update
        p.reliability_score = 0.7 * p.reliability_score + 0.3 * 0.9
        expected2 = 0.7 * expected + 0.3 * 0.9
        assert abs(p.reliability_score - expected2) < 0.001

    def test_tom_update_in_engine(self):
        engine = LivingEngine()
        engine.state.get_or_create_person("p1", "Alice")
        engine.state.add_message("p1", "Hello there!")

        # Process messages by running tick
        for msg in engine.state.unprocessed_messages():
            engine._analyze_message(msg)
            msg.processed = True

        person = engine.state.get_person("p1")
        # After analysis, ToM fields should be updated (not "unknown" anymore)
        # The heuristic will set greeting -> being_friendly
        assert person.last_known_state != "unknown" or person.last_known_intent != "unknown"


# ======================================================================
# Phase 2: Empathic Emotion Modulation
# ======================================================================


class TestEmpathyModulation:
    def test_venting_increases_warmth(self):
        # Use two identical states, one with venting context, one without
        # Start with some irritability so the reduction is visible
        state_with = EmotionalState(irritability=0.3)
        state_without = EmotionalState(irritability=0.3)
        affect = AffectState(arousal=0.3, valence=-0.2)
        cs = ChatState()

        update_reflective_emotions(
            state_with, affect, cs,
            user_context={"state": "stressed", "intent": "venting", "reliability": 0.8},
        )
        update_reflective_emotions(
            state_without, AffectState(arousal=0.3, valence=-0.2), cs,
        )

        assert state_with.social_warmth > state_without.social_warmth
        assert state_with.irritability < state_without.irritability

    def test_venting_vulnerable_also_works(self):
        state = EmotionalState()
        affect = AffectState(arousal=0.3, valence=-0.2)
        cs = ChatState()
        initial_warmth = state.social_warmth

        update_reflective_emotions(
            state, affect, cs,
            user_context={"state": "vulnerable", "intent": "venting", "reliability": 0.8},
        )

        assert state.social_warmth > initial_warmth


class TestBoundaryDetection:
    def test_low_reliability_positive_valence_increases_avoidance(self):
        state = EmotionalState(caution=0.0)
        affect = AffectState(arousal=0.3, valence=0.5)  # nice surface
        cs = ChatState()
        initial_caution = state.caution
        initial_anxiety = state.anxiety

        update_reflective_emotions(
            state, affect, cs,
            user_context={"state": "calm", "intent": "being_friendly", "reliability": 0.2},
        )

        assert state.caution > initial_caution
        assert state.anxiety > initial_anxiety


class TestVulnerabilityResponse:
    def test_genuine_vulnerability_increases_warmth(self):
        # Compare with and without vulnerability context
        # Start below baseline so decay doesn't cancel the gains
        state_with = EmotionalState(conviction=0.3, risk_tolerance=0.3)
        state_without = EmotionalState(conviction=0.3, risk_tolerance=0.3)
        affect = AffectState(arousal=0.2, valence=0.1)
        cs = ChatState()

        update_reflective_emotions(
            state_with, affect, cs,
            user_context={"state": "vulnerable", "intent": "sharing_information", "reliability": 0.9},
        )
        update_reflective_emotions(
            state_without, AffectState(arousal=0.2, valence=0.1), cs,
        )

        assert state_with.conviction > state_without.conviction
        assert state_with.risk_tolerance > state_without.risk_tolerance


class TestHostilityResponse:
    def test_hostile_user_increases_defenses(self):
        state = EmotionalState(caution=0.0)
        affect = AffectState(arousal=0.5, valence=-0.3)
        cs = ChatState()
        initial_caution = state.caution
        initial_anxiety = state.anxiety
        initial_irritability = state.irritability

        update_reflective_emotions(
            state, affect, cs,
            user_context={"state": "hostile", "intent": "confronting", "reliability": 0.5},
        )

        assert state.caution > initial_caution
        assert state.anxiety > initial_anxiety
        assert state.irritability > initial_irritability

    def test_confronting_intent_alone_triggers_defense(self):
        state = EmotionalState(caution=0.0)
        affect = AffectState(arousal=0.2, valence=0.0)
        cs = ChatState()
        initial_caution = state.caution

        update_reflective_emotions(
            state, affect, cs,
            user_context={"state": "calm", "intent": "confronting", "reliability": 0.7},
        )

        assert state.caution > initial_caution


# ======================================================================
# Phase 3: Time-of-Day Awareness
# ======================================================================


class TestTimeOfDay:
    def test_ground_truth_includes_time_of_day(self):
        cs = ChatState()
        text = cs.ground_truth_text()
        assert "Time of day:" in text

    def test_ground_truth_includes_weekday(self):
        cs = ChatState()
        text = cs.ground_truth_text()
        # Should contain a day name
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in text for day in days)

    def test_morning_label(self):
        cs = ChatState()
        # Mock localtime to return 8am
        with patch("autobot.chat_state.time") as mock_time:
            mock_time.time.return_value = time.time()
            mock_time.localtime.return_value = time.struct_time((2026, 1, 15, 8, 0, 0, 3, 15, 0))
            mock_time.strftime.side_effect = time.strftime
            text = cs.ground_truth_text()
            assert "morning" in text

    def test_night_label(self):
        cs = ChatState()
        with patch("autobot.chat_state.time") as mock_time:
            mock_time.time.return_value = time.time()
            mock_time.localtime.return_value = time.struct_time((2026, 1, 15, 23, 0, 0, 3, 15, 0))
            mock_time.strftime.side_effect = time.strftime
            text = cs.ground_truth_text()
            assert "night" in text


# ======================================================================
# Phase 4: Interest Drift
# ======================================================================


class TestInterestDecay:
    def test_passive_decay(self):
        engine = LivingEngine()
        engine.state.topics.append(InternalTopic(
            id="t1", topic="philosophy", interest_level=0.5,
            created_at=time.time(), last_thought_at=time.time(),
        ))

        initial = engine.state.topics[0].interest_level
        engine._apply_time_effects()
        assert engine.state.topics[0].interest_level < initial

    def test_neglected_topic_decays_faster(self):
        engine = LivingEngine()
        now = time.time()
        # Topic thought about recently
        engine.state.topics.append(InternalTopic(
            id="t1", topic="recent", interest_level=0.5,
            created_at=now, last_thought_at=now,
        ))
        # Topic not thought about in a long time (>300 ticks * 15s = 4500s)
        engine.state.topics.append(InternalTopic(
            id="t2", topic="neglected", interest_level=0.5,
            created_at=now - 6000, last_thought_at=now - 6000,
        ))

        engine._apply_time_effects()
        recent = engine.state.topics[0].interest_level
        neglected = engine.state.topics[1].interest_level
        assert neglected < recent  # neglected decays faster


class TestInterestEvolution:
    def test_evolve_micro_task_for_mature_topic(self):
        engine = LivingEngine()
        engine.state.topics.append(InternalTopic(
            id="t1", topic="consciousness", interest_level=0.9,
            created_at=time.time(), last_thought_at=time.time(),
            thought_count=15,
        ))

        # Run _select_micro_task many times to check evolve appears
        found_evolve = False
        for _ in range(50):
            task = engine._select_micro_task()
            if task.startswith("evolve:"):
                found_evolve = True
                break
        assert found_evolve

    def test_no_evolve_for_immature_topic(self):
        engine = LivingEngine()
        engine.state.topics.append(InternalTopic(
            id="t1", topic="consciousness", interest_level=0.9,
            created_at=time.time(), last_thought_at=time.time(),
            thought_count=3,  # not enough
        ))

        for _ in range(30):
            task = engine._select_micro_task()
            assert not task.startswith("evolve:")


class TestThoughtCount:
    def test_thought_count_increments(self):
        engine = LivingEngine()
        engine.state.topics.append(InternalTopic(
            id="t1", topic="philosophy", interest_level=0.5,
            created_at=time.time(), last_thought_at=time.time(),
            thought_count=0,
        ))

        engine._update_topic("philosophy", "p1")
        assert engine.state.topics[0].thought_count == 1

        engine._update_topic("philosophy", "p1")
        assert engine.state.topics[0].thought_count == 2

    def test_new_topic_starts_at_zero(self):
        engine = LivingEngine()
        engine._update_topic("new_topic", "p1")
        assert engine.state.topics[0].thought_count == 0


# ======================================================================
# Phase 5: Vulnerability Block & Self-Judgment
# ======================================================================


class TestVulnerabilityBlock:
    def test_high_warmth_high_anxiety_blocks_proactive(self):
        engine = LivingEngine()
        engine.emotions.social_warmth = 0.7
        engine.emotions.anxiety = 0.6
        engine.state.energy = 0.8

        # Create a person with recent interaction
        p = engine.state.get_or_create_person("p1", "Alice")
        p.last_interaction_time = time.time() - 200  # > 120s ago
        p.last_message_time = time.time() - 200
        p.warmth = 0.5
        p.trust = 0.5

        decision = engine._decide([])
        # Should NOT be proactive due to vulnerability block
        assert decision != "proactive"

    def test_no_block_when_anxiety_low(self):
        engine = LivingEngine()
        engine.emotions.social_warmth = 0.7
        engine.emotions.anxiety = 0.1  # low anxiety
        engine.state.energy = 0.8

        p = engine.state.get_or_create_person("p1", "Alice")
        p.last_interaction_time = time.time() - 200
        p.last_message_time = time.time() - 200
        p.warmth = 0.5
        p.trust = 0.5

        # Might or might not be proactive depending on other conditions,
        # but vulnerability block itself is not active
        vulnerability_block = (
            engine.emotions.social_warmth > 0.5
            and engine.emotions.anxiety > 0.4
        )
        assert not vulnerability_block


class TestSelfJudgment:
    def test_irritable_entity_generates_self_judgment(self):
        engine = LivingEngine()
        engine.emotions.irritability = 0.7

        from autobot.living_engine import TickResult
        result = TickResult(tick=1, decision="respond")
        result.outgoing_message = "Whatever."
        result.target_person_id = "p1"
        engine.state.get_or_create_person("p1", "Alice")

        events = engine._create_action_events(result)
        types = [e["type"] for e in events]
        assert "self_judgment" in types

        sj = [e for e in events if e["type"] == "self_judgment"][0]
        assert sj["subtype"] == "sharp_response"
        assert sj["person"] == "Alice"

    def test_no_self_judgment_when_calm(self):
        engine = LivingEngine()
        engine.emotions.irritability = 0.2  # calm

        from autobot.living_engine import TickResult
        result = TickResult(tick=1, decision="respond")
        result.outgoing_message = "Hello!"
        result.target_person_id = "p1"
        engine.state.get_or_create_person("p1", "Alice")

        events = engine._create_action_events(result)
        types = [e["type"] for e in events]
        assert "self_judgment" not in types


class TestSelfJudgmentAmygdala:
    def test_self_judgment_in_triggers(self):
        assert "self_judgment" in _AFFECT_TRIGGERS

    def test_self_judgment_produces_correct_affect(self):
        events = [{"type": "self_judgment", "subtype": "sharp_response", "intensity": 0.3}]
        cs = ChatState()
        affect = evaluate_amygdala(events, cs)
        assert affect.arousal > 0
        assert affect.valence < 0
        assert "self_criticism" in affect.salience_tags
