"""Tests for epistemic grounding: ground truth, epistemic tagging, grounding gate, micro-tasks."""

from __future__ import annotations

import time

from autobot.chat_state import ChatState
from autobot.grounding import check_grounding, ground_notes
from autobot.living_engine import LivingEngine
from autobot.memory import Episode, EpisodicMemory


class TestGroundTruthText:
    def test_empty_world(self):
        cs = ChatState()
        text = cs.ground_truth_text()
        assert "NOT met anyone" in text
        assert "GROUND TRUTH" in text

    def test_with_person_no_conversation(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        text = cs.ground_truth_text(person_id="p1")
        assert "Alice" in text
        assert "NEVER spoken" in text

    def test_with_person_after_conversation(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        cs.add_message("p1", "Hello!")
        cs.add_entity_message("p1", "Hi there!")
        text = cs.ground_truth_text(person_id="p1")
        assert "Alice" in text
        assert "2" in text  # 2 messages

    def test_multiple_people(self):
        cs = ChatState()
        cs.get_or_create_person("p1", "Alice")
        cs.get_or_create_person("p2", "Bob")
        text = cs.ground_truth_text()
        assert "Alice" in text
        assert "Bob" in text

    def test_not_established_fact_warning(self):
        cs = ChatState()
        text = cs.ground_truth_text()
        assert "NOT established fact" in text

    def test_uptime_and_tick(self):
        cs = ChatState()
        cs.tick_count = 42
        text = cs.ground_truth_text()
        assert "42" in text


class TestEpistemicTagging:
    def test_episode_default_source(self):
        ep = Episode(
            id=1, time=time.time(), summary="test",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
        )
        assert ep.source == "observed"
        assert ep.confidence == 1.0
        assert ep.grounded is True

    def test_imagined_episode(self):
        ep = Episode(
            id=1, time=time.time(), summary="thinking about life",
            involved_entities=[], emotion_arousal=0.3,
            emotion_valence=0.0, salience_tags=["self_reflection"],
            source="imagined", confidence=0.7, grounded=False,
        )
        assert ep.source == "imagined"
        assert ep.confidence == 0.7
        assert ep.grounded is False

    def test_imagined_episode_annotated_in_prompt(self):
        memory = EpisodicMemory()
        ep = Episode(
            id=1, time=time.time(), summary="wondering about stars",
            involved_entities=[], emotion_arousal=0.3,
            emotion_valence=0.2, salience_tags=[],
            source="imagined",
        )
        memory.episodes.append(ep)
        text = memory.to_prompt_text()
        assert "YOUR OWN THOUGHT" in text

    def test_observed_episode_no_annotation(self):
        memory = EpisodicMemory()
        ep = Episode(
            id=1, time=time.time(), summary="talked to Alice",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
            source="observed",
        )
        memory.episodes.append(ep)
        text = memory.to_prompt_text()
        assert "YOUR OWN THOUGHT" not in text

    def test_confidence_decay_imagined(self):
        memory = EpisodicMemory()
        ep = Episode(
            id=1, time=time.time(), summary="thought",
            involved_entities=[], emotion_arousal=0.3,
            emotion_valence=0.0, salience_tags=[],
            source="imagined", confidence=1.0,
        )
        memory.episodes.append(ep)
        memory.decay_confidence(rate=0.1)
        assert ep.confidence == 0.9

    def test_confidence_no_decay_observed(self):
        memory = EpisodicMemory()
        ep = Episode(
            id=1, time=time.time(), summary="real event",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=[],
            source="observed", confidence=1.0,
        )
        memory.episodes.append(ep)
        memory.decay_confidence(rate=0.1)
        assert ep.confidence == 1.0

    def test_confidence_clamps_to_zero(self):
        memory = EpisodicMemory()
        ep = Episode(
            id=1, time=time.time(), summary="thought",
            involved_entities=[], emotion_arousal=0.3,
            emotion_valence=0.0, salience_tags=[],
            source="imagined", confidence=0.05,
        )
        memory.episodes.append(ep)
        memory.decay_confidence(rate=0.1)
        assert ep.confidence == 0.0


class TestGroundingGate:
    def test_clean_text_no_flags(self):
        result = check_grounding(
            "I'm feeling curious today.",
            known_people=set(), known_topics=set(),
        )
        assert result.ungrounded_count == 0

    def test_flags_memory_claim(self):
        result = check_grounding(
            "I remember when we discussed quantum physics last week.",
            known_people=set(), known_topics=set(),
            conversation_exists=False,
        )
        assert result.ungrounded_count > 0
        assert any("memory_claim" in f for f in result.flags)

    def test_no_flag_with_known_person_and_conversation(self):
        result = check_grounding(
            "I remember Alice mentioned something interesting.",
            known_people={"Alice"}, known_topics=set(),
            conversation_exists=True,
        )
        # Known person + existing conversation = grounded
        assert result.ungrounded_count == 0

    def test_flags_fact_claim(self):
        result = check_grounding(
            "It turns out that the universe is actually infinite.",
            known_people=set(), known_topics=set(),
        )
        assert result.ungrounded_count > 0
        assert any("fact_claim" in f for f in result.flags)

    def test_no_flag_on_emotion(self):
        result = check_grounding(
            "I'm feeling restless and bored.",
            known_people=set(), known_topics=set(),
        )
        assert result.ungrounded_count == 0

    def test_subjective_notes_pass_through(self):
        notes = ground_notes(
            ["Alice seems interested in art"],
            known_people={"Alice"},
            conversation_messages=[],
        )
        assert notes[0] == "Alice seems interested in art"

    def test_invented_factual_notes_flagged(self):
        notes = ground_notes(
            ["Alice works at NASA and has a doctorate"],
            known_people={"Alice"},
            conversation_messages=[{"text": "Hello!", "role": "human"}],
        )
        assert "[unverified]" in notes[0]

    def test_grounded_notes_from_conversation(self):
        notes = ground_notes(
            ["Alice likes painting and art"],
            known_people={"Alice"},
            conversation_messages=[
                {"text": "I really enjoy painting and doing art", "role": "human"},
            ],
        )
        # "painting" and "art" appear in messages, so this is grounded
        assert "[unverified]" not in notes[0]


class TestMicroTaskSelector:
    def test_returns_a_task(self):
        engine = LivingEngine()
        task = engine._select_micro_task()
        assert isinstance(task, str)
        assert len(task) > 0

    def test_observe_when_silent(self):
        engine = LivingEngine()
        # No interactions = infinite silence > 60s
        task = engine._select_micro_task()
        # Should always have at least "observe" as a fallback
        assert task is not None

    def test_tidy_removes_stale_topics(self):
        engine = LivingEngine()
        engine._birth()
        # Add a low-interest topic
        from autobot.chat_state import InternalTopic
        engine.state.topics.append(InternalTopic(
            id="stale", topic="boring stuff", interest_level=0.1,
        ))
        initial_count = len(engine.state.topics)
        # Call multiple times to increase chance of getting "tidy"
        for _ in range(20):
            engine._select_micro_task()
        # Stale topic should have been removed
        remaining_topics = [t.topic for t in engine.state.topics]
        assert "boring stuff" not in remaining_topics

    def test_tidy_abandons_stalled_goals(self):
        engine = LivingEngine()
        g = engine.goal_engine.add_goal("stuck goal", priority=0.5)
        g.stall_counter = 35  # > 30 threshold
        for _ in range(20):
            engine._select_micro_task()
        assert g.abandoned

    def test_always_returns_something(self):
        engine = LivingEngine()
        task = engine._select_micro_task()
        assert task is not None
        assert ":" in task  # format is "type: description"
