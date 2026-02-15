"""Tests for the enhanced memory system (v2: recall penalty, diversity, suppression)."""

from __future__ import annotations

import time

from autobot.chat_state import ChatState
from autobot.emotions import AffectState
from autobot.memory import Episode, EpisodicMemory


class TestRecallPenalty:
    def test_recall_count_increases_on_retrieval(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.8, valence=0.5, salience_tags=["exchange"])
        events = [{"type": "meaningful_exchange", "person": "Alice"}]
        memory.maybe_create_episode(events, affect, cs)

        results = memory.retrieve(entities=["Alice"], limit=5)
        assert len(results) == 1
        assert results[0].recall_count == 1

        # Retrieve again
        results2 = memory.retrieve(entities=["Alice"], limit=5)
        assert results2[0].recall_count == 2

    def test_recall_penalty_reduces_score(self):
        ep = Episode(
            id=1, time=time.time(), summary="test",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
            recall_count=0,
        )
        score_fresh = ep.relevance_score(["Alice"], ["exchange"], 0.5)

        ep.recall_count = 5
        score_recalled = ep.relevance_score(["Alice"], ["exchange"], 0.5)
        assert score_recalled < score_fresh

    def test_recency_of_recall_penalty(self):
        ep = Episode(
            id=1, time=time.time(), summary="test",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
            recall_count=1, last_recalled_at=time.time(),  # just recalled
        )
        score = ep.relevance_score(["Alice"], ["exchange"], 0.5)

        ep2 = Episode(
            id=2, time=time.time(), summary="test2",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
            recall_count=1, last_recalled_at=time.time() - 7200,  # recalled 2h ago
        )
        score2 = ep2.relevance_score(["Alice"], ["exchange"], 0.5)
        assert score < score2  # recently recalled = lower score


class TestSuppression:
    def test_suppressed_episode_returns_negative_score(self):
        ep = Episode(
            id=1, time=time.time(), summary="suppressed",
            involved_entities=["Alice"], emotion_arousal=0.5,
            emotion_valence=0.3, salience_tags=["exchange"],
            suppressed_until=time.time() + 300,
        )
        score = ep.relevance_score(["Alice"], ["exchange"], 0.5)
        assert score == -1.0

    def test_suppress_looping_memories(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.8, valence=-0.3, salience_tags=["loneliness"])
        for i in range(3):
            events = [{"type": "self_reflection", "text": "feeling lonely"}]
            memory.maybe_create_episode(events, affect, cs)

        suppressed = memory.suppress_looping_memories("lonely", duration_seconds=60)
        assert suppressed > 0

        # Suppressed episodes should not appear in retrieval
        results = memory.retrieve(tags=["loneliness"], limit=5)
        for ep in results:
            assert ep.suppressed_until <= time.time()

    def test_suppressed_skipped_in_retrieve(self):
        memory = EpisodicMemory()
        cs = ChatState()

        # Create 2 episodes
        affect1 = AffectState(arousal=0.8, valence=0.5, salience_tags=["exchange"])
        events1 = [{"type": "meaningful_exchange", "person": "Alice"}]
        ep1 = memory.maybe_create_episode(events1, affect1, cs)

        affect2 = AffectState(arousal=0.7, valence=0.3, salience_tags=["novelty"])
        events2 = [{"type": "new_encounter", "person": "Bob"}]
        ep2 = memory.maybe_create_episode(events2, affect2, cs)

        # Suppress the first one
        ep1.suppressed_until = time.time() + 300

        results = memory.retrieve(entities=["Alice", "Bob"], limit=5)
        summaries = [r.summary for r in results]
        # ep1 should be excluded
        assert ep1.summary not in summaries or all(
            r.suppressed_until <= time.time() for r in results
        )


class TestDiversityFilter:
    def test_max_one_per_theme(self):
        memory = EpisodicMemory()
        cs = ChatState()

        # Create 3 episodes with same theme_tag
        for i in range(3):
            affect = AffectState(
                arousal=0.8 - i * 0.1, valence=0.5,
                salience_tags=["exchange"],
            )
            events = [{"type": "meaningful_exchange", "person": "Alice",
                       "details": f"convo {i}"}]
            memory.maybe_create_episode(events, affect, cs)

        results = memory.retrieve(
            entities=["Alice"], limit=5, enforce_diversity=True,
        )

        # With diversity filter, should get at most 1 per theme_tag
        theme_tags = [r.theme_tag for r in results if r.theme_tag]
        unique_tags = set(theme_tags)
        assert len(theme_tags) == len(unique_tags)

    def test_diversity_off_returns_all(self):
        memory = EpisodicMemory()
        cs = ChatState()

        for i in range(3):
            affect = AffectState(
                arousal=0.8 - i * 0.1, valence=0.5,
                salience_tags=["exchange"],
            )
            events = [{"type": "meaningful_exchange", "person": "Alice",
                       "details": f"convo {i}"}]
            memory.maybe_create_episode(events, affect, cs)

        results = memory.retrieve(
            entities=["Alice"], limit=5, enforce_diversity=False,
        )
        assert len(results) == 3


class TestThemeTag:
    def test_theme_tag_from_salience(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.8, valence=0.5, salience_tags=["novelty"])
        events = [{"type": "new_encounter", "person": "Alice"}]
        ep = memory.maybe_create_episode(events, affect, cs)
        assert ep is not None
        assert ep.theme_tag == "novelty"

    def test_theme_tag_from_person_when_no_tags(self):
        memory = EpisodicMemory()
        cs = ChatState()
        affect = AffectState(arousal=0.8, valence=0.5, salience_tags=[])
        events = [{"type": "meaningful_exchange", "person": "Alice"}]
        ep = memory.maybe_create_episode(events, affect, cs)
        assert ep is not None
        assert "Alice" in ep.theme_tag
