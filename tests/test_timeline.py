"""Tests for timeline export functionality."""

from __future__ import annotations

import time

from autobot.living_engine import LivingEngine
from autobot.memory import Episode, EpisodicMemory


class TestTimelineExport:
    def test_empty_timeline(self):
        engine = LivingEngine()
        result = engine.export_timeline()
        assert "metadata" in result
        assert "people" in result
        assert "timeline" in result
        assert isinstance(result["timeline"], list)

    def test_metadata_fields(self):
        engine = LivingEngine()
        result = engine.export_timeline()
        meta = result["metadata"]
        assert meta["entity_name"] == "Ryn"
        assert "session_start" in meta
        assert "session_start_iso" in meta
        assert "export_time" in meta
        assert "export_time_iso" in meta
        assert "session_duration_seconds" in meta
        assert meta["total_ticks"] == 0
        assert meta["tick_interval_seconds"] == 15
        assert "current_needs" in meta
        assert "current_emotions" in meta
        assert "current_energy" in meta
        assert "dominant_mood" in meta
        assert "prediction_summary" in meta
        assert "thinking_history_note" in meta

    def test_conversation_entries(self):
        engine = LivingEngine()
        engine.state.get_or_create_person("p1", "Alice")
        engine.state.add_message("p1", "Hello!")
        engine.state.add_entity_message("p1", "Hi there!")

        result = engine.export_timeline()
        conv_entries = [
            e for e in result["timeline"]
            if e["category"] == "conversation"
        ]
        assert len(conv_entries) == 2

        human_msg = [e for e in conv_entries if e["event_type"] == "human_message"]
        entity_msg = [e for e in conv_entries if e["event_type"] == "entity_message"]
        assert len(human_msg) == 1
        assert len(entity_msg) == 1
        assert human_msg[0]["content"]["text"] == "Hello!"
        assert human_msg[0]["content"]["person_name"] == "Alice"
        assert entity_msg[0]["content"]["text"] == "Hi there!"

    def test_thinking_entries(self):
        engine = LivingEngine()
        engine.thinking_history.append({
            "tick": 1,
            "time": time.time(),
            "decision": "idle",
            "thinking": "I wonder what's out there.",
        })

        result = engine.export_timeline()
        thought_entries = [
            e for e in result["timeline"]
            if e["category"] == "thought"
        ]
        assert len(thought_entries) == 1
        assert thought_entries[0]["event_type"] == "thought_idle"
        assert thought_entries[0]["content"]["thinking"] == "I wonder what's out there."

    def test_episode_entries(self):
        engine = LivingEngine()
        ep = Episode(
            id=1, time=time.time(), summary="Met Alice",
            involved_entities=["Alice"], emotion_arousal=0.6,
            emotion_valence=0.4, salience_tags=["novelty"],
            source="observed", confidence=1.0, grounded=True,
        )
        engine.memory.episodes.append(ep)

        result = engine.export_timeline()
        mem_entries = [
            e for e in result["timeline"]
            if e["category"] == "memory"
        ]
        assert len(mem_entries) == 1
        assert mem_entries[0]["content"]["summary"] == "Met Alice"
        assert mem_entries[0]["content"]["source"] == "observed"
        assert mem_entries[0]["content"]["confidence"] == 1.0
        assert mem_entries[0]["content"]["grounded"] is True

    def test_goal_entries(self):
        engine = LivingEngine()
        g = engine.goal_engine.add_goal("Understand the world", priority=0.6)

        result = engine.export_timeline()
        goal_entries = [
            e for e in result["timeline"]
            if e["category"] == "goal"
        ]
        assert len(goal_entries) == 1
        assert goal_entries[0]["event_type"] == "goal_created"
        assert goal_entries[0]["content"]["description"] == "Understand the world"
        assert goal_entries[0]["content"]["status"] == "active"

    def test_completed_goal_gets_completion_entry(self):
        engine = LivingEngine()
        g = engine.goal_engine.add_goal("Learn something", priority=0.5)
        g.completed = True
        g.progress_score = 1.0
        g.last_progress_at = time.time() + 10  # after creation

        result = engine.export_timeline()
        goal_entries = [
            e for e in result["timeline"]
            if e["category"] == "goal"
        ]
        assert len(goal_entries) == 2
        types = [e["event_type"] for e in goal_entries]
        assert "goal_created" in types
        assert "goal_completed" in types

    def test_event_entries(self):
        engine = LivingEngine()
        engine.state.record({
            "type": "boredom",
            "intensity": 0.3,
        })

        result = engine.export_timeline()
        event_entries = [
            e for e in result["timeline"]
            if e["category"] == "event"
        ]
        assert len(event_entries) == 1
        assert event_entries[0]["event_type"] == "boredom"
        assert event_entries[0]["content"]["intensity"] == 0.3

    def test_chronological_order(self):
        engine = LivingEngine()
        now = time.time()

        # Add entries with known timestamps in non-chronological order
        engine.state.record({"type": "early", "time": now - 100})
        engine.thinking_history.append({
            "tick": 5, "time": now - 50, "decision": "idle",
            "thinking": "middle thought",
        })
        engine.state.record({"type": "late", "time": now})

        result = engine.export_timeline()
        timestamps = [e["timestamp"] for e in result["timeline"]]
        assert timestamps == sorted(timestamps)

    def test_people_snapshot(self):
        engine = LivingEngine()
        p = engine.state.get_or_create_person("p1", "Alice")
        p.trust = 0.8
        p.warmth = 0.7
        p.notes = ["interesting person"]

        result = engine.export_timeline()
        assert "p1" in result["people"]
        alice = result["people"]["p1"]
        assert alice["name"] == "Alice"
        assert alice["trust"] == 0.8
        assert alice["warmth"] == 0.7
        assert alice["notes"] == ["interesting person"]


class TestMoodDownsampling:
    def test_empty_history(self):
        engine = LivingEngine()
        result = engine._downsample_mood_history()
        assert result == []

    def test_short_history_included_fully(self):
        engine = LivingEngine()
        for i in range(10):
            engine.state.mood_history.append({
                "tick": i,
                "time": time.time() + i,
                "emotions": {"anxiety": 0.1, "optimism": 0.5},
                "arousal": 0.2,
                "valence": 0.1,
                "decision": "idle",
            })

        result = engine._downsample_mood_history()
        assert len(result) == 10

    def test_long_flat_history_downsampled(self):
        engine = LivingEngine()
        # 100 identical mood entries -- should collapse to just first + last
        for i in range(100):
            engine.state.mood_history.append({
                "tick": i,
                "time": time.time() + i,
                "emotions": {"anxiety": 0.1, "optimism": 0.5},
                "arousal": 0.2,
                "valence": 0.1,
                "decision": "idle",
            })

        result = engine._downsample_mood_history()
        assert len(result) == 2  # first + last only

    def test_changes_preserved(self):
        engine = LivingEngine()
        base_time = time.time()

        # 60 entries, with a big emotion change in the middle
        for i in range(60):
            anxiety = 0.1 if i < 30 else 0.5  # big jump at i=30
            engine.state.mood_history.append({
                "tick": i,
                "time": base_time + i,
                "emotions": {"anxiety": anxiety, "optimism": 0.5},
                "arousal": 0.2,
                "valence": 0.1,
                "decision": "idle",
            })

        result = engine._downsample_mood_history()
        # Should include first, the change point, and last (at minimum)
        assert len(result) >= 3
        # First and last are always included
        assert result[0]["tick"] == 0
        assert result[-1]["tick"] == 59

    def test_decision_change_preserved(self):
        engine = LivingEngine()
        base_time = time.time()

        for i in range(60):
            decision = "idle" if i < 30 else "respond"
            engine.state.mood_history.append({
                "tick": i,
                "time": base_time + i,
                "emotions": {"anxiety": 0.1, "optimism": 0.5},
                "arousal": 0.2,
                "valence": 0.1,
                "decision": decision,
            })

        result = engine._downsample_mood_history()
        decisions = [e["decision"] for e in result]
        assert "idle" in decisions
        assert "respond" in decisions
