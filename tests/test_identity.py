"""Tests for the identity system and birth routine."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from autobot.chat_state import ChatState
from autobot.identity import Identity
from autobot.living_engine import LivingEngine


class TestIdentity:
    def test_default_values(self):
        ident = Identity.default()
        assert ident.name == "Ryn"
        assert ident.personality == ""
        assert ident.seed_interests == []
        assert ident.seed_goal == ""
        assert ident.seed_goal_category == "self"

    def test_from_file_missing_returns_default(self):
        ident = Identity.from_file("/nonexistent/path/identity.yaml")
        assert ident.name == "Ryn"
        assert ident.personality == ""

    def test_from_file_json(self):
        data = {
            "name": "TestBot",
            "personality": "Very curious.",
            "seed_interests": ["math", "music"],
            "seed_goal": "Learn things",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            ident = Identity.from_file(f.name)
        assert ident.name == "TestBot"
        assert ident.personality == "Very curious."
        assert ident.seed_interests == ["math", "music"]
        assert ident.seed_goal == "Learn things"

    def test_from_file_yaml(self):
        try:
            import yaml
        except ImportError:
            return  # skip if yaml not installed

        data = {
            "name": "YamlBot",
            "personality": "Thoughtful and reserved.",
            "seed_interests": ["philosophy"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            f.flush()
            ident = Identity.from_file(f.name)
        assert ident.name == "YamlBot"
        assert ident.personality == "Thoughtful and reserved."
        assert ident.seed_interests == ["philosophy"]
        assert ident.seed_goal == ""  # uses default

    def test_partial_config_uses_defaults(self):
        data = {"name": "PartialBot"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            ident = Identity.from_file(f.name)
        assert ident.name == "PartialBot"
        assert ident.personality == ""
        assert ident.seed_interests == []

    def test_malformed_file_returns_default(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            f.flush()
            ident = Identity.from_file(f.name)
        assert ident.name == "Ryn"


class TestBirthRoutine:
    def test_birth_sets_name(self):
        ident = Identity(name="TestEntity")
        engine = LivingEngine(identity=ident)
        engine._birth()
        assert engine.state.name == "TestEntity"

    def test_birth_creates_topics(self):
        ident = Identity(seed_interests=["math", "music", "art"])
        engine = LivingEngine(identity=ident)
        engine._birth()
        topic_names = [t.topic for t in engine.state.topics]
        assert "math" in topic_names
        assert "music" in topic_names
        assert "art" in topic_names

    def test_birth_creates_goal(self):
        ident = Identity(seed_goal="Find meaning", seed_goal_category="self")
        engine = LivingEngine(identity=ident)
        engine._birth()
        assert len(engine.goal_engine.goals) == 1
        assert engine.goal_engine.goals[0].description == "Find meaning"
        assert engine.goal_engine.goals[0].category == "self"

    def test_birth_creates_memory(self):
        ident = Identity(name="TestBot")
        engine = LivingEngine(identity=ident)
        engine._birth()
        assert len(engine.memory.episodes) == 1
        ep = engine.memory.episodes[0]
        assert "TestBot" in ep.summary
        assert "birth" in ep.salience_tags

    def test_birth_creates_thinking(self):
        ident = Identity(
            name="TestBot",
            seed_interests=["philosophy"],
        )
        engine = LivingEngine(identity=ident)
        engine._birth()
        assert len(engine.thinking_history) == 1
        assert engine.thinking_history[0]["decision"] == "birth"
        assert "philosophy" in engine.last_thinking

    def test_birth_runs_only_once(self):
        ident = Identity(seed_interests=["a", "b"])
        engine = LivingEngine(identity=ident)
        engine._birth()
        engine._birth()  # second call should be a no-op
        assert len(engine.state.topics) == 2  # not 4

    def test_backward_compat_without_identity(self):
        engine = LivingEngine()
        engine._birth()
        assert engine.state.name == "Ryn"
        assert len(engine.state.topics) == 0
        assert len(engine.goal_engine.goals) == 0
        # Birth memory still created
        assert len(engine.memory.episodes) == 1

    def test_birth_no_goal_when_empty(self):
        ident = Identity(seed_goal="")
        engine = LivingEngine(identity=ident)
        engine._birth()
        assert len(engine.goal_engine.goals) == 0


class TestFirstEncounterAwareness:
    def test_person_text_first_time(self):
        engine = LivingEngine()
        person = engine.state.get_or_create_person("p1", "Alice")
        assert person.familiarity == 0.0
        text = engine._person_text(person)
        assert "FIRST TIME" in text
        assert "NEVER spoken" in text

    def test_person_text_barely_met(self):
        engine = LivingEngine()
        person = engine.state.get_or_create_person("p1", "Alice")
        person.familiarity = 0.05
        text = engine._person_text(person)
        assert "barely met" in text
        assert "FIRST TIME" not in text

    def test_person_text_familiar(self):
        engine = LivingEngine()
        person = engine.state.get_or_create_person("p1", "Alice")
        person.familiarity = 0.5
        text = engine._person_text(person)
        assert "FIRST TIME" not in text
        assert "barely met" not in text

    def test_memories_text_empty(self):
        engine = LivingEngine()
        # No birth call, no memories at all
        text = engine._memories_text("Alice")
        assert "never interacted" in text.lower() or "Do NOT reference" in text

    def test_personality_text_empty(self):
        engine = LivingEngine()
        assert engine._personality_text() == ""

    def test_personality_text_with_identity(self):
        ident = Identity(personality="Curious and cautious.")
        engine = LivingEngine(identity=ident)
        text = engine._personality_text()
        assert "Curious and cautious." in text
