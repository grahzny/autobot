"""Tests for the enhanced goal system (v2: progress, stall, rumination, needs)."""

from __future__ import annotations

from autobot.chat_state import ChatState
from autobot.emotions import EmotionalState
from autobot.goals import Goal, GoalEngine
from autobot.needs import NeedsState
from autobot.repetition import RepetitionTracker


class TestGoalProgress:
    def test_record_progress(self):
        engine = GoalEngine()
        g = engine.add_goal("test goal", priority=0.5)
        engine.record_progress(g.id, delta=0.3, next_step="do thing")
        assert g.progress_score == 0.3
        assert g.stall_counter == 0
        assert "do thing" in g.next_steps

    def test_auto_complete_on_full_progress(self):
        engine = GoalEngine()
        g = engine.add_goal("test goal", priority=0.5)
        engine.record_progress(g.id, delta=1.0)
        assert g.completed

    def test_stall_counter_increments(self):
        engine = GoalEngine()
        g = engine.add_goal("test goal", priority=0.5)
        engine.tick_goals()
        assert g.stall_counter == 1
        engine.tick_goals()
        assert g.stall_counter == 2

    def test_stall_reset_on_progress(self):
        engine = GoalEngine()
        g = engine.add_goal("test goal", priority=0.5)
        for _ in range(10):
            engine.tick_goals()
        assert g.stall_counter == 10
        engine.record_progress(g.id, delta=0.1)
        assert g.stall_counter == 0


class TestGoalRumination:
    def test_rumination_penalty(self):
        g = Goal(id="r", description="think about life", priority=0.5,
                 rumination_counter=5)
        emotions = EmotionalState()
        ep = g.effective_priority(emotions)
        g2 = Goal(id="r2", description="think about life", priority=0.5,
                  rumination_counter=0)
        ep2 = g2.effective_priority(emotions)
        assert ep < ep2  # ruminating goal has lower effective priority

    def test_stall_penalty(self):
        g = Goal(id="s", description="stuck goal", priority=0.5, stall_counter=15)
        emotions = EmotionalState()
        ep = g.effective_priority(emotions)
        g2 = Goal(id="s2", description="fresh goal", priority=0.5, stall_counter=0)
        ep2 = g2.effective_priority(emotions)
        assert ep < ep2  # stalled goal has lower effective priority

    def test_rumination_syncs_from_tracker(self):
        engine = GoalEngine()
        g = engine.add_goal("explore philosophy", priority=0.5)
        rt = RepetitionTracker()
        for i in range(5):
            rt.record_goal_mention("explore philosophy", tick=i)
        engine.tick_goals(repetition_tracker=rt)
        assert g.rumination_counter > 0


class TestGoalStallAbandonment:
    def test_abandon_after_long_stall(self):
        engine = GoalEngine()
        g = engine.add_goal("impossible goal", priority=0.5)
        # Simulate 45 ticks of stalling with no progress
        for _ in range(45):
            engine.tick_goals()
        assert g.abandoned  # should be abandoned (> 40 ticks, < 0.2 progress)

    def test_no_abandon_if_progress(self):
        engine = GoalEngine()
        g = engine.add_goal("hard but progressing", priority=0.5)
        for _ in range(20):
            engine.tick_goals()
        engine.record_progress(g.id, delta=0.3)  # significant progress
        for _ in range(25):
            engine.tick_goals()
        assert not g.abandoned  # has progress > 0.2


class TestGoalActiveLimit:
    def test_enforce_active_limit(self):
        engine = GoalEngine()
        emotions = EmotionalState()
        for i in range(5):
            engine.add_goal(f"goal {i}", priority=0.5)
        engine._enforce_active_limit(emotions)
        # Active goals should have demoted priorities for excess goals
        active = engine.active_goals()
        assert len(active) == 5  # they're still active, just demoted
        # The lowest-priority goals should have reduced priority
        priorities = [g.priority for g in active]
        assert min(priorities) < 0.5  # at least some were demoted


class TestNeedsDrivenGeneration:
    def test_low_stimulation_generates_exploration_goal(self):
        cs = ChatState()
        emotions = EmotionalState()
        needs = NeedsState(stimulation=0.2)
        engine = GoalEngine()
        new_goals = engine.auto_generate(cs, emotions, needs=needs)
        descriptions = [g.description for g in engine.goals]
        assert any("new" in d.lower() or "interesting" in d.lower() for d in descriptions)

    def test_low_belonging_generates_connection_goal(self):
        cs = ChatState()
        emotions = EmotionalState()
        needs = NeedsState(belonging=0.2)
        engine = GoalEngine()
        new_goals = engine.auto_generate(cs, emotions, needs=needs)
        descriptions = [g.description for g in engine.goals]
        assert any("exchange" in d.lower() or "genuine" in d.lower() for d in descriptions)

    def test_no_goals_when_needs_satisfied(self):
        cs = ChatState()
        emotions = EmotionalState()
        needs = NeedsState()  # all defaults are above 0.35
        engine = GoalEngine()
        new_goals = engine.auto_generate(cs, emotions, needs=needs)
        # With no people, topics, or low needs, very few goals should be generated
        # (could be zero or energy-related if energy is low)
        assert len(engine.goals) <= 2


class TestGoalPromptText:
    def test_to_prompt_text_shows_progress(self):
        engine = GoalEngine()
        g = engine.add_goal("test goal", priority=0.5)
        engine.record_progress(g.id, delta=0.5)
        text = engine.to_prompt_text(EmotionalState())
        assert "50%" in text

    def test_to_prompt_text_shows_stalled(self):
        engine = GoalEngine()
        g = engine.add_goal("stuck goal", priority=0.5)
        for _ in range(25):
            engine.tick_goals()
        text = engine.to_prompt_text(EmotionalState())
        assert "STALLED" in text

    def test_to_prompt_text_caps_at_3(self):
        engine = GoalEngine()
        for i in range(5):
            engine.add_goal(f"goal {i}", priority=0.5)
        text = engine.to_prompt_text(EmotionalState())
        lines = [l for l in text.strip().split("\n") if l.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
        assert len(lines) <= 3
