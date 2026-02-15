"""Tests for the repetition detection system."""

from __future__ import annotations

from autobot.repetition import RepetitionTracker


class TestRepetitionTracker:
    def test_record_and_retrieve_themes(self):
        rt = RepetitionTracker()
        rt.record_themes(["space", "music"], tick=1, source="response")
        assert len(rt.recent_themes) == 2
        recent = rt.recently_used_themes(limit=5)
        assert "music" in recent
        assert "space" in recent

    def test_detect_loop_false_when_few_entries(self):
        rt = RepetitionTracker()
        rt.record_themes(["space"], tick=1, source="response")
        is_looping, theme = rt.detect_loop()
        assert not is_looping
        assert theme is None

    def test_detect_loop_true_when_repeated(self):
        rt = RepetitionTracker()
        for i in range(5):
            rt.record_themes(["space"], tick=i, source="response")
        is_looping, theme = rt.detect_loop()
        assert is_looping
        assert theme == "space"

    def test_detect_loop_false_with_variety(self):
        rt = RepetitionTracker()
        topics = ["space", "music", "cooking", "history", "math",
                  "art", "science", "poetry", "films", "games"]
        for i, t in enumerate(topics):
            rt.record_themes([t], tick=i, source="response")
        is_looping, theme = rt.detect_loop()
        assert not is_looping

    def test_is_theme_fresh(self):
        rt = RepetitionTracker()
        rt.record_themes(["space", "music"], tick=1, source="response")
        assert not rt.is_theme_fresh("space")
        assert rt.is_theme_fresh("cooking")

    def test_record_goal_mention_and_detect_rumination(self):
        rt = RepetitionTracker()
        for i in range(4):
            rt.record_goal_mention("explore philosophy", tick=i)
        count = rt.detect_goal_rumination("explore philosophy")
        assert count == 4

    def test_pivot_reason_no_loop(self):
        rt = RepetitionTracker()
        rt.record_themes(["space"], tick=1, source="response")
        assert rt.pivot_reason() == ""

    def test_pivot_reason_with_loop(self):
        rt = RepetitionTracker()
        for i in range(5):
            rt.record_themes(["loneliness"], tick=i, source="idle")
        reason = rt.pivot_reason()
        assert "loneliness" in reason
        assert len(reason) > 10

    def test_pivot_reason_goal_loop(self):
        rt = RepetitionTracker()
        for i in range(5):
            rt.record_goal_mention("learn python", tick=i)
        reason = rt.pivot_reason()
        assert "learn python" in reason

    def test_max_history_trimming(self):
        rt = RepetitionTracker(max_history=5)
        for i in range(10):
            rt.record_themes([f"topic_{i}"], tick=i, source="idle")
        assert len(rt.recent_themes) == 5

    def test_recently_used_themes_unique(self):
        rt = RepetitionTracker()
        rt.record_themes(["space"], tick=1, source="response")
        rt.record_themes(["space"], tick=2, source="response")
        rt.record_themes(["music"], tick=3, source="response")
        recent = rt.recently_used_themes(limit=5)
        # Should be unique
        assert len(recent) == len(set(recent))
