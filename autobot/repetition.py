"""Repetition Detection -- tracks recent output themes and detects loops.

Watches the entity's outputs for repeated themes, goal restating, and
memory retrieval patterns. When loops are detected, generates pivot
instructions to break the cycle.
"""

from __future__ import annotations

import time as _time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ThemeEntry:
    """A single theme observation."""

    theme: str
    tick: int
    timestamp: float
    source: str  # "response", "proactive", "reflection", "idle", "goal_mention"


@dataclass
class RepetitionTracker:
    """Tracks recent themes and detects repetitive patterns."""

    recent_themes: list[ThemeEntry] = field(default_factory=list)
    max_history: int = 20
    _loop_detected_at: float = 0.0

    def record_themes(self, themes: list[str], tick: int, source: str) -> None:
        """Record themes from an output."""
        now = _time.time()
        for theme in themes:
            self.recent_themes.append(ThemeEntry(
                theme=theme.lower().strip(),
                tick=tick,
                timestamp=now,
                source=source,
            ))
        # Trim to max_history
        if len(self.recent_themes) > self.max_history:
            self.recent_themes = self.recent_themes[-self.max_history:]

    def record_goal_mention(self, goal_description: str, tick: int) -> None:
        """Record when a goal is mentioned in output."""
        self.record_themes(
            [f"goal:{goal_description.lower()[:50]}"], tick, "goal_mention",
        )

    def detect_loop(self) -> tuple[bool, str | None]:
        """
        Check if the entity is looping on a theme.

        Returns (is_looping, repeated_theme_or_None).
        Criteria: same theme appearing 3+ times in the last 10 entries.
        """
        if len(self.recent_themes) < 5:
            return False, None

        recent = self.recent_themes[-10:]
        counts = Counter(entry.theme for entry in recent)
        for theme, count in counts.most_common(3):
            if count >= 3:
                self._loop_detected_at = _time.time()
                return True, theme
        return False, None

    def detect_goal_rumination(self, goal_description: str) -> int:
        """Count how many times a specific goal was mentioned recently."""
        key = f"goal:{goal_description.lower()[:50]}"
        return sum(1 for e in self.recent_themes if e.theme == key)

    def recently_used_themes(self, limit: int = 5) -> list[str]:
        """Return the most recent unique themes (for novelty checking)."""
        seen: list[str] = []
        for entry in reversed(self.recent_themes):
            if entry.theme not in seen:
                seen.append(entry.theme)
            if len(seen) >= limit:
                break
        return seen

    def is_theme_fresh(self, theme: str) -> bool:
        """Check if a theme has NOT appeared in recent history."""
        lower = theme.lower().strip()
        return lower not in [e.theme for e in self.recent_themes[-10:]]

    def seconds_since_loop_detected(self) -> float:
        """Seconds since a loop was last detected."""
        if self._loop_detected_at == 0:
            return float("inf")
        return _time.time() - self._loop_detected_at

    def pivot_reason(self) -> str:
        """Generate a reason string for breaking a loop, usable in prompts."""
        is_looping, theme = self.detect_loop()
        if not is_looping or not theme:
            return ""
        if theme.startswith("goal:"):
            return (
                f"You keep returning to '{theme[5:]}'. You've mentioned it "
                f"enough -- move on or take a concrete step."
            )
        return (
            f"You've been circling around '{theme}'. Change the subject "
            f"or go deeper with a specific question."
        )
