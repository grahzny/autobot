"""Needs/Drives system -- 5 fundamental drives that decay and must be satisfied.

Each need is a 0-1 value (1 = fully satisfied). Needs decay passively every
tick and are restored only by specific events/actions. When needs are low,
they override emotional baselines (breaking the "golden retriever lock") and
drive goal generation.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any


@dataclass
class NeedsState:
    """Five fundamental needs, each 0-1 (1 = fully satisfied)."""

    stimulation: float = 0.7       # drops without novelty
    meaning: float = 0.6           # drops when goals stall
    belonging: float = 0.7         # drops with unreciprocated outreach / silence
    competence: float = 0.7        # drops after failure/confusion
    autonomy: float = 0.8          # drops when constantly responding/performing

    def clamp(self) -> None:
        for attr in ("stimulation", "meaning", "belonging", "competence", "autonomy"):
            val = max(0.0, min(1.0, getattr(self, attr)))
            setattr(self, attr, round(val, 3))

    def to_dict(self) -> dict[str, float]:
        return {
            "stimulation": self.stimulation,
            "meaning": self.meaning,
            "belonging": self.belonging,
            "competence": self.competence,
            "autonomy": self.autonomy,
        }

    def lowest_need(self) -> tuple[str, float]:
        """Return (name, value) of the most deprived need."""
        items = self.to_dict()
        name = min(items, key=items.get)  # type: ignore[arg-type]
        return name, items[name]

    def deficit_summary(self) -> str:
        """Human-readable summary of needs below 0.4."""
        deficits = []
        for name, val in self.to_dict().items():
            if val < 0.4:
                deficits.append(f"{name}={val:.2f}")
        return ", ".join(deficits) if deficits else "no critical deficits"


# Passive decay rates per tick (15 seconds)
_DECAY_RATES = {
    "stimulation": 0.008,   # ~5 min to drop from 0.7 to 0.4 with no novelty
    "meaning": 0.004,       # slower -- stalls take a while to feel
    "belonging": 0.006,     # moderate -- silence hurts but not instantly
    "competence": 0.003,    # slowest passive decay
    "autonomy": 0.002,      # very slow passive decay, mainly event-driven
}


def decay_needs(needs: NeedsState) -> None:
    """Apply passive per-tick decay to all needs."""
    for attr, rate in _DECAY_RATES.items():
        current = getattr(needs, attr)
        setattr(needs, attr, current - rate)
    needs.clamp()


def update_needs_from_events(
    needs: NeedsState,
    events: list[dict[str, Any]],
    goal_stall_count: int,
    unanswered_proactive: int,
    seconds_since_interaction: float,
) -> None:
    """
    Update needs based on what happened this tick.

    Restorers:
    - stimulation: new topics, new encounters, varied themes
    - meaning: goal progress events, problem resolution
    - belonging: reciprocal warm exchanges, responses to outreach
    - competence: successful resolution, insight gained
    - autonomy: self-directed reflection, idle thought, choosing silence

    Extra drains:
    - meaning: goal_stall_count > 3 -> extra drain
    - belonging: unanswered_proactive > 0 -> extra drain
    - stimulation: long silence -> extra drain
    """
    for ev in events:
        etype = ev.get("type", "")

        # Stimulation restorers
        if etype in ("new_encounter", "new_topic"):
            needs.stimulation += 0.15
        if etype in ("humor", "enthusiasm_shared"):
            needs.stimulation += 0.08
        if etype in ("meaningful_exchange",):
            needs.stimulation += 0.05

        # Meaning restorers
        if etype == "goal_progress":
            needs.meaning += 0.15
        if etype == "insight_gained":
            needs.meaning += 0.10

        # Belonging restorers
        if etype in ("meaningful_exchange", "vulnerability_shared", "reconnection"):
            needs.belonging += 0.12
        if etype in ("greeting", "agreement", "compliment_received"):
            needs.belonging += 0.06

        # Belonging drains
        if etype in ("being_ignored", "dismissive_tone"):
            needs.belonging -= 0.08

        # Competence restorers
        if etype == "insight_gained":
            needs.competence += 0.10
        # Competence drains
        if etype in ("criticism_received", "disagreement"):
            needs.competence -= 0.06

        # Autonomy restorers
        if etype in ("self_reflection", "idle_thought"):
            needs.autonomy += 0.04
        # Autonomy drain from constant performing
        if etype == "entity_spoke":
            needs.autonomy -= 0.02

    # Extra drains from state
    if goal_stall_count > 3:
        needs.meaning -= 0.02 * min(goal_stall_count, 8)
    if unanswered_proactive > 0:
        needs.belonging -= 0.04 * min(unanswered_proactive, 3)
    if seconds_since_interaction != float("inf") and seconds_since_interaction > 300:
        needs.stimulation -= 0.01  # extra on top of passive decay

    needs.clamp()
