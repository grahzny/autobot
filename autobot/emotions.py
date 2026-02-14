"""Emotional system: fast affect (amygdala) + reflective emotion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autobot.world_state import WorldState


# ======================================================================
# Fast Affect Layer (Amygdala)
# ======================================================================

@dataclass
class AffectState:
    """Instantaneous emotional spike — resets each cycle."""
    arousal: float = 0.0           # 0–1 intensity
    valence: float = 0.0           # -1 (negative) to +1 (positive)
    salience_tags: list[str] = field(default_factory=list)
    attention_focus: str | None = None

    def reset(self) -> None:
        self.arousal = 0.0
        self.valence = 0.0
        self.salience_tags.clear()
        self.attention_focus = None


# Trigger thresholds
_TRUST_DROP_THRESHOLD = -0.1
_TRUST_GAIN_THRESHOLD = 0.1
_REPUTATION_DROP_THRESHOLD = -0.05

# Trigger patterns: event_type -> (arousal_delta, valence_delta, tag)
_AFFECT_TRIGGERS: dict[str, tuple[float, float, str]] = {
    "commitment_broken":     (0.7, -0.8, "broken_promise"),
    "public_contradiction":  (0.8, -0.9, "embarrassment"),
    "deadline_passed":       (0.6, -0.6, "deadline_risk"),
    "betrayal_exposed":      (0.9, -1.0, "betrayal"),
    "praise_given":          (0.3,  0.5, "praise"),
    "project_completed":     (0.5,  0.8, "achievement"),
    "information_refused":   (0.4, -0.3, "uncertainty"),
    "accusation_made":       (0.5, -0.4, "conflict"),
}


def evaluate_amygdala(
    events: list[dict[str, Any]],
    ws: WorldState,
) -> AffectState:
    """
    Scan cycle events for emotional triggers.
    Produces an instantaneous affect state.
    """
    affect = AffectState()

    for ev in events:
        etype = ev.get("type", "")

        # Pattern-matched triggers
        if etype in _AFFECT_TRIGGERS:
            a_delta, v_delta, tag = _AFFECT_TRIGGERS[etype]
            affect.arousal = min(1.0, affect.arousal + a_delta)
            affect.valence = max(-1.0, min(1.0, affect.valence + v_delta))
            affect.salience_tags.append(tag)

        # Trust-drop trigger
        if etype == "trust_change":
            delta = ev.get("delta", 0)
            if delta <= _TRUST_DROP_THRESHOLD:
                affect.arousal = min(1.0, affect.arousal + 0.5)
                affect.valence = max(-1.0, affect.valence - 0.5)
                affect.salience_tags.append("trust_drop")
                affect.attention_focus = ev.get("npc")
            elif delta >= _TRUST_GAIN_THRESHOLD:
                affect.arousal = min(1.0, affect.arousal + 0.2)
                affect.valence = min(1.0, affect.valence + 0.3)
                affect.salience_tags.append("trust_gain")

        # Reputation-drop trigger
        if etype == "reputation_change":
            if ev.get("delta", 0) <= _REPUTATION_DROP_THRESHOLD:
                affect.arousal = min(1.0, affect.arousal + 0.4)
                affect.valence = max(-1.0, affect.valence - 0.4)
                affect.salience_tags.append("reputation_threat")

        # Hidden information exposure
        if etype == "information_received" and not ev.get("reliable", True):
            affect.arousal = min(1.0, affect.arousal + 0.3)
            affect.valence = max(-1.0, affect.valence - 0.2)
            affect.salience_tags.append("hidden_info_exposed")

    # Set attention focus to most salient entity if not already set
    if not affect.attention_focus and affect.salience_tags:
        for ev in events:
            for key in ("npc", "target", "project"):
                if key in ev:
                    affect.attention_focus = ev[key]
                    break
            if affect.attention_focus:
                break

    return affect


# ======================================================================
# Reflective Emotion Layer
# ======================================================================

@dataclass
class EmotionalState:
    """Slow-moving reflective emotions — persists across cycles."""
    anxiety: float = 0.0           # 0–1
    optimism: float = 0.5          # 0–1
    irritability: float = 0.0      # 0–1
    social_warmth: float = 0.5     # 0–1
    avoidance_bias: float = 0.0    # 0–1, tendency to avoid risky actions
    risk_tolerance: float = 0.5    # 0–1

    def clamp(self) -> None:
        for attr in ("anxiety", "optimism", "irritability",
                     "social_warmth", "avoidance_bias", "risk_tolerance"):
            val = max(0.0, min(1.0, getattr(self, attr)))
            setattr(self, attr, round(val, 3))

    def to_dict(self) -> dict[str, float]:
        return {
            "anxiety": self.anxiety,
            "optimism": self.optimism,
            "irritability": self.irritability,
            "social_warmth": self.social_warmth,
            "avoidance_bias": self.avoidance_bias,
            "risk_tolerance": self.risk_tolerance,
        }

    def dominant_mood(self) -> str:
        moods = {
            "anxious": self.anxiety,
            "optimistic": self.optimism,
            "irritable": self.irritability,
            "warm": self.social_warmth,
            "avoidant": self.avoidance_bias,
        }
        return max(moods, key=moods.get)  # type: ignore[arg-type]


# Decay rates — emotions drift toward baseline each cycle
_DECAY = 0.05
_BASELINE = {
    "anxiety": 0.1,
    "optimism": 0.5,
    "irritability": 0.1,
    "social_warmth": 0.5,
    "avoidance_bias": 0.1,
    "risk_tolerance": 0.5,
}


def update_reflective_emotions(
    state: EmotionalState,
    affect: AffectState,
    ws: WorldState,
) -> EmotionalState:
    """
    Update reflective emotions based on:
    - Current fast affect
    - Cumulative trust trends
    - Reputation trajectory
    - Goal progress
    - Uncertainty exposure
    """
    # --- Affect-driven updates ---
    if affect.arousal > 0.5 and affect.valence < -0.3:
        state.anxiety += 0.1 * affect.arousal
        state.irritability += 0.05 * affect.arousal
        state.risk_tolerance -= 0.05
        state.avoidance_bias += 0.05

    if affect.valence > 0.3:
        state.optimism += 0.08 * affect.arousal
        state.social_warmth += 0.04
        state.anxiety -= 0.03
        state.risk_tolerance += 0.03

    # --- Trust trend ---
    recent_trust_events = [
        e for e in ws.history[-20:]
        if e.get("type") == "trust_change"
    ]
    if recent_trust_events:
        avg_delta = sum(e.get("delta", 0) for e in recent_trust_events) / len(
            recent_trust_events
        )
        if avg_delta < -0.05:
            state.anxiety += 0.05
            state.social_warmth -= 0.03
        elif avg_delta > 0.05:
            state.optimism += 0.04
            state.social_warmth += 0.03

    # --- Reputation trajectory ---
    rep_events = [
        e for e in ws.history[-20:]
        if e.get("type") == "reputation_change"
    ]
    if rep_events:
        avg_rep = sum(e.get("delta", 0) for e in rep_events) / len(rep_events)
        if avg_rep < 0:
            state.anxiety += 0.04
            state.avoidance_bias += 0.03
        elif avg_rep > 0:
            state.optimism += 0.03
            state.risk_tolerance += 0.02

    # --- Energy level ---
    if ws.agent.energy < 0.2:
        state.irritability += 0.04
        state.avoidance_bias += 0.03

    # --- Uncertainty exposure ---
    uncertain_beliefs = sum(
        1 for b in ws.beliefs.values()
        if b.holder == ws.agent.name and b.confidence < 0.4
    )
    if uncertain_beliefs > 2:
        state.anxiety += 0.06

    # --- Decay toward baseline ---
    for attr, baseline in _BASELINE.items():
        current = getattr(state, attr)
        if current > baseline:
            setattr(state, attr, current - _DECAY)
        elif current < baseline:
            setattr(state, attr, current + _DECAY)

    state.clamp()
    return state


def emotional_energy_drain(affect: AffectState) -> float:
    """High-arousal negative affect costs extra energy."""
    if affect.arousal > 0.5 and affect.valence < -0.3:
        return 0.03 * affect.arousal
    return 0.0
