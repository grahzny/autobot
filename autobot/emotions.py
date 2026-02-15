"""Emotional system: fast affect (amygdala) + reflective emotion layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from autobot.chat_state import ChatState
    from autobot.needs import NeedsState


# ======================================================================
# Fast Affect Layer (Amygdala)
# ======================================================================

@dataclass
class AffectState:
    """Instantaneous emotional spike -- resets each cycle."""
    arousal: float = 0.0           # 0-1 intensity
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

# Trigger patterns: event_type -> (arousal_delta, valence_delta, tag)
_AFFECT_TRIGGERS: dict[str, tuple[float, float, str]] = {
    # --- Carried from simulation (still useful) ---
    "praise_given":          (0.3,  0.5, "praise"),
    "information_refused":   (0.4, -0.3, "uncertainty"),
    "accusation_made":       (0.5, -0.4, "conflict"),

    # --- Chat-specific triggers ---
    "reconnection":          (0.2,  0.3, "reconnection"),
    "being_ignored":         (0.3, -0.3, "neglect"),
    "new_encounter":         (0.4,  0.2, "novelty"),
    "meaningful_exchange":   (0.3,  0.4, "depth"),
    "vulnerability_shared":  (0.4,  0.4, "intimacy"),
    "boundary_crossed":      (0.6, -0.5, "violation"),
    "boredom":               (0.15, -0.15, "stagnation"),
    "restlessness":          (0.25, -0.1, "restless"),
    "loneliness":            (0.4, -0.3, "isolation"),
    "humor":                 (0.2,  0.4, "levity"),
    "disagreement":          (0.4, -0.3, "friction"),
    "agreement":             (0.2,  0.3, "harmony"),
    "compliment_received":   (0.3,  0.5, "appreciation"),
    "criticism_received":    (0.5, -0.4, "criticism"),
    "personal_question":     (0.3,  0.1, "curiosity"),
    "dismissive_tone":       (0.3, -0.2, "rejection"),
    "enthusiasm_shared":     (0.3,  0.4, "excitement"),
    "farewell":              (0.2, -0.1, "parting"),
    "greeting":              (0.2,  0.2, "hello"),

    # --- Self-generated action triggers ---
    "entity_spoke":          (0.1,  0.2, "expression"),
    "entity_initiated":      (0.15, 0.2, "initiative"),
    "self_reflection":       (0.1,  0.1, "introspection"),
    "idle_thought":          (0.05, 0.0, "rumination"),
    "self_judgment":         (0.3, -0.3, "self_criticism"),
}


def evaluate_amygdala(
    events: list[dict[str, Any]],
    state: ChatState,
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

        # Trust-change trigger (from message analysis)
        if etype == "trust_change":
            delta = ev.get("delta", 0)
            if delta <= _TRUST_DROP_THRESHOLD:
                affect.arousal = min(1.0, affect.arousal + 0.5)
                affect.valence = max(-1.0, affect.valence - 0.5)
                affect.salience_tags.append("trust_drop")
                affect.attention_focus = ev.get("person", ev.get("npc"))
            elif delta >= _TRUST_GAIN_THRESHOLD:
                affect.arousal = min(1.0, affect.arousal + 0.2)
                affect.valence = min(1.0, affect.valence + 0.3)
                affect.salience_tags.append("trust_gain")

    # Set attention focus to most salient entity if not already set
    if not affect.attention_focus and affect.salience_tags:
        for ev in events:
            for key in ("person", "target", "npc"):
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
    """Slow-moving reflective emotions -- persists across cycles."""
    anxiety: float = 0.0           # 0-1
    optimism: float = 0.5          # 0-1
    irritability: float = 0.0      # 0-1
    social_warmth: float = 0.5     # 0-1
    avoidance_bias: float = 0.0    # 0-1, tendency to avoid engagement
    risk_tolerance: float = 0.5    # 0-1

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


# Decay rates -- emotions drift toward baseline each cycle
_DECAY = 0.05
_BASELINE = {
    "anxiety": 0.1,
    "optimism": 0.5,
    "irritability": 0.1,
    "social_warmth": 0.5,
    "avoidance_bias": 0.1,
    "risk_tolerance": 0.5,
}


def apply_emotional_weather(state: EmotionalState, tick_count: int) -> None:
    """Slow sinusoidal oscillation to prevent perfectly stable moods.

    Uses overlapping sine waves at different frequencies for each dimension.
    Amplitude is small (0.03-0.05) so it's a gentle drift, not chaos.
    """
    t = tick_count * 0.1  # scale ticks to a reasonable frequency
    state.anxiety += 0.04 * math.sin(t * 0.7 + 1.0)
    state.optimism += 0.05 * math.sin(t * 0.5)
    state.irritability += 0.03 * math.sin(t * 0.9 + 2.0)
    state.social_warmth += 0.04 * math.sin(t * 0.6 + 3.0)
    state.avoidance_bias += 0.03 * math.sin(t * 0.8 + 4.0)
    state.risk_tolerance += 0.04 * math.sin(t * 0.4 + 5.0)
    state.clamp()


def update_reflective_emotions(
    state: EmotionalState,
    affect: AffectState,
    cs: ChatState,
    needs: NeedsState | None = None,
    prediction_error: float | None = None,
    user_context: dict[str, Any] | None = None,
) -> EmotionalState:
    """
    Update reflective emotions based on:
    - Needs deficits (can override baselines)
    - Current fast affect
    - Prediction error from recent actions
    - Theory of Mind (user_context: state, intent, reliability)
    - Relationship trends
    - Energy level
    - Social isolation
    """
    # --- Needs deficits override baselines ---
    if needs:
        if needs.stimulation < 0.3:
            state.irritability += 0.06
            state.optimism -= 0.04
        if needs.belonging < 0.3:
            state.anxiety += 0.05
            state.social_warmth += 0.04  # craving, not satisfaction
        if needs.meaning < 0.3:
            state.anxiety += 0.04
            state.optimism -= 0.05
        if needs.competence < 0.3:
            state.anxiety += 0.05
            state.irritability += 0.03
        if needs.autonomy < 0.3:
            state.irritability += 0.04
            state.avoidance_bias += 0.05

    # --- Prediction error modulation ---
    if prediction_error is not None:
        if prediction_error < -0.2:  # worse than expected
            state.anxiety += 0.06
            state.optimism -= 0.08
            state.irritability += 0.04
        elif prediction_error > 0.2:  # better than expected
            state.optimism += 0.08
            state.anxiety -= 0.04
            state.social_warmth += 0.04

    # --- ToM-aware modulation ---
    if user_context:
        u_state = user_context.get("state", "")
        u_intent = user_context.get("intent", "")
        u_reliability = user_context.get("reliability", 0.8)

        # Empathy: venting/hurting user → warmth UP (not offense)
        if u_state in ("stressed", "hurting", "vulnerable") and u_intent == "venting":
            state.social_warmth += 0.06
            state.anxiety += 0.02
            state.irritability = max(0.0, state.irritability - 0.02)

        # Boundary: low reliability + nice surface → suspicion
        if u_reliability < 0.4 and affect.valence > 0.2:
            state.avoidance_bias += 0.04
            state.anxiety += 0.03
            state.social_warmth -= 0.02
            state.risk_tolerance -= 0.03

        # Vulnerability response: genuine vulnerability → open up
        if u_state == "vulnerable" and u_reliability > 0.7:
            state.social_warmth += 0.05
            state.risk_tolerance += 0.02

        # Hostility: hostile user or confrontation → defensive
        if u_state == "hostile" or u_intent == "confronting":
            state.avoidance_bias += 0.05
            state.anxiety += 0.04
            state.irritability += 0.03

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

    # --- Relationship trend ---
    recent_trust_events = [
        e for e in cs.history[-20:]
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

    # --- Energy level ---
    if cs.energy < 0.2:
        state.irritability += 0.04
        state.avoidance_bias += 0.03

    # --- Silence effects (reworked: pull back, don't cling) ---
    silence = cs.seconds_since_any_interaction()
    if silence > 90:  # 1.5 min: mild understimulation
        state.irritability += 0.01
        state.optimism -= 0.01
    if silence > 180:  # 3 min: measured response
        state.social_warmth += 0.01  # reduced from 0.03
        state.avoidance_bias += 0.01
        state.risk_tolerance += 0.01
    if silence > 600:  # 10 min: entity becomes self-directed, not desperate
        state.optimism -= 0.02
        state.avoidance_bias += 0.02
        state.social_warmth -= 0.01  # pull back instead of cling
    if silence > 3600:  # 1 hour: withdrawal
        state.anxiety += 0.02
        state.avoidance_bias += 0.03

    # --- Decay toward DYNAMIC baseline ---
    # When needs are low, the baseline shifts so the entity genuinely
    # becomes less happy rather than always snapping back to cheerful.
    dynamic_baseline = dict(_BASELINE)
    if needs:
        if needs.stimulation < 0.35:
            dynamic_baseline["optimism"] -= 0.15
            dynamic_baseline["irritability"] += 0.1
        if needs.belonging < 0.35:
            dynamic_baseline["anxiety"] += 0.1
            dynamic_baseline["optimism"] -= 0.1
        if needs.meaning < 0.35:
            dynamic_baseline["optimism"] -= 0.15
        if needs.competence < 0.35:
            dynamic_baseline["anxiety"] += 0.1

    for attr, baseline in dynamic_baseline.items():
        baseline = max(0.0, min(1.0, baseline))  # clamp dynamic baseline
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
