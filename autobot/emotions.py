"""Emotional system: fast affect (amygdala) + reflective emotion layer.

Market-linked version: triggers are market events (trade P&L, volatility,
cost pressure, news) rather than social interactions. The dual-layer
architecture (fast affect + slow reflective) is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from autobot.needs import EconomicState


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


# Trigger patterns: event_type -> (arousal_delta, valence_delta, tag)
_AFFECT_TRIGGERS: dict[str, tuple[float, float, str]] = {
    # --- Trade outcomes ---
    "trade_profit":          (0.4,  0.5, "profit"),
    "trade_loss":            (0.5, -0.5, "loss"),

    # --- Market signals ---
    "price_alert":           (0.4,  0.0, "alert"),
    "news_positive":         (0.2,  0.3, "opportunity"),
    "news_negative":         (0.3, -0.3, "risk"),

    # --- Thesis lifecycle ---
    "thesis_confirmed":      (0.3,  0.4, "validation"),
    "thesis_invalidated":    (0.4, -0.4, "invalidation"),
    "high_conviction_research": (0.2, 0.3, "insight"),

    # --- Prediction outcomes ---
    "prediction_accurate":   (0.2,  0.3, "accuracy"),
    "prediction_wrong":      (0.3, -0.3, "error"),

    # --- Economic pressure ---
    "starvation_warning":    (0.6, -0.5, "survival"),
    "drawdown":              (0.5, -0.4, "drawdown"),
    "recovery":              (0.3,  0.3, "recovery"),

    # --- Chris interaction ---
    "chris_message":         (0.2,  0.2, "chris"),
    "chris_insight":         (0.3,  0.4, "chris_insight"),

    # --- Self-generated ---
    "self_reflection":       (0.1,  0.1, "introspection"),
    "idle_thought":          (0.05, 0.0, "rumination"),
    "self_judgment":         (0.3, -0.3, "self_criticism"),
}


def evaluate_amygdala(
    events: list[dict[str, Any]],
    state: Any = None,  # MarketState (optional for backward compat)
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
            # Scale by intensity if provided
            intensity = ev.get("intensity", 1.0)
            affect.arousal = min(1.0, affect.arousal + a_delta * intensity)
            affect.valence = max(-1.0, min(1.0, affect.valence + v_delta * intensity))
            affect.salience_tags.append(tag)

        # Trade P&L magnitude scaling
        if etype in ("trade_profit", "trade_loss"):
            pnl = ev.get("pnl", 0)
            if abs(pnl) > 50:  # big trade
                affect.arousal = min(1.0, affect.arousal + 0.2)

    # Set attention focus to most salient ticker/entity
    if not affect.attention_focus and affect.salience_tags:
        for ev in events:
            for key in ("ticker", "person", "target"):
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
    """Slow-moving reflective emotions -- persists across cycles.

    Market-adapted fields:
    - conviction (was social_warmth): confidence in positions/theses
    - caution (was avoidance_bias): tendency to wait rather than act
    """
    anxiety: float = 0.0           # 0-1
    optimism: float = 0.5          # 0-1
    irritability: float = 0.0      # 0-1
    conviction: float = 0.5        # 0-1, confidence in positions
    caution: float = 0.2           # 0-1, tendency to wait
    risk_tolerance: float = 0.5    # 0-1

    # Backward-compat aliases for code that still uses old names
    @property
    def social_warmth(self) -> float:
        return self.conviction
    @social_warmth.setter
    def social_warmth(self, v: float) -> None:
        self.conviction = v
    @property
    def avoidance_bias(self) -> float:
        return self.caution
    @avoidance_bias.setter
    def avoidance_bias(self, v: float) -> None:
        self.caution = v

    def clamp(self) -> None:
        for attr in ("anxiety", "optimism", "irritability",
                     "conviction", "caution", "risk_tolerance"):
            val = max(0.0, min(1.0, getattr(self, attr)))
            setattr(self, attr, round(val, 3))

    def to_dict(self) -> dict[str, float]:
        return {
            "anxiety": self.anxiety,
            "optimism": self.optimism,
            "irritability": self.irritability,
            "conviction": self.conviction,
            "caution": self.caution,
            "risk_tolerance": self.risk_tolerance,
        }

    def dominant_mood(self) -> str:
        moods = {
            "anxious": self.anxiety,
            "optimistic": self.optimism,
            "irritable": self.irritability,
            "convicted": self.conviction,
            "cautious": self.caution,
        }
        return max(moods, key=moods.get)  # type: ignore[arg-type]


# Decay rates -- emotions drift toward baseline each cycle
_DECAY = 0.05
_BASELINE = {
    "anxiety": 0.1,
    "optimism": 0.5,
    "irritability": 0.1,
    "conviction": 0.5,
    "caution": 0.2,
    "risk_tolerance": 0.5,
}


def apply_emotional_weather(state: EmotionalState, tick_count: int) -> None:
    """Slow sinusoidal oscillation to prevent perfectly stable moods."""
    t = tick_count * 0.1
    state.anxiety += 0.04 * math.sin(t * 0.7 + 1.0)
    state.optimism += 0.05 * math.sin(t * 0.5)
    state.irritability += 0.03 * math.sin(t * 0.9 + 2.0)
    state.conviction += 0.04 * math.sin(t * 0.6 + 3.0)
    state.caution += 0.03 * math.sin(t * 0.8 + 4.0)
    state.risk_tolerance += 0.04 * math.sin(t * 0.4 + 5.0)
    state.clamp()


def update_reflective_emotions(
    state: EmotionalState,
    affect: AffectState,
    cs: Any,  # MarketState or ChatState
    needs: EconomicState | None = None,
    prediction_error: float | None = None,
    user_context: dict[str, Any] | None = None,
) -> EmotionalState:
    """
    Update reflective emotions based on:
    - Economic state deficits
    - Current fast affect
    - Prediction error from recent actions
    - Chris context (user_context)
    - Energy level
    - Market inactivity
    """
    # --- Economic state deficits ---
    if needs:
        # Low alpha (meaning) -> frustration
        if needs.alpha < 0.3:
            state.irritability += 0.06
            state.optimism -= 0.04
        # Low ROI (competence) -> anxiety + doubt
        if needs.roi < 0.3:
            state.anxiety += 0.05
            state.conviction -= 0.04
        # High volatility (market anxiety)
        if needs.volatility > 0.7:
            state.anxiety += 0.05
            state.caution += 0.04
            state.risk_tolerance -= 0.03
        # High cost pressure
        if needs.cost_pressure > 0.7:
            state.irritability += 0.04
            state.caution += 0.05

    # --- Prediction error modulation ---
    if prediction_error is not None:
        if prediction_error < -0.2:  # worse than expected
            state.anxiety += 0.06
            state.optimism -= 0.08
            state.irritability += 0.04
        elif prediction_error > 0.2:  # better than expected
            state.optimism += 0.08
            state.anxiety -= 0.04
            state.conviction += 0.04

    # --- Chris context (ToM-aware modulation) ---
    if user_context:
        u_state = user_context.get("state", "")
        u_intent = user_context.get("intent", "")
        u_reliability = user_context.get("reliability", 0.8)

        # Chris venting -> supportive mode
        if u_state in ("stressed", "hurting", "vulnerable") and u_intent == "venting":
            state.conviction += 0.06
            state.anxiety += 0.02
            state.irritability = max(0.0, state.irritability - 0.02)

        # Low reliability + nice surface -> suspicion
        if u_reliability < 0.4 and affect.valence > 0.2:
            state.caution += 0.04
            state.anxiety += 0.03
            state.conviction -= 0.02
            state.risk_tolerance -= 0.03

        # Genuine vulnerability -> open up
        if u_state == "vulnerable" and u_reliability > 0.7:
            state.conviction += 0.05
            state.risk_tolerance += 0.02

        # Hostility -> defensive
        if u_state == "hostile" or u_intent == "confronting":
            state.caution += 0.05
            state.anxiety += 0.04
            state.irritability += 0.03

    # --- Affect-driven updates ---
    if affect.arousal > 0.5 and affect.valence < -0.3:
        state.anxiety += 0.1 * affect.arousal
        state.irritability += 0.05 * affect.arousal
        state.risk_tolerance -= 0.05
        state.caution += 0.05

    if affect.valence > 0.3:
        state.optimism += 0.08 * affect.arousal
        state.conviction += 0.04
        state.anxiety -= 0.03
        state.risk_tolerance += 0.03

    # --- Market-specific triggers ---
    if "profit" in affect.salience_tags:
        state.conviction += 0.06
        state.optimism += 0.05
    if "loss" in affect.salience_tags:
        state.conviction -= 0.06
        state.caution += 0.04
    if "survival" in affect.salience_tags:
        state.caution += 0.08
        state.risk_tolerance -= 0.06
    if "drawdown" in affect.salience_tags:
        state.anxiety += 0.06
        state.risk_tolerance -= 0.04

    # --- Energy level ---
    if hasattr(cs, "energy") and cs.energy < 0.2:
        state.irritability += 0.04
        state.caution += 0.03

    # --- Decay toward dynamic baseline ---
    dynamic_baseline = dict(_BASELINE)
    if needs:
        if needs.alpha < 0.35:
            dynamic_baseline["optimism"] -= 0.15
            dynamic_baseline["irritability"] += 0.1
        if needs.roi < 0.35:
            dynamic_baseline["anxiety"] += 0.1
            dynamic_baseline["optimism"] -= 0.1
        if needs.volatility > 0.65:
            dynamic_baseline["anxiety"] += 0.1
            dynamic_baseline["caution"] += 0.1
        if needs.cost_pressure > 0.65:
            dynamic_baseline["irritability"] += 0.1

    for attr, baseline in dynamic_baseline.items():
        baseline = max(0.0, min(1.0, baseline))
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
