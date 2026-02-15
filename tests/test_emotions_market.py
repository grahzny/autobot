"""Tests for market-linked emotional system.

Verifies that market events (trade P&L, alerts, starvation, etc.) correctly
drive the amygdala and reflective emotion layers.
"""

from __future__ import annotations

from autobot.chat_state import ChatState, MarketState
from autobot.emotions import (
    AffectState,
    EmotionalState,
    _AFFECT_TRIGGERS,
    apply_emotional_weather,
    emotional_energy_drain,
    evaluate_amygdala,
    update_reflective_emotions,
)
from autobot.needs import EconomicState


# ======================================================================
# Amygdala trigger tests
# ======================================================================


class TestMarketTriggers:
    """Market events must produce correct affect responses."""

    def test_trade_profit_positive_valence(self):
        events = [{"type": "trade_profit", "ticker": "BHP.AX", "pnl": 25.0}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence > 0
        assert affect.arousal > 0
        assert "profit" in affect.salience_tags

    def test_trade_loss_negative_valence(self):
        events = [{"type": "trade_loss", "ticker": "CBA.AX", "pnl": -30.0}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence < 0
        assert affect.arousal > 0
        assert "loss" in affect.salience_tags

    def test_big_trade_extra_arousal(self):
        small = [{"type": "trade_profit", "pnl": 10.0}]
        big = [{"type": "trade_profit", "pnl": 100.0}]
        a_small = evaluate_amygdala(small, MarketState())
        a_big = evaluate_amygdala(big, MarketState())
        assert a_big.arousal > a_small.arousal

    def test_starvation_warning_high_arousal(self):
        events = [{"type": "starvation_warning", "intensity": 0.8}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.arousal > 0.3
        assert affect.valence < 0
        assert "survival" in affect.salience_tags

    def test_price_alert_neutral_valence(self):
        events = [{"type": "price_alert", "ticker": "AAPL", "intensity": 0.5}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.arousal > 0
        assert "alert" in affect.salience_tags

    def test_news_positive(self):
        events = [{"type": "news_positive", "ticker": "MSFT", "intensity": 0.6}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence > 0
        assert "opportunity" in affect.salience_tags

    def test_news_negative(self):
        events = [{"type": "news_negative", "ticker": "MSFT", "intensity": 0.6}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence < 0
        assert "risk" in affect.salience_tags

    def test_thesis_confirmed(self):
        events = [{"type": "thesis_confirmed", "intensity": 0.5}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence > 0
        assert "validation" in affect.salience_tags

    def test_thesis_invalidated(self):
        events = [{"type": "thesis_invalidated", "intensity": 0.5}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence < 0
        assert "invalidation" in affect.salience_tags

    def test_drawdown_event(self):
        events = [{"type": "drawdown", "intensity": 0.6}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence < 0
        assert "drawdown" in affect.salience_tags

    def test_recovery_event(self):
        events = [{"type": "recovery", "intensity": 0.5}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.valence > 0
        assert "recovery" in affect.salience_tags

    def test_all_market_triggers_registered(self):
        """All expected market event types should be in the trigger table."""
        expected = {
            "trade_profit", "trade_loss", "price_alert",
            "news_positive", "news_negative",
            "thesis_confirmed", "thesis_invalidated",
            "starvation_warning", "drawdown", "recovery",
            "chris_message", "chris_insight",
        }
        for trigger in expected:
            assert trigger in _AFFECT_TRIGGERS, f"Missing trigger: {trigger}"

    def test_intensity_scaling(self):
        """Event intensity should scale the affect proportionally."""
        low = [{"type": "trade_profit", "pnl": 10.0, "intensity": 0.2}]
        high = [{"type": "trade_profit", "pnl": 10.0, "intensity": 1.0}]
        a_low = evaluate_amygdala(low, MarketState())
        a_high = evaluate_amygdala(high, MarketState())
        assert a_high.arousal > a_low.arousal

    def test_attention_focus_set_to_ticker(self):
        events = [{"type": "trade_profit", "ticker": "BHP.AX", "pnl": 10.0}]
        affect = evaluate_amygdala(events, MarketState())
        assert affect.attention_focus == "BHP.AX"


# ======================================================================
# Reflective emotion modulation with economic state
# ======================================================================


class TestEconomicModulation:
    """Economic state deficits must modulate reflective emotions."""

    def test_low_alpha_increases_irritability(self):
        state = EmotionalState(irritability=0.0)
        eco = EconomicState(alpha=0.2)  # low alpha
        affect = AffectState()
        cs = MarketState()
        initial = state.irritability
        update_reflective_emotions(state, affect, cs, needs=eco)
        assert state.irritability > initial

    def test_low_roi_increases_anxiety(self):
        state = EmotionalState(anxiety=0.0)
        eco = EconomicState(roi=0.2)  # low ROI
        affect = AffectState()
        cs = MarketState()
        initial = state.anxiety
        update_reflective_emotions(state, affect, cs, needs=eco)
        assert state.anxiety > initial

    def test_high_volatility_increases_caution(self):
        state = EmotionalState(caution=0.0)
        eco = EconomicState(volatility=0.8)  # high volatility
        affect = AffectState()
        cs = MarketState()
        initial = state.caution
        update_reflective_emotions(state, affect, cs, needs=eco)
        assert state.caution > initial

    def test_high_cost_pressure_increases_irritability(self):
        state = EmotionalState(irritability=0.0)
        eco = EconomicState(cost_pressure=0.8)  # high cost pressure
        affect = AffectState()
        cs = MarketState()
        initial = state.irritability
        update_reflective_emotions(state, affect, cs, needs=eco)
        assert state.irritability > initial

    def test_profit_salience_boosts_conviction(self):
        state = EmotionalState(conviction=0.3)
        affect = AffectState(arousal=0.5, valence=0.5, salience_tags=["profit"])
        cs = MarketState()
        initial = state.conviction
        update_reflective_emotions(state, affect, cs)
        assert state.conviction > initial

    def test_loss_salience_reduces_conviction(self):
        state = EmotionalState(conviction=0.8)
        affect = AffectState(arousal=0.5, valence=-0.5, salience_tags=["loss"])
        cs = MarketState()
        initial = state.conviction
        update_reflective_emotions(state, affect, cs)
        assert state.conviction < initial

    def test_survival_tag_increases_caution(self):
        state = EmotionalState(caution=0.1)
        affect = AffectState(
            arousal=0.6, valence=-0.5, salience_tags=["survival"]
        )
        cs = MarketState()
        initial = state.caution
        update_reflective_emotions(state, affect, cs)
        assert state.caution > initial

    def test_drawdown_tag_increases_anxiety(self):
        state = EmotionalState(anxiety=0.0)
        affect = AffectState(
            arousal=0.5, valence=-0.4, salience_tags=["drawdown"]
        )
        cs = MarketState()
        initial = state.anxiety
        update_reflective_emotions(state, affect, cs)
        assert state.anxiety > initial


# ======================================================================
# Backward compatibility
# ======================================================================


class TestEmotionBackwardCompat:
    """Backward-compat aliases must work correctly."""

    def test_social_warmth_alias(self):
        state = EmotionalState(conviction=0.7)
        assert state.social_warmth == 0.7
        state.social_warmth = 0.3
        assert state.conviction == 0.3

    def test_avoidance_bias_alias(self):
        state = EmotionalState(caution=0.4)
        assert state.avoidance_bias == 0.4
        state.avoidance_bias = 0.6
        assert state.caution == 0.6

    def test_to_dict_uses_new_names(self):
        state = EmotionalState(conviction=0.7, caution=0.3)
        d = state.to_dict()
        assert "conviction" in d
        assert "caution" in d
        assert d["conviction"] == 0.7
        assert d["caution"] == 0.3
