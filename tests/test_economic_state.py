"""Tests for EconomicState -- decay, event-driven updates, deficit summary."""

from __future__ import annotations

from autobot.needs import EconomicState, decay_economic_state, update_economic_state


class TestEconomicStateBasics:
    def test_defaults(self):
        state = EconomicState()
        assert state.alpha == 0.5
        assert state.roi == 0.5
        assert state.volatility == 0.3
        assert state.cost_pressure == 0.3

    def test_clamp_upper(self):
        state = EconomicState(alpha=1.5, roi=2.0)
        state.clamp()
        assert state.alpha == 1.0
        assert state.roi == 1.0

    def test_clamp_lower(self):
        state = EconomicState(alpha=-0.5, roi=-1.0)
        state.clamp()
        assert state.alpha == 0.0
        assert state.roi == 0.0

    def test_to_dict(self):
        state = EconomicState()
        d = state.to_dict()
        assert "alpha" in d
        assert "roi" in d
        assert "volatility" in d
        assert "cost_pressure" in d


class TestLowestNeed:
    def test_low_alpha_is_critical(self):
        state = EconomicState(alpha=0.1, roi=0.5, volatility=0.3, cost_pressure=0.3)
        name, val = state.lowest_need()
        assert name == "alpha"

    def test_high_volatility_is_critical(self):
        state = EconomicState(alpha=0.8, roi=0.8, volatility=0.9, cost_pressure=0.3)
        name, val = state.lowest_need()
        assert name == "volatility"
        assert abs(val - 0.1) < 0.01  # 1.0 - 0.9 = 0.1

    def test_high_cost_pressure_is_critical(self):
        state = EconomicState(alpha=0.8, roi=0.8, volatility=0.2, cost_pressure=0.95)
        name, val = state.lowest_need()
        assert name == "cost_pressure"


class TestDeficitSummary:
    def test_no_deficits(self):
        state = EconomicState()
        assert state.deficit_summary() == "no critical deficits"

    def test_alpha_deficit(self):
        state = EconomicState(alpha=0.2)
        summary = state.deficit_summary()
        assert "alpha" in summary

    def test_high_volatility_deficit(self):
        state = EconomicState(volatility=0.8)
        summary = state.deficit_summary()
        assert "volatility" in summary

    def test_multiple_deficits(self):
        state = EconomicState(alpha=0.1, roi=0.1, volatility=0.9, cost_pressure=0.9)
        summary = state.deficit_summary()
        assert "alpha" in summary
        assert "roi" in summary
        assert "volatility" in summary
        assert "cost_pressure" in summary


class TestDecay:
    def test_alpha_decays(self):
        state = EconomicState(alpha=0.5)
        decay_economic_state(state)
        assert state.alpha < 0.5

    def test_roi_decays(self):
        state = EconomicState(roi=0.5)
        decay_economic_state(state)
        assert state.roi < 0.5

    def test_volatility_settles(self):
        state = EconomicState(volatility=0.5)
        decay_economic_state(state)
        assert state.volatility < 0.5  # settles down

    def test_cost_pressure_settles(self):
        state = EconomicState(cost_pressure=0.5)
        decay_economic_state(state)
        assert state.cost_pressure < 0.5  # settles down

    def test_values_stay_clamped(self):
        state = EconomicState(alpha=0.001, roi=0.001, volatility=0.001, cost_pressure=0.001)
        decay_economic_state(state)
        assert state.alpha >= 0.0
        assert state.roi >= 0.0
        assert state.volatility >= 0.0
        assert state.cost_pressure >= 0.0


class TestEventDrivenUpdates:
    def test_trade_profit_boosts_alpha_and_roi(self):
        state = EconomicState(alpha=0.5, roi=0.5)
        events = [{"type": "trade_profit"}]
        update_economic_state(state, events)
        assert state.alpha > 0.5
        assert state.roi > 0.5

    def test_trade_loss_drains_alpha_and_roi(self):
        state = EconomicState(alpha=0.5, roi=0.5)
        events = [{"type": "trade_loss"}]
        update_economic_state(state, events)
        assert state.alpha < 0.5
        assert state.roi < 0.5

    def test_thesis_confirmed_boosts_alpha(self):
        state = EconomicState(alpha=0.5)
        events = [{"type": "thesis_confirmed"}]
        update_economic_state(state, events)
        assert state.alpha > 0.5

    def test_thesis_invalidated_drains_alpha(self):
        state = EconomicState(alpha=0.5)
        events = [{"type": "thesis_invalidated"}]
        update_economic_state(state, events)
        assert state.alpha < 0.5

    def test_price_alert_raises_volatility(self):
        state = EconomicState(volatility=0.3)
        events = [{"type": "price_alert"}]
        update_economic_state(state, events)
        assert state.volatility > 0.3

    def test_starvation_warning_raises_both(self):
        state = EconomicState(volatility=0.3, cost_pressure=0.3)
        events = [{"type": "starvation_warning"}]
        update_economic_state(state, events)
        assert state.volatility > 0.3
        assert state.cost_pressure > 0.3

    def test_recovery_lowers_volatility(self):
        state = EconomicState(volatility=0.5)
        events = [{"type": "recovery"}]
        update_economic_state(state, events)
        assert state.volatility < 0.5


class TestMetricEffects:
    def test_positive_trade_pnl(self):
        state = EconomicState(alpha=0.5)
        update_economic_state(state, [], trade_pnl=25.0)
        assert state.alpha > 0.5

    def test_negative_trade_pnl(self):
        state = EconomicState(alpha=0.5)
        update_economic_state(state, [], trade_pnl=-25.0)
        assert state.alpha < 0.5

    def test_good_sharpe_boosts_roi(self):
        state = EconomicState(roi=0.5)
        update_economic_state(state, [], sharpe=1.5)
        assert state.roi > 0.5

    def test_bad_sharpe_drains_roi(self):
        state = EconomicState(roi=0.5)
        update_economic_state(state, [], sharpe=-0.5)
        assert state.roi < 0.5

    def test_high_budget_spend_raises_cost_pressure(self):
        state = EconomicState(cost_pressure=0.3)
        update_economic_state(state, [], daily_budget_pct=0.85)
        assert state.cost_pressure > 0.3

    def test_drawdown_raises_volatility(self):
        state = EconomicState(volatility=0.3)
        update_economic_state(state, [], drawdown_pct=0.15)
        assert state.volatility > 0.3


class TestBackwardCompatibility:
    def test_needsstate_alias(self):
        from autobot.needs import NeedsState
        # NeedsState is just EconomicState
        state = NeedsState()
        assert hasattr(state, "alpha")
        assert hasattr(state, "roi")

    def test_decay_needs_alias(self):
        from autobot.needs import decay_needs
        state = EconomicState(alpha=0.5)
        decay_needs(state)
        assert state.alpha < 0.5
