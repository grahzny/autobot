"""Integration tests for the market entity.

Tests end-to-end scenarios: trade lifecycle, starvation, Sunday protocol,
Chris interaction, and adaptive tick intervals.
"""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

from autobot.identity import Identity
from autobot.living_engine import MarketEngine, MarketTickResult
from autobot.portfolio import Position


# ======================================================================
# Trade Lifecycle
# ======================================================================


class TestTradeLifecycle:
    """Full cycle: analyze -> buy -> monitor -> sell -> post-mortem."""

    def test_buy_records_rationale(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.portfolio.cash = 1000.0
        engine.monitor.add_ticker("BHP.AX")
        engine.state.last_prices["BHP.AX"] = 42.0

        # Simulate a buy
        pos = engine.portfolio.execute_buy(
            "BHP.AX", 42.0, 4, rationale="Test thesis",
        )
        assert pos is not None
        assert engine.portfolio.cash < 1000.0
        assert len(engine.portfolio.open_positions()) == 1

    def test_sell_generates_post_mortem(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.portfolio.cash = 1000.0
        engine.monitor.add_ticker("BHP.AX")

        # Buy
        pos = engine.portfolio.execute_buy(
            "BHP.AX", 40.0, 4, rationale="Long thesis",
        )
        assert pos is not None

        # Sell at profit (big enough gap to cover $12 brokerage + slippage)
        engine.state.last_prices["BHP.AX"] = 50.0
        pnl = engine.portfolio.execute_sell("BHP.AX", 50.0, rationale="Target hit")
        assert pnl is not None
        assert pnl > 0  # profitable after fees

        # Record post-mortem
        engine._record_post_mortem("BHP.AX", pnl)
        assert len(engine.research.post_mortems) == 1
        assert engine.research.post_mortems[0].pnl > 0

    def test_losing_trade_generates_loss_post_mortem(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.portfolio.cash = 1000.0

        pos = engine.portfolio.execute_buy("CBA.AX", 100.0, 1, rationale="Test")
        assert pos is not None

        pnl = engine.portfolio.execute_sell("CBA.AX", 90.0, rationale="Stop loss")
        assert pnl is not None
        assert pnl < 0  # loss

        engine._record_post_mortem("CBA.AX", pnl)
        assert engine.research.post_mortems[0].pnl < 0

    def test_trade_updates_predictions(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.portfolio.cash = 1000.0

        # Register prediction
        engine.trade_predictions.register_trade_prediction(
            ticker="BHP.AX", direction="long",
            entry_price=42.0, target_price=50.0, conviction=0.7,
        )

        # Buy and sell
        pos = engine.portfolio.execute_buy("BHP.AX", 42.0, 4, rationale="Test")
        pnl = engine.portfolio.execute_sell("BHP.AX", 46.0, rationale="Partial target")

        # Resolve prediction
        error = engine.trade_predictions.resolve_trade_prediction(
            "BHP.AX", 46.0, pnl,
        )
        assert error is not None  # prediction error calculated

        accuracy = engine.trade_predictions.direction_accuracy()
        assert accuracy is not None
        assert accuracy == 1.0  # got direction right (long + profit)


# ======================================================================
# Starvation
# ======================================================================


class TestStarvationIntegration:
    """Entity must die when capital reaches $0."""

    def test_death_at_zero_capital(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 0.0
        engine._born = True

        result = engine.tick()
        assert result.is_dead is True
        assert result.decision == "dead"
        assert result.starvation_level == "FATAL"

    def test_death_at_negative_capital(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = -1.0
        engine._born = True

        result = engine.tick()
        assert result.is_dead is True

    def test_survival_mode_low_capital(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 3.0
        engine._born = True

        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
                    with patch.object(engine.monitor, "market_session_label", return_value="ASX open"):
                        result = engine.tick()

        # Should be idle (survival mode), not analyzing
        assert result.decision == "idle"
        assert not result.is_dead

    def test_starvation_warning_levels(self):
        engine = MarketEngine()

        engine.accountant.operating_capital = 100.0
        assert engine.accountant.starvation_warning() is None

        engine.accountant.operating_capital = 15.0
        assert engine.accountant.starvation_warning() == "WARNING"

        engine.accountant.operating_capital = 3.0
        assert engine.accountant.starvation_warning() == "CRITICAL"

        engine.accountant.operating_capital = 0.0
        assert engine.accountant.starvation_warning() == "FATAL"

    def test_starvation_event_generated(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 3.0
        engine._born = True

        with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
            with patch.object(engine.monitor, "is_market_open", return_value=False):
                with patch.object(engine.monitor, "is_sunday", return_value=False):
                    with patch.object(engine.monitor, "market_session_label", return_value="closed"):
                        events = engine._appraise()

        starvation_events = [e for e in events if e["type"] == "starvation_warning"]
        assert len(starvation_events) == 1
        assert starvation_events[0]["level"] == "CRITICAL"


# ======================================================================
# Sunday Protocol
# ======================================================================


class TestSundayProtocol:
    """Sunday should trigger strategic synthesis."""

    def test_sunday_generates_synthesis_objective(self):
        engine = MarketEngine()
        engine.strategy.last_sunday_tick = 0

        engine.strategy.auto_generate(
            is_sunday=True, tick_count=50,
        )

        sunday_objs = [
            o for o in engine.strategy.objectives
            if o.obj_type == "sunday"
        ]
        assert len(sunday_objs) == 1
        assert sunday_objs[0].priority == 0.7

    def test_sunday_cooldown_respected(self):
        engine = MarketEngine()
        engine.strategy.last_sunday_tick = 45

        engine.strategy.auto_generate(
            is_sunday=True, tick_count=50,
        )

        sunday_objs = [
            o for o in engine.strategy.objectives
            if o.obj_type == "sunday"
        ]
        assert len(sunday_objs) == 0  # too soon

    def test_sunday_synthesis_records_weekly_review(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.accountant.refresh_daily_budget()

        result = MarketTickResult(tick=50, decision="synthesize")

        with patch("autobot.living_engine.strategic_synthesis") as mock_synth:
            mock_synth.return_value = MagicMock(
                summary="Good week",
                watchlist_changes=["add FMG.AX"],
                research_notes=["Iron ore demand strong"],
                adjustments=[],
                conviction_updates={},
                used_llm=True,
                raw="",
            )
            engine._do_synthesize(result)

        assert len(engine.research.weekly_reviews) == 1
        assert engine.research.weekly_reviews[0]["summary"] == "Good week"

    def test_sunday_synthesis_applies_watchlist_changes(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.accountant.refresh_daily_budget()
        engine.monitor.add_ticker("OLD.AX")

        result = MarketTickResult(tick=50, decision="synthesize")

        with patch("autobot.living_engine.strategic_synthesis") as mock_synth:
            mock_synth.return_value = MagicMock(
                summary="Adjustments needed",
                watchlist_changes=["add NEW.AX", "remove OLD.AX"],
                research_notes=[],
                adjustments=[],
                conviction_updates={},
                used_llm=True,
                raw="",
            )
            engine._do_synthesize(result)

        assert "NEW.AX" in engine.monitor.watchlist
        assert "OLD.AX" not in engine.monitor.watchlist


# ======================================================================
# Chris Interaction
# ======================================================================


class TestChrisInteraction:
    """Chris messages must be processed and responded to."""

    def test_message_queued(self):
        engine = MarketEngine()
        engine.receive_message("chris", "Chris", "Hello!")
        assert len(engine.state.unprocessed_messages()) == 1
        assert engine.state.chris is not None

    def test_chris_message_triggers_respond_decision(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.accountant.refresh_daily_budget()
        engine.receive_message("chris", "Chris", "How's the portfolio?")

        with patch("autobot.living_engine.respond_to_chris") as mock_respond:
            mock_respond.return_value = MagicMock(
                thinking="Chris wants portfolio update",
                response="Portfolio is doing well.",
                topics_of_interest=[],
                used_llm=True,
            )
            result = engine.tick()

        assert result.decision == "respond_chris"
        assert result.outgoing_message == "Portfolio is doing well."

    def test_chris_theory_of_mind_updated(self):
        engine = MarketEngine()
        engine.state.get_or_create_person("chris", "Chris")
        engine.state.add_message("chris", "What do you think about BHP?")

        with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
            with patch.object(engine.monitor, "is_market_open", return_value=False):
                with patch.object(engine.monitor, "market_session_label", return_value="closed"):
                    events = engine._perceive()

        # After processing, Chris's ToM should be updated
        chris = engine.state.chris
        assert chris.last_known_intent == "probing"  # has question mark


# ======================================================================
# Adaptive Tick Interval
# ======================================================================


class TestAdaptiveTickIntegration:
    """Tick interval must adapt to market conditions."""

    def test_interval_table(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0

        # Market open, no positions
        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                assert engine.calculate_tick_interval() == 60.0

        # Market open, with positions
        engine.portfolio.positions.append(Position(
            ticker="BHP.AX", shares=10, entry_price=42.0,
        ))
        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                assert engine.calculate_tick_interval() == 30.0

        # Market closed
        with patch.object(engine.monitor, "is_market_open", return_value=False):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                with patch.object(engine.monitor, "_now_local") as mock_now:
                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    # Monday evening
                    mock_now.return_value = datetime(
                        2026, 2, 16, 20, 0, 0,
                        tzinfo=ZoneInfo("Australia/Melbourne")
                    )
                    assert engine.calculate_tick_interval() == 300.0

    def test_chris_message_overrides_interval(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0
        engine.receive_message("chris", "Chris", "Hello!")
        assert engine.calculate_tick_interval() == 30.0


# ======================================================================
# End-to-end tick
# ======================================================================


class TestEndToEndTick:
    """Verify the full tick cycle runs without errors."""

    def test_multiple_ticks(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0

        # Run 5 ticks
        results = []
        for _ in range(5):
            result = engine.tick()
            results.append(result)
            assert not result.is_dead

        assert engine.state.tick_count == 5
        assert len(engine.state.mood_history) == 5

    def test_tick_with_watchlist(self):
        identity = Identity(
            name="TestBot",
            starting_capital=1000.0,
            seed_watchlist=["BHP.AX", "CBA.AX"],
        )
        engine = MarketEngine.from_identity(identity)

        result = engine.tick()
        assert not result.is_dead
        assert engine.state.tick_count == 1
        assert "BHP.AX" in engine.monitor.watchlist

    def test_economic_state_decays_over_ticks(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.economic.alpha = 0.8  # high alpha

        initial_alpha = engine.economic.alpha
        for _ in range(10):
            engine.tick()

        # Alpha should have decayed
        assert engine.economic.alpha < initial_alpha

    def test_debug_state_complete(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.tick()

        state = engine.debug_state()
        required_keys = [
            "tick", "energy", "emotions", "economic_state",
            "capital", "portfolio", "market_session", "watchlist",
            "research", "strategy", "is_alive",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"
