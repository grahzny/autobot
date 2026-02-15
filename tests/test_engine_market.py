"""Tests for MarketEngine -- tick cycle, decision logic, adaptive interval, starvation."""

from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

from autobot.accountant import Accountant
from autobot.chat_state import MarketState
from autobot.emotions import EmotionalState
from autobot.identity import Identity
from autobot.living_engine import MarketEngine, MarketTickResult
from autobot.market import MarketMonitor
from autobot.needs import EconomicState
from autobot.portfolio import Portfolio
from autobot.strategy import StrategyEngine


# ======================================================================
# Initialization
# ======================================================================


class TestMarketEngineInit:
    def test_default_creation(self):
        engine = MarketEngine()
        assert engine.state.tick_count == 0
        assert isinstance(engine.accountant, Accountant)
        assert isinstance(engine.portfolio, Portfolio)
        assert isinstance(engine.monitor, MarketMonitor)
        assert isinstance(engine.strategy, StrategyEngine)

    def test_from_identity(self):
        identity = Identity(
            name="TestBot",
            starting_capital=500.0,
            timezone="Australia/Melbourne",
            brokerage_fee=10.0,
            slippage_pct=0.001,
            max_position_pct=0.15,
            seed_watchlist=["BHP.AX", "AAPL"],
        )
        engine = MarketEngine.from_identity(identity)
        assert engine.accountant.operating_capital == 500.0
        assert engine.portfolio.brokerage_fee == 10.0
        assert engine.portfolio.slippage_pct == 0.001
        assert engine.portfolio.max_position_pct == 0.15
        assert engine.portfolio.cash == 500.0
        assert "BHP.AX" in engine.monitor.watchlist
        assert "AAPL" in engine.monitor.watchlist
        assert engine.state.name == "TestBot"

    def test_needs_alias(self):
        engine = MarketEngine()
        assert engine.needs is engine.economic
        engine.needs.alpha = 0.3
        assert engine.economic.alpha == 0.3


# ======================================================================
# Birth routine
# ======================================================================


class TestMarketEngineBirth:
    def test_birth_runs_once(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine._birth()
        assert engine._born is True
        assert len(engine.memory.episodes) == 1
        assert "birth" in engine.memory.episodes[0].salience_tags

    def test_birth_sets_name(self):
        engine = MarketEngine()
        engine.identity = Identity(name="TestBot")
        engine._birth()
        assert engine.state.name == "TestBot"

    def test_birth_refreshes_budget(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 900.0
        engine._birth()
        assert engine.accountant.daily_budget > 0


# ======================================================================
# Tick cycle basics
# ======================================================================


class TestMarketEngineTick:
    def test_tick_increments_count(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        result = engine.tick()
        assert engine.state.tick_count == 1

    def test_tick_returns_result(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        result = engine.tick()
        assert isinstance(result, MarketTickResult)
        assert result.tick == 1

    def test_tick_records_mood(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.tick()
        assert len(engine.state.mood_history) == 1

    def test_idle_tick_produces_thinking(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        result = engine.tick()
        assert result.thinking != ""

    def test_economic_state_in_result(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        result = engine.tick()
        assert "alpha" in result.economic_state
        assert "roi" in result.economic_state


# ======================================================================
# Decision logic
# ======================================================================


class TestDecisionLogic:
    def test_respond_chris_when_message_pending(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.accountant.refresh_daily_budget()
        engine.state.get_or_create_person("chris", "Chris")
        engine.state.add_message("chris", "What's the portfolio looking like?")

        # Run tick -- should respond to Chris
        with patch("autobot.living_engine.respond_to_chris") as mock_respond:
            mock_respond.return_value = MagicMock(
                thinking="Chris asked about portfolio",
                response="We're doing well.",
                topics_of_interest=[],
                used_llm=True,
            )
            result = engine.tick()

        assert result.decision == "respond_chris"

    def test_idle_when_no_messages(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        # No messages, no market open, no sunday
        with patch.object(engine.monitor, "is_market_open", return_value=False):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                result = engine.tick()
        assert result.decision == "idle"

    def test_survival_mode_when_capital_low(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 3.0  # below $5 threshold
        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                result = engine.tick()
        assert result.decision == "idle"

    def test_synthesize_on_sunday(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        engine.strategy.last_sunday_tick = 0
        engine.state.tick_count = 49  # tick() increments to 50 -> 50-0>40
        engine._born = True  # skip birth
        with patch.object(engine.monitor, "is_sunday", return_value=True):
            with patch.object(engine.monitor, "is_market_open", return_value=False):
                with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
                    with patch.object(engine.monitor, "market_session_label", return_value="Sunday (strategic synthesis)"):
                        with patch("autobot.living_engine.strategic_synthesis") as mock_synth:
                            mock_synth.return_value = MagicMock(
                                summary="Weekly review complete",
                                watchlist_changes=[],
                                research_notes=[],
                                adjustments=[],
                                conviction_updates={},
                                used_llm=True,
                                raw="",
                            )
                            result = engine.tick()
        assert result.decision == "synthesize"


# ======================================================================
# Starvation / Death
# ======================================================================


class TestStarvation:
    def test_death_when_capital_zero(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 0.0
        engine._born = True  # skip birth
        engine.state.tick_count = 0
        result = engine.tick()
        assert result.is_dead is True
        assert result.starvation_level == "FATAL"
        assert result.decision == "dead"

    def test_death_when_capital_negative(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = -5.0
        engine._born = True
        engine.state.tick_count = 0
        result = engine.tick()
        assert result.is_dead is True

    def test_alive_when_capital_positive(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0
        result = engine.tick()
        assert result.is_dead is False


# ======================================================================
# Adaptive tick interval
# ======================================================================


class TestAdaptiveInterval:
    def test_fast_with_chris_message(self):
        engine = MarketEngine()
        engine.state.get_or_create_person("chris", "Chris")
        engine.state.add_message("chris", "Hello")
        assert engine.calculate_tick_interval() == 30.0

    def test_fast_with_positions_market_open(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0
        # Mock an open position
        from autobot.portfolio import Position
        engine.portfolio.positions.append(Position(
            ticker="BHP.AX", shares=10, entry_price=42.0,
        ))
        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                interval = engine.calculate_tick_interval()
        assert interval == 30.0

    def test_medium_market_open_no_positions(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0
        with patch.object(engine.monitor, "is_market_open", return_value=True):
            with patch.object(engine.monitor, "is_sunday", return_value=False):
                interval = engine.calculate_tick_interval()
        assert interval == 60.0

    def test_slow_when_starvation(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 3.0  # < $5
        interval = engine.calculate_tick_interval()
        assert interval == 900.0

    def test_sunday_interval(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 100.0
        with patch.object(engine.monitor, "is_sunday", return_value=True):
            with patch.object(engine.monitor, "is_market_open", return_value=False):
                interval = engine.calculate_tick_interval()
        assert interval == 300.0


# ======================================================================
# Perceive step
# ======================================================================


class TestPerceive:
    def test_session_transition_detected(self):
        engine = MarketEngine()
        engine.state.last_session = "Markets closed"

        with patch.object(engine.monitor, "market_session_label", return_value="ASX open"):
            with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
                with patch.object(engine.monitor, "is_market_open", return_value=True):
                    events = engine._perceive()

        transition_events = [e for e in events if e["type"] == "session_transition"]
        assert len(transition_events) == 1
        assert transition_events[0]["from"] == "Markets closed"
        assert transition_events[0]["to"] == "ASX open"

    def test_chris_message_processed(self):
        engine = MarketEngine()
        engine.state.get_or_create_person("chris", "Chris")
        engine.state.add_message("chris", "Hello there!")

        with patch.object(engine.monitor, "should_fetch_prices", return_value=False):
            with patch.object(engine.monitor, "is_market_open", return_value=False):
                with patch.object(engine.monitor, "market_session_label", return_value="closed"):
                    events = engine._perceive()

        # Message should be processed
        assert engine.state.pending_messages[0].processed is True


# ======================================================================
# Monitor step
# ======================================================================


class TestMonitor:
    def test_monitor_no_positions(self):
        engine = MarketEngine()
        result = MarketTickResult(tick=1, decision="monitor")
        engine._do_monitor(result)
        assert result.decision == "idle"  # no positions -> idle

    def test_monitor_with_positions(self):
        engine = MarketEngine()
        from autobot.portfolio import Position
        engine.portfolio.positions.append(Position(
            id="pos1", ticker="BHP.AX", shares=10, entry_price=42.0,
        ))
        engine.state.last_prices["BHP.AX"] = 43.0

        result = MarketTickResult(tick=1, decision="monitor")
        engine._do_monitor(result)
        assert "BHP.AX" in result.thinking
        assert "$43.00" in result.thinking


# ======================================================================
# Debug state
# ======================================================================


class TestDebugState:
    def test_debug_state_keys(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 1000.0
        state = engine.debug_state()
        assert "tick" in state
        assert "capital" in state
        assert "portfolio" in state
        assert "emotions" in state
        assert "economic_state" in state
        assert "market_session" in state
        assert "is_alive" in state

    def test_debug_state_reflects_capital(self):
        engine = MarketEngine()
        engine.accountant.operating_capital = 750.0
        state = engine.debug_state()
        assert state["capital"]["operating_capital"] == 750.0
        assert state["is_alive"] is True


# ======================================================================
# Receive message
# ======================================================================


class TestReceiveMessage:
    def test_receive_message_queues(self):
        engine = MarketEngine()
        engine.receive_message("chris", "Chris", "Buy more BHP!")
        assert len(engine.state.unprocessed_messages()) == 1
        assert engine.state.chris is not None
        assert engine.state.chris.name == "Chris"
