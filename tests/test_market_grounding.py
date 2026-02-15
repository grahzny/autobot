"""Tests for market-specific grounding and trade predictions."""

from __future__ import annotations

from autobot.grounding import check_market_grounding
from autobot.prediction import TradePredictionTracker


class TestMarketGrounding:
    def test_flags_certainty_claim(self):
        result = check_market_grounding("The stock will definitely go up.")
        assert result.ungrounded_count > 0
        assert any("market_prediction" in f for f in result.flags)

    def test_flags_rally_claim(self):
        result = check_market_grounding("It's going to rally hard next week.")
        assert result.ungrounded_count > 0

    def test_flags_guaranteed_return(self):
        result = check_market_grounding("This is a guaranteed return of 20%.")
        assert result.ungrounded_count > 0

    def test_flags_cant_lose(self):
        result = check_market_grounding("You can't lose on this trade.")
        assert result.ungrounded_count > 0

    def test_clean_analysis_no_flags(self):
        result = check_market_grounding(
            "BHP has a PE ratio of 12 and solid fundamentals."
        )
        assert result.ungrounded_count == 0

    def test_hedged_language_ok(self):
        result = check_market_grounding(
            "The stock could potentially rally if earnings beat expectations."
        )
        assert result.ungrounded_count == 0

    def test_price_claim_near_known(self):
        result = check_market_grounding(
            "BHP is trading at $42.50",
            known_prices={"BHP.AX": 42.30},
        )
        assert result.ungrounded_count == 0  # within 5%

    def test_price_claim_far_from_known(self):
        result = check_market_grounding(
            "BHP is trading at $100.00",
            known_prices={"BHP.AX": 42.30},
        )
        assert result.ungrounded_count > 0
        assert any("price_claim" in f for f in result.flags)


class TestTradePredictions:
    def test_register_prediction(self):
        tracker = TradePredictionTracker()
        pred = tracker.register_trade_prediction(
            ticker="BHP.AX", direction="long",
            entry_price=42.0, target_price=48.0,
            conviction=0.7, timeframe="weeks",
        )
        assert pred.ticker == "BHP.AX"
        assert len(tracker.predictions) == 1

    def test_resolve_prediction(self):
        tracker = TradePredictionTracker()
        tracker.register_trade_prediction(
            ticker="BHP.AX", direction="long",
            entry_price=42.0, target_price=48.0,
        )
        error = tracker.resolve_trade_prediction("BHP.AX", 46.0, 30.0)
        assert error is not None
        assert error >= 0  # error is absolute difference

    def test_resolve_nonexistent(self):
        tracker = TradePredictionTracker()
        error = tracker.resolve_trade_prediction("AAPL", 150.0, 10.0)
        assert error is None

    def test_direction_accuracy_all_correct(self):
        tracker = TradePredictionTracker()
        tracker.register_trade_prediction("A", "long", 10.0, 15.0)
        tracker.resolve_trade_prediction("A", 14.0, 30.0)  # long, positive pnl = correct
        tracker.register_trade_prediction("B", "long", 20.0, 25.0)
        tracker.resolve_trade_prediction("B", 23.0, 20.0)
        assert tracker.direction_accuracy() == 1.0

    def test_direction_accuracy_mixed(self):
        tracker = TradePredictionTracker()
        tracker.register_trade_prediction("A", "long", 10.0, 15.0)
        tracker.resolve_trade_prediction("A", 14.0, 30.0)  # correct
        tracker.register_trade_prediction("B", "long", 20.0, 25.0)
        tracker.resolve_trade_prediction("B", 18.0, -15.0)  # wrong
        assert abs(tracker.direction_accuracy() - 0.5) < 0.01

    def test_direction_accuracy_no_data(self):
        tracker = TradePredictionTracker()
        assert tracker.direction_accuracy() is None

    def test_average_prediction_error(self):
        tracker = TradePredictionTracker()
        tracker.register_trade_prediction("A", "long", 100.0, 110.0)
        tracker.resolve_trade_prediction("A", 108.0, 60.0)
        avg = tracker.average_prediction_error()
        assert avg is not None
        assert avg >= 0

    def test_max_history_trimming(self):
        tracker = TradePredictionTracker(max_history=5)
        for i in range(10):
            tracker.register_trade_prediction(f"T{i}", "long", 10.0, 15.0)
        assert len(tracker.predictions) == 5
