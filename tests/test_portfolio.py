"""Tests for Portfolio -- buy/sell with fees, P&L, Sharpe, position limits."""

from __future__ import annotations

from autobot.portfolio import Portfolio, Position


class TestPositionBasics:
    def test_open_position(self):
        p = Position(ticker="BHP.AX", shares=10, entry_price=42.0)
        assert p.is_open is True
        assert p.realized_pnl is None

    def test_closed_position(self):
        p = Position(
            ticker="BHP.AX", shares=10,
            entry_price=42.0, exit_price=45.0,
            entry_brokerage=6.0, exit_brokerage=6.0,
        )
        assert p.is_open is False
        # P&L = (45-42)*10 - 6 - 6 = 30 - 12 = 18
        assert abs(p.realized_pnl - 18.0) < 0.01

    def test_cost_basis(self):
        p = Position(
            ticker="AAPL", shares=5,
            entry_price=150.0,
            entry_brokerage=6.0, entry_slippage=0.375,
        )
        # 150*5 + 6 + 0.375 = 756.375
        assert abs(p.cost_basis - 756.375) < 0.01

    def test_unrealized_pnl(self):
        p = Position(
            ticker="AAPL", shares=5,
            entry_price=150.0,
            entry_brokerage=6.0, entry_slippage=0.375,
        )
        # At $155: (155-150)*5 - 6 - 0.375 = 25 - 6.375 = 18.625
        assert abs(p.unrealized_pnl(155.0) - 18.625) < 0.01

    def test_mark_to_market(self):
        p = Position(ticker="AAPL", shares=10, entry_price=150.0)
        assert p.mark_to_market(160.0) == 1600.0


class TestBuyExecution:
    def test_basic_buy(self):
        port = Portfolio(cash=1000.0, brokerage_fee=6.0, slippage_pct=0.0005)
        pos = port.execute_buy("BHP.AX", 42.0, 4, "test thesis")
        assert pos is not None
        assert pos.ticker == "BHP.AX"
        assert pos.shares == 4
        assert pos.entry_brokerage == 6.0
        assert pos.entry_slippage > 0
        assert port.cash < 1000.0

    def test_buy_deducts_correct_amount(self):
        port = Portfolio(cash=1000.0, brokerage_fee=6.0, slippage_pct=0.0005)
        pos = port.execute_buy("BHP.AX", 42.0, 4)
        # Total cost = 42*4 + 6 + 42*0.0005*4 = 168 + 6 + 0.084 = 174.084
        expected_remaining = 1000.0 - 168.0 - 6.0 - (42.0 * 0.0005 * 4)
        assert abs(port.cash - expected_remaining) < 0.01

    def test_buy_fails_insufficient_cash(self):
        port = Portfolio(cash=100.0)
        pos = port.execute_buy("AAPL", 150.0, 10)
        assert pos is None
        assert port.cash == 100.0  # unchanged

    def test_buy_fails_position_too_large(self):
        port = Portfolio(cash=1000.0, max_position_pct=0.20)
        # Try to buy 100% of portfolio
        pos = port.execute_buy("AAPL", 150.0, 6)  # $900 > 20% of $1000
        assert pos is None

    def test_buy_within_position_limit(self):
        port = Portfolio(cash=1000.0, max_position_pct=0.20)
        # 20% of $1000 = $200, so 1 share at $150 = $150 < $200
        pos = port.execute_buy("AAPL", 150.0, 1)
        assert pos is not None

    def test_slippage_increases_entry_price(self):
        port = Portfolio(cash=1000.0, slippage_pct=0.001)
        pos = port.execute_buy("BHP.AX", 100.0, 1)
        assert pos is not None
        assert pos.entry_price > 100.0  # slippage raises effective price


class TestSellExecution:
    def test_basic_sell(self):
        port = Portfolio(cash=1000.0, brokerage_fee=6.0, slippage_pct=0.0005)
        port.execute_buy("BHP.AX", 42.0, 4)
        cash_before_sell = port.cash
        pnl = port.execute_sell("BHP.AX", 45.0, "take profit")
        assert pnl is not None
        assert port.cash > cash_before_sell

    def test_sell_moves_to_history(self):
        port = Portfolio(cash=1000.0)
        port.execute_buy("BHP.AX", 42.0, 4)
        port.execute_sell("BHP.AX", 45.0)
        assert len(port.trade_history) == 1
        assert port.trade_history[0].exit_price is not None

    def test_sell_nonexistent_position(self):
        port = Portfolio(cash=1000.0)
        pnl = port.execute_sell("BHP.AX", 45.0)
        assert pnl is None

    def test_profitable_trade_pnl(self):
        port = Portfolio(cash=1000.0, brokerage_fee=6.0, slippage_pct=0.0)
        port.execute_buy("BHP.AX", 40.0, 4)
        pnl = port.execute_sell("BHP.AX", 45.0)
        assert pnl is not None
        assert pnl > 0

    def test_losing_trade_pnl(self):
        port = Portfolio(cash=1000.0, brokerage_fee=6.0, slippage_pct=0.0)
        port.execute_buy("BHP.AX", 45.0, 4)
        pnl = port.execute_sell("BHP.AX", 40.0)
        assert pnl is not None
        assert pnl < 0


class TestPortfolioValue:
    def test_cash_only(self):
        port = Portfolio(cash=1000.0)
        assert port.total_value({}) == 1000.0

    def test_with_position(self):
        port = Portfolio(cash=1000.0, brokerage_fee=0.0, slippage_pct=0.0)
        port.execute_buy("AAPL", 150.0, 1)
        # Cash = 1000 - 150 = 850, position at 160 = 160
        value = port.total_value({"AAPL": 160.0})
        assert abs(value - 1010.0) < 0.01  # 850 + 160


class TestPerformanceMetrics:
    def test_sharpe_insufficient_data(self):
        port = Portfolio()
        assert port.sharpe_ratio() is None

    def test_sharpe_with_data(self):
        port = Portfolio()
        # Simulate daily returns
        port.daily_returns = [0.01, 0.02, -0.005, 0.015, 0.008, -0.01, 0.012]
        sharpe = port.sharpe_ratio()
        assert sharpe is not None
        assert isinstance(sharpe, float)

    def test_win_rate_no_trades(self):
        port = Portfolio()
        assert port.win_rate() == 0.0

    def test_win_rate_all_winners(self):
        port = Portfolio(cash=10000.0, brokerage_fee=0.0, slippage_pct=0.0)
        port.execute_buy("A", 10.0, 10)
        port.execute_sell("A", 15.0)
        port.execute_buy("B", 20.0, 5)
        port.execute_sell("B", 25.0)
        assert port.win_rate() == 1.0

    def test_win_rate_mixed(self):
        port = Portfolio(cash=10000.0, brokerage_fee=0.0, slippage_pct=0.0)
        port.execute_buy("A", 10.0, 10)
        port.execute_sell("A", 15.0)  # win
        port.execute_buy("B", 20.0, 5)
        port.execute_sell("B", 15.0)  # loss
        assert abs(port.win_rate() - 0.5) < 0.01

    def test_total_realized_pnl(self):
        port = Portfolio(cash=10000.0, brokerage_fee=0.0, slippage_pct=0.0)
        port.execute_buy("A", 10.0, 10)
        port.execute_sell("A", 15.0)  # +50
        port.execute_buy("B", 20.0, 5)
        port.execute_sell("B", 18.0)  # -10
        pnl = port.total_realized_pnl()
        assert abs(pnl - 40.0) < 0.01


class TestCanBuy:
    def test_can_buy_affordable(self):
        port = Portfolio(cash=1000.0)
        ok, msg = port.can_buy("BHP.AX", 42.0, 2)
        assert ok is True

    def test_cannot_buy_too_expensive(self):
        port = Portfolio(cash=100.0)
        ok, msg = port.can_buy("AAPL", 150.0, 10)
        assert ok is False
        assert "Insufficient" in msg

    def test_cannot_buy_exceeds_position_limit(self):
        port = Portfolio(cash=1000.0, max_position_pct=0.10)
        ok, msg = port.can_buy("AAPL", 150.0, 1)  # $150 = 15% > 10%
        assert ok is False
        assert "too large" in msg.lower() or "Position" in msg


class TestSerialization:
    def test_to_dict(self):
        port = Portfolio(cash=1000.0)
        port.execute_buy("BHP.AX", 42.0, 4)
        d = port.to_dict()
        assert d["cash"] == port.cash
        assert len(d["positions"]) == 1

    def test_summary(self):
        port = Portfolio(cash=1000.0)
        s = port.summary()
        assert s["cash"] == 1000.0
        assert s["open_positions"] == 0
        assert s["total_trades"] == 0
