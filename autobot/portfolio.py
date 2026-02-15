"""Portfolio -- Simulated trading with realistic costs.

Manages positions, executes simulated buys/sells with brokerage fees and
slippage, tracks P&L, and calculates performance metrics (Sharpe, win rate).
"""

from __future__ import annotations

import math
import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    """An open or closed stock position."""

    id: str = ""
    ticker: str = ""
    shares: int = 0
    entry_price: float = 0.0
    entry_time: float = 0.0
    entry_rationale: str = ""
    entry_brokerage: float = 0.0
    entry_slippage: float = 0.0

    # Filled on close
    exit_price: float | None = None
    exit_time: float | None = None
    exit_rationale: str = ""
    exit_brokerage: float = 0.0
    exit_slippage: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def cost_basis(self) -> float:
        """Total cost to enter (price * shares + fees)."""
        return (self.entry_price * self.shares) + self.entry_brokerage + self.entry_slippage

    @property
    def market_value(self) -> float:
        """Current value at entry price (use mark_to_market for live)."""
        return self.entry_price * self.shares

    def mark_to_market(self, current_price: float) -> float:
        """Current market value of the position."""
        return current_price * self.shares

    @property
    def realized_pnl(self) -> float | None:
        """P&L if position is closed, including all fees."""
        if self.exit_price is None:
            return None
        gross = (self.exit_price - self.entry_price) * self.shares
        total_fees = (
            self.entry_brokerage + self.entry_slippage
            + self.exit_brokerage + self.exit_slippage
        )
        return gross - total_fees

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized P&L at a given price (excluding exit fees)."""
        gross = (current_price - self.entry_price) * self.shares
        return gross - self.entry_brokerage - self.entry_slippage

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "entry_rationale": self.entry_rationale,
            "entry_brokerage": self.entry_brokerage,
            "entry_slippage": self.entry_slippage,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "exit_rationale": self.exit_rationale,
            "exit_brokerage": self.exit_brokerage,
            "exit_slippage": self.exit_slippage,
            "is_open": self.is_open,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Portfolio:
    """Manages cash, positions, and trade history."""

    cash: float = 1000.00
    brokerage_fee: float = 6.00
    slippage_pct: float = 0.0005        # 0.05%
    max_position_pct: float = 0.20      # max 20% of portfolio value

    positions: list[Position] = field(default_factory=list)
    trade_history: list[Position] = field(default_factory=list)

    # Performance tracking
    daily_returns: list[float] = field(default_factory=list)
    last_total_value: float | None = None

    def open_positions(self) -> list[Position]:
        """Return all currently open positions."""
        return [p for p in self.positions if p.is_open]

    def position_for_ticker(self, ticker: str) -> Position | None:
        """Find an open position for a ticker."""
        for p in self.positions:
            if p.ticker == ticker and p.is_open:
                return p
        return None

    def total_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + mark-to-market of open positions."""
        value = self.cash
        for p in self.open_positions():
            price = prices.get(p.ticker, p.entry_price)
            value += p.mark_to_market(price)
        return value

    def can_buy(
        self, ticker: str, price: float, shares: int,
    ) -> tuple[bool, str]:
        """Check if a buy is feasible (cash, position size limit)."""
        slippage = price * self.slippage_pct * shares
        total_cost = (price * shares) + self.brokerage_fee + slippage

        if total_cost > self.cash:
            return False, f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"

        # Position size check against current portfolio value
        position_value = price * shares
        # Use cash as proxy for portfolio value (conservative)
        portfolio_value = self.cash
        for p in self.open_positions():
            portfolio_value += p.entry_price * p.shares

        if portfolio_value > 0 and (position_value / portfolio_value) > self.max_position_pct:
            return False, (
                f"Position too large: ${position_value:.2f} is "
                f"{position_value/portfolio_value*100:.0f}% of portfolio "
                f"(max {self.max_position_pct*100:.0f}%)"
            )

        return True, "OK"

    def execute_buy(
        self, ticker: str, price: float, shares: int, rationale: str = "",
    ) -> Position | None:
        """Execute a simulated buy with brokerage and slippage.

        Returns the Position if successful, None if cannot afford.
        """
        can, reason = self.can_buy(ticker, price, shares)
        if not can:
            return None

        slippage = round(price * self.slippage_pct * shares, 4)
        # Slippage increases effective entry price
        effective_price = price + (price * self.slippage_pct)

        position = Position(
            id=str(uuid.uuid4())[:8],
            ticker=ticker,
            shares=shares,
            entry_price=round(effective_price, 4),
            entry_time=_time.time(),
            entry_rationale=rationale,
            entry_brokerage=self.brokerage_fee,
            entry_slippage=slippage,
        )

        total_cost = (price * shares) + self.brokerage_fee + slippage
        self.cash -= total_cost
        self.cash = round(self.cash, 4)
        self.positions.append(position)

        return position

    def execute_sell(
        self, ticker: str, price: float, rationale: str = "",
    ) -> float | None:
        """Sell an open position at the given price.

        Returns realized P&L (including all fees), or None if no position.
        """
        position = self.position_for_ticker(ticker)
        if position is None:
            return None

        slippage = round(price * self.slippage_pct * position.shares, 4)
        # Slippage decreases effective exit price
        effective_price = price - (price * self.slippage_pct)

        position.exit_price = round(effective_price, 4)
        position.exit_time = _time.time()
        position.exit_rationale = rationale
        position.exit_brokerage = self.brokerage_fee
        position.exit_slippage = slippage

        # Credit cash: sale proceeds minus fees
        proceeds = (price * position.shares) - self.brokerage_fee - slippage
        self.cash += proceeds
        self.cash = round(self.cash, 4)

        # Move to trade history
        self.trade_history.append(position)

        return position.realized_pnl

    def record_daily_return(self, prices: dict[str, float]) -> None:
        """Record daily return for Sharpe calculation."""
        current = self.total_value(prices)
        if self.last_total_value is not None and self.last_total_value > 0:
            daily_ret = (current - self.last_total_value) / self.last_total_value
            self.daily_returns.append(daily_ret)
        self.last_total_value = current

    def sharpe_ratio(self) -> float | None:
        """Annualized Sharpe ratio from daily returns.

        Returns None if insufficient data (need at least 5 returns).
        """
        if len(self.daily_returns) < 5:
            return None

        mean_ret = sum(self.daily_returns) / len(self.daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in self.daily_returns) / len(self.daily_returns)
        std_ret = math.sqrt(variance)

        if std_ret == 0:
            return None

        # Annualize: multiply by sqrt(252 trading days)
        return round((mean_ret / std_ret) * math.sqrt(252), 4)

    def win_rate(self) -> float:
        """Percentage of closed trades that were profitable."""
        closed = [t for t in self.trade_history if t.realized_pnl is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.realized_pnl > 0)  # type: ignore[operator]
        return round(wins / len(closed), 4)

    def total_realized_pnl(self) -> float:
        """Sum of all realized P&L from closed trades."""
        return sum(
            t.realized_pnl for t in self.trade_history
            if t.realized_pnl is not None
        )

    def summary(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        """Portfolio summary for API/display."""
        prices = prices or {}
        return {
            "cash": round(self.cash, 2),
            "total_value": round(self.total_value(prices), 2),
            "open_positions": len(self.open_positions()),
            "total_trades": len(self.trade_history),
            "total_realized_pnl": round(self.total_realized_pnl(), 2),
            "win_rate": self.win_rate(),
            "sharpe_ratio": self.sharpe_ratio(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "cash": self.cash,
            "brokerage_fee": self.brokerage_fee,
            "slippage_pct": self.slippage_pct,
            "max_position_pct": self.max_position_pct,
            "positions": [p.to_dict() for p in self.positions],
            "trade_history": [p.to_dict() for p in self.trade_history],
            "daily_returns": self.daily_returns[-252:],  # keep ~1 year
            "last_total_value": self.last_total_value,
        }
