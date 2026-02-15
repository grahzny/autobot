"""MarketMonitor -- yfinance wrapper, watchlist, market hours, price alerts.

Provides market data access with timezone-aware market session detection
for both ASX (10:00-16:00 AEST) and US (09:30-16:00 ET = ~00:30-07:00 AEST).
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]


@dataclass
class PriceSnapshot:
    """A point-in-time price observation for a ticker."""

    ticker: str
    price: float
    change_pct: float       # daily percentage change
    volume: int
    timestamp: float        # unix timestamp of observation


@dataclass
class WatchlistEntry:
    """A ticker on the entity's watchlist with cached data."""

    ticker: str
    name: str = ""
    sector: str = ""
    exchange: str = ""       # "ASX" or "US"
    last_price: float | None = None
    price_history: list[PriceSnapshot] = field(default_factory=list)

    # Alert thresholds
    alert_above: float | None = None
    alert_below: float | None = None

    # Cached fundamentals (refreshed every 24h)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    fundamentals_fetched_at: float = 0.0


@dataclass
class MarketMonitor:
    """Market data access and session detection."""

    watchlist: dict[str, WatchlistEntry] = field(default_factory=dict)
    timezone: str = "Australia/Melbourne"

    # Rate limiting
    last_price_fetch: float = 0.0
    last_news_fetch: float = 0.0
    price_fetch_interval: float = 300.0     # 5 minutes
    news_fetch_interval: float = 1800.0     # 30 minutes

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def _now_local(self) -> datetime:
        return datetime.now(self._tz())

    def add_ticker(self, ticker: str) -> WatchlistEntry:
        """Add a ticker to the watchlist."""
        if ticker not in self.watchlist:
            exchange = "ASX" if ticker.endswith(".AX") else "US"
            self.watchlist[ticker] = WatchlistEntry(
                ticker=ticker, exchange=exchange,
            )
        return self.watchlist[ticker]

    def remove_ticker(self, ticker: str) -> None:
        """Remove a ticker from the watchlist."""
        self.watchlist.pop(ticker, None)

    # --- Market hours detection ---

    def is_asx_open(self, now: datetime | None = None) -> bool:
        """ASX is open Mon-Fri 10:00-16:00 AEST/AEDT."""
        now = now or self._now_local()
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return 10 <= now.hour < 16

    def is_us_open(self, now: datetime | None = None) -> bool:
        """US markets open 09:30-16:00 ET.

        In AEST that's roughly 00:30-07:00 (EST) or 23:30-06:00 (EDT).
        We check by converting to US/Eastern directly.
        """
        now = now or self._now_local()
        us_tz = ZoneInfo("US/Eastern")
        us_now = now.astimezone(us_tz)
        if us_now.weekday() >= 5:
            return False
        us_hour = us_now.hour
        us_minute = us_now.minute
        # Open from 09:30 to 16:00 ET
        if us_hour < 9 or us_hour >= 16:
            return False
        if us_hour == 9 and us_minute < 30:
            return False
        return True

    def is_market_open(self, exchange: str = "any", now: datetime | None = None) -> bool:
        """Check if a specific exchange (or any) is open."""
        if exchange == "ASX":
            return self.is_asx_open(now)
        if exchange == "US":
            return self.is_us_open(now)
        # "any" -- either open
        return self.is_asx_open(now) or self.is_us_open(now)

    def is_sunday(self, now: datetime | None = None) -> bool:
        """Sunday = strategic synthesis day."""
        now = now or self._now_local()
        return now.weekday() == 6

    def market_session_label(self, now: datetime | None = None) -> str:
        """Human-readable market session label for ground truth."""
        now = now or self._now_local()

        if self.is_sunday(now):
            return "Sunday (strategic synthesis)"

        parts = []
        if self.is_asx_open(now):
            parts.append("ASX open")
        if self.is_us_open(now):
            parts.append("US open")

        if parts:
            return ", ".join(parts)

        if now.weekday() >= 5:
            return "Weekend (markets closed)"

        return "Markets closed"

    # --- Price fetching ---

    def should_fetch_prices(self) -> bool:
        """Rate limit: only fetch prices every price_fetch_interval seconds."""
        return (_time.time() - self.last_price_fetch) >= self.price_fetch_interval

    def fetch_prices(self, tickers: list[str] | None = None) -> dict[str, PriceSnapshot]:
        """Fetch current prices for tickers (defaults to watchlist).

        Returns dict of ticker -> PriceSnapshot.
        """
        if yf is None:
            logger.warning("yfinance not installed -- cannot fetch prices")
            return {}

        tickers = tickers or list(self.watchlist.keys())
        if not tickers:
            return {}

        now = _time.time()
        results: dict[str, PriceSnapshot] = {}

        try:
            data = yf.download(
                " ".join(tickers),
                period="1d",
                interval="1d",
                progress=False,
                threads=True,
            )

            if data.empty:
                return results

            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        close = float(data["Close"].iloc[-1])
                        prev_close = float(data["Open"].iloc[-1])
                        volume = int(data["Volume"].iloc[-1])
                    else:
                        close = float(data["Close"][ticker].iloc[-1])
                        prev_close = float(data["Open"][ticker].iloc[-1])
                        volume = int(data["Volume"][ticker].iloc[-1])

                    change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0.0

                    snap = PriceSnapshot(
                        ticker=ticker,
                        price=round(close, 4),
                        change_pct=round(change_pct, 2),
                        volume=volume,
                        timestamp=now,
                    )
                    results[ticker] = snap

                    # Update watchlist entry
                    if ticker in self.watchlist:
                        entry = self.watchlist[ticker]
                        entry.last_price = snap.price
                        entry.price_history.append(snap)
                        # Keep last 100 snapshots
                        if len(entry.price_history) > 100:
                            entry.price_history = entry.price_history[-100:]

                except (KeyError, IndexError) as exc:
                    logger.debug("Failed to parse price for %s: %s", ticker, exc)

        except Exception as exc:
            logger.warning("yfinance download failed: %s", exc)

        self.last_price_fetch = now
        return results

    def fetch_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental data for a ticker (cached 24h)."""
        if yf is None:
            return {}

        entry = self.watchlist.get(ticker)
        if entry and (_time.time() - entry.fundamentals_fetched_at) < 86400:
            return entry.fundamentals

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            fundamentals = {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "name": info.get("shortName", ticker),
            }

            if entry:
                entry.fundamentals = fundamentals
                entry.fundamentals_fetched_at = _time.time()
                entry.name = fundamentals.get("name", "")
                entry.sector = fundamentals.get("sector", "")

            return fundamentals

        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)
            return {}

    # --- Alerts ---

    def set_alert(
        self, ticker: str, above: float | None = None, below: float | None = None,
    ) -> None:
        """Set price alert thresholds for a ticker."""
        entry = self.watchlist.get(ticker)
        if entry:
            if above is not None:
                entry.alert_above = above
            if below is not None:
                entry.alert_below = below

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check all watchlist entries for triggered alerts."""
        triggered = []
        for ticker, entry in self.watchlist.items():
            if entry.last_price is None:
                continue
            if entry.alert_above is not None and entry.last_price >= entry.alert_above:
                triggered.append({
                    "type": "price_alert",
                    "subtype": "above_threshold",
                    "ticker": ticker,
                    "price": entry.last_price,
                    "threshold": entry.alert_above,
                })
                entry.alert_above = None  # one-shot
            if entry.alert_below is not None and entry.last_price <= entry.alert_below:
                triggered.append({
                    "type": "price_alert",
                    "subtype": "below_threshold",
                    "ticker": ticker,
                    "price": entry.last_price,
                    "threshold": entry.alert_below,
                })
                entry.alert_below = None  # one-shot

        return triggered

    def ground_truth_text(self) -> str:
        """Market status for inclusion in ground truth."""
        now = self._now_local()
        lines = [
            f"Market session: {self.market_session_label(now)}",
            f"Local time: {now.strftime('%H:%M %Z')} ({now.strftime('%A')})",
        ]

        # Watchlist prices
        priced = [
            (t, e) for t, e in self.watchlist.items() if e.last_price is not None
        ]
        if priced:
            lines.append("Watchlist:")
            for ticker, entry in priced:
                hist = entry.price_history
                change = f" ({hist[-1].change_pct:+.1f}%)" if hist else ""
                lines.append(f"  {ticker}: ${entry.last_price:.2f}{change}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "timezone": self.timezone,
            "watchlist": {
                ticker: {
                    "ticker": e.ticker,
                    "name": e.name,
                    "sector": e.sector,
                    "exchange": e.exchange,
                    "last_price": e.last_price,
                    "alert_above": e.alert_above,
                    "alert_below": e.alert_below,
                    "fundamentals": e.fundamentals,
                    "fundamentals_fetched_at": e.fundamentals_fetched_at,
                }
                for ticker, e in self.watchlist.items()
            },
            "last_price_fetch": self.last_price_fetch,
            "last_news_fetch": self.last_news_fetch,
        }
