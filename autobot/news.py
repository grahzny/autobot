"""NewsHarvester -- yfinance news fetching with deduplication.

Fetches news headlines from yfinance (free, unlimited) and deduplicates
by URL. Keeps the last 100 items.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]


@dataclass
class NewsItem:
    """A single news headline."""

    title: str = ""
    url: str = ""
    publisher: str = ""
    ticker: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "ticker": self.ticker,
            "timestamp": self.timestamp,
        }


@dataclass
class NewsHarvester:
    """Fetches and deduplicates news from yfinance."""

    items: list[NewsItem] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    max_items: int = 100

    def fetch_yfinance_news(self, tickers: list[str]) -> list[NewsItem]:
        """Fetch news for tickers from yfinance. Returns new items only."""
        if yf is None:
            logger.warning("yfinance not installed -- cannot fetch news")
            return []

        new_items: list[NewsItem] = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news or []

                for article in news:
                    url = article.get("link", "") or article.get("url", "")
                    if not url or url in self.seen_urls:
                        continue

                    item = NewsItem(
                        title=article.get("title", ""),
                        url=url,
                        publisher=article.get("publisher", ""),
                        ticker=ticker,
                        timestamp=article.get("providerPublishTime", _time.time()),
                    )

                    self.seen_urls.add(url)
                    self.items.append(item)
                    new_items.append(item)

            except Exception as exc:
                logger.debug("Failed to fetch news for %s: %s", ticker, exc)

        # Trim to max_items
        if len(self.items) > self.max_items:
            removed = self.items[:-self.max_items]
            self.items = self.items[-self.max_items:]
            # Clean up seen_urls for removed items
            removed_urls = {i.url for i in removed}
            current_urls = {i.url for i in self.items}
            self.seen_urls = current_urls | (self.seen_urls - removed_urls)

        return new_items

    def recent_for_ticker(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        """Get recent news items for a specific ticker."""
        return [
            item for item in reversed(self.items)
            if item.ticker == ticker
        ][:limit]

    def recent(self, limit: int = 10) -> list[NewsItem]:
        """Get the most recent news items across all tickers."""
        return list(reversed(self.items[-limit:]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "items": [i.to_dict() for i in self.items],
        }
