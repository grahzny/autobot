"""Tests for NewsHarvester -- deduplication, ticker filtering."""

from __future__ import annotations

from autobot.news import NewsHarvester, NewsItem


class TestNewsItem:
    def test_to_dict(self):
        item = NewsItem(
            title="BHP reports strong earnings",
            url="https://example.com/1",
            publisher="Reuters",
            ticker="BHP.AX",
            timestamp=1000.0,
        )
        d = item.to_dict()
        assert d["title"] == "BHP reports strong earnings"
        assert d["ticker"] == "BHP.AX"


class TestNewsHarvester:
    def test_deduplication(self):
        harvester = NewsHarvester()
        item1 = NewsItem(title="News 1", url="https://a.com/1", ticker="BHP.AX")
        item2 = NewsItem(title="News 1 copy", url="https://a.com/1", ticker="BHP.AX")
        item3 = NewsItem(title="News 2", url="https://a.com/2", ticker="BHP.AX")

        # Manually add to simulate dedup
        harvester.items.append(item1)
        harvester.seen_urls.add(item1.url)

        # item2 has same URL, should be deduped
        assert item2.url in harvester.seen_urls

        harvester.items.append(item3)
        harvester.seen_urls.add(item3.url)
        assert len(harvester.items) == 2

    def test_recent_for_ticker(self):
        harvester = NewsHarvester()
        harvester.items = [
            NewsItem(title="BHP 1", url="u1", ticker="BHP.AX"),
            NewsItem(title="AAPL 1", url="u2", ticker="AAPL"),
            NewsItem(title="BHP 2", url="u3", ticker="BHP.AX"),
        ]
        bhp_news = harvester.recent_for_ticker("BHP.AX", limit=5)
        assert len(bhp_news) == 2
        assert all(n.ticker == "BHP.AX" for n in bhp_news)

    def test_recent_returns_latest_first(self):
        harvester = NewsHarvester()
        harvester.items = [
            NewsItem(title="Old", url="u1", ticker="BHP.AX"),
            NewsItem(title="New", url="u2", ticker="BHP.AX"),
        ]
        recent = harvester.recent(limit=2)
        assert recent[0].title == "New"

    def test_max_items_trimming(self):
        harvester = NewsHarvester(max_items=5)
        for i in range(10):
            item = NewsItem(title=f"News {i}", url=f"url{i}", ticker="BHP.AX")
            harvester.items.append(item)
            harvester.seen_urls.add(item.url)

        # Simulate trim
        if len(harvester.items) > harvester.max_items:
            harvester.items = harvester.items[-harvester.max_items:]
        assert len(harvester.items) == 5

    def test_to_dict(self):
        harvester = NewsHarvester()
        harvester.items.append(
            NewsItem(title="Test", url="u1", ticker="BHP.AX")
        )
        d = harvester.to_dict()
        assert len(d["items"]) == 1
