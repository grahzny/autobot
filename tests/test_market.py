"""Tests for MarketMonitor -- market hours, session labels, alerts."""

from __future__ import annotations

from datetime import datetime

from autobot.market import MarketMonitor, PriceSnapshot, WatchlistEntry

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


def _melb(year, month, day, hour, minute=0) -> datetime:
    """Create a Melbourne-timezone datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("Australia/Melbourne"))


class TestMarketHoursASX:
    def test_asx_open_midday(self):
        mon = MarketMonitor(timezone="Australia/Melbourne")
        # Wednesday 12:00 AEST
        dt = _melb(2026, 2, 11, 12, 0)  # Wednesday
        assert mon.is_asx_open(dt) is True

    def test_asx_open_at_10(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 11, 10, 0)
        assert mon.is_asx_open(dt) is True

    def test_asx_closed_at_16(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 11, 16, 0)
        assert mon.is_asx_open(dt) is False

    def test_asx_closed_before_10(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 11, 9, 30)
        assert mon.is_asx_open(dt) is False

    def test_asx_closed_saturday(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 14, 12, 0)  # Saturday
        assert mon.is_asx_open(dt) is False

    def test_asx_closed_sunday(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 15, 12, 0)  # Sunday
        assert mon.is_asx_open(dt) is False


class TestMarketHoursUS:
    def test_us_open_during_aest_night(self):
        mon = MarketMonitor()
        # US market 09:30 ET = ~01:30 AEDT (Feb, DST active in Australia)
        # Wednesday 02:00 AEDT -> should be ~10:00 ET (roughly)
        # Let's use a known good time: Thu 03:00 AEDT = Wed 11:00 ET
        dt = _melb(2026, 2, 12, 3, 0)  # Thursday 3am AEDT
        assert mon.is_us_open(dt) is True

    def test_us_closed_during_aest_afternoon(self):
        mon = MarketMonitor()
        # 14:00 AEST = ~22:00 ET previous day (market closed)
        dt = _melb(2026, 2, 11, 14, 0)  # Wednesday 2pm AEST
        assert mon.is_us_open(dt) is False

    def test_us_closed_on_weekend(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 15, 3, 0)  # Sunday 3am AEDT
        assert mon.is_us_open(dt) is False


class TestSunday:
    def test_sunday_detected(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 15, 12, 0)  # Sunday
        assert mon.is_sunday(dt) is True

    def test_monday_not_sunday(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 16, 12, 0)  # Monday
        assert mon.is_sunday(dt) is False

    def test_saturday_not_sunday(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 14, 12, 0)  # Saturday
        assert mon.is_sunday(dt) is False


class TestSessionLabel:
    def test_sunday_label(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 15, 12, 0)
        label = mon.market_session_label(dt)
        assert "Sunday" in label
        assert "synthesis" in label

    def test_asx_open_label(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 11, 12, 0)  # Wednesday midday
        label = mon.market_session_label(dt)
        assert "ASX open" in label

    def test_weekend_label(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 14, 12, 0)  # Saturday
        label = mon.market_session_label(dt)
        assert "Weekend" in label or "closed" in label

    def test_weekday_closed_label(self):
        mon = MarketMonitor()
        dt = _melb(2026, 2, 11, 18, 0)  # Wednesday 6pm AEST, both markets closed
        label = mon.market_session_label(dt)
        assert "closed" in label.lower()


class TestWatchlist:
    def test_add_asx_ticker(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("BHP.AX")
        assert entry.ticker == "BHP.AX"
        assert entry.exchange == "ASX"
        assert "BHP.AX" in mon.watchlist

    def test_add_us_ticker(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("AAPL")
        assert entry.ticker == "AAPL"
        assert entry.exchange == "US"

    def test_remove_ticker(self):
        mon = MarketMonitor()
        mon.add_ticker("AAPL")
        mon.remove_ticker("AAPL")
        assert "AAPL" not in mon.watchlist

    def test_add_duplicate_returns_existing(self):
        mon = MarketMonitor()
        e1 = mon.add_ticker("AAPL")
        e1.name = "Apple"
        e2 = mon.add_ticker("AAPL")
        assert e2.name == "Apple"  # same object


class TestAlerts:
    def test_alert_above_triggers(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("BHP.AX")
        entry.last_price = 50.0
        mon.set_alert("BHP.AX", above=45.0)
        alerts = mon.check_alerts()
        assert len(alerts) == 1
        assert alerts[0]["subtype"] == "above_threshold"
        # One-shot: alert cleared
        assert entry.alert_above is None

    def test_alert_below_triggers(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("AAPL")
        entry.last_price = 140.0
        mon.set_alert("AAPL", below=150.0)
        alerts = mon.check_alerts()
        assert len(alerts) == 1
        assert alerts[0]["subtype"] == "below_threshold"

    def test_no_alert_when_within_range(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("AAPL")
        entry.last_price = 150.0
        mon.set_alert("AAPL", above=160.0, below=140.0)
        alerts = mon.check_alerts()
        assert len(alerts) == 0

    def test_alert_not_triggered_without_price(self):
        mon = MarketMonitor()
        mon.add_ticker("AAPL")
        mon.set_alert("AAPL", above=100.0)
        alerts = mon.check_alerts()
        assert len(alerts) == 0  # no price yet


class TestGroundTruth:
    def test_ground_truth_includes_session(self):
        mon = MarketMonitor()
        text = mon.ground_truth_text()
        assert "Market session:" in text

    def test_ground_truth_includes_watchlist_prices(self):
        mon = MarketMonitor()
        entry = mon.add_ticker("BHP.AX")
        entry.last_price = 42.50
        text = mon.ground_truth_text()
        assert "BHP.AX" in text
        assert "42.50" in text


class TestRateLimiting:
    def test_should_fetch_prices_initially(self):
        mon = MarketMonitor()
        assert mon.should_fetch_prices() is True

    def test_should_not_fetch_prices_too_soon(self):
        import time
        mon = MarketMonitor()
        mon.last_price_fetch = time.time()
        assert mon.should_fetch_prices() is False


class TestSerialization:
    def test_to_dict(self):
        mon = MarketMonitor()
        mon.add_ticker("BHP.AX")
        d = mon.to_dict()
        assert d["timezone"] == "Australia/Melbourne"
        assert "BHP.AX" in d["watchlist"]
