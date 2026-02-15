"""Tests for MarketState -- ground truth rendering, session tracking, Chris management."""

from __future__ import annotations

import time
from unittest.mock import patch

from autobot.chat_state import MarketState, PersonProfile


class TestMarketStateBasics:
    def test_default_state(self):
        ms = MarketState()
        assert ms.tick_count == 0
        assert ms.name == "Ryn"
        assert ms.energy == 1.0
        assert ms.chris is None

    def test_get_or_create_person_creates_chris(self):
        ms = MarketState()
        person = ms.get_or_create_person("chris", "Chris")
        assert person.name == "Chris"
        assert person.id == "chris"
        assert ms.chris is person

    def test_second_create_returns_same(self):
        ms = MarketState()
        p1 = ms.get_or_create_person("chris", "Chris")
        p2 = ms.get_or_create_person("chris", "Chris")
        assert p1 is p2

    def test_add_message(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        msg = ms.add_message("chris", "Hello!")
        assert msg.text == "Hello!"
        assert msg.person_id == "chris"
        assert len(ms.pending_messages) == 1
        assert len(ms.conversations["chris"]) == 1

    def test_add_entity_message(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        ms.add_entity_message("chris", "Acknowledged.")
        assert len(ms.conversations["chris"]) == 1
        assert ms.conversations["chris"][0]["role"] == "entity"

    def test_unprocessed_messages(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        ms.add_message("chris", "Hello!")
        ms.add_message("chris", "How are you?")
        assert len(ms.unprocessed_messages()) == 2
        ms.pending_messages[0].processed = True
        assert len(ms.unprocessed_messages()) == 1


class TestMarketStatePeopleCompat:
    def test_people_property_empty(self):
        ms = MarketState()
        assert ms.people == {}

    def test_people_property_with_chris(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        assert "chris" in ms.people
        assert ms.people["chris"].name == "Chris"

    def test_get_person_returns_chris(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        assert ms.get_person("chris") is ms.chris

    def test_get_person_returns_none_for_unknown(self):
        ms = MarketState()
        assert ms.get_person("alice") is None


class TestMarketStateGroundTruth:
    def test_basic_ground_truth(self):
        ms = MarketState()
        text = ms.ground_truth_text()
        assert "GROUND TRUTH" in text
        assert "Tick count:" in text

    def test_ground_truth_with_market_text(self):
        ms = MarketState()
        text = ms.ground_truth_text(market_text="Market session: ASX open")
        assert "ASX open" in text

    def test_ground_truth_with_portfolio_text(self):
        ms = MarketState()
        text = ms.ground_truth_text(portfolio_text="Cash: $1000.00")
        assert "Cash: $1000.00" in text

    def test_ground_truth_with_capital_text(self):
        ms = MarketState()
        text = ms.ground_truth_text(capital_text="Operating capital: $950.00")
        assert "$950.00" in text

    def test_ground_truth_with_chris(self):
        ms = MarketState()
        ms.get_or_create_person("chris", "Chris")
        text = ms.ground_truth_text()
        assert "Chris" in text


class TestMarketStateRecordThinking:
    def test_record_thinking(self):
        ms = MarketState()
        ms.record_thinking("Analyzing BHP.AX", "analysis")
        assert len(ms.thinking_log) == 1
        assert ms.thinking_log[0]["thought"] == "Analyzing BHP.AX"
        assert ms.thinking_log[0]["category"] == "analysis"

    def test_thinking_log_capped(self):
        ms = MarketState()
        for i in range(250):
            ms.record_thinking(f"Thought {i}")
        assert len(ms.thinking_log) == 200

    def test_record_event(self):
        ms = MarketState()
        ms.record({"type": "trade_profit", "pnl": 10.0})
        assert len(ms.history) == 1
        assert ms.history[0]["type"] == "trade_profit"

    def test_seconds_since_interaction(self):
        ms = MarketState()
        assert ms.seconds_since_any_interaction() == float("inf")

        ms.get_or_create_person("chris", "Chris")
        ms.chris.last_interaction_time = time.time() - 60
        secs = ms.seconds_since_any_interaction()
        assert 59 <= secs <= 62
