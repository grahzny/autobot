"""Tests for Accountant -- token cost tracking, budget management, starvation."""

from __future__ import annotations

from autobot.accountant import Accountant


class TestCostCalculation:
    def test_zero_tokens(self):
        acc = Accountant()
        assert acc.calculate_cost(0, 0) == 0.0

    def test_input_cost(self):
        acc = Accountant(token_cost_input=3.00, token_cost_output=15.00)
        # 1M input tokens = $3.00
        cost = acc.calculate_cost(1_000_000, 0)
        assert abs(cost - 3.00) < 0.001

    def test_output_cost(self):
        acc = Accountant(token_cost_input=3.00, token_cost_output=15.00)
        # 1M output tokens = $15.00
        cost = acc.calculate_cost(0, 1_000_000)
        assert abs(cost - 15.00) < 0.001

    def test_combined_cost(self):
        acc = Accountant(token_cost_input=3.00, token_cost_output=15.00)
        # 500 input + 256 output
        cost = acc.calculate_cost(500, 256)
        expected = (500 / 1_000_000) * 3.00 + (256 / 1_000_000) * 15.00
        assert abs(cost - expected) < 0.001

    def test_typical_call_cost(self):
        acc = Accountant()
        # Typical small call: 500 input, 256 output
        cost = acc.calculate_cost(500, 256)
        assert cost < 0.01  # very cheap


class TestRecordUsage:
    def test_deducts_from_capital(self):
        acc = Accountant(operating_capital=100.00)
        cost = acc.record_usage("analysis", 1000, 500, "test call")
        assert acc.operating_capital < 100.00
        assert acc.total_spent == cost
        assert len(acc.usage_log) == 1

    def test_tracks_daily_spending(self):
        acc = Accountant(operating_capital=100.00)
        acc.record_usage("analysis", 1000, 500)
        acc.record_usage("chat", 500, 200)
        assert acc.daily_spent > 0
        assert "analysis" in acc.daily_spent_by_role
        assert "chat" in acc.daily_spent_by_role

    def test_multiple_calls_accumulate(self):
        acc = Accountant(operating_capital=100.00)
        cost1 = acc.record_usage("analysis", 1000, 500)
        cost2 = acc.record_usage("analysis", 1000, 500)
        assert abs(acc.total_spent - (cost1 + cost2)) < 0.0001
        assert len(acc.usage_log) == 2


class TestCanAfford:
    def test_can_afford_small_call(self):
        acc = Accountant(operating_capital=100.00)
        assert acc.can_afford(256) is True

    def test_cannot_afford_when_broke(self):
        acc = Accountant(operating_capital=0.0)
        assert acc.can_afford(256) is False

    def test_cannot_afford_massive_call(self):
        acc = Accountant(operating_capital=0.001)
        assert acc.can_afford(1_000_000) is False


class TestIsAlive:
    def test_alive_with_capital(self):
        acc = Accountant(operating_capital=100.00)
        assert acc.is_alive() is True

    def test_dead_at_zero(self):
        acc = Accountant(operating_capital=0.0)
        assert acc.is_alive() is False

    def test_dead_when_negative(self):
        acc = Accountant(operating_capital=-1.0)
        assert acc.is_alive() is False

    def test_alive_with_tiny_amount(self):
        acc = Accountant(operating_capital=0.01)
        assert acc.is_alive() is True


class TestStarvationWarning:
    def test_healthy(self):
        acc = Accountant(operating_capital=100.00)
        assert acc.starvation_warning() is None

    def test_warning_at_20(self):
        acc = Accountant(operating_capital=20.00)
        assert acc.starvation_warning() == "WARNING"

    def test_warning_at_15(self):
        acc = Accountant(operating_capital=15.00)
        assert acc.starvation_warning() == "WARNING"

    def test_critical_at_5(self):
        acc = Accountant(operating_capital=5.00)
        assert acc.starvation_warning() == "CRITICAL"

    def test_critical_at_3(self):
        acc = Accountant(operating_capital=3.00)
        assert acc.starvation_warning() == "CRITICAL"

    def test_fatal_at_zero(self):
        acc = Accountant(operating_capital=0.0)
        assert acc.starvation_warning() == "FATAL"

    def test_fatal_when_negative(self):
        acc = Accountant(operating_capital=-5.0)
        assert acc.starvation_warning() == "FATAL"


class TestDailyBudget:
    def test_refresh_sets_budget(self):
        acc = Accountant(operating_capital=300.00)
        acc.refresh_daily_budget()
        assert abs(acc.daily_budget - 10.00) < 0.01  # 300/30 = 10

    def test_refresh_resets_daily_spent(self):
        acc = Accountant(operating_capital=300.00)
        acc.daily_spent = 5.00
        acc.daily_spent_by_role = {"analysis": 3.0, "chat": 2.0}
        acc.last_budget_day = -1  # force refresh
        acc.refresh_daily_budget()
        assert acc.daily_spent == 0.0
        assert acc.daily_spent_by_role == {}

    def test_role_budget_allocation(self):
        acc = Accountant(operating_capital=300.00)
        acc.refresh_daily_budget()
        # analysis gets 40% of daily budget
        analysis_budget = acc.role_budget("analysis")
        assert abs(analysis_budget - 4.00) < 0.01  # 10 * 0.40 = 4

    def test_role_remaining(self):
        acc = Accountant(operating_capital=300.00)
        acc.refresh_daily_budget()
        acc.daily_spent_by_role["analysis"] = 2.00
        remaining = acc.role_remaining("analysis")
        assert abs(remaining - 2.00) < 0.01  # 4.00 - 2.00


class TestShouldUseLLM:
    def test_allows_when_healthy(self):
        acc = Accountant(operating_capital=100.00)
        acc.refresh_daily_budget()
        assert acc.should_use_llm("analysis", 256) is True

    def test_blocks_when_dead(self):
        acc = Accountant(operating_capital=0.0)
        acc.refresh_daily_budget()
        assert acc.should_use_llm("analysis", 256) is False

    def test_blocks_when_cannot_afford(self):
        acc = Accountant(operating_capital=0.0001)
        acc.refresh_daily_budget()
        assert acc.should_use_llm("analysis", 1_000_000) is False

    def test_blocks_when_role_budget_exhausted(self):
        acc = Accountant(operating_capital=100.00)
        acc.refresh_daily_budget()
        # Exhaust analysis budget
        analysis_budget = acc.role_budget("analysis")
        acc.daily_spent_by_role["analysis"] = analysis_budget + 1.0
        assert acc.should_use_llm("analysis", 256) is False


class TestSerialization:
    def test_to_dict(self):
        acc = Accountant(operating_capital=500.00)
        acc.record_usage("analysis", 1000, 500, "test")
        d = acc.to_dict()
        assert d["operating_capital"] == acc.operating_capital
        assert d["total_spent"] == acc.total_spent
        assert len(d["usage_log"]) == 1

    def test_summary(self):
        acc = Accountant(operating_capital=500.00)
        acc.refresh_daily_budget()
        s = acc.summary()
        assert s["operating_capital"] == 500.00
        assert s["is_alive"] is True
        assert s["starvation_warning"] is None
