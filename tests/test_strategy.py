"""Tests for StrategyEngine -- objective generation, cost-aware prioritization."""

from __future__ import annotations

from autobot.strategy import StrategyEngine, TradingObjective


class TestObjectiveGeneration:
    def test_chris_message_generates_respond_objective(self):
        engine = StrategyEngine()
        objs = engine.auto_generate(has_chris_message=True, tick_count=1)
        assert any(o.obj_type == "respond_chris" for o in objs)

    def test_chris_message_highest_priority(self):
        engine = StrategyEngine()
        engine.auto_generate(has_chris_message=True, tick_count=1)
        top = engine.prioritized()[0]
        assert top.obj_type == "respond_chris"
        assert top.priority >= 0.9

    def test_sunday_generates_synthesis(self):
        engine = StrategyEngine()
        engine.last_sunday_tick = 0
        objs = engine.auto_generate(is_sunday=True, tick_count=50)
        assert any(o.obj_type == "sunday" for o in objs)

    def test_sunday_cooldown(self):
        engine = StrategyEngine()
        engine.last_sunday_tick = 45  # only 5 ticks ago
        objs = engine.auto_generate(is_sunday=True, tick_count=50)
        assert not any(o.obj_type == "sunday" for o in objs)

    def test_market_open_positions_generates_review(self):
        engine = StrategyEngine()
        objs = engine.auto_generate(
            market_open=True, has_positions=True, tick_count=1,
        )
        assert any(o.obj_type == "review" for o in objs)

    def test_market_open_generates_scan(self):
        engine = StrategyEngine()
        engine.last_scan_tick = 0
        objs = engine.auto_generate(market_open=True, tick_count=100)
        assert any(o.obj_type == "scan" for o in objs)

    def test_scan_cooldown(self):
        engine = StrategyEngine()
        engine.last_scan_tick = 95  # only 5 ticks ago
        objs = engine.auto_generate(market_open=True, tick_count=100)
        assert not any(o.obj_type == "scan" for o in objs)

    def test_low_alpha_generates_analyze(self):
        engine = StrategyEngine()
        objs = engine.auto_generate(
            alpha_low=True, tick_count=1,
            watchlist_tickers=["BHP.AX", "CBA.AX"],
        )
        assert any(o.obj_type == "analyze" for o in objs)

    def test_capital_low_skips_scan(self):
        engine = StrategyEngine()
        objs = engine.auto_generate(
            market_open=True, capital_low=True, tick_count=100,
        )
        assert not any(o.obj_type == "scan" for o in objs)

    def test_capital_low_skips_analyze(self):
        engine = StrategyEngine()
        objs = engine.auto_generate(
            alpha_low=True, capital_low=True, tick_count=1,
            watchlist_tickers=["BHP.AX"],
        )
        assert not any(o.obj_type == "analyze" for o in objs)

    def test_no_duplicate_objective_types(self):
        engine = StrategyEngine()
        engine.auto_generate(has_chris_message=True, tick_count=1)
        engine.auto_generate(has_chris_message=True, tick_count=2)
        respond_objs = [
            o for o in engine.objectives
            if o.obj_type == "respond_chris" and not o.completed
        ]
        assert len(respond_objs) == 1


class TestObjectivePrioritization:
    def test_prioritized_returns_highest_first(self):
        engine = StrategyEngine()
        engine.objectives = [
            TradingObjective(id="low", priority=0.3),
            TradingObjective(id="high", priority=0.9),
            TradingObjective(id="mid", priority=0.5),
        ]
        ordered = engine.prioritized()
        assert ordered[0].id == "high"
        assert ordered[-1].id == "low"

    def test_completed_excluded_from_prioritized(self):
        engine = StrategyEngine()
        engine.objectives = [
            TradingObjective(id="done", priority=0.9, completed=True),
            TradingObjective(id="active", priority=0.5),
        ]
        ordered = engine.prioritized()
        assert len(ordered) == 1
        assert ordered[0].id == "active"


class TestObjectiveCompletion:
    def test_complete_objective(self):
        engine = StrategyEngine()
        engine.objectives.append(
            TradingObjective(id="test_1", obj_type="scan", priority=0.5)
        )
        engine.complete_objective("test_1")
        assert engine.objectives[0].completed

    def test_cleanup_removes_old_completed(self):
        import time
        engine = StrategyEngine()
        # Old completed objective
        old = TradingObjective(
            id="old", priority=0.5, completed=True,
            created_at=time.time() - 7200,  # 2 hours ago
        )
        # Recent active objective
        active = TradingObjective(
            id="active", priority=0.5,
            created_at=time.time(),
        )
        engine.objectives = [old, active]
        engine.cleanup()
        assert len(engine.objectives) == 1
        assert engine.objectives[0].id == "active"


class TestObjectiveToDict:
    def test_to_dict_fields(self):
        obj = TradingObjective(
            id="test", obj_type="analyze", ticker="BHP.AX",
            description="Test", priority=0.7, conviction=0.8,
        )
        d = obj.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "analyze"
        assert d["ticker"] == "BHP.AX"
        assert d["priority"] == 0.7
