"""Tests for backward-compatible needs aliases.

The original NeedsState has been replaced by EconomicState.
These tests verify the backward-compatibility aliases still work
so existing code doesn't break during the transition.
See test_economic_state.py for comprehensive EconomicState tests.
"""

from __future__ import annotations

from autobot.needs import NeedsState, decay_needs, update_needs_from_events


class TestBackwardCompatAliases:
    def test_needsstate_is_economic_state(self):
        from autobot.needs import EconomicState
        assert NeedsState is EconomicState

    def test_needsstate_creates_instance(self):
        n = NeedsState()
        assert hasattr(n, "alpha")
        assert hasattr(n, "roi")
        assert hasattr(n, "volatility")
        assert hasattr(n, "cost_pressure")

    def test_decay_needs_works(self):
        n = NeedsState(alpha=0.5)
        decay_needs(n)
        assert n.alpha < 0.5

    def test_update_needs_from_events_works(self):
        n = NeedsState(alpha=0.5)
        events = [{"type": "trade_profit"}]
        update_needs_from_events(n, events, 0, 0, 0)
        assert n.alpha > 0.5

    def test_clamp(self):
        n = NeedsState(alpha=1.5, roi=-0.1)
        n.clamp()
        assert n.alpha == 1.0
        assert n.roi == 0.0

    def test_lowest_need(self):
        n = NeedsState(alpha=0.1, roi=0.5, volatility=0.3, cost_pressure=0.3)
        name, val = n.lowest_need()
        assert name == "alpha"

    def test_deficit_summary_no_deficits(self):
        n = NeedsState()
        assert n.deficit_summary() == "no critical deficits"

    def test_to_dict(self):
        n = NeedsState()
        d = n.to_dict()
        assert set(d.keys()) == {"alpha", "roi", "volatility", "cost_pressure"}
