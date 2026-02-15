"""StrategyEngine -- trading objectives and cost-aware prioritization.

Replaces GoalEngine for the market entity. Generates trading objectives
based on market conditions, portfolio state, and economic indicators.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Literal

ObjectiveType = Literal["scan", "analyze", "trade", "review", "sunday", "respond_chris"]


@dataclass
class TradingObjective:
    """A single trading objective to pursue."""

    id: str = ""
    obj_type: ObjectiveType = "scan"
    ticker: str = ""
    description: str = ""
    priority: float = 0.5          # 0-1
    conviction: float = 0.0        # 0-1
    estimated_token_cost: float = 0.0
    created_at: float = 0.0
    completed: bool = False

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.obj_type,
            "ticker": self.ticker,
            "description": self.description,
            "priority": self.priority,
            "conviction": self.conviction,
            "estimated_token_cost": self.estimated_token_cost,
            "completed": self.completed,
        }


@dataclass
class StrategyEngine:
    """Generates and prioritizes trading objectives."""

    objectives: list[TradingObjective] = field(default_factory=list)
    last_scan_tick: int = 0
    last_analysis_tick: int = 0
    last_sunday_tick: int = 0

    def auto_generate(
        self,
        has_chris_message: bool = False,
        is_sunday: bool = False,
        market_open: bool = False,
        has_positions: bool = False,
        capital_low: bool = False,
        alpha_low: bool = False,
        tick_count: int = 0,
        watchlist_tickers: list[str] | None = None,
    ) -> list[TradingObjective]:
        """Generate trading objectives based on current state.

        Returns newly created objectives.
        """
        new_objs: list[TradingObjective] = []
        existing_types = {o.obj_type for o in self.objectives if not o.completed}

        # Always respond to Chris
        if has_chris_message and "respond_chris" not in existing_types:
            obj = TradingObjective(
                id=f"chris_{tick_count}",
                obj_type="respond_chris",
                description="Respond to Chris's message",
                priority=0.95,
                estimated_token_cost=0.01,
            )
            self.objectives.append(obj)
            new_objs.append(obj)

        # Sunday protocol
        if is_sunday and "sunday" not in existing_types:
            if tick_count - self.last_sunday_tick > 40:  # ~10 min
                obj = TradingObjective(
                    id=f"sunday_{tick_count}",
                    obj_type="sunday",
                    description="Sunday strategic synthesis",
                    priority=0.7,
                    estimated_token_cost=0.02,
                )
                self.objectives.append(obj)
                new_objs.append(obj)

        # Market open + positions -> monitor
        if market_open and has_positions and "review" not in existing_types:
            obj = TradingObjective(
                id=f"review_{tick_count}",
                obj_type="review",
                description="Monitor open positions",
                priority=0.8,
                estimated_token_cost=0.005,
            )
            self.objectives.append(obj)
            new_objs.append(obj)

        # Market open + budget available -> scan
        if market_open and not capital_low:
            if "scan" not in existing_types:
                if tick_count - self.last_scan_tick > 60:  # ~15 min
                    obj = TradingObjective(
                        id=f"scan_{tick_count}",
                        obj_type="scan",
                        description="Scan market for opportunities",
                        priority=0.5,
                        estimated_token_cost=0.005,
                    )
                    self.objectives.append(obj)
                    new_objs.append(obj)

        # Low alpha -> seek research to restore meaning
        if alpha_low and not capital_low and "analyze" not in existing_types:
            tickers = watchlist_tickers or []
            if tickers:
                ticker = tickers[tick_count % len(tickers)]
                obj = TradingObjective(
                    id=f"analyze_{tick_count}",
                    obj_type="analyze",
                    ticker=ticker,
                    description=f"Deep analysis of {ticker}",
                    priority=0.6,
                    estimated_token_cost=0.01,
                )
                self.objectives.append(obj)
                new_objs.append(obj)

        return new_objs

    def prioritized(self) -> list[TradingObjective]:
        """Return uncompleted objectives sorted by priority (highest first)."""
        return sorted(
            [o for o in self.objectives if not o.completed],
            key=lambda o: o.priority,
            reverse=True,
        )

    def complete_objective(self, obj_id: str) -> None:
        """Mark an objective as completed."""
        for o in self.objectives:
            if o.id == obj_id:
                o.completed = True
                if o.obj_type == "scan":
                    self.last_scan_tick = o.created_at
                if o.obj_type == "sunday":
                    self.last_sunday_tick = o.created_at
                break

    def cleanup(self) -> None:
        """Remove old completed objectives."""
        cutoff = _time.time() - 3600  # older than 1 hour
        self.objectives = [
            o for o in self.objectives
            if not o.completed or o.created_at > cutoff
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": [o.to_dict() for o in self.objectives[-20:]],
            "last_scan_tick": self.last_scan_tick,
            "last_analysis_tick": self.last_analysis_tick,
            "last_sunday_tick": self.last_sunday_tick,
        }
