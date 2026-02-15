"""EconomicState -- Market-driven needs that replace conversational drives.

Four economic indicators (0-1 each) that decay passively and are restored
by market events. These map onto the original need categories:

- alpha (Meaning): restored by profitable trades + high-conviction research
- roi (Competence): tied to Sharpe ratio + prediction accuracy
- volatility (Anxiety): rises with market uncertainty + drawdown
- cost_pressure (Irritability): rises when token spend exceeds budget

When economic state is poor, the entity becomes more cautious and
cost-conscious in its decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EconomicState:
    """Four economic indicators, each 0-1."""

    alpha: float = 0.5           # meaning -- restored by profits + conviction
    roi: float = 0.5             # competence -- tied to Sharpe + accuracy
    volatility: float = 0.3      # anxiety -- rises with uncertainty + drawdown
    cost_pressure: float = 0.3   # irritability -- rises with high token spend

    # Backward-compat properties mapping old NeedsState names -> economic fields.
    # Removed in Phase 3 when all consumers are rewritten.
    @property
    def stimulation(self) -> float:
        return self.alpha
    @stimulation.setter
    def stimulation(self, v: float) -> None:
        self.alpha = v
    @property
    def meaning(self) -> float:
        return self.alpha
    @meaning.setter
    def meaning(self, v: float) -> None:
        self.alpha = v
    @property
    def belonging(self) -> float:
        return self.roi
    @belonging.setter
    def belonging(self, v: float) -> None:
        self.roi = v
    @property
    def competence(self) -> float:
        return self.roi
    @competence.setter
    def competence(self, v: float) -> None:
        self.roi = v
    @property
    def autonomy(self) -> float:
        return 1.0 - self.cost_pressure
    @autonomy.setter
    def autonomy(self, v: float) -> None:
        self.cost_pressure = 1.0 - v

    def clamp(self) -> None:
        for attr in ("alpha", "roi", "volatility", "cost_pressure"):
            val = max(0.0, min(1.0, getattr(self, attr)))
            setattr(self, attr, round(val, 3))

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "roi": self.roi,
            "volatility": self.volatility,
            "cost_pressure": self.cost_pressure,
        }

    def lowest_need(self) -> tuple[str, float]:
        """Return (name, value) of the most critical economic indicator.

        For alpha/roi: low is bad. For volatility/cost_pressure: high is bad.
        We normalize: for vol/cost, we invert (1-x) so low = critical.
        """
        normalized = {
            "alpha": self.alpha,
            "roi": self.roi,
            "volatility": 1.0 - self.volatility,       # high vol = bad
            "cost_pressure": 1.0 - self.cost_pressure,  # high cost = bad
        }
        name = min(normalized, key=normalized.get)  # type: ignore[arg-type]
        return name, normalized[name]

    def deficit_summary(self) -> str:
        """Human-readable summary of critical economic indicators."""
        deficits = []
        if self.alpha < 0.3:
            deficits.append(f"alpha={self.alpha:.2f}")
        if self.roi < 0.3:
            deficits.append(f"roi={self.roi:.2f}")
        if self.volatility > 0.7:
            deficits.append(f"volatility={self.volatility:.2f}")
        if self.cost_pressure > 0.7:
            deficits.append(f"cost_pressure={self.cost_pressure:.2f}")
        return ", ".join(deficits) if deficits else "no critical deficits"


# Passive decay/settle rates per tick (~15 seconds)
_DECAY_RATES = {
    "alpha": 0.003,          # meaning decays without profitable activity
    "roi": 0.002,            # competence decays slowly
    "volatility": 0.005,     # settles down (anxiety decreases)
    "cost_pressure": 0.003,  # settles down (irritability decreases)
}


def decay_economic_state(state: EconomicState) -> None:
    """Apply passive per-tick decay/settling to all indicators."""
    for attr, rate in _DECAY_RATES.items():
        current = getattr(state, attr)
        setattr(state, attr, current - rate)
    state.clamp()


def update_economic_state(
    state: EconomicState,
    events: list[dict[str, Any]],
    token_cost: float = 0.0,
    daily_budget_pct: float = 0.0,   # fraction of daily budget spent (0-1)
    drawdown_pct: float = 0.0,       # current drawdown from peak (0-1)
    sharpe: float | None = None,
    trade_pnl: float | None = None,
) -> None:
    """Update economic state based on events and market metrics.

    Args:
        state: The economic state to update.
        events: List of market events from this tick.
        token_cost: Cost of LLM calls this tick.
        daily_budget_pct: Fraction of daily budget spent so far (0-1).
        drawdown_pct: Current portfolio drawdown from peak (0-1).
        sharpe: Current Sharpe ratio (None if insufficient data).
        trade_pnl: P&L from any trade closed this tick.
    """
    for ev in events:
        etype = ev.get("type", "")

        # Alpha (meaning) restorers
        if etype == "trade_profit":
            state.alpha += 0.15
        if etype == "thesis_confirmed":
            state.alpha += 0.10
        if etype == "high_conviction_research":
            state.alpha += 0.05

        # Alpha drains
        if etype == "trade_loss":
            state.alpha -= 0.10
        if etype == "thesis_invalidated":
            state.alpha -= 0.08

        # ROI (competence) restorers
        if etype == "prediction_accurate":
            state.roi += 0.08
        if etype == "trade_profit":
            state.roi += 0.06

        # ROI drains
        if etype == "prediction_wrong":
            state.roi -= 0.06
        if etype == "trade_loss":
            state.roi -= 0.08

        # Volatility (anxiety) drivers
        if etype in ("price_alert", "news_negative"):
            state.volatility += 0.06
        if etype == "drawdown":
            state.volatility += 0.10
        if etype == "starvation_warning":
            state.volatility += 0.15

        # Volatility reducers
        if etype in ("recovery", "news_positive"):
            state.volatility -= 0.04
        if etype == "trade_profit":
            state.volatility -= 0.06

        # Cost pressure drivers
        if etype == "starvation_warning":
            state.cost_pressure += 0.12

    # Direct metric effects
    if trade_pnl is not None:
        if trade_pnl > 0:
            state.alpha += min(0.10, trade_pnl / 50.0)  # scale with P&L
        else:
            state.alpha -= min(0.10, abs(trade_pnl) / 50.0)

    if sharpe is not None:
        # Sharpe > 1 is good, < 0 is bad
        if sharpe > 1.0:
            state.roi += 0.05
        elif sharpe < 0:
            state.roi -= 0.05

    # Cost pressure from token spending
    if daily_budget_pct > 0.8:
        state.cost_pressure += 0.08
    elif daily_budget_pct > 0.5:
        state.cost_pressure += 0.03

    # Drawdown raises volatility
    if drawdown_pct > 0.1:
        state.volatility += min(0.15, drawdown_pct * 0.5)

    state.clamp()


# --- Backward compatibility aliases ---
# Other modules (emotions, living_engine) still import these during Phase 1.
# These aliases will be removed in Phase 3 when all consumers are rewritten.
NeedsState = EconomicState
decay_needs = decay_economic_state


def update_needs_from_events(
    needs: EconomicState,
    events: list[dict[str, Any]],
    goal_stall_count: int = 0,
    unanswered_proactive: int = 0,
    seconds_since_interaction: float = 0.0,
) -> None:
    """Backward-compatible shim mapping old needs events to economic state updates."""
    update_economic_state(needs, events)
