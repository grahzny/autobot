"""Accountant -- Token cost tracking, budget management, starvation check.

Every LLM call costs real money deducted from Operating Capital. The Accountant
tracks all spending, enforces daily budgets per role, and signals starvation
when capital runs out (entity dies at $0).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Record of a single LLM call's token usage and cost."""

    timestamp: float
    role: str               # MarketAnalyst, TradeReasoner, etc.
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    context: str = ""       # brief description of what this call was for


# Budget allocation percentages by role
_BUDGET_ALLOCATION: dict[str, float] = {
    "market_scan": 0.15,
    "analysis": 0.40,
    "trading": 0.15,
    "synthesis": 0.15,
    "chat": 0.10,
    "emergency": 0.05,
}


@dataclass
class Accountant:
    """Tracks operating capital, token costs, and budget enforcement."""

    operating_capital: float = 1000.00
    token_cost_input: float = 3.00      # $/1M input tokens
    token_cost_output: float = 15.00    # $/1M output tokens

    usage_log: list[TokenUsage] = field(default_factory=list)
    total_spent: float = 0.0

    # Daily budget tracking
    daily_budget: float = 0.0           # set by refresh_daily_budget()
    daily_spent: float = 0.0
    daily_spent_by_role: dict[str, float] = field(default_factory=dict)
    last_budget_day: int = -1           # day-of-year of last refresh

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the USD cost for a given token count."""
        input_cost = (prompt_tokens / 1_000_000) * self.token_cost_input
        output_cost = (completion_tokens / 1_000_000) * self.token_cost_output
        return round(input_cost + output_cost, 6)

    def record_usage(
        self,
        role: str,
        prompt_tokens: int,
        completion_tokens: int,
        context: str = "",
    ) -> float:
        """Record a token usage event and deduct from operating capital.

        Returns the cost in USD.
        """
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        usage = TokenUsage(
            timestamp=_time.time(),
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            context=context,
        )
        self.usage_log.append(usage)

        self.operating_capital -= cost
        self.total_spent += cost
        self.daily_spent += cost
        self.daily_spent_by_role[role] = (
            self.daily_spent_by_role.get(role, 0.0) + cost
        )

        return cost

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if we can afford an estimated number of output tokens.

        Uses a conservative estimate assuming equal input/output split.
        """
        estimated_cost = self.calculate_cost(estimated_tokens, estimated_tokens)
        return self.operating_capital >= estimated_cost

    def is_alive(self) -> bool:
        """Entity is alive as long as it has operating capital."""
        return self.operating_capital > 0

    def refresh_daily_budget(self) -> None:
        """Refresh daily budget if a new day has started.

        Daily budget = operating_capital / 30 (survive for a month).
        """
        today = _time.localtime().tm_yday
        if today != self.last_budget_day:
            self.daily_budget = max(0.0, self.operating_capital / 30.0)
            self.daily_spent = 0.0
            self.daily_spent_by_role = {}
            self.last_budget_day = today

    def role_budget(self, role: str) -> float:
        """Get the daily budget allocation for a specific role."""
        allocation = _BUDGET_ALLOCATION.get(role, 0.05)
        return self.daily_budget * allocation

    def role_remaining(self, role: str) -> float:
        """How much budget remains for a role today."""
        budget = self.role_budget(role)
        spent = self.daily_spent_by_role.get(role, 0.0)
        return max(0.0, budget - spent)

    def should_use_llm(self, role: str, estimated_output_tokens: int) -> bool:
        """Gate check: should we spend tokens on this LLM call?

        Checks: alive, can afford, within daily role budget.
        """
        if not self.is_alive():
            return False

        estimated_cost = self.calculate_cost(
            estimated_output_tokens, estimated_output_tokens
        )

        if self.operating_capital < estimated_cost:
            return False

        if self.role_remaining(role) < estimated_cost:
            return False

        return True

    def starvation_warning(self) -> str | None:
        """Return a warning level string, or None if healthy.

        - FATAL: capital <= 0
        - CRITICAL: capital <= 5
        - WARNING: capital <= 20
        - None: healthy
        """
        if self.operating_capital <= 0:
            return "FATAL"
        if self.operating_capital <= 5:
            return "CRITICAL"
        if self.operating_capital <= 20:
            return "WARNING"
        return None

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for API/display."""
        return {
            "operating_capital": round(self.operating_capital, 2),
            "total_spent": round(self.total_spent, 4),
            "daily_budget": round(self.daily_budget, 4),
            "daily_spent": round(self.daily_spent, 4),
            "is_alive": self.is_alive(),
            "starvation_warning": self.starvation_warning(),
            "calls_today": len([
                u for u in self.usage_log
                if _time.localtime(u.timestamp).tm_yday == _time.localtime().tm_yday
            ]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "operating_capital": self.operating_capital,
            "total_spent": self.total_spent,
            "token_cost_input": self.token_cost_input,
            "token_cost_output": self.token_cost_output,
            "daily_budget": self.daily_budget,
            "daily_spent": self.daily_spent,
            "daily_spent_by_role": dict(self.daily_spent_by_role),
            "last_budget_day": self.last_budget_day,
            "usage_log": [
                {
                    "timestamp": u.timestamp,
                    "role": u.role,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "cost_usd": u.cost_usd,
                    "context": u.context,
                }
                for u in self.usage_log[-100:]  # keep last 100 entries
            ],
        }
