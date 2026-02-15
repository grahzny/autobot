"""Reward Prediction Error -- lightweight learning signal.

Tracks expected vs actual satisfaction for the entity's actions.
Prediction error (PE) = actual - expected:
  Positive PE = better than expected (reinforcing)
  Negative PE = worse than expected (discouraging)

PE feeds into emotion drift, goal reprioritization, and memory salience.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field


@dataclass
class PendingPrediction:
    """An action whose outcome hasn't been evaluated yet."""

    action_type: str            # "response", "proactive", "reflection"
    expected_satisfaction: float  # 0-1, what the entity expected
    created_at: float
    target_person_id: str | None = None
    goal_id: str | None = None
    context: str = ""           # brief context for evaluation

    def is_expired(self, timeout_seconds: float = 120) -> bool:
        """Has this prediction gone unanswered long enough to count as failure?"""
        return (_time.time() - self.created_at) > timeout_seconds


@dataclass
class PredictionEngine:
    """Tracks expectations and computes prediction errors."""

    pending: list[PendingPrediction] = field(default_factory=list)
    recent_errors: list[float] = field(default_factory=list)  # last N PEs
    max_history: int = 20

    # Running average for adaptive expectations
    _running_satisfaction: float = 0.5

    def register_prediction(
        self,
        action_type: str,
        expected_satisfaction: float,
        target_person_id: str | None = None,
        goal_id: str | None = None,
        context: str = "",
    ) -> None:
        """Register an expected outcome for a pending action."""
        self.pending.append(PendingPrediction(
            action_type=action_type,
            expected_satisfaction=expected_satisfaction,
            created_at=_time.time(),
            target_person_id=target_person_id,
            goal_id=goal_id,
            context=context,
        ))

    def evaluate_response_outcome(
        self,
        person_id: str,
        got_reply: bool,
        reply_warmth: float = 0.0,
    ) -> float | None:
        """
        Evaluate the outcome of a response/proactive action.

        Returns prediction_error or None if no matching prediction.
        prediction_error = actual_satisfaction - expected_satisfaction
        Positive = better than expected, Negative = worse than expected.
        """
        # Find matching prediction
        match = None
        for pred in self.pending:
            if pred.target_person_id == person_id and not pred.is_expired():
                match = pred
                break

        if not match:
            return None

        self.pending.remove(match)

        # Compute actual satisfaction
        if match.action_type == "proactive":
            if not got_reply:
                actual = 0.1  # silence after outreach = very low
            else:
                actual = 0.4 + reply_warmth * 0.5  # reply + warmth bonus
        elif match.action_type == "response":
            actual = 0.3 + reply_warmth * 0.4  # base + warmth
            if got_reply:
                actual += 0.2  # conversation continued
        else:
            actual = 0.5  # neutral for other action types

        actual = max(0.0, min(1.0, actual))
        pe = round(actual - match.expected_satisfaction, 3)
        self._record_error(pe, actual)

        return pe

    def evaluate_stale_predictions(self) -> list[float]:
        """Check for expired predictions (no response received) and score them."""
        errors: list[float] = []
        expired = [p for p in self.pending if p.is_expired()]
        for pred in expired:
            self.pending.remove(pred)
            # Expired = no response = disappointment
            actual = 0.1
            pe = round(actual - pred.expected_satisfaction, 3)
            self._record_error(pe, actual)
            errors.append(pe)
        return errors

    def _record_error(self, pe: float, actual: float) -> None:
        """Record a PE value and update running average."""
        self.recent_errors.append(pe)
        if len(self.recent_errors) > self.max_history:
            self.recent_errors = self.recent_errors[-self.max_history:]
        # Update running average
        self._running_satisfaction = round(
            0.9 * self._running_satisfaction + 0.1 * actual, 3,
        )

    def average_recent_error(self) -> float:
        """Average PE over recent history. Negative = chronic disappointment."""
        if not self.recent_errors:
            return 0.0
        return round(sum(self.recent_errors) / len(self.recent_errors), 3)

    def expected_satisfaction_for(
        self,
        action_type: str,
        person_id: str | None = None,
    ) -> float:
        """Adaptive expectation based on recent history."""
        base = self._running_satisfaction
        if action_type == "proactive":
            base *= 0.7  # proactive is inherently riskier
        return round(max(0.0, min(1.0, base)), 2)


# ======================================================================
# Trade Predictions -- market-specific prediction tracking
# ======================================================================


@dataclass
class TradePrediction:
    """A prediction about a trade outcome."""

    id: str = ""
    ticker: str = ""
    direction: str = "long"     # "long" or "short"
    entry_price: float = 0.0
    target_price: float = 0.0
    conviction: float = 0.5
    timeframe: str = ""         # "days", "weeks", "months"
    created_at: float = 0.0
    resolved: bool = False
    actual_exit: float | None = None
    actual_pnl: float | None = None
    prediction_error: float | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _time.time()


@dataclass
class TradePredictionTracker:
    """Tracks trade predictions and calculates accuracy."""

    predictions: list[TradePrediction] = field(default_factory=list)
    max_history: int = 50

    def register_trade_prediction(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        target_price: float,
        conviction: float = 0.5,
        timeframe: str = "",
    ) -> TradePrediction:
        """Register a new trade prediction."""
        pred = TradePrediction(
            id=f"{ticker}_{int(_time.time())}",
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            target_price=target_price,
            conviction=conviction,
            timeframe=timeframe,
        )
        self.predictions.append(pred)
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history:]
        return pred

    def resolve_trade_prediction(
        self,
        ticker: str,
        actual_exit: float,
        actual_pnl: float,
    ) -> float | None:
        """Resolve a prediction and calculate error.

        Returns the prediction error (0 = perfect, higher = worse).
        """
        # Find most recent unresolved prediction for this ticker
        pred = None
        for p in reversed(self.predictions):
            if p.ticker == ticker and not p.resolved:
                pred = p
                break

        if pred is None:
            return None

        pred.resolved = True
        pred.actual_exit = actual_exit
        pred.actual_pnl = actual_pnl

        # Prediction error: how far off was the target?
        if pred.entry_price > 0:
            expected_return = (pred.target_price - pred.entry_price) / pred.entry_price
            actual_return = (actual_exit - pred.entry_price) / pred.entry_price
            pred.prediction_error = abs(expected_return - actual_return)
        else:
            pred.prediction_error = 0.0

        return pred.prediction_error

    def direction_accuracy(self) -> float | None:
        """What fraction of resolved predictions got the direction right?"""
        resolved = [p for p in self.predictions if p.resolved and p.actual_pnl is not None]
        if not resolved:
            return None
        correct = sum(
            1 for p in resolved
            if (p.direction == "long" and p.actual_pnl > 0)
            or (p.direction == "short" and p.actual_pnl < 0)
        )
        return round(correct / len(resolved), 4)

    def average_prediction_error(self) -> float | None:
        """Average prediction error across resolved trades."""
        resolved = [p for p in self.predictions if p.prediction_error is not None]
        if not resolved:
            return None
        return round(
            sum(p.prediction_error for p in resolved) / len(resolved), 4  # type: ignore[misc]
        )
