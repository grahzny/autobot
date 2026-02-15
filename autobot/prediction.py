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
