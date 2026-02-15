"""Tests for the reward prediction error system."""

from __future__ import annotations

import time

from autobot.prediction import PendingPrediction, PredictionEngine


class TestPredictionEngine:
    def test_register_prediction(self):
        pe = PredictionEngine()
        pe.register_prediction("response", 0.6, target_person_id="p1")
        assert len(pe.pending) == 1
        assert pe.pending[0].action_type == "response"
        assert pe.pending[0].expected_satisfaction == 0.6

    def test_evaluate_response_positive_pe(self):
        pe = PredictionEngine()
        pe.register_prediction("response", 0.3, target_person_id="p1")
        # Warm reply = actual > expected
        error = pe.evaluate_response_outcome("p1", got_reply=True, reply_warmth=0.5)
        assert error is not None
        assert error > 0  # better than expected
        assert len(pe.pending) == 0

    def test_evaluate_response_negative_pe(self):
        pe = PredictionEngine()
        pe.register_prediction("response", 0.8, target_person_id="p1")
        # Cold reply = actual < expected
        error = pe.evaluate_response_outcome("p1", got_reply=True, reply_warmth=0.0)
        assert error is not None
        assert error < 0  # worse than expected

    def test_evaluate_no_matching_prediction(self):
        pe = PredictionEngine()
        error = pe.evaluate_response_outcome("p1", got_reply=True)
        assert error is None

    def test_evaluate_stale_predictions(self):
        pe = PredictionEngine()
        # Create an already-expired prediction
        pred = PendingPrediction(
            action_type="proactive",
            expected_satisfaction=0.6,
            created_at=time.time() - 200,  # expired (> 120s)
            target_person_id="p1",
        )
        pe.pending.append(pred)
        errors = pe.evaluate_stale_predictions()
        assert len(errors) == 1
        assert errors[0] < 0  # disappointment
        assert len(pe.pending) == 0

    def test_average_recent_error(self):
        pe = PredictionEngine()
        pe.recent_errors = [0.1, -0.2, 0.3]
        avg = pe.average_recent_error()
        assert abs(avg - 0.067) < 0.01

    def test_average_recent_error_empty(self):
        pe = PredictionEngine()
        assert pe.average_recent_error() == 0.0

    def test_expected_satisfaction_adapts(self):
        pe = PredictionEngine()
        base = pe.expected_satisfaction_for("response")
        # After recording some outcomes, expectations should shift
        pe._running_satisfaction = 0.8
        higher = pe.expected_satisfaction_for("response")
        assert higher > base

    def test_proactive_expectation_lower_than_response(self):
        pe = PredictionEngine()
        resp = pe.expected_satisfaction_for("response")
        proactive = pe.expected_satisfaction_for("proactive")
        assert proactive < resp

    def test_max_history_trimming(self):
        pe = PredictionEngine(max_history=5)
        for i in range(10):
            pe.recent_errors.append(float(i) * 0.1)
        pe._record_error(0.0, 0.5)
        assert len(pe.recent_errors) <= 6  # 5 max + 1 just added, then trimmed


class TestPendingPrediction:
    def test_not_expired_when_fresh(self):
        pred = PendingPrediction(
            action_type="response",
            expected_satisfaction=0.5,
            created_at=time.time(),
        )
        assert not pred.is_expired()

    def test_expired_after_timeout(self):
        pred = PendingPrediction(
            action_type="response",
            expected_satisfaction=0.5,
            created_at=time.time() - 200,
        )
        assert pred.is_expired()
