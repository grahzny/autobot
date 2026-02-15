"""Tests for brain.py market roles -- LLM mocked, cost tracking, budget gating."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autobot.accountant import Accountant
from autobot.brain import (
    MarketAnalysis,
    TradeDecision,
    SynthesisResult,
    BrainResponse,
    analyze_market_data,
    reason_trade,
    strategic_synthesis,
    respond_to_chris,
    _parse_json,
    _llm_call,
    is_llm_available,
)


class TestParseJson:
    """_parse_json must robustly extract JSON from LLM output."""

    def test_plain_json(self):
        result = _parse_json('{"action": "buy", "shares": 10}')
        assert result is not None
        assert result["action"] == "buy"

    def test_json_in_markdown(self):
        raw = '```json\n{"thesis": "bullish"}\n```'
        result = _parse_json(raw)
        assert result is not None
        assert result["thesis"] == "bullish"

    def test_json_with_surrounding_text(self):
        raw = 'Here is my analysis: {"action": "hold"} based on data.'
        result = _parse_json(raw)
        assert result is not None
        assert result["action"] == "hold"

    def test_invalid_json_returns_none(self):
        result = _parse_json("this is not json at all")
        assert result is None

    def test_nested_json(self):
        raw = '{"outer": {"inner": 42}}'
        result = _parse_json(raw)
        assert result is not None
        assert result["outer"]["inner"] == 42


class TestLLMCallCostTracking:
    """_llm_call must track costs via Accountant when provided."""

    @patch("autobot.brain._get_client")
    @patch("autobot.brain._get_model", return_value="test-model")
    def test_records_usage(self, mock_model, mock_client):
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_response.usage = mock_usage

        client = MagicMock()
        client.chat.completions.create.return_value = mock_response
        mock_client.return_value = client

        accountant = Accountant(operating_capital=100.0)
        text, usage = _llm_call(
            system="test", user="test",
            accountant=accountant, role="analysis",
        )

        assert text == "test response"
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert len(accountant.usage_log) == 1
        assert accountant.usage_log[0].role == "analysis"

    @patch("autobot.brain._get_client")
    @patch("autobot.brain._get_model", return_value="test-model")
    def test_deducts_from_capital(self, mock_model, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500
        mock_response.usage = mock_usage

        client = MagicMock()
        client.chat.completions.create.return_value = mock_response
        mock_client.return_value = client

        accountant = Accountant(operating_capital=100.0)
        initial_cap = accountant.operating_capital
        _llm_call(
            system="test", user="test",
            accountant=accountant, role="trading",
        )
        assert accountant.operating_capital < initial_cap

    def test_llm_failure_returns_none(self):
        """When LLM fails, should return None without crashing."""
        with patch("autobot.brain._get_client", side_effect=Exception("no server")):
            text, usage = _llm_call("test", "test")
            assert text is None
            assert usage["prompt_tokens"] == 0


class TestAnalyzeMarketData:
    """analyze_market_data must return structured MarketAnalysis."""

    @patch("autobot.brain._llm_call")
    def test_returns_analysis(self, mock_call):
        mock_call.return_value = (
            '{"thesis": "bullish on BHP", "conviction": 0.7, '
            '"signals": ["strong earnings"], "risks": ["commodity downturn"], '
            '"action": "buy"}',
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        result = analyze_market_data(
            ticker="BHP.AX",
            price_data="Current: $45.00",
            fundamentals="PE: 12.5",
        )

        assert isinstance(result, MarketAnalysis)
        assert result.thesis == "bullish on BHP"
        assert result.conviction == 0.7
        assert result.action == "buy"
        assert result.used_llm is True

    @patch("autobot.brain._llm_call")
    def test_llm_failure_returns_default(self, mock_call):
        mock_call.return_value = (None, {"prompt_tokens": 0, "completion_tokens": 0})
        result = analyze_market_data("BHP.AX", "", "")
        assert result.used_llm is False
        assert result.action == "watch"


class TestReasonTrade:
    """reason_trade must return structured TradeDecision."""

    @patch("autobot.brain._llm_call")
    def test_returns_decision(self, mock_call):
        mock_call.return_value = (
            '{"action": "buy", "rationale": "Strong thesis", '
            '"shares": 10, "stop_loss": 40.0, "target": 50.0, '
            '"confidence": 0.8}',
            {"prompt_tokens": 200, "completion_tokens": 80},
        )

        result = reason_trade(
            ticker="BHP.AX",
            thesis="Bullish on iron ore",
            portfolio_text="Cash: $1000",
            market_context="ASX open",
        )

        assert isinstance(result, TradeDecision)
        assert result.action == "buy"
        assert result.shares == 10
        assert result.stop_loss == 40.0
        assert result.confidence == 0.8
        assert result.used_llm is True


class TestStrategicSynthesis:
    """strategic_synthesis must return structured SynthesisResult."""

    @patch("autobot.brain._llm_call")
    def test_returns_synthesis(self, mock_call):
        mock_call.return_value = (
            '{"summary": "Good week overall", '
            '"watchlist_changes": ["add FMG.AX"], '
            '"research_notes": ["Iron ore demand strong"], '
            '"adjustments": ["Increase position limits"], '
            '"conviction_updates": {"BHP.AX": 0.8}}',
            {"prompt_tokens": 500, "completion_tokens": 200},
        )

        result = strategic_synthesis(
            portfolio_text="Cash: $900",
            research_text="BHP thesis: bullish",
            performance_text="Win rate: 60%",
            lessons_text="Cut losses faster",
        )

        assert isinstance(result, SynthesisResult)
        assert result.summary == "Good week overall"
        assert "add FMG.AX" in result.watchlist_changes
        assert result.used_llm is True


class TestRespondToChris:
    """respond_to_chris must handle conversation with Chris."""

    @patch("autobot.brain._llm_call")
    def test_returns_response(self, mock_call):
        mock_call.return_value = (
            '{"thinking": "Chris is asking about portfolio", '
            '"response": "We are up 5% this week.", '
            '"topics_of_interest": ["performance"]}',
            {"prompt_tokens": 300, "completion_tokens": 80},
        )

        result = respond_to_chris(
            entity_name="Ryn",
            message="How are we doing?",
            emotions_text="Mood: optimistic",
            portfolio_text="Total: $1050",
            market_text="Markets open",
            research_text="No theses",
            conversation_text="",
        )

        assert isinstance(result, BrainResponse)
        assert result.response == "We are up 5% this week."
        assert result.used_llm is True

    @patch("autobot.brain._llm_call")
    def test_llm_failure_fallback(self, mock_call):
        mock_call.return_value = (None, {"prompt_tokens": 0, "completion_tokens": 0})
        result = respond_to_chris(
            entity_name="Ryn", message="Hello",
            emotions_text="", portfolio_text="",
            market_text="", research_text="",
            conversation_text="",
        )
        assert result.response is not None
        assert result.used_llm is False


class TestBackwardCompat:
    """Backward compat stubs must not crash."""

    def test_generate_response_returns_brain_response(self):
        from autobot.brain import generate_response
        result = generate_response()
        assert isinstance(result, BrainResponse)
        assert result.used_llm is False

    def test_generate_proactive_returns_none(self):
        from autobot.brain import generate_proactive
        result = generate_proactive()
        assert result is None

    def test_generate_reflection_returns_none(self):
        from autobot.brain import generate_reflection
        result = generate_reflection()
        assert result is None

    def test_generate_idle_thought_returns_none(self):
        from autobot.brain import generate_idle_thought
        result = generate_idle_thought()
        assert result is None
