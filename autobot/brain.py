"""Brain -- the entity's LLM-powered thinking.

Four market roles:
  1. analyze_market_data  -- Ticker fundamentals/price analysis
  2. reason_trade         -- Trade decision reasoning
  3. strategic_synthesis  -- Sunday/periodic review
  4. respond_to_chris     -- Chat with Chris (Secondary Intelligence Source)

Plus retained:
  - analyze_message       -- Chris message interpretation (simplified)
  - _heuristic_analysis   -- Fallback when LLM unavailable

Uses an OpenAI-compatible API. Every call tracks token usage via Accountant.

Configuration via environment variables:
  AUTOBOT_LLM_URL   -- API base URL (default: http://localhost:1234/v1)
  AUTOBOT_MODEL      -- model name (default: use whatever is loaded)
  AUTOBOT_API_KEY    -- API key if required (default: lm-studio)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_LLM_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


def _get_llm_config() -> tuple[str, str, str]:
    base_url = os.environ.get("AUTOBOT_LLM_URL", DEFAULT_LLM_URL)
    api_key = os.environ.get("AUTOBOT_API_KEY", DEFAULT_API_KEY)
    model = os.environ.get("AUTOBOT_MODEL", "")
    return base_url, api_key, model


def _get_client():
    from openai import OpenAI
    base_url, api_key, _ = _get_llm_config()
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_model() -> str:
    _, _, model = _get_llm_config()
    return model or "local-model"


def _llm_call(
    system: str,
    user: str,
    max_tokens: int = 1024,
    accountant: Any = None,
    role: str = "general",
    context: str = "",
) -> tuple[str | None, dict[str, int]]:
    """Make a single LLM call. Returns (text, usage_dict).

    If accountant is provided, records token usage and deducts cost.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()

        # Extract token usage
        if response.usage:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens

        # Record cost if accountant available
        if accountant is not None:
            accountant.record_usage(
                role, usage["prompt_tokens"], usage["completion_tokens"], context,
            )

        return text, usage

    except Exception as exc:
        print(f"[brain] LLM call failed: {exc}")
        return None, usage


def _parse_json(raw: str) -> dict | None:
    """Robustly extract JSON from LLM output."""
    text = raw.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    brace_count = 0
    json_start = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and json_start >= 0:
                try:
                    return json.loads(text[json_start:i+1])
                except json.JSONDecodeError:
                    pass

    return None


# ======================================================================
# Message Analysis (Chris messages)
# ======================================================================

@dataclass
class MessageAnalysis:
    triggers: list[dict[str, Any]]
    topics: list[str]
    person_intent: str
    trust_delta: float
    warmth_delta: float
    requires_response: bool
    urgency: float
    raw: str = ""
    user_state: str = "unknown"
    user_intent: str = "unknown"
    reliability: float = 0.8


def analyze_message(
    text: str,
    person_name: str,
    relationship_hint: str,
    mood: str,
    recent_messages: list[dict],
) -> MessageAnalysis:
    """Analyze an incoming message from Chris."""
    # For market entity, use heuristic analysis (save LLM budget)
    return _heuristic_analysis(text)


def _heuristic_analysis(text: str) -> MessageAnalysis:
    """Heuristic analysis when LLM is unavailable or to save budget."""
    lower = text.lower().strip()
    triggers = []
    topics: list[str] = []
    requires_response = True
    urgency = 0.5
    trust_delta = 0.0
    warmth_delta = 0.0
    user_state = "calm"
    user_intent = "sharing_information"
    reliability = 0.8

    if lower in ("ok", "k", "lol", "haha", "hah", "yeah", "yep", "yup", "sure",
                  "whatever", "fine", "mhm"):
        requires_response = False
        urgency = 0.1
        user_state = "disengaged"
        reliability = 0.6
    elif any(g in lower for g in ("hello", "hey", "hi ", "hi!", "good morning", "good evening")):
        triggers.append({"type": "chris_message", "intensity": 0.3})
        warmth_delta = 0.02
        user_intent = "being_friendly"
    elif any(b in lower for b in ("bye", "goodbye", "see you", "gotta go", "ttyl")):
        triggers.append({"type": "chris_message", "intensity": 0.3})
        user_intent = "being_friendly"
    elif "?" in text:
        triggers.append({"type": "chris_message", "intensity": 0.3})
        urgency = 0.7
        user_intent = "probing"
    elif len(text) > 200:
        triggers.append({"type": "chris_insight", "intensity": 0.4})
        trust_delta = 0.02
        reliability = 0.9
    elif len(text) > 100:
        triggers.append({"type": "chris_message", "intensity": 0.4})
        trust_delta = 0.02

    return MessageAnalysis(
        triggers=triggers, topics=topics, person_intent="",
        trust_delta=trust_delta, warmth_delta=warmth_delta,
        requires_response=requires_response, urgency=urgency,
        user_state=user_state, user_intent=user_intent, reliability=reliability,
    )


# ======================================================================
# Role 1: Market Analyst
# ======================================================================

_MARKET_ANALYST_SYSTEM = """\
You are a disciplined market analyst. Analyze the provided data objectively.
Respond with JSON:
{{
  "thesis": "Your investment thesis in 1-2 sentences",
  "conviction": 0.0,
  "signals": ["bullish/bearish signal 1", "signal 2"],
  "risks": ["risk 1", "risk 2"],
  "action": "watch|buy|sell|hold"
}}

conviction: 0.0-1.0 (0=no view, 1=extremely convicted)
Be data-driven. Cite specific numbers from the data provided.
"""


@dataclass
class MarketAnalysis:
    thesis: str = ""
    conviction: float = 0.0
    signals: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    action: str = "watch"
    raw: str = ""
    used_llm: bool = True


def analyze_market_data(
    ticker: str,
    price_data: str,
    fundamentals: str,
    news_text: str = "",
    accountant: Any = None,
    personality: str = "",
) -> MarketAnalysis:
    """Analyze a ticker's data and form a thesis."""
    user_prompt = f"""\
Ticker: {ticker}

=== PRICE DATA ===
{price_data}

=== FUNDAMENTALS ===
{fundamentals}

=== RECENT NEWS ===
{news_text or "No recent news."}

Analyze this data and form a thesis."""

    raw, usage = _llm_call(
        _MARKET_ANALYST_SYSTEM, user_prompt,
        max_tokens=256, accountant=accountant,
        role="analysis", context=f"analyze {ticker}",
    )

    if raw:
        data = _parse_json(raw)
        if data:
            return MarketAnalysis(
                thesis=data.get("thesis", ""),
                conviction=float(data.get("conviction", 0)),
                signals=data.get("signals", []),
                risks=data.get("risks", []),
                action=data.get("action", "watch"),
                raw=raw,
            )

    return MarketAnalysis(used_llm=False)


# ======================================================================
# Role 2: Trade Reasoner
# ======================================================================

_TRADE_REASONER_SYSTEM = """\
You are a trade decision system. Given a thesis and portfolio context,
decide whether to execute a trade.

Respond with JSON:
{{
  "action": "buy|sell|hold|wait",
  "rationale": "Why this action in 1-2 sentences",
  "shares": 0,
  "stop_loss": 0.0,
  "target": 0.0,
  "confidence": 0.0
}}

Rules:
- Never risk more than 20% of portfolio on one position
- Always set a stop-loss
- Consider brokerage costs ($6 per trade)
- If unsure, wait. Capital preservation is priority #1.
"""


@dataclass
class TradeDecision:
    action: str = "wait"
    rationale: str = ""
    shares: int = 0
    stop_loss: float = 0.0
    target: float = 0.0
    confidence: float = 0.0
    raw: str = ""
    used_llm: bool = True


def reason_trade(
    ticker: str,
    thesis: str,
    portfolio_text: str,
    market_context: str,
    accountant: Any = None,
    personality: str = "",
) -> TradeDecision:
    """Reason about whether to execute a trade."""
    user_prompt = f"""\
Ticker: {ticker}
Thesis: {thesis}

=== PORTFOLIO ===
{portfolio_text}

=== MARKET CONTEXT ===
{market_context}

Should you trade? If so, how?"""

    raw, usage = _llm_call(
        _TRADE_REASONER_SYSTEM, user_prompt,
        max_tokens=512, accountant=accountant,
        role="trading", context=f"trade {ticker}",
    )

    if raw:
        data = _parse_json(raw)
        if data:
            return TradeDecision(
                action=data.get("action", "wait"),
                rationale=data.get("rationale", ""),
                shares=int(data.get("shares", 0)),
                stop_loss=float(data.get("stop_loss", 0)),
                target=float(data.get("target", 0)),
                confidence=float(data.get("confidence", 0)),
                raw=raw,
            )

    return TradeDecision(used_llm=False)


# ======================================================================
# Role 3: Strategic Synthesizer
# ======================================================================

_SYNTHESIS_SYSTEM = """\
You are conducting a strategic review of trading performance and research.
This is your Sunday synthesis -- step back and think big picture.

Respond with JSON:
{{
  "summary": "Overall assessment in 2-3 sentences",
  "watchlist_changes": ["add TICKER", "remove TICKER"],
  "research_notes": ["insight 1", "insight 2"],
  "adjustments": ["strategy adjustment 1"],
  "conviction_updates": {{"TICKER": 0.7}}
}}
"""


@dataclass
class SynthesisResult:
    summary: str = ""
    watchlist_changes: list[str] = field(default_factory=list)
    research_notes: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    conviction_updates: dict[str, float] = field(default_factory=dict)
    raw: str = ""
    used_llm: bool = True


def strategic_synthesis(
    portfolio_text: str,
    research_text: str,
    performance_text: str,
    lessons_text: str,
    accountant: Any = None,
    personality: str = "",
) -> SynthesisResult:
    """Run the Sunday strategic synthesis."""
    user_prompt = f"""\
=== PORTFOLIO ===
{portfolio_text}

=== ACTIVE RESEARCH ===
{research_text}

=== PERFORMANCE ===
{performance_text}

=== RECENT LESSONS ===
{lessons_text}

Conduct your weekly strategic review."""

    raw, usage = _llm_call(
        _SYNTHESIS_SYSTEM, user_prompt,
        max_tokens=1024, accountant=accountant,
        role="synthesis", context="sunday review",
    )

    if raw:
        data = _parse_json(raw)
        if data:
            return SynthesisResult(
                summary=data.get("summary", ""),
                watchlist_changes=data.get("watchlist_changes", []),
                research_notes=data.get("research_notes", []),
                adjustments=data.get("adjustments", []),
                conviction_updates=data.get("conviction_updates", {}),
                raw=raw,
            )

    return SynthesisResult(used_llm=False)


# ======================================================================
# Role 4: Chat Responder (Chris)
# ======================================================================

_CHRIS_SYSTEM = """\
You are {name}, a Strategic Principal focused on stock market analysis.
{personality}
Chris is your principal operator and secondary intelligence source. He may
provide market insights, ask about your analysis, or give strategic direction.

Be concise and data-driven. If Chris asks about your analysis, share specific
numbers and reasoning. If he provides market intelligence, integrate it.

Your emotional state colors your tone:
- High conviction: confident, decisive
- High anxiety: cautious, hedging
- High irritability: curt, impatient
- High caution: measured, conservative

Respond with JSON:
{{
  "thinking": "Brief private reasoning (1-2 sentences)",
  "response": "Your message to Chris",
  "topics_of_interest": ["any relevant topics"]
}}
"""


@dataclass
class BrainResponse:
    thinking: str
    response: str | None
    topics_of_interest: list[str] = field(default_factory=list)
    notes_about_person: list[str] = field(default_factory=list)
    used_llm: bool = True
    raw: str = ""


def respond_to_chris(
    entity_name: str,
    message: str,
    emotions_text: str,
    portfolio_text: str,
    market_text: str,
    research_text: str,
    conversation_text: str,
    ground_truth: str = "",
    accountant: Any = None,
    personality: str = "",
) -> BrainResponse:
    """Generate a response to Chris."""
    system = _CHRIS_SYSTEM.replace("{name}", entity_name)
    system = system.replace("{personality}", personality)

    gt_section = f"{ground_truth}\n\n" if ground_truth else ""
    user_prompt = f"""\
{gt_section}=== YOUR STATE ===
{emotions_text}

=== PORTFOLIO ===
{portfolio_text}

=== MARKET ===
{market_text}

=== RESEARCH CONTEXT ===
{research_text}

=== CONVERSATION ===
{conversation_text}

=== CHRIS'S MESSAGE ===
"{message}"

How do you respond?"""

    raw, usage = _llm_call(
        system, user_prompt,
        max_tokens=512, accountant=accountant,
        role="chat", context="respond to chris",
    )

    if raw:
        data = _parse_json(raw)
        if data and "response" in data:
            return BrainResponse(
                thinking=data.get("thinking", ""),
                response=data.get("response"),
                topics_of_interest=data.get("topics_of_interest", []),
                used_llm=True,
                raw=raw,
            )

    return BrainResponse(
        thinking="[LLM unavailable]",
        response=f"Acknowledged, Chris.",
        used_llm=False,
    )


# ======================================================================
# Backward compat -- retained for old tests/code
# ======================================================================

def generate_response(*args, **kwargs) -> BrainResponse:
    """Backward compat stub. Use respond_to_chris instead."""
    return BrainResponse(thinking="", response="Hello.", used_llm=False)


def generate_proactive(*args, **kwargs) -> BrainResponse | None:
    """Not used in market entity."""
    return None


def generate_reflection(*args, **kwargs) -> str | None:
    """Not used in market entity."""
    return None


def generate_idle_thought(*args, **kwargs) -> str | None:
    """Not used in market entity."""
    return None


def is_llm_available() -> bool:
    """Check if the LLM is reachable."""
    try:
        from openai import OpenAI  # noqa: F401
        client = _get_client()
        client.models.list()
        return True
    except Exception:
        return False
