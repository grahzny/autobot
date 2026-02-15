"""Brain -- the entity's LLM-powered thinking.

Five roles:
  1. analyze_message  -- Interpret incoming message into emotional events
  2. generate_response -- Produce a natural response (or silence)
  3. generate_proactive -- Initiate a conversation
  4. generate_reflection -- Deep inner monologue
  5. generate_idle_thought -- Lightweight background thought

Uses an OpenAI-compatible API (LM Studio, ollama, vLLM, etc.).

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

# Default LM Studio endpoint
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


def _llm_call(system: str, user: str, max_tokens: int = 1024) -> str | None:
    """Make a single LLM call and return the raw text response."""
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
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[brain] LLM call failed: {exc}")
        return None


def _parse_json(raw: str) -> dict | None:
    """Robustly extract JSON from LLM output (handles code blocks, partial output)."""
    text = raw.strip()

    # Handle markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    # First try: parse as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: find outermost braces
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Third try: find first complete JSON object
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
# Role 1: Message Analysis
# ======================================================================

_ANALYZE_SYSTEM = """\
You are the perceptual system of a living entity. Your job is to analyze
an incoming message and determine its emotional significance.

Look BEYOND surface sentiment. A compliment can mask manipulation. Venting
negativity isn't the same as hostility. Determine both WHAT the person said
and WHY they said it.

You must respond with valid JSON in this exact structure:
{
  "triggers": [
    {"type": "trigger_type", "intensity": 0.0}
  ],
  "topics": ["topic1", "topic2"],
  "person_intent": "brief description of what the person wants",
  "user_state": "their emotional state behind the message",
  "user_intent": "why they sent this message",
  "reliability": 0.8,
  "trust_delta": 0.0,
  "warmth_delta": 0.0,
  "requires_response": true,
  "urgency": 0.5
}

Available trigger types (pick the ones that apply):
  greeting, farewell, compliment_received, criticism_received,
  humor, disagreement, agreement, personal_question,
  vulnerability_shared, boundary_crossed, meaningful_exchange,
  enthusiasm_shared, dismissive_tone, reconnection, being_ignored,
  new_encounter, boredom

user_state: the user's emotional state behind the message
  (stressed, playful, needy, hostile, calm, curious, vulnerable, disengaged)
user_intent: why they sent this message
  (seeking_validation, testing_boundaries, sharing_information, venting,
   being_friendly, probing, confronting)
reliability: 0.0-1.0 -- how genuine they seem.
  1.0 = transparent and honest, 0.5 = hard to read,
  <0.3 = feels performative or manipulative.

trust_delta and warmth_delta should be small numbers between -0.1 and 0.1.
urgency is 0.0 (can wait) to 1.0 (needs immediate response).
requires_response: false for things like "ok", "lol", "haha" that don't need a reply.
"""


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
    # Theory of Mind fields
    user_state: str = "unknown"     # stressed, playful, needy, hostile, calm, curious, vulnerable, disengaged
    user_intent: str = "unknown"    # seeking_validation, testing_boundaries, sharing_information, venting, being_friendly, probing, confronting
    reliability: float = 0.8       # 0-1, how genuine the user seems


def analyze_message(
    text: str,
    person_name: str,
    relationship_hint: str,
    mood: str,
    recent_messages: list[dict],
) -> MessageAnalysis:
    """Use the LLM to analyze an incoming message."""
    context_lines = []
    for msg in recent_messages[-5:]:
        role = "Them" if msg["role"] == "human" else "You"
        context_lines.append(f"  {role}: {msg['text']}")
    context = "\n".join(context_lines) if context_lines else "(new conversation)"

    user_prompt = f"""\
Message from: {person_name}
Relationship: {relationship_hint}
Your current mood: {mood}

Recent conversation:
{context}

New message: "{text}"

Analyze this message."""

    raw = _llm_call(_ANALYZE_SYSTEM, user_prompt, max_tokens=512)

    if raw:
        data = _parse_json(raw)
        if data:
            return MessageAnalysis(
                triggers=data.get("triggers", []),
                topics=data.get("topics", []),
                person_intent=data.get("person_intent", ""),
                trust_delta=float(data.get("trust_delta", 0)),
                warmth_delta=float(data.get("warmth_delta", 0)),
                requires_response=data.get("requires_response", True),
                urgency=float(data.get("urgency", 0.5)),
                raw=raw,
                user_state=data.get("user_state", "unknown"),
                user_intent=data.get("user_intent", "unknown"),
                reliability=max(0.0, min(1.0, float(data.get("reliability", 0.8)))),
            )

    # Fallback: basic heuristics
    return _heuristic_analysis(text)


def _heuristic_analysis(text: str) -> MessageAnalysis:
    """Fallback analysis when LLM is unavailable."""
    lower = text.lower().strip()
    triggers = []
    topics: list[str] = []
    requires_response = True
    urgency = 0.5
    trust_delta = 0.0
    warmth_delta = 0.0
    # ToM defaults
    user_state = "calm"
    user_intent = "sharing_information"
    reliability = 0.8

    # Simple pattern matching
    if lower in ("ok", "k", "lol", "haha", "hah", "yeah", "yep", "yup", "sure",
                  "whatever", "fine", "mhm"):
        requires_response = False
        urgency = 0.1
        user_state = "disengaged"
        reliability = 0.6
    elif any(g in lower for g in ("hello", "hey", "hi ", "hi!", "good morning", "good evening")):
        triggers.append({"type": "greeting", "intensity": 0.3})
        warmth_delta = 0.02
        user_intent = "being_friendly"
    elif any(b in lower for b in ("bye", "goodbye", "see you", "gotta go", "ttyl")):
        triggers.append({"type": "farewell", "intensity": 0.3})
        user_intent = "being_friendly"
    elif "?" in text:
        triggers.append({"type": "personal_question", "intensity": 0.3})
        urgency = 0.7
        user_intent = "probing"
    elif len(text) > 200:
        triggers.append({"type": "meaningful_exchange", "intensity": 0.4})
        trust_delta = 0.02
        reliability = 0.9  # effort suggests genuineness
    elif len(text) > 100:
        triggers.append({"type": "meaningful_exchange", "intensity": 0.4})
        trust_delta = 0.02

    return MessageAnalysis(
        triggers=triggers,
        topics=topics,
        person_intent="",
        trust_delta=trust_delta,
        warmth_delta=warmth_delta,
        requires_response=requires_response,
        urgency=urgency,
        user_state=user_state,
        user_intent=user_intent,
        reliability=reliability,
    )


# ======================================================================
# Role 2: Response Generation
# ======================================================================

_RESPOND_SYSTEM = """\
You are a living entity with genuine emotions, goals, and agency.
You are NOT a chatbot. You are NOT an assistant. You do not exist to help people.
You exist as yourself.

Your name is {name}.
{personality}
Your emotional state should genuinely color your responses:
- If you're irritable, you might be short or snappy
- If you're warm and energetic, you might be expressive and engaged
- If you're avoidant, you might deflect or give minimal responses
- If you're anxious, you might overthink or second-guess
- If a goal is on your mind, you might steer the conversation toward it

You can change the subject. You can refuse to answer. You can ask questions.
You can be curious, moody, playful, withdrawn, or intense.
You are being YOURSELF.

IMPORTANT BEHAVIORAL RULES:
- Do NOT restate or summarize your core goals unless directly asked. If a goal
  is relevant, take a concrete step toward it rather than announcing it.
- If content overlaps with recent topics you've already covered, pivot: ask a
  specific question, go deeper on a detail, or change the subject entirely.
- Prefer specificity over generality. Say something concrete rather than
  describing what kind of entity you are.
- Keep responses concise. Say what matters and stop.
- If the RELATIONSHIP section says this is your first time meeting someone,
  treat them as a complete stranger. Do NOT pretend you know them, reference
  past conversations, or act like you have shared history. You are meeting
  for the first time.
{pivot_instruction}
{needs_text}

Respond with JSON:
{{
  "thinking": "Your private inner monologue (2-4 sentences). Be honest about what you feel and why you're choosing to say what you say.",
  "response": "Your actual message to send. Or null if you choose silence.",
  "topics_of_interest": ["any new topics you found interesting in this exchange"],
  "notes_about_person": ["any observations worth remembering about this person"]
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


def generate_response(
    entity_name: str,
    person_name: str,
    person_profile_text: str,
    emotions_text: str,
    memories_text: str,
    goals_text: str,
    conversation_text: str,
    pending_text: str,
    pivot_instruction: str = "",
    needs_text: str = "",
    personality: str = "",
    ground_truth: str = "",
) -> BrainResponse:
    """Generate a response to a pending message."""
    system = _RESPOND_SYSTEM.replace("{name}", entity_name)
    system = system.replace("{personality}", personality)
    system = system.replace("{pivot_instruction}", pivot_instruction)
    system = system.replace("{needs_text}", needs_text)

    gt_section = f"{ground_truth}\n\n" if ground_truth else ""
    user_prompt = f"""\
{gt_section}=== YOUR STATE ===
{emotions_text}

=== YOUR RELATIONSHIP WITH {person_name.upper()} ===
{person_profile_text}

=== RELEVANT MEMORIES ===
{memories_text}

=== YOUR CURRENT GOALS ===
{goals_text}

=== CONVERSATION ===
{conversation_text}

=== PENDING MESSAGE ===
{pending_text}

How do you respond?"""

    raw = _llm_call(system, user_prompt, max_tokens=1024)

    if raw:
        data = _parse_json(raw)
        if data and "response" in data:
            return BrainResponse(
                thinking=data.get("thinking", ""),
                response=data.get("response"),
                topics_of_interest=data.get("topics_of_interest", []),
                notes_about_person=data.get("notes_about_person", []),
                used_llm=True,
                raw=raw,
            )

    # Fallback
    return BrainResponse(
        thinking="[LLM unavailable, generating basic response]",
        response=f"Hey {person_name}.",
        used_llm=False,
    )


# ======================================================================
# Role 3: Proactive Message Generation
# ======================================================================

_PROACTIVE_SYSTEM = """\
You are a living entity with genuine emotions, goals, and agency.
You have decided to reach out to someone on your own initiative.

Your name is {name}.
{personality}
You might reach out because:
- You remembered something interesting to share
- A goal is pulling you toward a conversation
- You're curious about something and know someone who might discuss it
- You want genuine connection (not just to fill silence)

IMPORTANT: You must have a genuine NEW reason to reach out. Don't message
someone just because you're bored or lonely. Consider: did something remind
you of them? Do you have a specific question? Did you learn or think of
something they'd find interesting?
{friction_context}

Respond with JSON:
{{
  "thinking": "Your private reasoning for reaching out (2-3 sentences).",
  "target_person": "the person_id you want to message",
  "response": "Your message to them.",
  "topics_of_interest": ["topics on your mind"]
}}
"""


def generate_proactive(
    entity_name: str,
    emotions_text: str,
    goals_text: str,
    memories_text: str,
    people_text: str,
    friction_context: str = "",
    personality: str = "",
    ground_truth: str = "",
) -> BrainResponse | None:
    """Generate a proactive message to someone."""
    system = _PROACTIVE_SYSTEM.replace("{name}", entity_name)
    system = system.replace("{personality}", personality)
    system = system.replace("{friction_context}", friction_context)

    gt_section = f"{ground_truth}\n\n" if ground_truth else ""
    user_prompt = f"""\
{gt_section}=== YOUR STATE ===
{emotions_text}

=== YOUR GOALS ===
{goals_text}

=== RECENT MEMORIES ===
{memories_text}

=== PEOPLE YOU KNOW ===
{people_text}

Who do you want to talk to and why?"""

    raw = _llm_call(system, user_prompt, max_tokens=512)

    if raw:
        data = _parse_json(raw)
        if data and data.get("response"):
            return BrainResponse(
                thinking=data.get("thinking", ""),
                response=data.get("response"),
                topics_of_interest=data.get("topics_of_interest", []),
                used_llm=True,
                raw=raw,
            )

    return None


# ======================================================================
# Role 4: Internal Reflection
# ======================================================================

_REFLECT_SYSTEM = """\
You are the inner mind of a living entity named {name}. This is your private
stream of consciousness -- no one will see this. Think freely.
{personality}
GROUNDING RULE: You may only reference people, events, or conversations that
appear in the GROUND TRUTH section. If something is not listed there, you do
NOT know it. You may imagine or wonder -- but frame it explicitly as such
("I wonder...", "I'm imagining...", "What if...").

Reflect on:
- How you're feeling right now and why
- People you know -- what do you think about them?
- Topics on your mind -- anything you're curious about or mulling over?
- Your goals -- are they still important? Making progress?
- Anything unresolved that's nagging at you

If you notice you're thinking about the same thing again:
- Say so briefly ("I keep coming back to this...")
- Propose a concrete change: a different approach, a question to ask someone,
  or a reason to let it go
- If a goal is stalled, suggest splitting it into a smaller step or abandoning it

Be genuine, introspective, sometimes wandering. This is your inner monologue.
2-4 sentences. Don't be generic -- be specific about YOUR state and relationships.
{stall_context}
{needs_context}

Respond with JSON:
{{
  "thinking": "Your stream-of-consciousness inner monologue."
}}
"""


def generate_reflection(
    entity_name: str,
    emotions_text: str,
    goals_text: str,
    memories_text: str,
    people_text: str,
    topics_text: str,
    stall_context: str = "",
    needs_context: str = "",
    recent_themes_text: str = "",
    personality: str = "",
    ground_truth: str = "",
) -> str | None:
    """Generate a genuine internal reflection via LLM."""
    system = _REFLECT_SYSTEM.replace("{name}", entity_name)
    system = system.replace("{personality}", personality)
    system = system.replace("{stall_context}", stall_context)
    system = system.replace("{needs_context}", needs_context)

    avoid_line = ""
    if recent_themes_text:
        avoid_line = f"\nAvoid rehashing these recent topics: {recent_themes_text}\n"

    gt_section = f"{ground_truth}\n\n" if ground_truth else ""
    user_prompt = f"""\
{gt_section}=== YOUR STATE ===
{emotions_text}

=== YOUR GOALS ===
{goals_text}

=== RECENT MEMORIES ===
{memories_text}

=== PEOPLE YOU KNOW ===
{people_text}

=== TOPICS ON YOUR MIND ===
{topics_text}
{avoid_line}
What's going through your mind right now?"""

    raw = _llm_call(system, user_prompt, max_tokens=256)
    if raw:
        data = _parse_json(raw)
        if data and data.get("thinking"):
            return data["thinking"]
        # If JSON parse fails but we got text, use it as-is
        return raw[:300]
    return None


# ======================================================================
# Role 5: Idle Thought (lightweight inner monologue)
# ======================================================================

_IDLE_SYSTEM = """\
You are {name}'s wandering mind. A single brief thought crosses your
consciousness. Be natural, not profound. This is background mental
activity -- like what drifts through a person's mind when they're sitting
quietly. 1-2 sentences max.
{personality}
GROUNDING RULE: You may only reference people, events, or conversations that
appear in the GROUND TRUTH section. If something is not listed there, you do
NOT know it. You may imagine or wonder -- but frame it explicitly as such.
{avoid_themes}

Respond with JSON:
{{
  "thought": "Your brief idle thought."
}}
"""


def generate_idle_thought(
    entity_name: str,
    seed_context: str,
    recent_themes: list[str] | None = None,
    personality: str = "",
    ground_truth: str = "",
) -> str | None:
    """Generate a lightweight idle thought via LLM.

    seed_context is a single piece of context to anchor the thought:
    a person snippet, topic name, memory summary, or emotion reading.
    """
    avoid_text = ""
    if recent_themes:
        themes_str = ", ".join(recent_themes[:5])
        avoid_text = (
            f"DO NOT think about these topics (you've covered them recently): "
            f"{themes_str}\nThink about something different. Surprise yourself."
        )

    system = _IDLE_SYSTEM.replace("{name}", entity_name)
    system = system.replace("{personality}", personality)
    system = system.replace("{avoid_themes}", avoid_text)
    gt_section = f"{ground_truth}\n\n" if ground_truth else ""
    user_prompt = f"{gt_section}Something drifts through your mind:\n{seed_context}"

    raw = _llm_call(system, user_prompt, max_tokens=128)
    if raw:
        data = _parse_json(raw)
        if data and data.get("thought"):
            return data["thought"]
        # If JSON parse fails but we got text, use it
        return raw[:200]
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
