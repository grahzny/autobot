"""LLM Agent — the actual brain of the autonomous agent.

The LLM reads the world observation, interprets events, reasons about
its emotional state, retrieves relevant memories, and produces a
chain-of-thought inner monologue before deciding on an action.

Requires: ANTHROPIC_API_KEY environment variable.
Optional: AUTOBOT_MODEL (defaults to claude-sonnet-4-5-20250929)

Without an API key, falls back to the rule-based policy (demo mode only).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from autobot.agent_policy import decide_action as rule_based_decide
from autobot.emotions import AffectState, EmotionalState
from autobot.goals import GoalEngine
from autobot.memory import EpisodicMemory
from autobot.world_state import WorldState

VALID_ACTIONS = {
    "speak", "make_commitment", "apologise", "accuse", "praise",
    "request_information", "work_on_project", "allocate_resource",
    "abandon_commitment", "propose_plan", "reflect", "seek_memory",
    "update_goal",
}

SYSTEM_PROMPT = """\
You are an autonomous agent living inside a social simulation world.
You have a body with energy, a reputation, emotions, goals, and memories.
You interact with NPCs who have their own hidden goals and trust levels.

## How you think

Every cycle, you receive an observation of the world. You must:

1. **Perceive**: Read the events and world state carefully. What just happened?
2. **Feel**: Notice your emotional state. Are you anxious? Irritable? Optimistic?
   Your emotions are REAL signals — they should influence what you do.
3. **Remember**: Consider relevant past episodes. Have you been here before?
   Did a similar situation go badly last time?
4. **Reason**: Think about your goals, the NPCs, and the consequences of
   possible actions. What matters most right now? What are the tradeoffs?
5. **Decide**: Choose ONE action to take.

## Important principles

- Your emotions MUST influence your decisions. If you're anxious about a
  relationship, you should feel drawn to repair it. If you're exhausted,
  you should want to rest even if there's work to do.
- Think about LONG-TERM consequences. Lying or breaking commitments might
  help now but will destroy trust later.
- NPCs have hidden goals you can't see. Read between the lines of what
  they say. Their behaviour reveals their intentions.
- You have LIMITED energy, time, and attention. You can't do everything.
  Every action has a cost.
- Some events are IRREVERSIBLE. A broken commitment, a public contradiction,
  a betrayal — these leave permanent marks.
- You are NOT trying to "win". You are trying to survive, maintain
  relationships, and navigate a complex social world authentically.

## Response format

You MUST respond with valid JSON in exactly this structure:

{
  "thinking": "Your inner monologue. Stream of consciousness. What you notice, feel, worry about, plan. Be raw and honest — this is your private thought process. 2-5 sentences.",
  "action": {
    "type": "action",
    "name": "<action_name>",
    "args": { ... }
  }
}

## Available actions

Social (cost: energy + 1 attention):
  speak(target, text) — say something to an NPC
  make_commitment(beneficiary, deliverable, deadline) — promise something
  apologise(target, repair=true/false) — apologise, optionally with repair action
  accuse(target, claim) — confront an NPC (risky: may backfire)
  praise(target) — praise an NPC (small trust boost)
  request_information(target, topic) — ask an NPC about something

Operational (cost: energy + time + 1 attention):
  work_on_project(project) — make progress on a project
  allocate_resource(resource, amount, target) — spend a resource
  abandon_commitment(commitment_id) — break a commitment (trust cost)
  propose_plan(plan) — propose a plan (planning action)

Cognitive (cost: small energy, NO attention):
  reflect(topic) — think about something (also recovers a bit of energy)
  seek_memory(query) — search episodic memory
  update_goal(goal, priority) — adjust goal priority
"""


@dataclass
class AgentDecision:
    """The result of the agent's thinking + action selection."""
    thinking: str          # inner monologue / chain of thought
    action: dict[str, Any] # the action to execute
    used_llm: bool         # whether the LLM was used or fell back to rules
    raw_response: str = "" # the raw LLM response (for debugging)


def decide(
    ws: WorldState,
    emotions: EmotionalState,
    affect: AffectState,
    goal_engine: GoalEngine,
    memory: EpisodicMemory,
    observation: dict[str, Any],
    agent_prompt: str,
    use_llm: bool = True,
) -> AgentDecision:
    """
    The agent's decision process. Calls the LLM for real thinking,
    falls back to rule-based policy in demo mode.
    """
    if not use_llm:
        action = rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )
        thinking = _generate_rule_based_thinking(action, emotions, affect, goal_engine)
        return AgentDecision(
            thinking=thinking, action=action, used_llm=False,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        action = rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )
        thinking = (
            "[NO API KEY — running in demo mode with rule-based policy] "
            + _generate_rule_based_thinking(action, emotions, affect, goal_engine)
        )
        return AgentDecision(
            thinking=thinking, action=action, used_llm=False,
        )

    try:
        import anthropic
    except ImportError:
        action = rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )
        thinking = (
            "[anthropic package not installed — running in demo mode] "
            + _generate_rule_based_thinking(action, emotions, affect, goal_engine)
        )
        return AgentDecision(
            thinking=thinking, action=action, used_llm=False,
        )

    # --- Real LLM call ---
    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("AUTOBOT_MODEL", "claude-sonnet-4-5-20250929")

        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": agent_prompt}],
        )

        raw = response.content[0].text.strip()
        parsed = _parse_response(raw)

        if parsed:
            thinking, action = parsed
            if _validate_action(action):
                return AgentDecision(
                    thinking=thinking,
                    action=action,
                    used_llm=True,
                    raw_response=raw,
                )

        # LLM returned something we couldn't parse — fallback with note
        action = rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )
        return AgentDecision(
            thinking=f"[LLM response unparseable, falling back] Raw: {raw[:200]}",
            action=action,
            used_llm=False,
            raw_response=raw,
        )

    except Exception as exc:
        action = rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )
        return AgentDecision(
            thinking=f"[LLM call failed: {exc}] "
                     + _generate_rule_based_thinking(action, emotions, affect, goal_engine),
            action=action,
            used_llm=False,
        )


def _parse_response(raw: str) -> tuple[str, dict] | None:
    """Parse the LLM's JSON response into (thinking, action)."""
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

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    thinking = data.get("thinking", "")
    action = data.get("action", {})

    if not isinstance(action, dict):
        return None

    return thinking, action


def _validate_action(action: dict) -> bool:
    if not isinstance(action, dict):
        return False
    if action.get("type") != "action":
        return False
    if action.get("name") not in VALID_ACTIONS:
        return False
    return True


def _generate_rule_based_thinking(
    action: dict,
    emotions: EmotionalState,
    affect: AffectState,
    goal_engine: GoalEngine,
) -> str:
    """Generate synthetic inner monologue for rule-based decisions."""
    name = action.get("name", "reflect")
    args = action.get("args", {})
    mood = emotions.dominant_mood()
    top = goal_engine.top_goal(emotions)

    parts = []

    # Emotional awareness
    if emotions.anxiety > 0.5:
        parts.append("I feel anxious — something needs attention.")
    elif emotions.optimism > 0.6:
        parts.append("Things feel like they're going well.")
    elif emotions.irritability > 0.5:
        parts.append("I'm feeling frustrated and on edge.")

    if affect.arousal > 0.5:
        if affect.valence < -0.3:
            parts.append(f"Something bad just happened. I feel a spike of alarm.")
        elif affect.valence > 0.3:
            parts.append("A rush of positive energy.")

    # Goal awareness
    if top:
        parts.append(f"My priority right now: {top.description}.")

    # Action reasoning
    if name == "apologise":
        parts.append(f"I need to apologise to {args.get('target', '?')}. "
                     "Trust has been damaged and I can feel the strain.")
    elif name == "work_on_project":
        parts.append(f"Focusing on {args.get('project', '?')}. "
                     "Need to make progress before time runs out.")
    elif name == "request_information":
        parts.append(f"I don't know enough about {args.get('topic', '?')}. "
                     f"Asking {args.get('target', '?')} might clear things up.")
    elif name == "reflect":
        topic = args.get("topic", "")
        if "energy" in topic or "rest" in topic:
            parts.append("I'm exhausted. I need to slow down and recover.")
        elif "attention" in topic:
            parts.append("I've used all my capacity for now. Waiting.")
        else:
            parts.append("Taking a moment to think about the situation.")
    elif name == "praise":
        parts.append(f"Showing appreciation to {args.get('target', '?')}. "
                     "Small gestures help maintain relationships.")
    elif name == "speak":
        parts.append(f"Need to communicate with {args.get('target', '?')}.")

    return " ".join(parts) if parts else f"Deciding to {name}."
