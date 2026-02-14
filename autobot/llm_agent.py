"""LLM Agent — calls Claude (or falls back to rule-based policy).

Set ANTHROPIC_API_KEY to enable LLM mode.
Without it, the rule-based policy runs automatically.
"""

from __future__ import annotations

import json
import os
from typing import Any

from autobot.agent_policy import decide_action as rule_based_decide
from autobot.emotions import AffectState, EmotionalState
from autobot.goals import GoalEngine
from autobot.memory import EpisodicMemory
from autobot.world_state import WorldState

# Valid action names the LLM can choose from
VALID_ACTIONS = {
    "speak", "make_commitment", "apologise", "accuse", "praise",
    "request_information", "work_on_project", "allocate_resource",
    "abandon_commitment", "propose_plan", "reflect", "seek_memory",
    "update_goal",
}


def decide_action_llm(
    ws: WorldState,
    emotions: EmotionalState,
    affect: AffectState,
    goal_engine: GoalEngine,
    memory: EpisodicMemory,
    observation: dict[str, Any],
    agent_prompt: str,
) -> dict[str, Any]:
    """
    Try to get an action from the LLM. Falls back to rule-based if:
    - No API key is set
    - The anthropic package isn't installed
    - The LLM returns an invalid action
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )

    try:
        import anthropic
    except ImportError:
        return rule_based_decide(
            ws, emotions, affect, goal_engine, memory, observation,
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("AUTOBOT_MODEL", "claude-sonnet-4-5-20250929")

        system_prompt = (
            "You are an autonomous agent in a social simulation world.\n"
            "You must respond with ONLY a valid JSON action object.\n"
            "Format: {\"type\": \"action\", \"name\": \"<action_name>\", \"args\": {...}}\n\n"
            "Available actions:\n"
            "  Social: speak(target, text), make_commitment(beneficiary, deliverable, deadline),\n"
            "    apologise(target, repair=bool), accuse(target, claim), praise(target),\n"
            "    request_information(target, topic)\n"
            "  Operational: work_on_project(project), allocate_resource(resource, amount, target),\n"
            "    abandon_commitment(commitment_id), propose_plan(plan)\n"
            "  Cognitive: reflect(topic), seek_memory(query), update_goal(goal, priority)\n\n"
            "Your emotional state MUST influence your decisions.\n"
            "High anxiety should drive you to resolve uncertainty and repair trust.\n"
            "Low energy should make you conserve resources.\n"
            "Consider long-term consequences over short-term gains.\n"
        )

        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": agent_prompt}],
        )

        text = response.content[0].text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        action = json.loads(text)
        if _validate_llm_action(action):
            return action

    except Exception:
        pass

    # Fallback
    return rule_based_decide(
        ws, emotions, affect, goal_engine, memory, observation,
    )


def _validate_llm_action(action: dict) -> bool:
    """Check that the LLM produced a structurally valid action."""
    if not isinstance(action, dict):
        return False
    if action.get("type") != "action":
        return False
    if action.get("name") not in VALID_ACTIONS:
        return False
    return True
