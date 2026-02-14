"""Agent Policy — the autonomous decision-maker.

This is a rule-based policy that acts as the agent's "brain".
It reads the observation, emotional state, goals, and memories,
then selects the best action.

Designed to be replaceable with an LLM call — the engine only cares
about receiving a valid action dict back.
"""

from __future__ import annotations

from typing import Any

from autobot.emotions import AffectState, EmotionalState
from autobot.entities import CommitmentStatus
from autobot.goals import Goal, GoalEngine
from autobot.memory import EpisodicMemory
from autobot.world_state import WorldState


def decide_action(
    ws: WorldState,
    emotions: EmotionalState,
    affect: AffectState,
    goal_engine: GoalEngine,
    memory: EpisodicMemory,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """
    Select the best action for this cycle based on the agent's internal state.
    Returns a valid action dict: {"type": "action", "name": ..., "args": {...}}
    """
    top = goal_engine.top_goal(emotions)

    # --- Energy critical: rest unless a high-priority goal overrides ---
    if ws.agent.energy < 0.08:
        return _action("reflect", topic="resting to conserve energy")

    # --- No attention left for actions that cost attention ---
    if ws.agent.resources["attention"] < 1:
        # Cognitive actions (reflect, seek_memory, update_goal) cost 0 attention.
        # But we can still do goal-directed cognitive work.
        if top and top.category in ("trust", "reputation"):
            # Social actions need attention — try cognitive alternatives
            return _action("reflect", topic=f"planning how to address: {top.description}")
        return _action("reflect", topic="no attention remaining, waiting")

    # --- Goal-directed behaviour: try goals in ranked order ---
    for goal in goal_engine.ranked_goals(emotions):
        action = _goal_to_action(goal, ws, emotions, affect, memory, observation)
        if action:
            return action

    # --- Default: reflect ---
    return _action("reflect", topic="assessing situation")


def _goal_to_action(
    goal: Goal,
    ws: WorldState,
    emotions: EmotionalState,
    affect: AffectState,
    memory: EpisodicMemory,
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    """Map a goal to a concrete action."""

    # ---- Trust repair ----
    if goal.category == "trust" and goal.target_entity:
        npc = ws.get_npc_by_name(goal.target_entity)
        if not npc:
            return None

        # Check if we have broken commitments to this NPC
        broken = [
            c for c in ws.commitments.values()
            if c.beneficiary == npc.id
            and c.status == CommitmentStatus.BROKEN
        ]
        if broken:
            # Check memories for prior apologies
            past_apologies = memory.retrieve(
                entities=[npc.name], tags=["broken_promise"], limit=3,
            )
            already_apologised = any(
                "apology" in (ep.summary or "") for ep in past_apologies
            )
            if not already_apologised:
                return _action("apologise", target=npc.name, repair=True)

        # If trust is low and no broken commitments, try praising
        if npc.trust_in_agent < 0.4:
            return _action("praise", target=npc.name)

        # Speak to maintain relationship
        return _action("speak", target=npc.name,
                        text="I want to make things right between us.")

    # ---- Reputation ----
    if goal.category == "reputation":
        # Work on most important project to build public success
        projects = sorted(
            ws.projects.values(),
            key=lambda p: p.importance_weight,
            reverse=True,
        )
        for proj in projects:
            if proj.progress < 1.0:
                return _action("work_on_project", project=proj.name)
        return _action("reflect", topic="how to rebuild reputation")

    # ---- Project ----
    if goal.category == "project" and goal.target_entity:
        proj = ws.get_project_by_name(goal.target_entity)
        if proj and proj.progress < 1.0:
            return _action("work_on_project", project=proj.name)
        return None

    # ---- Uncertainty ----
    if goal.category == "uncertainty":
        # Find NPC with lowest trust (most uncertain relationship)
        uncertain_npcs = sorted(
            ws.npcs.values(), key=lambda n: n.trust_in_agent,
        )
        if uncertain_npcs:
            target = uncertain_npcs[0]
            # Check if anxiety is high enough to seek information
            if emotions.anxiety > 0.3:
                return _action(
                    "request_information",
                    target=target.name,
                    topic="your intentions",
                )
            else:
                return _action("reflect", topic="uncertain relationships")
        # Check beliefs
        uncertain_beliefs = [
            b for b in ws.beliefs.values()
            if b.holder == ws.agent.name and b.confidence < 0.4
        ]
        if uncertain_beliefs:
            b = uncertain_beliefs[0]
            # Try to find an NPC who might know
            if ws.npcs:
                npc = next(iter(ws.npcs.values()))
                return _action(
                    "request_information",
                    target=npc.name,
                    topic=b.proposition,
                )
        return _action("reflect", topic="reducing uncertainty")

    # ---- Energy ----
    if goal.category == "energy":
        # Rest — but if reputation is critical, might override
        if emotions.avoidance_bias > 0.6 or ws.agent.energy < 0.2:
            return _action("reflect", topic="resting to preserve energy")
        # Avoidance bias low, maybe push through
        return None

    return None


def _action(name: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": "action", "name": name, "args": kwargs}
