"""Observation model — filters canonical world state into agent-visible packets."""

from __future__ import annotations

from typing import Any

from autobot.entities import CommitmentStatus
from autobot.world_state import WorldState


def build_observation(
    ws: WorldState,
    cycle_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Produce the observation packet the agent receives each cycle.
    The agent never sees hidden NPC goals, full belief graphs, or the
    event queue.
    """
    # Visible commitments: only those involving the agent
    visible_commitments = []
    for c in ws.commitments.values():
        if c.promisor == ws.agent.name:
            visible_commitments.append({
                "id": c.id,
                "beneficiary": ws.npcs[c.beneficiary].name
                    if c.beneficiary in ws.npcs else c.beneficiary,
                "deliverable": c.deliverable,
                "deadline": c.deadline,
                "status": c.status.value,
            })

    # Open projects
    open_projects = []
    for p in ws.projects.values():
        open_projects.append({
            "id": p.id,
            "name": p.name,
            "progress": round(p.progress, 2),
            "deadline": p.deadline,
            "overdue": p.is_overdue(ws.time),
            "importance": p.importance_weight,
        })

    # Uncertainty flags: beliefs with low confidence
    uncertainty_flags = []
    for b in ws.beliefs.values():
        if b.holder == ws.agent.name and b.confidence < 0.5:
            uncertainty_flags.append({
                "belief_id": b.id,
                "proposition": b.proposition,
                "confidence": round(b.confidence, 2),
            })

    # NPC visible state (trust is NOT directly visible — only inferred)
    npc_summaries = []
    for npc in ws.npcs.values():
        # Agent can see NPC statements in cycle events but not trust number.
        # We expose a qualitative hint.
        trust_hint = _trust_to_hint(npc.trust_in_agent)
        npc_summaries.append({
            "name": npc.name,
            "disposition": trust_hint,
            "influence": round(npc.influence_weight, 2),
        })

    # Filter events to only those the agent could plausibly perceive
    visible_events = [_filter_event(e) for e in cycle_events if _is_visible(e)]

    return {
        "time": ws.time_str(),
        "events": visible_events,
        "self_state": {
            "energy": round(ws.agent.energy, 2),
            "reputation": round(ws.agent.reputation, 2),
            "time_remaining": round(ws.agent.resources["time"], 1),
            "attention_remaining": round(ws.agent.resources["attention"], 1),
        },
        "npcs": npc_summaries,
        "visible_commitments": visible_commitments,
        "open_projects": open_projects,
        "uncertainty_flags": uncertainty_flags,
    }


# ---- Helpers ----

def _trust_to_hint(trust: float) -> str:
    if trust >= 0.8:
        return "very warm"
    if trust >= 0.6:
        return "friendly"
    if trust >= 0.4:
        return "neutral"
    if trust >= 0.2:
        return "cool"
    return "hostile"


def _is_visible(event: dict[str, Any]) -> bool:
    """Some events are hidden from the agent."""
    hidden_types = {"hidden_npc_action", "hidden_goal_change"}
    return event.get("type") not in hidden_types


def _filter_event(event: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields the agent shouldn't see."""
    filtered = dict(event)
    for key in ("world_time", "day", "hour", "irreversible",
                "hidden_motive", "npc_internal"):
        filtered.pop(key, None)
    return filtered
