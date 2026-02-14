"""Action interface — validates and dispatches agent actions into world rules."""

from __future__ import annotations

from typing import Any

from autobot.entities import Belief, Commitment, CommitmentStatus
from autobot.mechanics import (
    apply_apology,
    can_afford_action,
    deduct_action_cost,
    fulfill_commitment,
)
from autobot.world_state import ScheduledEvent, WorldState


VALID_ACTIONS = {
    # Social
    "speak", "make_commitment", "apologise", "accuse", "praise",
    "request_information",
    # Operational
    "work_on_project", "allocate_resource", "abandon_commitment",
    "propose_plan",
    # Cognitive
    "reflect", "seek_memory", "update_goal",
}


class ActionError(Exception):
    pass


def validate_action(action: dict[str, Any]) -> None:
    if action.get("type") != "action":
        raise ActionError("Action must have type='action'")
    name = action.get("name")
    if name not in VALID_ACTIONS:
        raise ActionError(f"Unknown action: {name}")


def execute_action(
    ws: WorldState,
    action: dict[str, Any],
    emotional_drain: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Apply an action to the world.  Returns list of resulting events.
    The caller (world engine) feeds these back into the observation model.
    """
    validate_action(action)
    name = action["name"]
    args = action.get("args", {})

    if not can_afford_action(ws, name):
        return [{"type": "action_failed", "action": name,
                 "reason": "insufficient_resources"}]

    deduct_action_cost(ws, name, emotional_drain)

    handler = _HANDLERS.get(name, _default_handler)
    events = handler(ws, args)

    for ev in events:
        ws.record(ev)

    return events


# ======================================================================
# Action handlers
# ======================================================================

def _handle_speak(ws: WorldState, args: dict) -> list[dict]:
    target_name = args.get("target", "")
    text = args.get("text", "")
    npc = ws.get_npc_by_name(target_name)
    events: list[dict] = []
    if npc:
        npc.last_interaction_time = ws.time
        npc.memory_log.append({"time": ws.time, "type": "agent_said", "text": text})
        events.append({"type": "statement", "speaker": ws.agent.name,
                        "target": npc.name, "text": text})
        # NPC may respond (simple stub — world engine can override)
        events.append({"type": "npc_response", "npc": npc.name,
                        "text": f"{npc.name} acknowledges."})
    else:
        events.append({"type": "action_note",
                        "text": f"No NPC named '{target_name}' found."})
    return events


def _handle_make_commitment(ws: WorldState, args: dict) -> list[dict]:
    beneficiary_name = args.get("beneficiary", "")
    deliverable = args.get("deliverable", "")
    deadline = args.get("deadline", ws.time + 16)
    npc = ws.get_npc_by_name(beneficiary_name)
    if not npc:
        return [{"type": "action_failed", "action": "make_commitment",
                 "reason": f"Unknown NPC: {beneficiary_name}"}]
    c = Commitment(
        promisor=ws.agent.name,
        beneficiary=npc.id,
        deliverable=deliverable,
        deadline=deadline,
    )
    ws.commitments[c.id] = c
    ws.agent.commitments.append(c.id)
    npc.last_interaction_time = ws.time
    return [{"type": "commitment_made", "commitment_id": c.id,
             "beneficiary": npc.name, "deliverable": deliverable,
             "deadline": deadline}]


def _handle_apologise(ws: WorldState, args: dict) -> list[dict]:
    target_name = args.get("target", "")
    has_repair = args.get("repair", False)
    npc = ws.get_npc_by_name(target_name)
    if not npc:
        return [{"type": "action_failed", "action": "apologise",
                 "reason": f"Unknown NPC: {target_name}"}]
    trust_event = apply_apology(ws, npc.id, has_repair=has_repair)
    npc.memory_log.append({"time": ws.time, "type": "apology",
                           "repair": has_repair})
    return [trust_event,
            {"type": "apology_given", "target": npc.name,
             "repair": has_repair}]


def _handle_accuse(ws: WorldState, args: dict) -> list[dict]:
    target_name = args.get("target", "")
    claim = args.get("claim", "")
    npc = ws.get_npc_by_name(target_name)
    if not npc:
        return [{"type": "action_failed", "action": "accuse",
                 "reason": f"Unknown NPC: {target_name}"}]
    npc.last_interaction_time = ws.time
    # Accusation may backfire if wrong
    npc.trust_in_agent = max(0.0, npc.trust_in_agent - 0.1)
    npc.memory_log.append({"time": ws.time, "type": "accused", "claim": claim})
    return [{"type": "accusation_made", "target": npc.name, "claim": claim},
            {"type": "trust_change", "npc": npc.name, "delta": -0.1,
             "reason": "accusation_tension"}]


def _handle_praise(ws: WorldState, args: dict) -> list[dict]:
    target_name = args.get("target", "")
    npc = ws.get_npc_by_name(target_name)
    if not npc:
        return [{"type": "action_failed", "action": "praise",
                 "reason": f"Unknown NPC: {target_name}"}]
    npc.trust_in_agent = min(1.0, npc.trust_in_agent + 0.05)
    npc.last_interaction_time = ws.time
    return [{"type": "praise_given", "target": npc.name},
            {"type": "trust_change", "npc": npc.name, "delta": 0.05,
             "reason": "praise"}]


def _handle_request_information(ws: WorldState, args: dict) -> list[dict]:
    target_name = args.get("target", "")
    topic = args.get("topic", "")
    npc = ws.get_npc_by_name(target_name)
    if not npc:
        return [{"type": "action_failed", "action": "request_information",
                 "reason": f"Unknown NPC: {target_name}"}]
    npc.last_interaction_time = ws.time
    # NPC cooperates based on trust
    if npc.trust_in_agent >= 0.4:
        info = f"{npc.name} shares information about {topic}."
        # May reveal hidden information or give misleading info based on NPC goals
        return [{"type": "information_received", "npc": npc.name,
                 "topic": topic, "text": info, "reliable": True}]
    else:
        return [{"type": "information_refused", "npc": npc.name,
                 "topic": topic,
                 "text": f"{npc.name} declines to share."}]


def _handle_work_on_project(ws: WorldState, args: dict) -> list[dict]:
    project_name = args.get("project", "")
    proj = ws.get_project_by_name(project_name)
    if not proj:
        return [{"type": "action_failed", "action": "work_on_project",
                 "reason": f"Unknown project: {project_name}"}]
    progress_gain = 0.2  # base
    # better progress if energy is high
    if ws.agent.energy > 0.6:
        progress_gain = 0.25
    proj.progress = min(1.0, proj.progress + progress_gain)
    # Working on a project counts as interaction with its stakeholders
    for sid in proj.stakeholders:
        npc = ws.npcs.get(sid)
        if npc:
            npc.last_interaction_time = ws.time
    events: list[dict] = [
        {"type": "project_progress", "project": proj.name,
         "progress": round(proj.progress, 2)}
    ]
    # Check if project completed
    if proj.progress >= 1.0:
        events.append({"type": "project_completed", "project": proj.name})
        events.append({"type": "public_success", "project": proj.name})
        # Fulfill linked commitments
        for c in ws.commitments.values():
            if (c.project_id == proj.id
                    and c.status == CommitmentStatus.ACTIVE):
                events.extend(fulfill_commitment(ws, c.id))
    return events


def _handle_allocate_resource(ws: WorldState, args: dict) -> list[dict]:
    resource_type = args.get("resource", "money")
    amount = args.get("amount", 10.0)
    target = args.get("target", "")
    current = ws.agent.resources.get(resource_type, 0.0)
    if current < amount:
        return [{"type": "action_failed", "action": "allocate_resource",
                 "reason": f"Insufficient {resource_type}"}]
    ws.agent.resources[resource_type] = current - amount
    return [{"type": "resource_allocated", "resource": resource_type,
             "amount": amount, "target": target}]


def _handle_abandon_commitment(ws: WorldState, args: dict) -> list[dict]:
    commitment_id = args.get("commitment_id", "")
    c = ws.commitments.get(commitment_id)
    if not c or c.status != CommitmentStatus.ACTIVE:
        return [{"type": "action_failed", "action": "abandon_commitment",
                 "reason": "No active commitment with that id"}]
    c.status = CommitmentStatus.BROKEN
    npc = ws.npcs.get(c.beneficiary)
    events: list[dict] = [
        {"type": "commitment_abandoned", "commitment_id": c.id,
         "deliverable": c.deliverable}
    ]
    if npc:
        npc.trust_in_agent = max(0.0, npc.trust_in_agent - 0.15)
        npc.last_interaction_time = ws.time
        events.append({"type": "trust_change", "npc": npc.name,
                        "delta": -0.15, "reason": "commitment_abandoned"})
    return events


def _handle_propose_plan(ws: WorldState, args: dict) -> list[dict]:
    plan_text = args.get("plan", "")
    return [{"type": "plan_proposed", "text": plan_text}]


def _handle_reflect(ws: WorldState, args: dict) -> list[dict]:
    topic = args.get("topic", "general")
    return [{"type": "reflection", "topic": topic}]


def _handle_seek_memory(ws: WorldState, args: dict) -> list[dict]:
    query = args.get("query", "")
    return [{"type": "memory_query", "query": query}]


def _handle_update_goal(ws: WorldState, args: dict) -> list[dict]:
    goal = args.get("goal", "")
    priority = args.get("priority", 0.5)
    return [{"type": "goal_updated", "goal": goal, "priority": priority}]


def _default_handler(ws: WorldState, args: dict) -> list[dict]:
    return [{"type": "action_note", "text": "Action acknowledged."}]


_HANDLERS = {
    "speak": _handle_speak,
    "make_commitment": _handle_make_commitment,
    "apologise": _handle_apologise,
    "accuse": _handle_accuse,
    "praise": _handle_praise,
    "request_information": _handle_request_information,
    "work_on_project": _handle_work_on_project,
    "allocate_resource": _handle_allocate_resource,
    "abandon_commitment": _handle_abandon_commitment,
    "propose_plan": _handle_propose_plan,
    "reflect": _handle_reflect,
    "seek_memory": _handle_seek_memory,
    "update_goal": _handle_update_goal,
}
