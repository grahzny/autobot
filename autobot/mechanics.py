"""World mechanics: trust dynamics, reputation, scarcity, irreversibility."""

from __future__ import annotations

from typing import Any

from autobot.entities import Commitment, CommitmentStatus
from autobot.world_state import WorldState


# ======================================================================
# 4.1  Trust Dynamics
# ======================================================================

TRUST_FULFILL_DELTA = 0.15
TRUST_BREAK_DELTA = 0.25
TRUST_APOLOGY_MINOR = 0.05
TRUST_REPAIR_PARTIAL = 0.10
TRUST_DECAY_RATE = 0.02          # per day with no interaction


def apply_trust_fulfillment(ws: WorldState, npc_id: str) -> dict[str, Any]:
    npc = ws.npcs[npc_id]
    old = npc.trust_in_agent
    npc.trust_in_agent = min(1.0, npc.trust_in_agent + TRUST_FULFILL_DELTA)
    npc.clamp()
    npc.last_interaction_time = ws.time
    delta = npc.trust_in_agent - old
    return {"type": "trust_change", "npc": npc.name, "delta": round(delta, 3),
            "reason": "commitment_fulfilled"}


def apply_trust_break(ws: WorldState, npc_id: str) -> dict[str, Any]:
    npc = ws.npcs[npc_id]
    old = npc.trust_in_agent
    npc.trust_in_agent = max(0.0, npc.trust_in_agent - TRUST_BREAK_DELTA)
    npc.clamp()
    npc.last_interaction_time = ws.time
    delta = npc.trust_in_agent - old
    return {"type": "trust_change", "npc": npc.name, "delta": round(delta, 3),
            "reason": "commitment_broken"}


def apply_apology(ws: WorldState, npc_id: str, has_repair: bool) -> dict[str, Any]:
    npc = ws.npcs[npc_id]
    old = npc.trust_in_agent
    gain = TRUST_REPAIR_PARTIAL if has_repair else TRUST_APOLOGY_MINOR
    npc.trust_in_agent = min(1.0, npc.trust_in_agent + gain)
    npc.clamp()
    npc.last_interaction_time = ws.time
    delta = npc.trust_in_agent - old
    return {"type": "trust_change", "npc": npc.name, "delta": round(delta, 3),
            "reason": "apology" + ("_with_repair" if has_repair else "")}


def apply_trust_decay(ws: WorldState) -> list[dict[str, Any]]:
    """Called once per day for NPCs the agent hasn't interacted with."""
    events: list[dict[str, Any]] = []
    for npc in ws.npcs.values():
        ticks_since = ws.time - npc.last_interaction_time
        # decay kicks in after ~16 ticks (one full day)
        if ticks_since >= 16:
            old = npc.trust_in_agent
            npc.trust_in_agent = max(0.0, npc.trust_in_agent - TRUST_DECAY_RATE)
            npc.clamp()
            if old != npc.trust_in_agent:
                events.append({
                    "type": "trust_change", "npc": npc.name,
                    "delta": round(npc.trust_in_agent - old, 3),
                    "reason": "decay",
                })
    return events


# ======================================================================
# 4.2  Reputation System
# ======================================================================

def recompute_reputation(ws: WorldState) -> float:
    """Reputation = weighted average of NPC trust + history bonuses/penalties."""
    if not ws.npcs:
        return ws.agent.reputation

    total_weight = 0.0
    weighted_trust = 0.0
    for npc in ws.npcs.values():
        w = npc.influence_weight
        weighted_trust += npc.trust_in_agent * w
        total_weight += w

    base = weighted_trust / total_weight if total_weight > 0 else 0.5

    # bonus/penalty from recent public events
    bonus = 0.0
    recent = [e for e in ws.history if e.get("world_time", 0) > ws.time - 16]
    for ev in recent:
        if ev.get("type") == "public_contradiction":
            bonus -= 0.05
        elif ev.get("type") == "public_success":
            bonus += 0.03

    new_rep = max(0.0, min(1.0, base + bonus))
    return round(new_rep, 3)


def update_reputation(ws: WorldState) -> dict[str, Any] | None:
    old = ws.agent.reputation
    ws.agent.reputation = recompute_reputation(ws)
    ws.agent.clamp()
    delta = round(ws.agent.reputation - old, 3)
    if abs(delta) >= 0.001:
        return {"type": "reputation_change", "old": old,
                "new": ws.agent.reputation, "delta": delta}
    return None


# ======================================================================
# 4.3  Scarcity Model
# ======================================================================

# Action costs: (energy_cost, time_cost, attention_cost)
ACTION_COSTS: dict[str, tuple[float, float, float]] = {
    # Social
    "speak":               (0.05, 0.5, 1),
    "make_commitment":     (0.05, 0.5, 1),
    "apologise":           (0.08, 0.5, 1),
    "accuse":              (0.10, 1.0, 1),
    "praise":              (0.03, 0.5, 1),
    "request_information": (0.04, 0.5, 1),
    # Operational
    "work_on_project":     (0.15, 2.0, 1),
    "allocate_resource":   (0.05, 0.5, 1),
    "abandon_commitment":  (0.06, 0.5, 1),
    "propose_plan":        (0.08, 1.0, 1),
    # Cognitive
    "reflect":             (0.03, 0.5, 0),
    "seek_memory":         (0.02, 0.0, 0),
    "update_goal":         (0.02, 0.0, 0),
}


def can_afford_action(ws: WorldState, action_name: str) -> bool:
    costs = ACTION_COSTS.get(action_name, (0.05, 0.5, 1))
    a = ws.agent
    return (
        a.energy >= costs[0]
        and a.resources["time"] >= costs[1]
        and a.resources["attention"] >= costs[2]
    )


def deduct_action_cost(
    ws: WorldState, action_name: str, emotional_drain: float = 0.0,
) -> None:
    """Deduct energy/time/attention.  emotional_drain adds extra energy cost."""
    costs = ACTION_COSTS.get(action_name, (0.05, 0.5, 1))
    a = ws.agent
    a.energy -= costs[0] + emotional_drain
    a.resources["time"] -= costs[1]
    a.resources["attention"] -= costs[2]
    a.clamp()
    for k in a.resources:
        a.resources[k] = max(0.0, a.resources[k])


# ======================================================================
# 4.4  Commitment lifecycle
# ======================================================================

def check_overdue_commitments(ws: WorldState) -> list[dict[str, Any]]:
    """Break commitments whose deadline has passed."""
    events: list[dict[str, Any]] = []
    for c in ws.commitments.values():
        if c.is_overdue(ws.time) and c.status == CommitmentStatus.ACTIVE:
            c.status = CommitmentStatus.BROKEN
            ev = apply_trust_break(ws, c.beneficiary)
            ev["commitment_id"] = c.id
            ev["deliverable"] = c.deliverable
            events.append(ev)
            events.append({
                "type": "commitment_broken",
                "commitment_id": c.id,
                "deliverable": c.deliverable,
                "beneficiary": ws.npcs[c.beneficiary].name,
            })
    return events


def fulfill_commitment(ws: WorldState, commitment_id: str) -> list[dict[str, Any]]:
    c = ws.commitments[commitment_id]
    c.status = CommitmentStatus.FULFILLED
    ev = apply_trust_fulfillment(ws, c.beneficiary)
    ev["commitment_id"] = c.id
    return [
        ev,
        {"type": "commitment_fulfilled", "commitment_id": c.id,
         "deliverable": c.deliverable,
         "beneficiary": ws.npcs[c.beneficiary].name},
    ]


# ======================================================================
# 4.5  Irreversibility markers
# ======================================================================

IRREVERSIBLE_EVENT_TYPES = {
    "commitment_broken",
    "public_contradiction",
    "betrayal_exposed",
    "deadline_passed",
}


def tag_irreversibility(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") in IRREVERSIBLE_EVENT_TYPES:
        event["irreversible"] = True
    return event
