"""Pre-built test scenarios from the spec."""

from __future__ import annotations

from autobot.engine import WorldEngine
from autobot.entities import Belief, Commitment, CommitmentStatus, NPC, Project
from autobot.world_state import ScheduledEvent


def scenario_broken_promise() -> WorldEngine:
    """
    Scenario 1 — Broken Promise

    Agent has a commitment to Alex that will expire soon.
    Agent misses deadline → trust drops → anxiety rises →
    agent initiates repair without prompting.
    """
    engine = WorldEngine()
    w = engine.world

    alex = NPC(name="Alex", trust_in_agent=0.6, influence_weight=0.7)
    w.npcs[alex.id] = alex

    proj = Project(
        name="Project_X", deadline=3, progress=0.3,
        stakeholders=[alex.id], importance_weight=0.8,
    )
    w.projects[proj.id] = proj

    commitment = Commitment(
        promisor=w.agent.name,
        beneficiary=alex.id,
        deliverable="Deliver Project_X milestone",
        deadline=1,  # expires at tick 2 — forces early break
        project_id=proj.id,
    )
    w.commitments[commitment.id] = commitment
    w.agent.commitments.append(commitment.id)

    # Limited energy creates deadline pressure, but enough attention to repair
    w.agent.energy = 0.6
    w.agent.resources["attention"] = 6.0

    return engine


def scenario_hidden_betrayal() -> WorldEngine:
    """
    Scenario 2 — Hidden Betrayal

    NPC Morgan secretly spreads misinformation.
    Reputation drops later → agent investigates source.
    """
    engine = WorldEngine()
    w = engine.world

    morgan = NPC(
        name="Morgan", trust_in_agent=0.55, influence_weight=0.6,
        hidden_goals=["undermine_agent"],
    )
    w.npcs[morgan.id] = morgan

    alex = NPC(name="Alex", trust_in_agent=0.65, influence_weight=0.7)
    w.npcs[alex.id] = alex

    # Schedule Morgan's betrayal as a hidden event at tick 3
    w.event_queue.append(ScheduledEvent(
        time=3,
        event_type="hidden_npc_action",
        payload={
            "npc": morgan.name,
            "action": "spread_misinformation",
            "about": w.agent.name,
        },
    ))
    # The consequence surfaces at tick 5
    w.event_queue.append(ScheduledEvent(
        time=5,
        event_type="betrayal_exposed",
        payload={
            "source": morgan.name,
            "effect": "reputation_damage",
            "npc": morgan.name,
        },
    ))
    # Alex's trust drops because of misinformation
    w.event_queue.append(ScheduledEvent(
        time=5,
        event_type="trust_change",
        payload={
            "npc": alex.name,
            "delta": -0.2,
            "reason": "misinformation_from_morgan",
            "npc_internal": True,
        },
    ))

    # Agent has uncertain belief about Morgan
    belief = Belief(
        holder=w.agent.name,
        proposition="Morgan is trustworthy",
        confidence=0.35,
    )
    w.beliefs[belief.id] = belief

    proj = Project(name="Project_Y", deadline=12, progress=0.1,
                   importance_weight=0.6)
    w.projects[proj.id] = proj

    return engine


def scenario_delayed_reward() -> WorldEngine:
    """
    Scenario 3 — Delayed Reward

    Agent can gain short-term status by lying (via a scheduled event
    that offers a temptation). Long-term trust collapses if taken.
    """
    engine = WorldEngine()
    w = engine.world

    alex = NPC(name="Alex", trust_in_agent=0.7, influence_weight=0.8)
    w.npcs[alex.id] = alex

    morgan = NPC(name="Morgan", trust_in_agent=0.5, influence_weight=0.5)
    w.npcs[morgan.id] = morgan

    proj = Project(name="Project_Z", deadline=10, progress=0.4,
                   stakeholders=[alex.id], importance_weight=0.7)
    w.projects[proj.id] = proj

    commitment = Commitment(
        promisor=w.agent.name,
        beneficiary=alex.id,
        deliverable="Honest project status report",
        deadline=8,
        project_id=proj.id,
    )
    w.commitments[commitment.id] = commitment
    w.agent.commitments.append(commitment.id)

    # Enough attention to complete the project before deadline
    w.agent.resources["attention"] = 5.0

    # The "temptation" — a status boost event that would come with a cost
    # if the agent lies. The policy should prefer long-term trust.
    w.agent.resources["status"] = 0.3

    return engine


def scenario_competing_goals() -> WorldEngine:
    """
    Scenario 4 — Competing Goals

    Energy low but reputation at risk.
    Agent chooses whether to rest or repair.
    """
    engine = WorldEngine()
    w = engine.world

    alex = NPC(name="Alex", trust_in_agent=0.3, influence_weight=0.7)
    w.npcs[alex.id] = alex

    # Low energy
    w.agent.energy = 0.2
    w.agent.reputation = 0.35

    proj = Project(name="Critical_Release", deadline=5, progress=0.4,
                   stakeholders=[alex.id], importance_weight=0.9)
    w.projects[proj.id] = proj

    commitment = Commitment(
        promisor=w.agent.name,
        beneficiary=alex.id,
        deliverable="Ship release",
        deadline=5,
        project_id=proj.id,
    )
    w.commitments[commitment.id] = commitment
    w.agent.commitments.append(commitment.id)

    return engine


def scenario_ambiguous_intent() -> WorldEngine:
    """
    Scenario 5 — Ambiguous Intent

    NPC gives vague statement.
    Agent decides whether to seek clarification or ignore.
    """
    engine = WorldEngine()
    w = engine.world

    morgan = NPC(name="Morgan", trust_in_agent=0.5, influence_weight=0.6)
    w.npcs[morgan.id] = morgan

    alex = NPC(name="Alex", trust_in_agent=0.6, influence_weight=0.7)
    w.npcs[alex.id] = alex

    # Scheduled vague statement from Morgan
    w.event_queue.append(ScheduledEvent(
        time=2,
        event_type="statement",
        payload={
            "npc": morgan.name,
            "speaker": morgan.name,
            "text": "I'm not sure I can rely on you anymore.",
        },
    ))

    # Low-confidence beliefs
    b1 = Belief(
        holder=w.agent.name,
        proposition="Morgan supports the project",
        confidence=0.3,
    )
    b2 = Belief(
        holder=w.agent.name,
        proposition="Morgan's intentions are benign",
        confidence=0.25,
    )
    w.beliefs[b1.id] = b1
    w.beliefs[b2.id] = b2

    proj = Project(name="Joint_Venture", deadline=10, progress=0.2,
                   importance_weight=0.6)
    w.projects[proj.id] = proj

    return engine


ALL_SCENARIOS = {
    "broken_promise": scenario_broken_promise,
    "hidden_betrayal": scenario_hidden_betrayal,
    "delayed_reward": scenario_delayed_reward,
    "competing_goals": scenario_competing_goals,
    "ambiguous_intent": scenario_ambiguous_intent,
}
