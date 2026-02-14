"""Simulation loop — runs the world autonomously without user input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from autobot.agent_policy import decide_action
from autobot.engine import CycleResult, WorldEngine


@dataclass
class SimulationConfig:
    max_cycles: int = 50
    verbose: bool = True
    log_file: str | None = None


@dataclass
class SimulationLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, cycle: int, result: CycleResult, action: dict | None) -> None:
        self.entries.append({
            "cycle": cycle,
            "time": result.observation["time"],
            "action": action,
            "events": result.events,
            "energy": result.observation["self_state"]["energy"],
            "reputation": result.observation["self_state"]["reputation"],
            "emotions": result.emotions.to_dict(),
            "arousal": result.affect.arousal,
            "valence": result.affect.valence,
            "new_episode": result.new_episode,
            "goals": result.goals_text,
        })

    def to_json(self) -> str:
        return json.dumps(self.entries, indent=2, default=str)


def run_simulation(
    engine: WorldEngine,
    config: SimulationConfig | None = None,
    on_cycle: Callable[[int, CycleResult, dict | None], None] | None = None,
) -> SimulationLog:
    """
    The autonomous self-prompt loop.

    Each cycle:
      1. Observe world (tick with no action to get baseline events)
      2. Amygdala evaluates
      3. Reflective mood updates
      4. Memory retrieval triggered
      5. Goal engine reassesses
      6. Agent policy selects action
      7. Execute action via another tick

    No user input required. The world continues evolving through time events.
    """
    cfg = config or SimulationConfig()
    log = SimulationLog()

    if cfg.verbose:
        _print_header(engine)

    for cycle in range(1, cfg.max_cycles + 1):
        # --- Agent decides ---
        action = decide_action(
            ws=engine.world,
            emotions=engine.emotions,
            affect=_last_affect(engine),
            goal_engine=engine.goal_engine,
            memory=engine.memory,
            observation={},
        )

        # --- World ticks with the action ---
        result = engine.tick(agent_action=action)

        # --- Log ---
        log.append(cycle, result, action)

        if cfg.verbose:
            _print_cycle(cycle, result, action)

        if on_cycle:
            on_cycle(cycle, result, action)

    if cfg.verbose:
        _print_footer(engine, log)

    if cfg.log_file:
        with open(cfg.log_file, "w") as f:
            f.write(log.to_json())

    return log


def _last_affect(engine: WorldEngine):
    """Get a neutral affect for the first tick."""
    from autobot.emotions import AffectState
    return AffectState()


# ======================================================================
# Pretty printing
# ======================================================================

def _print_header(engine: WorldEngine) -> None:
    w = engine.world
    print("=" * 70)
    print("  AUTOBOT — AI World Model Simulation")
    print("=" * 70)
    print(f"  Agent: {w.agent.name}")
    print(f"  NPCs:  {', '.join(n.name for n in w.npcs.values())}")
    print(f"  Projects: {', '.join(p.name for p in w.projects.values())}")
    print("=" * 70)
    print()


def _print_cycle(cycle: int, result: CycleResult, action: dict | None) -> None:
    obs = result.observation
    emo = result.emotions
    aff = result.affect

    print(f"--- Cycle {cycle} | {obs['time']} ---")

    if action:
        name = action.get("name", "?")
        args = action.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  ACTION: {name}({args_str})")

    # Significant events
    for ev in result.events:
        etype = ev.get("type", "")
        if etype in ("trust_change", "reputation_change", "commitment_broken",
                      "commitment_fulfilled", "deadline_passed",
                      "project_completed", "action_failed",
                      "npc_response", "statement", "information_received",
                      "information_refused", "apology_given"):
            detail = _event_summary(ev)
            print(f"  EVENT: {detail}")

    print(f"  STATE: energy={obs['self_state']['energy']:.2f}  "
          f"rep={obs['self_state']['reputation']:.2f}  "
          f"mood={emo.dominant_mood()}  "
          f"arousal={aff.arousal:.2f}  valence={aff.valence:.2f}")

    if result.new_episode:
        print("  ** New episodic memory created **")

    print()


def _event_summary(ev: dict) -> str:
    etype = ev.get("type", "")
    if etype == "trust_change":
        return (f"Trust with {ev.get('npc','?')}: "
                f"{ev.get('delta',0):+.2f} ({ev.get('reason','?')})")
    if etype == "reputation_change":
        return f"Reputation: {ev.get('old',0):.2f} -> {ev.get('new',0):.2f}"
    if etype == "commitment_broken":
        return (f"BROKEN: commitment to {ev.get('beneficiary','?')} "
                f"({ev.get('deliverable','?')})")
    if etype == "commitment_fulfilled":
        return (f"FULFILLED: commitment to {ev.get('beneficiary','?')} "
                f"({ev.get('deliverable','?')})")
    if etype == "deadline_passed":
        return f"DEADLINE PASSED: {ev.get('project','?')}"
    if etype == "project_completed":
        return f"PROJECT COMPLETED: {ev.get('project','?')}"
    if etype == "action_failed":
        return f"FAILED: {ev.get('action','?')} — {ev.get('reason','?')}"
    if etype == "npc_response":
        return f"{ev.get('npc','?')}: \"{ev.get('text','')}\""
    if etype == "statement":
        return (f"{ev.get('speaker','?')} -> {ev.get('target','?')}: "
                f"\"{ev.get('text','')}\"")
    if etype == "information_received":
        return f"INFO from {ev.get('npc','?')}: {ev.get('text','')}"
    if etype == "information_refused":
        return f"REFUSED by {ev.get('npc','?')}: {ev.get('text','')}"
    if etype == "apology_given":
        repair = " (with repair)" if ev.get("repair") else ""
        return f"Apologised to {ev.get('target','?')}{repair}"
    return f"{etype}: {ev}"


def _print_footer(engine: WorldEngine, log: SimulationLog) -> None:
    w = engine.world
    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Final energy:     {w.agent.energy:.2f}")
    print(f"  Final reputation: {w.agent.reputation:.2f}")
    for npc in w.npcs.values():
        print(f"  Trust ({npc.name}):   {npc.trust_in_agent:.2f}")
    print(f"  Episodes created: {len(engine.memory.episodes)}")
    print(f"  Goals total:      {len(engine.goal_engine.goals)}")
    active = engine.goal_engine.active_goals()
    print(f"  Goals active:     {len(active)}")
    completed = [g for g in engine.goal_engine.goals if g.completed]
    print(f"  Goals completed:  {len(completed)}")
    abandoned = [g for g in engine.goal_engine.goals if g.abandoned]
    print(f"  Goals abandoned:  {len(abandoned)}")
    print("=" * 70)
