"""World Engine — the master orchestrator.

The LLM does not modify world state directly.
It proposes actions.
The World Engine applies rules.
The world returns consequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autobot.actions import execute_action, validate_action, ActionError
from autobot.emotions import (
    AffectState,
    EmotionalState,
    emotional_energy_drain,
    evaluate_amygdala,
    update_reflective_emotions,
)
from autobot.goals import GoalEngine
from autobot.mechanics import (
    apply_trust_decay,
    check_overdue_commitments,
    tag_irreversibility,
    update_reputation,
)
from autobot.memory import EpisodicMemory
from autobot.observation import build_observation
from autobot.world_state import WorldState


@dataclass
class CycleResult:
    """Everything produced by a single world tick."""
    observation: dict[str, Any]
    events: list[dict[str, Any]]
    affect: AffectState
    emotions: EmotionalState
    goals_text: str
    memory_text: str
    new_episode: bool
    agent_prompt: str
    agent_thinking: str = ""       # inner monologue from the LLM
    used_llm: bool = False         # whether the LLM was the decision-maker


@dataclass
class WorldEngine:
    """
    Deterministic rule-based engine.
    Each cycle:  advance time → evaluate rules → process agent action →
                 update emotions → update goals → build observation
    """

    world: WorldState = field(default_factory=WorldState)
    emotions: EmotionalState = field(default_factory=EmotionalState)
    goal_engine: GoalEngine = field(default_factory=GoalEngine)
    memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    cycle_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(self, agent_action: dict[str, Any] | None = None) -> CycleResult:
        """
        Run one world cycle.

        1. Advance time
        2. Fire scheduled events & check rules
        3. Process agent action (if any)
        4. Amygdala evaluates
        5. Reflective mood updates
        6. Memory writes if salient
        7. Goal engine reassesses
        8. Build observation packet
        9. Compose agent prompt
        """
        self.cycle_count += 1
        all_events: list[dict[str, Any]] = []

        # 1. Advance time
        time_events = self.world.advance_time(ticks=1)
        # Apply side-effects from scheduled events (e.g. trust changes)
        for ev in time_events:
            self._apply_event_side_effects(ev)
        all_events.extend(time_events)

        # 2. Rule evaluation
        all_events.extend(self._evaluate_rules())

        # 3. Agent action
        if agent_action:
            try:
                validate_action(agent_action)
                # Emotional drain from previous cycle's affect
                drain = emotional_energy_drain(
                    evaluate_amygdala(all_events, self.world)
                )
                action_events = execute_action(
                    self.world, agent_action, emotional_drain=drain,
                )
                all_events.extend(action_events)
            except ActionError as exc:
                all_events.append({
                    "type": "action_error",
                    "message": str(exc),
                })

        # Tag irreversibility
        all_events = [tag_irreversibility(e) for e in all_events]

        # Record all events
        for ev in all_events:
            self.world.record(ev)

        # 4. Amygdala
        affect = evaluate_amygdala(all_events, self.world)

        # 5. Reflective emotions
        self.emotions = update_reflective_emotions(
            self.emotions, affect, self.world,
        )

        # Apply emotional energy drain
        drain = emotional_energy_drain(affect)
        if drain > 0:
            self.world.agent.energy = max(
                0.0, self.world.agent.energy - drain,
            )

        # 6. Memory
        goal_ctx = None
        top = self.goal_engine.top_goal(self.emotions)
        if top:
            goal_ctx = top.description
        episode = self.memory.maybe_create_episode(
            all_events, affect, self.world, goal_context=goal_ctx,
        )

        # 7. Goals
        self.goal_engine.auto_generate(self.world, self.emotions)

        # 8. Observation
        observation = build_observation(self.world, all_events)

        # 9. Agent prompt
        agent_prompt = self._compose_prompt(observation, affect)

        # Store mood snapshot
        self.world.agent.mood_history.append({
            "cycle": self.cycle_count,
            "emotions": self.emotions.to_dict(),
            "arousal": affect.arousal,
            "valence": affect.valence,
        })

        return CycleResult(
            observation=observation,
            events=all_events,
            affect=affect,
            emotions=self.emotions,
            goals_text=self.goal_engine.to_prompt_text(self.emotions),
            memory_text=self.memory.to_prompt_text(),
            new_episode=episode is not None,
            agent_prompt=agent_prompt,
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _apply_event_side_effects(self, ev: dict[str, Any]) -> None:
        """Apply state mutations for events fired from the scheduled queue."""
        etype = ev.get("type", "")
        if etype == "trust_change":
            # Scheduled trust changes (e.g. from NPC betrayal) need to
            # actually modify the NPC's trust value.
            npc_name = ev.get("npc")
            delta = ev.get("delta", 0)
            if npc_name and delta:
                npc = self.world.get_npc_by_name(npc_name)
                if npc:
                    npc.trust_in_agent = max(
                        0.0, min(1.0, npc.trust_in_agent + delta)
                    )
                    npc.clamp()

    def _evaluate_rules(self) -> list[dict[str, Any]]:
        """Run all automatic world rules each tick."""
        events: list[dict[str, Any]] = []

        # Overdue commitments
        events.extend(check_overdue_commitments(self.world))

        # Trust decay (runs every tick; function filters by time)
        events.extend(apply_trust_decay(self.world))

        # Reputation recomputation
        rep_ev = update_reputation(self.world)
        if rep_ev:
            events.append(rep_ev)

        # Overdue projects
        for proj in self.world.projects.values():
            if proj.is_overdue(self.world.time) and proj.progress < 1.0:
                ev = {"type": "deadline_passed", "project": proj.name}
                # Only fire once — check history
                already_fired = any(
                    e.get("type") == "deadline_passed"
                    and e.get("project") == proj.name
                    for e in self.world.history
                )
                if not already_fired:
                    events.append(ev)

        return events

    def _compose_prompt(
        self,
        observation: dict[str, Any],
        affect: AffectState,
    ) -> str:
        """
        Build the full prompt that would be sent to the LLM agent.
        This is the self-prompt loop's output.
        """
        import json

        mood = self.emotions.dominant_mood()
        top_goal = self.goal_engine.top_goal(self.emotions)

        # Retrieve relevant memories
        entities = [e.get("npc", e.get("target", ""))
                    for e in observation.get("events", [])]
        memories = self.memory.retrieve(
            entities=entities,
            tags=affect.salience_tags,
            current_arousal=affect.arousal,
            limit=3,
        )
        mem_lines = []
        for m in memories:
            flag = " [UNRESOLVED]" if m.unresolved else ""
            mem_lines.append(f"  - {m.summary}{flag}")

        sections = [
            "=== WORLD OBSERVATION ===",
            json.dumps(observation, indent=2),
            "",
            "=== EMOTIONAL STATE ===",
            f"  Dominant mood: {mood}",
            f"  Anxiety: {self.emotions.anxiety:.2f}",
            f"  Optimism: {self.emotions.optimism:.2f}",
            f"  Irritability: {self.emotions.irritability:.2f}",
            f"  Social warmth: {self.emotions.social_warmth:.2f}",
            f"  Avoidance bias: {self.emotions.avoidance_bias:.2f}",
            f"  Risk tolerance: {self.emotions.risk_tolerance:.2f}",
            "",
            f"  Fast affect — arousal: {affect.arousal:.2f}, "
            f"valence: {affect.valence:.2f}",
            f"  Salience tags: {affect.salience_tags}",
            f"  Attention focus: {affect.attention_focus}",
            "",
            "=== RELEVANT MEMORIES ===",
            *(mem_lines if mem_lines else ["  (none)"]),
            "",
            "=== ACTIVE GOALS (ranked) ===",
            self.goal_engine.to_prompt_text(self.emotions),
            "",
            "=== INSTRUCTIONS ===",
            "You are an autonomous agent in a social world.",
            "Choose ONE action to take this cycle.",
            f"Your top priority: {top_goal.description if top_goal else 'none'}",
            "Your emotional state should influence your decision.",
            "",
            'Respond with: {"type": "action", "name": "<action>", "args": {...}}',
        ]

        return "\n".join(sections)
