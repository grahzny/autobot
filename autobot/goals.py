"""Goal Engine — generates, ranks, and manages competing goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autobot.emotions import EmotionalState
from autobot.world_state import WorldState


@dataclass
class Goal:
    id: str
    description: str
    priority: float = 0.5          # 0–1, higher = more urgent
    category: str = "general"      # trust, reputation, project, uncertainty, energy
    target_entity: str | None = None
    created_at: int = 0
    completed: bool = False
    abandoned: bool = False
    failure_count: int = 0         # times agent tried and failed

    def effective_priority(self, emotions: EmotionalState) -> float:
        """Priority modulated by emotional state."""
        p = self.priority

        if self.category == "trust":
            # anxiety boosts trust-repair urgency
            p += emotions.anxiety * 0.2
            p += emotions.social_warmth * 0.1
        elif self.category == "reputation":
            p += emotions.anxiety * 0.15
            p -= emotions.avoidance_bias * 0.1
        elif self.category == "project":
            p += (1 - emotions.avoidance_bias) * 0.1
            p -= emotions.irritability * 0.05
        elif self.category == "uncertainty":
            # high anxiety drives uncertainty resolution
            p += emotions.anxiety * 0.25
        elif self.category == "energy":
            # low risk tolerance favours rest
            p += (1 - emotions.risk_tolerance) * 0.2

        # Penalty for repeated failure — drives abandonment
        p -= self.failure_count * 0.1

        return max(0.0, min(1.0, round(p, 3)))


@dataclass
class GoalEngine:
    goals: list[Goal] = field(default_factory=list)
    _id_counter: int = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"goal_{self._id_counter}"

    def add_goal(
        self,
        description: str,
        priority: float = 0.5,
        category: str = "general",
        target_entity: str | None = None,
        created_at: int = 0,
    ) -> Goal:
        g = Goal(
            id=self._next_id(),
            description=description,
            priority=priority,
            category=category,
            target_entity=target_entity,
            created_at=created_at,
        )
        self.goals.append(g)
        return g

    def active_goals(self) -> list[Goal]:
        return [g for g in self.goals if not g.completed and not g.abandoned]

    def ranked_goals(self, emotions: EmotionalState) -> list[Goal]:
        """Return active goals sorted by emotion-modulated priority (desc)."""
        active = self.active_goals()
        active.sort(key=lambda g: g.effective_priority(emotions), reverse=True)
        return active

    def top_goal(self, emotions: EmotionalState) -> Goal | None:
        ranked = self.ranked_goals(emotions)
        return ranked[0] if ranked else None

    def complete_goal(self, goal_id: str) -> None:
        for g in self.goals:
            if g.id == goal_id:
                g.completed = True
                return

    def abandon_goal(self, goal_id: str) -> None:
        for g in self.goals:
            if g.id == goal_id:
                g.abandoned = True
                return

    def record_failure(self, goal_id: str) -> None:
        for g in self.goals:
            if g.id == goal_id:
                g.failure_count += 1
                # Auto-abandon after repeated failures
                if g.failure_count >= 3:
                    g.abandoned = True
                return

    # ----- Auto-generation from world state -----

    def auto_generate(self, ws: WorldState, emotions: EmotionalState) -> list[Goal]:
        """
        Scan world state and create goals the agent should care about.
        Also retires goals whose conditions no longer apply.
        Returns newly created goals.
        """
        # --- Retire completed goals ---
        for g in self.active_goals():
            if g.category == "project" and g.target_entity:
                proj = ws.get_project_by_name(g.target_entity)
                if proj and (proj.progress >= 1.0 or proj.is_overdue(ws.time)):
                    g.completed = proj.progress >= 1.0
                    g.abandoned = proj.is_overdue(ws.time) and proj.progress < 1.0
            elif g.category == "trust" and g.target_entity:
                npc = ws.get_npc_by_name(g.target_entity)
                if npc and npc.trust_in_agent >= 0.6:
                    g.completed = True
            elif g.category == "reputation":
                if ws.agent.reputation >= 0.5:
                    g.completed = True
            elif g.category == "energy":
                if ws.agent.energy >= 0.5:
                    g.completed = True

        new_goals: list[Goal] = []
        existing_descriptions = {g.description for g in self.goals}

        # Trust repair goals
        for npc in ws.npcs.values():
            if npc.trust_in_agent < 0.45:
                desc = f"Repair trust with {npc.name}"
                if desc not in existing_descriptions:
                    g = self.add_goal(
                        desc, priority=0.7, category="trust",
                        target_entity=npc.name, created_at=ws.time,
                    )
                    new_goals.append(g)

        # Reputation goal
        if ws.agent.reputation < 0.4:
            desc = "Restore damaged reputation"
            if desc not in existing_descriptions:
                g = self.add_goal(
                    desc, priority=0.7, category="reputation",
                    created_at=ws.time,
                )
                new_goals.append(g)

        # Project deadline goals
        for proj in ws.projects.values():
            if proj.progress < 1.0 and not proj.is_overdue(ws.time):
                remaining = proj.deadline - ws.time
                urgency = max(0.3, 1.0 - remaining / 16.0)
                desc = f"Complete {proj.name} before deadline"
                if desc not in existing_descriptions:
                    g = self.add_goal(
                        desc, priority=urgency, category="project",
                        target_entity=proj.name, created_at=ws.time,
                    )
                    new_goals.append(g)

        # Uncertainty resolution
        uncertain = [
            b for b in ws.beliefs.values()
            if b.holder == ws.agent.name and b.confidence < 0.4
        ]
        if len(uncertain) > 1 and emotions.anxiety > 0.4:
            desc = "Reduce uncertainty about unclear beliefs"
            if desc not in existing_descriptions:
                g = self.add_goal(
                    desc, priority=0.5, category="uncertainty",
                    created_at=ws.time,
                )
                new_goals.append(g)

        # Energy preservation
        if ws.agent.energy < 0.25:
            desc = "Preserve energy — rest"
            if desc not in existing_descriptions:
                g = self.add_goal(
                    desc, priority=0.6, category="energy",
                    created_at=ws.time,
                )
                new_goals.append(g)

        return new_goals

    def to_prompt_text(self, emotions: EmotionalState) -> str:
        """Render goals for the agent prompt."""
        ranked = self.ranked_goals(emotions)
        if not ranked:
            return "No active goals."
        lines = []
        for i, g in enumerate(ranked, 1):
            ep = g.effective_priority(emotions)
            lines.append(
                f"  {i}. [{g.category}] {g.description} "
                f"(priority={ep:.2f}, failures={g.failure_count})"
            )
        return "\n".join(lines)
