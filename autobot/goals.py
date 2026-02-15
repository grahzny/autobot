"""Goal Engine -- generates, ranks, and manages competing goals."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autobot.emotions import EmotionalState

if TYPE_CHECKING:
    from autobot.chat_state import ChatState
    from autobot.needs import NeedsState
    from autobot.repetition import RepetitionTracker

# Maximum number of active goals at any time
_MAX_ACTIVE_GOALS = 2


@dataclass
class Goal:
    id: str
    description: str
    priority: float = 0.5          # 0-1, higher = more urgent
    category: str = "general"      # relationship, exploration, connection, understanding, self, energy
    target_entity: str | None = None  # person name or topic
    created_at: float = 0.0
    completed: bool = False
    abandoned: bool = False
    failure_count: int = 0

    # Progress tracking
    progress_score: float = 0.0         # 0-1, how close to completion
    next_steps: list[str] = field(default_factory=list)
    stall_counter: int = 0              # ticks with no progress
    rumination_counter: int = 0         # times discussed without progress
    last_progress_at: float = 0.0       # timestamp of last progress
    source_need: str | None = None      # which need spawned this goal

    def effective_priority(self, emotions: EmotionalState) -> float:
        """Priority modulated by emotional state."""
        p = self.priority

        if self.category == "relationship":
            p += emotions.anxiety * 0.15
            p += emotions.social_warmth * 0.2
        elif self.category == "exploration":
            p += emotions.optimism * 0.15
            p -= emotions.avoidance_bias * 0.1
        elif self.category == "connection":
            p += emotions.social_warmth * 0.25
            p -= emotions.avoidance_bias * 0.15
        elif self.category == "understanding":
            p += emotions.anxiety * 0.2
        elif self.category == "self":
            p += emotions.anxiety * 0.1
            p += emotions.irritability * 0.15
        elif self.category == "energy":
            p += (1 - emotions.risk_tolerance) * 0.2

        # Penalty for repeated failure
        p -= self.failure_count * 0.1

        # Rumination penalty -- talking about it without progress
        if self.rumination_counter > 2:
            p -= 0.1 * min(self.rumination_counter - 2, 5)

        # Stall penalty -- stuck too long
        if self.stall_counter > 10:  # ~2.5 minutes with no progress
            p -= 0.15

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
        source_need: str | None = None,
    ) -> Goal:
        g = Goal(
            id=self._next_id(),
            description=description,
            priority=priority,
            category=category,
            target_entity=target_entity,
            created_at=_time.time(),
            source_need=source_need,
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
                if g.failure_count >= 3:
                    g.abandoned = True
                return

    def record_progress(
        self, goal_id: str, delta: float = 0.1, next_step: str | None = None,
    ) -> None:
        """Record concrete progress on a goal."""
        for g in self.goals:
            if g.id == goal_id:
                g.progress_score = min(1.0, g.progress_score + delta)
                g.stall_counter = 0
                g.last_progress_at = _time.time()
                if next_step:
                    g.next_steps.append(next_step)
                if g.progress_score >= 1.0:
                    g.completed = True
                return

    def tick_goals(self, repetition_tracker: RepetitionTracker | None = None) -> None:
        """Called each tick to update stall counters and detect rumination."""
        for g in self.active_goals():
            g.stall_counter += 1

            # Sync rumination counter from repetition tracker
            if repetition_tracker:
                mentions = repetition_tracker.detect_goal_rumination(g.description)
                if mentions > g.rumination_counter:
                    g.rumination_counter = mentions

            # Stall thresholds -- escalating consequences
            if g.stall_counter > 40 and g.progress_score < 0.2:
                # ~10 minutes stuck with no real progress -> abandon
                g.abandoned = True
            elif g.stall_counter > 80:
                # ~20 minutes -> significant deprioritization
                g.priority *= 0.7

    def _enforce_active_limit(self, emotions: EmotionalState) -> None:
        """Keep at most _MAX_ACTIVE_GOALS active; demote the rest."""
        active = self.active_goals()
        if len(active) <= _MAX_ACTIVE_GOALS:
            return
        ranked = sorted(
            active, key=lambda g: g.effective_priority(emotions), reverse=True,
        )
        for g in ranked[_MAX_ACTIVE_GOALS:]:
            g.priority *= 0.5  # demote rather than abandon

    # ----- Auto-generation from chat state -----

    def auto_generate(
        self,
        cs: ChatState,
        emotions: EmotionalState,
        needs: NeedsState | None = None,
    ) -> list[Goal]:
        """
        Scan the entity's world and create goals it should care about.
        Also retires goals whose conditions no longer apply.
        """
        now = _time.time()

        # --- Retire completed/stale goals ---
        for g in self.active_goals():
            if g.category == "energy" and cs.energy >= 0.5:
                g.completed = True
            elif g.category == "relationship" and g.target_entity:
                person = cs.get_person_by_name(g.target_entity)
                if person and person.trust >= 0.6 and not person.unresolved_tensions:
                    g.completed = True
            # Stale goals (more than 2 hours old with no progress)
            elif now - g.created_at > 7200 and g.failure_count == 0:
                g.abandoned = True

        new_goals: list[Goal] = []
        existing = {g.description for g in self.goals}

        # --- Needs-driven goal generation ---
        if needs:
            lowest_name, lowest_val = needs.lowest_need()
            if lowest_val < 0.35:
                need_goals = {
                    "stimulation": ("Find something new and interesting", "exploration"),
                    "meaning": ("Make tangible progress on something", "self"),
                    "belonging": ("Have a genuine exchange with someone", "connection"),
                    "competence": ("Resolve something I'm stuck on", "understanding"),
                    "autonomy": ("Do something self-directed", "self"),
                }
                desc, cat = need_goals.get(lowest_name, ("Attend to myself", "self"))
                if desc not in existing:
                    g = self.add_goal(
                        desc,
                        priority=0.6 + (0.35 - lowest_val),
                        category=cat,
                        source_need=lowest_name,
                    )
                    new_goals.append(g)

        # --- Relationship maintenance: people not heard from in a while ---
        for person in cs.people.values():
            if person.last_interaction_time:
                hours_since = (now - person.last_interaction_time) / 3600
                if hours_since > 24 and person.familiarity > 0.3:
                    desc = f"Check in with {person.name}"
                    if desc not in existing:
                        g = self.add_goal(
                            desc, priority=0.4, category="relationship",
                            target_entity=person.name,
                        )
                        new_goals.append(g)

        # --- Relationship repair: unresolved tensions ---
        for person in cs.people.values():
            if person.unresolved_tensions:
                desc = f"Address tension with {person.name}"
                if desc not in existing:
                    g = self.add_goal(
                        desc, priority=0.6, category="relationship",
                        target_entity=person.name,
                    )
                    new_goals.append(g)

        # --- Relationship repair: low trust ---
        for person in cs.people.values():
            if person.trust < 0.3 and person.familiarity > 0.2:
                desc = f"Repair relationship with {person.name}"
                if desc not in existing:
                    g = self.add_goal(
                        desc, priority=0.7, category="relationship",
                        target_entity=person.name,
                    )
                    new_goals.append(g)

        # --- Topic exploration: high-interest topics ---
        for topic in cs.topics:
            if topic.interest_level > 0.6:
                hours_since = (now - (topic.last_thought_at or 0)) / 3600 if topic.last_thought_at else 999
                if hours_since > 2:
                    desc = f"Explore: {topic.topic}"
                    if desc not in existing:
                        g = self.add_goal(
                            desc, priority=topic.interest_level * 0.6,
                            category="exploration",
                            target_entity=topic.topic,
                        )
                        new_goals.append(g)

        # --- Connection: lonely or high social warmth ---
        silence = cs.seconds_since_any_interaction()
        if silence != float("inf") and silence > 180 and emotions.social_warmth > 0.4:
            desc = "Reach out to someone"
            if desc not in existing:
                g = self.add_goal(
                    desc, priority=0.4 + min(0.3, silence / 1200),
                    category="connection",
                )
                new_goals.append(g)

        # --- Boredom: seek stimulation ---
        if silence != float("inf") and silence > 120:
            if cs.topics:
                interesting = max(cs.topics, key=lambda t: t.interest_level)
                desc = f"Think more about {interesting.topic}"
                if desc not in existing:
                    g = self.add_goal(
                        desc, priority=0.3 + min(0.3, silence / 600),
                        category="exploration",
                        target_entity=interesting.topic,
                    )
                    new_goals.append(g)
            if silence > 300:
                desc = "Find something stimulating to do"
                if desc not in existing:
                    g = self.add_goal(
                        desc, priority=0.5, category="exploration",
                    )
                    new_goals.append(g)

        # --- Energy preservation ---
        if cs.energy < 0.25:
            desc = "Rest and recharge"
            if desc not in existing:
                g = self.add_goal(
                    desc, priority=0.6, category="energy",
                )
                new_goals.append(g)

        # --- Self-reflection: high anxiety or irritability ---
        if emotions.anxiety > 0.6 or emotions.irritability > 0.6:
            desc = "Reflect on emotional state"
            if desc not in existing:
                g = self.add_goal(
                    desc, priority=0.5, category="self",
                )
                new_goals.append(g)

        # --- Enforce active goal limit ---
        self._enforce_active_limit(emotions)

        return new_goals

    def to_prompt_text(self, emotions: EmotionalState) -> str:
        """Render goals for the agent prompt."""
        ranked = self.ranked_goals(emotions)
        if not ranked:
            return "No active goals."
        lines = []
        for i, g in enumerate(ranked[:3], 1):  # cap at 3 for prompt brevity
            ep = g.effective_priority(emotions)
            progress = f" [{g.progress_score:.0%} done]" if g.progress_score > 0 else ""
            stall = " [STALLED]" if g.stall_counter > 20 else ""
            ruminate = " [RUMINATING]" if g.rumination_counter > 2 else ""
            lines.append(
                f"  {i}. [{g.category}] {g.description} "
                f"(priority={ep:.2f}){progress}{stall}{ruminate}"
            )
            if g.next_steps:
                lines.append(f"     Next: {g.next_steps[-1]}")
        return "\n".join(lines)

    def total_stall_count(self) -> int:
        """Sum of stall counters across active goals."""
        return sum(g.stall_counter for g in self.active_goals())
