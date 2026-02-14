"""Canonical World State — the hidden ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autobot.entities import Agent, Belief, Commitment, NPC, Project


@dataclass
class ScheduledEvent:
    """An event queued for future execution."""
    time: int
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """
    The full canonical state of the world.
    The LLM never sees this directly — only filtered observation packets.
    """

    # Time
    time: int = 0                  # discrete tick counter
    day: int = 1
    hour: int = 8                  # 8:00 start

    # Entity registry
    agent: Agent = field(default_factory=Agent)
    npcs: dict[str, NPC] = field(default_factory=dict)           # id -> NPC
    projects: dict[str, Project] = field(default_factory=dict)   # id -> Project
    commitments: dict[str, Commitment] = field(default_factory=dict)
    beliefs: dict[str, Belief] = field(default_factory=dict)     # id -> Belief

    # Event queue (future consequences)
    event_queue: list[ScheduledEvent] = field(default_factory=list)

    # History log — append-only record of everything that happened
    history: list[dict[str, Any]] = field(default_factory=list)

    # Relationship graph: (entity_a, entity_b) -> relationship metadata
    relationships: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )

    # --------------- helpers ---------------

    def advance_time(self, ticks: int = 1) -> list[dict[str, Any]]:
        """Advance the clock and return events that fire."""
        fired: list[dict[str, Any]] = []
        for _ in range(ticks):
            self.time += 1
            self.hour += 1
            # Passive energy recovery each tick (rest)
            self.agent.energy = min(1.0, self.agent.energy + 0.02)
            if self.hour >= 24:
                self.hour = 8
                self.day += 1
                # new-day resource reset
                self.agent.resources["time"] = 8.0
                self.agent.resources["attention"] = 3.0
                # Overnight energy recovery
                self.agent.energy = min(1.0, self.agent.energy + 0.3)
            # fire scheduled events
            remaining = []
            for ev in self.event_queue:
                if ev.time <= self.time:
                    fired.append({
                        "type": ev.event_type,
                        **ev.payload,
                    })
                else:
                    remaining.append(ev)
            self.event_queue = remaining
        return fired

    def get_npc_by_name(self, name: str) -> NPC | None:
        for npc in self.npcs.values():
            if npc.name == name:
                return npc
        return None

    def get_project_by_name(self, name: str) -> Project | None:
        for proj in self.projects.values():
            if proj.name == name:
                return proj
        return None

    def record(self, event: dict[str, Any]) -> None:
        """Append an event to history."""
        event["world_time"] = self.time
        event["day"] = self.day
        event["hour"] = self.hour
        self.history.append(event)

    def time_str(self) -> str:
        return f"Day {self.day}, {self.hour:02d}:00"
