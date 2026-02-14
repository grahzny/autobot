"""Core world entities: Agent, NPC, Project, Commitment, Resource, Belief."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommitmentStatus(Enum):
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    BROKEN = "broken"


class ResourceType(Enum):
    TIME = "time"
    MONEY = "money"
    ATTENTION = "attention"
    STATUS = "status"


# ---------------------------------------------------------------------------
# Agent (the LLM-controlled self)
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    name: str = "Self"
    energy: float = 1.0          # 0–1
    reputation: float = 0.5      # 0–1
    mood_history: list[dict[str, Any]] = field(default_factory=list)
    known_beliefs: list[str] = field(default_factory=list)  # belief ids
    commitments: list[str] = field(default_factory=list)     # commitment ids
    resources: dict[str, float] = field(default_factory=lambda: {
        "time": 8.0,
        "money": 100.0,
        "attention": 3.0,   # max actions per cycle
        "status": 0.5,
    })

    def clamp(self) -> None:
        self.energy = max(0.0, min(1.0, self.energy))
        self.reputation = max(0.0, min(1.0, self.reputation))


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------
@dataclass
class NPC:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    trust_in_agent: float = 0.5       # 0–1
    beliefs_about_agent: set[str] = field(default_factory=set)
    hidden_goals: list[str] = field(default_factory=list)
    memory_log: list[dict[str, Any]] = field(default_factory=list)
    influence_weight: float = 0.5
    last_interaction_time: int = 0

    def clamp(self) -> None:
        self.trust_in_agent = max(0.0, min(1.0, self.trust_in_agent))
        self.influence_weight = max(0.0, min(1.0, self.influence_weight))


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    deadline: int = 10             # world-time units
    progress: float = 0.0         # 0–1
    stakeholders: list[str] = field(default_factory=list)  # npc ids
    importance_weight: float = 0.5

    def is_overdue(self, current_time: int) -> bool:
        return current_time > self.deadline and self.progress < 1.0


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------
@dataclass
class Commitment:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    promisor: str = ""             # entity name/id
    beneficiary: str = ""          # npc id
    deliverable: str = ""          # description
    deadline: int = 10
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    project_id: str | None = None  # optional link to a project

    def is_overdue(self, current_time: int) -> bool:
        return (
            current_time > self.deadline
            and self.status == CommitmentStatus.ACTIVE
        )


# ---------------------------------------------------------------------------
# Resource pool entry
# ---------------------------------------------------------------------------
@dataclass
class Resource:
    type: ResourceType = ResourceType.TIME
    quantity: float = 0.0


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------
@dataclass
class Belief:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    holder: str = ""               # entity name/id
    proposition: str = ""
    confidence: float = 0.5        # 0–1

    def clamp(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))
