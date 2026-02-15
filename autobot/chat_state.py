"""Chat State -- the entity's world is conversations, relationships, and its own mind."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class PersonProfile:
    """What the entity knows and feels about a real person."""
    id: str
    name: str
    trust: float = 0.5              # 0-1, how much the entity trusts them
    familiarity: float = 0.0        # 0-1, how well it knows them
    warmth: float = 0.5             # 0-1, how positively it feels about them
    last_message_time: float | None = None   # Unix timestamp of their last message
    last_interaction_time: float | None = None  # Unix timestamp of last exchange
    conversation_count: int = 0
    topics_discussed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # entity's private notes
    unresolved_tensions: list[str] = field(default_factory=list)

    # Theory of Mind tracking
    last_known_state: str = "unknown"
    last_known_intent: str = "unknown"
    reliability_score: float = 0.5      # running average

    # Social friction tracking
    proactive_attempt_count: int = 0
    last_proactive_at: float | None = None
    unanswered_proactive_count: int = 0
    last_proactive_topic: str = ""

    def clamp(self) -> None:
        self.trust = max(0.0, min(1.0, self.trust))
        self.familiarity = max(0.0, min(1.0, self.familiarity))
        self.warmth = max(0.0, min(1.0, self.warmth))

    def disposition_hint(self) -> str:
        """Qualitative description of the relationship (entity's perspective)."""
        if self.trust >= 0.8 and self.warmth >= 0.7:
            return "close"
        if self.trust >= 0.6:
            return "friendly"
        if self.trust >= 0.4:
            return "neutral"
        if self.trust >= 0.2:
            return "wary"
        return "distrustful"


@dataclass
class PendingMessage:
    """A message from a human that hasn't been fully processed yet."""
    id: str
    person_id: str
    text: str
    received_at: float
    processed: bool = False


@dataclass
class InternalTopic:
    """Something the entity is thinking about or interested in."""
    id: str
    topic: str
    interest_level: float = 0.5     # 0-1
    created_at: float = 0.0
    last_thought_at: float | None = None
    thought_count: int = 0          # how many times entity has thought about this
    related_people: list[str] = field(default_factory=list)


@dataclass
class ChatState:
    """The entity's view of its world -- conversations, relationships, mind."""

    # Time
    tick_count: int = 0
    start_time: float = field(default_factory=time.time)

    # The entity itself
    name: str = "Ryn"
    energy: float = 1.0             # 0-1

    # People it knows
    people: dict[str, PersonProfile] = field(default_factory=dict)

    # Messages waiting to be dealt with
    pending_messages: list[PendingMessage] = field(default_factory=list)

    # Conversation history per person (list of {role, text, time} dicts)
    conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Internal interests and topics
    topics: list[InternalTopic] = field(default_factory=list)

    # Append-only event log (same role as old WorldState.history)
    history: list[dict[str, Any]] = field(default_factory=list)

    # Mood snapshots over time
    mood_history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_person(self, person_id: str) -> PersonProfile | None:
        return self.people.get(person_id)

    def get_person_by_name(self, name: str) -> PersonProfile | None:
        for p in self.people.values():
            if p.name == name:
                return p
        return None

    def get_or_create_person(self, person_id: str, name: str) -> PersonProfile:
        if person_id not in self.people:
            self.people[person_id] = PersonProfile(id=person_id, name=name)
        return self.people[person_id]

    def add_message(self, person_id: str, text: str) -> PendingMessage:
        """Queue an incoming message from a person."""
        msg = PendingMessage(
            id=str(uuid4())[:8],
            person_id=person_id,
            text=text,
            received_at=time.time(),
        )
        self.pending_messages.append(msg)

        # Also store in conversation history
        self.conversations.setdefault(person_id, []).append({
            "role": "human",
            "text": text,
            "time": time.time(),
        })

        # Update person's last message time
        person = self.get_person(person_id)
        if person:
            person.last_message_time = time.time()

        return msg

    def add_entity_message(self, person_id: str, text: str) -> None:
        """Record an outgoing message from the entity."""
        self.conversations.setdefault(person_id, []).append({
            "role": "entity",
            "text": text,
            "time": time.time(),
        })
        person = self.get_person(person_id)
        if person:
            person.last_interaction_time = time.time()
            person.conversation_count += 1

    def unprocessed_messages(self) -> list[PendingMessage]:
        return [m for m in self.pending_messages if not m.processed]

    def recent_conversation(self, person_id: str, limit: int = 10) -> list[dict]:
        """Get the last N messages with a person."""
        return self.conversations.get(person_id, [])[-limit:]

    def record(self, event: dict[str, Any]) -> None:
        """Append an event to the history log."""
        event.setdefault("time", time.time())
        event.setdefault("tick", self.tick_count)
        self.history.append(event)

    def seconds_since_any_interaction(self) -> float:
        """How long since the entity talked to anyone."""
        last = 0.0
        for p in self.people.values():
            if p.last_interaction_time and p.last_interaction_time > last:
                last = p.last_interaction_time
        if last == 0.0:
            return float("inf")
        return time.time() - last

    def ground_truth_text(self, person_id: str | None = None) -> str:
        """Render verifiable facts about the world for LLM grounding.

        This is the ONLY source of truth the entity can reference.
        Anything not stated here is NOT established fact.
        """
        now = time.time()
        uptime = now - self.start_time
        lines = ["=== GROUND TRUTH (only these facts are established) ==="]
        lines.append(
            f"Current time: {time.strftime('%H:%M', time.localtime(now))}"
        )

        # Time-of-day awareness
        hour = time.localtime(now).tm_hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        weekday = time.strftime("%A", time.localtime(now))
        lines.append(f"Time of day: {time_of_day} ({weekday})")

        lines.append(f"You have been alive for {int(uptime / 60)} minutes.")
        lines.append(f"Tick count: {self.tick_count}")

        # People facts
        if self.people:
            names = ", ".join(p.name for p in self.people.values())
            lines.append(f"People you have met: {names}")
            for p in self.people.values():
                if p.last_message_time:
                    mins = int((now - p.last_message_time) / 60)
                    last = f"{mins} min ago"
                else:
                    last = "never"
                lines.append(
                    f"  {p.name}: last spoke to you {last}, "
                    f"{p.conversation_count} conversations"
                )
        else:
            lines.append("You have NOT met anyone yet. You know NO ONE.")

        # Per-person conversation facts
        if person_id:
            conv = self.conversations.get(person_id, [])
            if conv:
                lines.append(
                    f"Messages exchanged with this person: {len(conv)}"
                )
            else:
                lines.append(
                    "You have NEVER spoken to this person before."
                )

        lines.append("")
        lines.append(
            "IMPORTANT: Anything not listed above is NOT established fact."
        )
        lines.append(
            "If you think about something not listed, frame it as "
            "imagination, curiosity, or a question -- NOT as a memory "
            "or fact."
        )
        return "\n".join(lines)


# ======================================================================
# MarketState -- market entity's view of the world
# ======================================================================


@dataclass
class MarketState:
    """The market entity's view of its world.

    Replaces ChatState for the market entity. Chris is the only
    human the entity talks to.
    """

    # Time
    tick_count: int = 0
    start_time: float = field(default_factory=time.time)
    timezone: str = "Australia/Melbourne"

    # The entity itself
    name: str = "Ryn"
    energy: float = 1.0

    # Chris (the only human)
    chris: PersonProfile | None = None
    pending_messages: list[PendingMessage] = field(default_factory=list)
    conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Market tracking
    last_prices: dict[str, float] = field(default_factory=dict)
    last_session: str = ""           # track market session transitions
    last_price_fetch_tick: int = 0

    # History and mood
    history: list[dict[str, Any]] = field(default_factory=list)
    mood_history: list[dict[str, Any]] = field(default_factory=list)

    # Thinking log (for UI display)
    thinking_log: list[dict[str, Any]] = field(default_factory=list)

    # --- People compatibility (for code that uses self.state.people) ---
    @property
    def people(self) -> dict[str, PersonProfile]:
        if self.chris:
            return {"chris": self.chris}
        return {}

    def get_person(self, person_id: str) -> PersonProfile | None:
        if person_id == "chris" and self.chris:
            return self.chris
        return None

    def get_or_create_person(self, person_id: str, name: str) -> PersonProfile:
        if self.chris is None:
            self.chris = PersonProfile(id="chris", name=name)
        return self.chris

    def add_message(self, person_id: str, text: str) -> PendingMessage:
        msg = PendingMessage(
            id=str(uuid4())[:8],
            person_id="chris",
            text=text,
            received_at=time.time(),
        )
        self.pending_messages.append(msg)
        self.conversations.setdefault("chris", []).append({
            "role": "human", "text": text, "time": time.time(),
        })
        if self.chris:
            self.chris.last_message_time = time.time()
        return msg

    def add_entity_message(self, person_id: str, text: str) -> None:
        self.conversations.setdefault("chris", []).append({
            "role": "entity", "text": text, "time": time.time(),
        })
        if self.chris:
            self.chris.last_interaction_time = time.time()
            self.chris.conversation_count += 1

    def unprocessed_messages(self) -> list[PendingMessage]:
        return [m for m in self.pending_messages if not m.processed]

    def recent_conversation(self, person_id: str, limit: int = 10) -> list[dict]:
        return self.conversations.get("chris", [])[-limit:]

    def record(self, event: dict[str, Any]) -> None:
        event.setdefault("time", time.time())
        event.setdefault("tick", self.tick_count)
        self.history.append(event)

    def record_thinking(self, thought: str, category: str = "general") -> None:
        """Log an internal thought for UI display."""
        self.thinking_log.append({
            "thought": thought,
            "category": category,
            "time": time.time(),
            "tick": self.tick_count,
        })
        if len(self.thinking_log) > 200:
            self.thinking_log = self.thinking_log[-200:]

    def seconds_since_any_interaction(self) -> float:
        if self.chris and self.chris.last_interaction_time:
            return time.time() - self.chris.last_interaction_time
        return float("inf")

    def ground_truth_text(
        self,
        market_text: str = "",
        portfolio_text: str = "",
        capital_text: str = "",
    ) -> str:
        """Render ground truth including market status."""
        now = time.time()
        uptime = now - self.start_time
        lines = ["=== GROUND TRUTH (only these facts are established) ==="]
        lines.append(f"Tick count: {self.tick_count}")
        lines.append(f"Uptime: {int(uptime / 60)} minutes")

        if market_text:
            lines.append("")
            lines.append(market_text)

        if portfolio_text:
            lines.append("")
            lines.append(portfolio_text)

        if capital_text:
            lines.append("")
            lines.append(capital_text)

        # Chris contact info
        if self.chris:
            if self.chris.last_message_time:
                mins = int((now - self.chris.last_message_time) / 60)
                lines.append(f"\nChris: last message {mins} min ago")
            else:
                lines.append("\nChris: no messages yet")

        lines.append("")
        lines.append("Anything not listed above is NOT established fact.")
        return "\n".join(lines)
