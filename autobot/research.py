"""ResearchLedger -- notes, trade rationales, post-mortems, weekly reviews.

The entity's structured research record. Every thesis, observation, and
trade decision is logged here for later review and learning.
"""

from __future__ import annotations

import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


NoteType = Literal["thesis", "observation", "fundamental", "technical", "macro"]
SourceType = Literal["ground_truth", "analysis", "speculation"]


@dataclass
class ResearchNote:
    """A single research observation or thesis."""

    id: str = ""
    timestamp: float = 0.0
    ticker: str = ""
    note_type: NoteType = "observation"
    content: str = ""
    conviction: float = 0.5     # 0-1
    source: SourceType = "analysis"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = _time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "note_type": self.note_type,
            "content": self.content,
            "conviction": self.conviction,
            "source": self.source,
        }


@dataclass
class TradeRationale:
    """Why a trade was entered -- thesis, catalysts, risks, targets."""

    trade_id: str = ""
    ticker: str = ""
    direction: str = "long"     # "long" or "short"
    thesis: str = ""
    conviction: float = 0.5
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    stop_loss: float | None = None
    target: float | None = None
    time_horizon: str = ""      # "days", "weeks", "months"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.trade_id:
            self.trade_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = _time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "thesis": self.thesis,
            "conviction": self.conviction,
            "catalysts": self.catalysts,
            "risks": self.risks,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "time_horizon": self.time_horizon,
            "timestamp": self.timestamp,
        }


@dataclass
class PostMortem:
    """Post-trade analysis -- what went right/wrong, lessons learned."""

    trade_id: str = ""
    ticker: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    what_went_right: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "what_went_right": self.what_went_right,
            "what_went_wrong": self.what_went_wrong,
            "lessons": self.lessons,
            "timestamp": self.timestamp,
        }


@dataclass
class ResearchLedger:
    """The entity's research record."""

    notes: list[ResearchNote] = field(default_factory=list)
    rationales: list[TradeRationale] = field(default_factory=list)
    post_mortems: list[PostMortem] = field(default_factory=list)
    weekly_reviews: list[dict[str, Any]] = field(default_factory=list)

    def add_note(
        self,
        ticker: str,
        content: str,
        note_type: NoteType = "observation",
        conviction: float = 0.5,
        source: SourceType = "analysis",
    ) -> ResearchNote:
        """Add a research note."""
        note = ResearchNote(
            ticker=ticker,
            content=content,
            note_type=note_type,
            conviction=conviction,
            source=source,
        )
        self.notes.append(note)
        # Keep last 200 notes
        if len(self.notes) > 200:
            self.notes = self.notes[-200:]
        return note

    def add_rationale(self, rationale: TradeRationale) -> None:
        """Record a trade rationale."""
        self.rationales.append(rationale)

    def add_post_mortem(self, post_mortem: PostMortem) -> None:
        """Record a post-mortem."""
        self.post_mortems.append(post_mortem)

    def add_weekly_review(self, review: dict[str, Any]) -> None:
        """Record a weekly review summary."""
        review.setdefault("timestamp", _time.time())
        self.weekly_reviews.append(review)
        if len(self.weekly_reviews) > 52:
            self.weekly_reviews = self.weekly_reviews[-52:]

    def notes_for_ticker(self, ticker: str, limit: int = 10) -> list[ResearchNote]:
        """Get recent notes for a specific ticker."""
        return [
            n for n in reversed(self.notes)
            if n.ticker == ticker
        ][:limit]

    def active_theses(self) -> list[ResearchNote]:
        """Get high-conviction thesis notes (conviction > 0.6)."""
        return [
            n for n in self.notes
            if n.note_type == "thesis" and n.conviction > 0.6
        ]

    def rationale_for_trade(self, trade_id: str) -> TradeRationale | None:
        """Find the rationale for a specific trade."""
        for r in self.rationales:
            if r.trade_id == trade_id:
                return r
        return None

    def post_mortem_for_trade(self, trade_id: str) -> PostMortem | None:
        """Find the post-mortem for a specific trade."""
        for pm in self.post_mortems:
            if pm.trade_id == trade_id:
                return pm
        return None

    def recent_lessons(self, limit: int = 5) -> list[str]:
        """Get recent lessons from post-mortems."""
        lessons = []
        for pm in reversed(self.post_mortems):
            lessons.extend(pm.lessons)
            if len(lessons) >= limit:
                break
        return lessons[:limit]

    def summary(self) -> dict[str, Any]:
        """Summary for API/display."""
        return {
            "total_notes": len(self.notes),
            "active_theses": len(self.active_theses()),
            "total_rationales": len(self.rationales),
            "total_post_mortems": len(self.post_mortems),
            "weekly_reviews": len(self.weekly_reviews),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "notes": [n.to_dict() for n in self.notes[-200:]],
            "rationales": [r.to_dict() for r in self.rationales[-50:]],
            "post_mortems": [pm.to_dict() for pm in self.post_mortems[-50:]],
            "weekly_reviews": self.weekly_reviews[-52:],
        }
