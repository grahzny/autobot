"""Episodic Memory -- stores and retrieves emotionally salient episodes."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from autobot.emotions import AffectState

if TYPE_CHECKING:
    from autobot.chat_state import ChatState


@dataclass
class Episode:
    """A single episodic memory."""
    id: int
    time: float                    # Unix timestamp when encoded
    summary: str
    involved_entities: list[str]   # person names
    emotion_arousal: float
    emotion_valence: float
    salience_tags: list[str]
    goal_context: str | None = None
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    unresolved: bool = False       # flags open loops for retrieval bias

    # Anti-repetition tracking
    recall_count: int = 0
    last_recalled_at: float = 0.0
    suppressed_until: float = 0.0  # skip this episode until this timestamp
    theme_tag: str = ""            # primary theme for diversity filtering

    # Epistemic grounding
    source: str = "observed"       # observed | user_told | inferred | imagined
    confidence: float = 1.0        # 0.0-1.0, decays for imagined/inferred
    grounded: bool = True          # False = from entity's own imagination

    def relevance_score(
        self,
        query_entities: list[str],
        query_tags: list[str],
        current_arousal: float,
    ) -> float:
        """Score how relevant this episode is to a retrieval query."""
        # Suppression check -- effectively hidden
        if _time.time() < self.suppressed_until:
            return -1.0

        score = 0.0
        # Entity overlap
        overlap = set(self.involved_entities) & set(query_entities)
        score += len(overlap) * 0.3
        # Tag overlap
        tag_overlap = set(self.salience_tags) & set(query_tags)
        score += len(tag_overlap) * 0.25
        # Unresolved memories are stickier
        if self.unresolved:
            score += 0.2
        # High-arousal memories are more retrievable
        score += self.emotion_arousal * 0.15
        # Recency bias (more recent = higher score)
        age_hours = (_time.time() - self.time) / 3600
        recency = max(0.0, 0.2 - age_hours * 0.01)
        score += recency

        # Recall penalty -- diminishing returns on repeated retrieval
        if self.recall_count > 0:
            score -= 0.1 * min(self.recall_count, 5)  # up to -0.5

        # Recency-of-recall penalty (recently recalled = less interesting)
        if self.last_recalled_at > 0:
            hours_since_recall = (_time.time() - self.last_recalled_at) / 3600
            if hours_since_recall < 0.5:  # recalled in last 30 min
                score -= 0.3
            elif hours_since_recall < 2:
                score -= 0.1

        return round(score, 3)


# Thresholds for episode creation
_SALIENCE_THRESHOLD = 0.4
_TRUST_CHANGE_THRESHOLD = 0.1

# Event types that always create episodes
_EPISODE_TRIGGERS = {
    "meaningful_exchange", "boundary_crossed", "vulnerability_shared",
    "trust_change", "new_encounter", "reconnection",
    "disagreement", "criticism_received", "compliment_received",
    "farewell",
    # Self-generated action triggers
    "entity_spoke", "entity_initiated", "self_reflection",
}


@dataclass
class EpisodicMemory:
    episodes: list[Episode] = field(default_factory=list)
    _id_counter: int = 0

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def maybe_create_episode(
        self,
        events: list[dict[str, Any]],
        affect: AffectState,
        cs: ChatState,
        goal_context: str | None = None,
        source: str = "observed",
    ) -> Episode | None:
        """
        Evaluate whether this cycle's events warrant an episodic memory.
        Returns the episode if created, else None.
        """
        should_create = False

        if affect.arousal >= _SALIENCE_THRESHOLD:
            should_create = True

        for ev in events:
            if ev.get("type") in _EPISODE_TRIGGERS:
                should_create = True
                break
            if ev.get("type") == "trust_change":
                if abs(ev.get("delta", 0)) >= _TRUST_CHANGE_THRESHOLD:
                    should_create = True
                    break

        if not should_create:
            return None

        # Build summary
        summary_parts = []
        involved = set()
        for ev in events:
            etype = ev.get("type", "")
            if etype in _EPISODE_TRIGGERS or etype == "trust_change":
                person = ev.get("person", ev.get("target", ""))
                detail = ev.get("details", ev.get("text", ""))
                if detail:
                    summary_parts.append(f"{etype}: {detail[:80]}")
                elif person:
                    summary_parts.append(f"{etype}: {person}")
                else:
                    summary_parts.append(etype)
            for key in ("person", "target", "speaker"):
                if key in ev:
                    involved.add(ev[key])

        summary = "; ".join(summary_parts) if summary_parts else "notable moment"

        # Snapshot key state
        snapshot = {
            "energy": cs.energy,
            "relationships": {
                p.name: {"trust": round(p.trust, 2), "warmth": round(p.warmth, 2)}
                for p in cs.people.values()
            },
        }

        # Derive a theme tag from the most significant salience tag
        theme_tag = ""
        if affect.salience_tags:
            theme_tag = affect.salience_tags[0]
        elif involved:
            theme_tag = f"person:{next(iter(involved))}"

        is_grounded = source in ("observed", "user_told")
        ep = Episode(
            id=self._next_id(),
            time=_time.time(),
            summary=summary,
            involved_entities=list(involved),
            emotion_arousal=affect.arousal,
            emotion_valence=affect.valence,
            salience_tags=list(affect.salience_tags),
            goal_context=goal_context,
            state_snapshot=snapshot,
            unresolved=affect.valence < -0.3,
            theme_tag=theme_tag,
            source=source,
            confidence=1.0 if is_grounded else 0.7,
            grounded=is_grounded,
        )
        self.episodes.append(ep)
        return ep

    def retrieve(
        self,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        current_arousal: float = 0.0,
        limit: int = 5,
        enforce_diversity: bool = True,
    ) -> list[Episode]:
        """Retrieve most relevant episodes with diversity filtering."""
        query_entities = entities or []
        query_tags = tags or []
        scored = [
            (ep, ep.relevance_score(query_entities, query_tags, current_arousal))
            for ep in self.episodes
            if _time.time() >= ep.suppressed_until  # skip suppressed
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not enforce_diversity:
            results = [ep for ep, _ in scored[:limit]]
        else:
            # Max 1 episode per theme_tag to prevent thematic loops
            results: list[Episode] = []
            seen_themes: set[str] = set()
            for ep, sc in scored:
                if sc <= 0:
                    continue
                if ep.theme_tag and ep.theme_tag in seen_themes:
                    continue
                results.append(ep)
                if ep.theme_tag:
                    seen_themes.add(ep.theme_tag)
                if len(results) >= limit:
                    break

        # Update recall tracking for retrieved episodes
        now = _time.time()
        for ep in results:
            ep.recall_count += 1
            ep.last_recalled_at = now

        return results

    def suppress_looping_memories(
        self, theme: str, duration_seconds: float = 300,
    ) -> int:
        """Suppress memories matching a theme that's been looping."""
        suppressed = 0
        until = _time.time() + duration_seconds
        theme_lower = theme.lower()
        for ep in self.episodes:
            if theme_lower in ep.summary.lower() or ep.theme_tag == theme_lower:
                ep.suppressed_until = until
                suppressed += 1
        return suppressed

    def decay_confidence(self, rate: float = 0.01) -> None:
        """Reduce confidence on imagined/inferred episodes each tick.

        Observed and user_told episodes don't decay -- they're grounded.
        """
        for ep in self.episodes:
            if ep.source in ("imagined", "inferred"):
                ep.confidence = max(0.0, ep.confidence - rate)

    def to_prompt_text(self, limit: int = 5) -> str:
        """Render recent episodes for the agent prompt."""
        recent = self.episodes[-limit:]
        if not recent:
            return "No memories yet."
        lines = []
        for ep in reversed(recent):
            flag = " [UNRESOLVED]" if ep.unresolved else ""
            source_tag = ""
            if ep.source == "imagined":
                source_tag = " [YOUR OWN THOUGHT -- not an external event]"
            elif ep.source == "inferred":
                source_tag = " [YOUR INFERENCE]"
            lines.append(
                f"  - {ep.summary} "
                f"(arousal={ep.emotion_arousal:.1f}, "
                f"valence={ep.emotion_valence:.1f}){flag}{source_tag}"
            )
        return "\n".join(lines)
