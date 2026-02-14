"""Episodic Memory — stores and retrieves emotionally salient episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autobot.emotions import AffectState
from autobot.world_state import WorldState


@dataclass
class Episode:
    """A single episodic memory."""
    id: int
    time: int
    summary: str
    involved_entities: list[str]
    emotion_arousal: float
    emotion_valence: float
    salience_tags: list[str]
    goal_context: str | None = None
    world_state_snapshot: dict[str, Any] = field(default_factory=dict)
    unresolved: bool = False       # flags open loops for retrieval bias

    def relevance_score(
        self,
        query_entities: list[str],
        query_tags: list[str],
        current_arousal: float,
    ) -> float:
        """Score how relevant this episode is to a retrieval query."""
        score = 0.0
        # Entity overlap
        overlap = set(self.involved_entities) & set(query_entities)
        score += len(overlap) * 0.3
        # Emotion similarity
        tag_overlap = set(self.salience_tags) & set(query_tags)
        score += len(tag_overlap) * 0.25
        # Unresolved memories are stickier
        if self.unresolved:
            score += 0.2
        # High-arousal memories are more retrievable
        score += self.emotion_arousal * 0.15
        # Recency bias
        score += 0.1  # small base
        return round(score, 3)


# Thresholds for episode creation
_SALIENCE_THRESHOLD = 0.4
_TRUST_CHANGE_THRESHOLD = 0.1

# Event types that always create episodes
_EPISODE_TRIGGERS = {
    "commitment_broken", "commitment_fulfilled", "commitment_abandoned",
    "public_contradiction", "betrayal_exposed", "project_completed",
    "reputation_change",
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
        ws: WorldState,
        goal_context: str | None = None,
    ) -> Episode | None:
        """
        Evaluate whether this cycle's events warrant an episodic memory.
        Returns the episode if created, else None.
        """
        # Check triggers
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
                summary_parts.append(
                    f"{etype}: {ev.get('npc', ev.get('project', ''))}"
                )
            for key in ("npc", "target", "project", "beneficiary", "speaker"):
                if key in ev:
                    involved.add(ev[key])

        summary = "; ".join(summary_parts) if summary_parts else "notable cycle"

        # Snapshot key world metrics
        snapshot = {
            "energy": ws.agent.energy,
            "reputation": ws.agent.reputation,
            "trust_levels": {
                npc.name: round(npc.trust_in_agent, 2)
                for npc in ws.npcs.values()
            },
        }

        ep = Episode(
            id=self._next_id(),
            time=ws.time,
            summary=summary,
            involved_entities=list(involved),
            emotion_arousal=affect.arousal,
            emotion_valence=affect.valence,
            salience_tags=list(affect.salience_tags),
            goal_context=goal_context,
            world_state_snapshot=snapshot,
            unresolved=affect.valence < -0.3,
        )
        self.episodes.append(ep)
        return ep

    def retrieve(
        self,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        current_arousal: float = 0.0,
        limit: int = 5,
    ) -> list[Episode]:
        """Retrieve most relevant episodes."""
        query_entities = entities or []
        query_tags = tags or []
        scored = [
            (ep, ep.relevance_score(query_entities, query_tags, current_arousal))
            for ep in self.episodes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def to_prompt_text(self, limit: int = 5) -> str:
        """Render recent episodes for the agent prompt."""
        recent = self.episodes[-limit:]
        if not recent:
            return "No episodic memories yet."
        lines = []
        for ep in reversed(recent):
            flag = " [UNRESOLVED]" if ep.unresolved else ""
            lines.append(
                f"  - T={ep.time}: {ep.summary} "
                f"(arousal={ep.emotion_arousal:.1f}, "
                f"valence={ep.emotion_valence:.1f}){flag}"
            )
        return "\n".join(lines)
