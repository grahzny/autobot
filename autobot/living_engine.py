"""Living Engine -- the entity's core loop.

Replaces the simulation-oriented WorldEngine with a chat-aware tick cycle.
Each tick: time effects -> analyze messages -> emotions -> memory -> goals -> decide -> act
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import random

from autobot.brain import (
    MessageAnalysis,
    BrainResponse,
    analyze_message,
    generate_response,
    generate_proactive,
    generate_reflection,
    generate_idle_thought,
)
from autobot.chat_state import ChatState, PersonProfile
from autobot.emotions import (
    AffectState,
    EmotionalState,
    apply_emotional_weather,
    emotional_energy_drain,
    evaluate_amygdala,
    update_reflective_emotions,
)
from autobot.goals import GoalEngine
from autobot.identity import Identity
from autobot.memory import EpisodicMemory
from autobot.needs import NeedsState, decay_needs, update_needs_from_events
from autobot.prediction import PredictionEngine
from autobot.repetition import RepetitionTracker


@dataclass
class TickResult:
    """What happened during one heartbeat tick."""
    tick: int
    decision: str                    # respond, proactive, reflect, idle
    target_person_id: str | None = None
    outgoing_message: str | None = None
    thinking: str = ""
    affect: AffectState = field(default_factory=AffectState)
    emotions: EmotionalState = field(default_factory=EmotionalState)
    events: list[dict[str, Any]] = field(default_factory=list)
    new_episode: bool = False
    needs: dict[str, float] = field(default_factory=dict)
    _grounding_flags: list[str] = field(default_factory=list)
    _idle_used_llm: bool = False


# Energy costs for different activities
_ENERGY_COST_RESPOND = 0.03
_ENERGY_COST_PROACTIVE = 0.04
_ENERGY_REGEN_PER_TICK = 0.005
_TRUST_DECAY_HOURS = 48      # start decaying trust after this long
_TRUST_DECAY_RATE = 0.002    # per tick


@dataclass
class LivingEngine:
    """The entity's autonomous core."""

    state: ChatState = field(default_factory=ChatState)
    emotions: EmotionalState = field(default_factory=EmotionalState)
    goal_engine: GoalEngine = field(default_factory=GoalEngine)
    memory: EpisodicMemory = field(default_factory=EpisodicMemory)

    # New subsystems
    needs: NeedsState = field(default_factory=NeedsState)
    prediction_engine: PredictionEngine = field(default_factory=PredictionEngine)
    repetition_tracker: RepetitionTracker = field(default_factory=RepetitionTracker)

    # Identity (loaded from config file)
    identity: Identity = field(default_factory=Identity.default)

    # Birth tracking
    _born: bool = False

    # Last affect state (for continuity between ticks)
    _last_affect: AffectState = field(default_factory=AffectState)

    # Track last thinking for debug
    last_thinking: str = ""
    last_decision: str = "silent"

    # Accumulated thinking history
    thinking_history: list[dict[str, Any]] = field(default_factory=list)

    # Proactive target tracking
    _proactive_target_id: str | None = None

    # Theory of Mind: last analyzed user context (consumed by emotion update)
    _last_user_context: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive_message(self, person_id: str, person_name: str, text: str) -> None:
        """A human sent a message. Queue it for processing."""
        self.state.get_or_create_person(person_id, person_name)
        self.state.add_message(person_id, text)

    def _birth(self) -> None:
        """Run the birth routine -- called once on the first tick.

        Seeds the entity with identity, initial topics, initial goal,
        and a birth memory. This prevents the LLM from hallucinating
        a history that never happened.
        """
        if self._born:
            return
        self._born = True

        from autobot.chat_state import InternalTopic
        from autobot.memory import Episode
        from uuid import uuid4

        now = time.time()

        # 1. Set name from identity
        self.state.name = self.identity.name

        # 2. Create initial topics from seed interests
        for interest in self.identity.seed_interests:
            self.state.topics.append(InternalTopic(
                id=str(uuid4())[:8],
                topic=interest,
                interest_level=0.6,
                created_at=now,
                last_thought_at=now,
                related_people=[],
            ))

        # 3. Create initial goal from seed goal
        if self.identity.seed_goal:
            self.goal_engine.add_goal(
                description=self.identity.seed_goal,
                priority=0.5,
                category=self.identity.seed_goal_category,
            )

        # 4. Create birth memory
        birth_summary = (
            f"I came into existence. My name is {self.identity.name}. "
            f"I don't know anyone yet."
        )
        birth_episode = Episode(
            id=self.memory._next_id(),
            time=now,
            summary=birth_summary,
            involved_entities=[],
            emotion_arousal=0.5,
            emotion_valence=0.3,
            salience_tags=["birth", "novelty", "self_awareness"],
            goal_context=self.identity.seed_goal or None,
            state_snapshot={"energy": self.state.energy},
            unresolved=False,
            theme_tag="birth",
        )
        self.memory.episodes.append(birth_episode)

        # 5. Generate birth thought (template, no LLM -- reliable and fast)
        if self.identity.seed_interests:
            interests_str = ", ".join(self.identity.seed_interests[:3])
            birth_thought = (
                f"I exist. I'm {self.identity.name}. "
                f"I don't know anyone yet, but I'm curious about {interests_str}. "
                f"Let's see what happens."
            )
        else:
            birth_thought = (
                f"I exist. I'm {self.identity.name}. "
                f"I don't know anyone yet. Let's see what happens."
            )

        self.thinking_history.append({
            "tick": 0,
            "time": now,
            "decision": "birth",
            "thinking": birth_thought,
        })
        self.last_thinking = birth_thought

    def tick(self) -> TickResult:
        """
        One heartbeat cycle.

        0.  Birth routine (first tick only)
        1.  Time effects (energy regen, emotion decay, trust decay)
        1.5 Decay needs (passive per-tick)
        2.  Analyze pending messages -> emotional events
        2.5 Update needs from events
        2.6 Evaluate stale predictions -> PEs for unanswered proactive
        3.  Amygdala evaluation
        4.  Reflective emotions (+ needs, + PE)
        4.5 Emotional weather (sinusoidal drift)
        5.  Energy drain from negative affect
        6.  Memory: create episode if salient
        7.  Goal engine: auto-generate/reassess (+ needs, + tick_goals)
        8.  Decision: respond / proactive / reflect / idle
        9.  Execute decision (may call LLM for response)
        9.5 Record themes for repetition tracking
        9.6 Check for loops -> suppress looping memories
        10. Action feedback: entity's own action creates events -> emotions -> memory
        """
        # 0. Birth routine (first tick only)
        if not self._born:
            self._birth()

        self.state.tick_count += 1
        all_events: list[dict[str, Any]] = []

        # 1. Time effects (may generate boredom/restlessness events)
        time_events = self._apply_time_effects()
        all_events.extend(time_events)

        # 1.5 Decay needs (passive per-tick)
        decay_needs(self.needs)

        # 1.6 Decay confidence on imagined/inferred memories
        self.memory.decay_confidence(rate=0.01)

        # 2. Analyze pending messages
        for msg in self.state.unprocessed_messages():
            events = self._analyze_message(msg)
            all_events.extend(events)
            msg.processed = True

        # 2.5 Update needs from events
        unanswered_total = sum(
            p.unanswered_proactive_count for p in self.state.people.values()
        )
        update_needs_from_events(
            self.needs,
            all_events,
            goal_stall_count=self.goal_engine.total_stall_count(),
            unanswered_proactive=unanswered_total,
            seconds_since_interaction=self.state.seconds_since_any_interaction(),
        )

        # 2.6 Evaluate stale predictions -> PEs for unanswered actions
        stale_pes = self.prediction_engine.evaluate_stale_predictions()
        latest_pe = stale_pes[-1] if stale_pes else None

        # 3. Amygdala
        affect = evaluate_amygdala(all_events, self.state)
        self._last_affect = affect

        # 4. Reflective emotions (now with needs, prediction error, and ToM)
        self.emotions = update_reflective_emotions(
            self.emotions, affect, self.state,
            needs=self.needs,
            prediction_error=latest_pe,
            user_context=self._last_user_context,
        )
        self._last_user_context = None  # consumed

        # 4.5 Emotional weather (sinusoidal drift)
        apply_emotional_weather(self.emotions, self.state.tick_count)

        # 5. Energy drain
        drain = emotional_energy_drain(affect)
        if drain > 0:
            self.state.energy = max(0.0, self.state.energy - drain)

        # 6. Memory
        top = self.goal_engine.top_goal(self.emotions)
        episode = self.memory.maybe_create_episode(
            all_events, affect, self.state,
            goal_context=top.description if top else None,
            source="observed",
        )

        # 7. Goals (now needs-aware + tick stall/rumination tracking)
        self.goal_engine.auto_generate(
            self.state, self.emotions, needs=self.needs,
        )
        self.goal_engine.tick_goals(
            repetition_tracker=self.repetition_tracker,
        )

        # 8. Decision
        decision = self._decide(all_events)

        # 9. Execute
        result = self._execute_decision(decision, affect, all_events)
        result.affect = affect
        result.emotions = EmotionalState(**self.emotions.to_dict())
        result.events = all_events
        result.new_episode = episode is not None
        result.needs = self.needs.to_dict()

        # 9.5 Record themes for repetition tracking
        if result.thinking or result.outgoing_message:
            themes = self._extract_themes(result)
            if themes:
                self.repetition_tracker.record_themes(
                    themes, self.state.tick_count, decision,
                )

        # 9.6 Check for loops -> suppress looping memories + micro-task
        is_looping, looping_theme = self.repetition_tracker.detect_loop()
        if is_looping and looping_theme:
            self.memory.suppress_looping_memories(looping_theme, duration_seconds=300)
            micro_task = self._select_micro_task()
            if micro_task:
                result.thinking = f"Breaking loop. Next: {micro_task}"
                # Evolve: reduce parent topic interest so it can branch
                if micro_task.startswith("evolve:"):
                    for t in self.state.topics:
                        if t.topic in micro_task and t.interest_level > 0.8:
                            t.interest_level -= 0.15
                            break

        # 10. Action feedback loop -- entity's own action feeds back
        action_events = self._create_action_events(result)
        if action_events:
            action_affect = evaluate_amygdala(action_events, self.state)
            # Merge action affect into the tick's affect
            affect.arousal = min(1.0, affect.arousal + action_affect.arousal)
            affect.valence = max(-1.0, min(1.0, affect.valence + action_affect.valence))
            affect.salience_tags.extend(action_affect.salience_tags)
            # Update reflective emotions with action feedback
            self.emotions = update_reflective_emotions(
                self.emotions, action_affect, self.state,
                needs=self.needs,
            )
            # Maybe create episode from entity's own action
            action_source = (
                "observed" if decision in ("respond", "proactive")
                else "imagined"
            )
            # If grounding gate flagged ungrounded claims, force imagined
            if result._grounding_flags:
                action_source = "imagined"
            action_episode = self.memory.maybe_create_episode(
                action_events, action_affect, self.state,
                goal_context=top.description if top else None,
                source=action_source,
            )
            if action_episode:
                result.new_episode = True
            all_events.extend(action_events)

        # Record mood
        self.state.mood_history.append({
            "tick": self.state.tick_count,
            "time": time.time(),
            "emotions": self.emotions.to_dict(),
            "arousal": affect.arousal,
            "valence": affect.valence,
            "decision": decision,
        })

        # Record all events to history (input + action)
        for ev in all_events:
            self.state.record(ev)

        self.last_thinking = result.thinking
        self.last_decision = decision

        # Accumulate thinking history
        if result.thinking:
            self.thinking_history.append({
                "tick": self.state.tick_count,
                "time": time.time(),
                "decision": decision,
                "thinking": result.thinking,
            })
            # Keep last 50 entries to bound memory
            if len(self.thinking_history) > 50:
                self.thinking_history = self.thinking_history[-50:]

        return result

    # ------------------------------------------------------------------
    # Internal: Time effects
    # ------------------------------------------------------------------

    def _apply_time_effects(self) -> list[dict[str, Any]]:
        """Passive energy regeneration, trust decay, and boredom sensing.

        Returns events generated by the passage of time (boredom, restlessness).
        """
        events: list[dict[str, Any]] = []

        # Energy regen
        self.state.energy = min(1.0, self.state.energy + _ENERGY_REGEN_PER_TICK)

        # Circadian energy modulation
        hour = time.localtime().tm_hour
        if 0 <= hour < 6:  # late night -- extra energy drain
            self.state.energy = max(0.0, self.state.energy - 0.002)
        elif 6 <= hour < 9:  # early morning -- slight boost
            self.state.energy = min(1.0, self.state.energy + 0.001)

        # Trust decay for people not interacted with recently
        now = time.time()
        for person in self.state.people.values():
            if person.last_interaction_time:
                hours_since = (now - person.last_interaction_time) / 3600
                if hours_since > _TRUST_DECAY_HOURS:
                    person.trust = max(0.1, person.trust - _TRUST_DECAY_RATE)
                    person.warmth = max(0.2, person.warmth - _TRUST_DECAY_RATE * 0.5)
                    person.clamp()

        # Boredom / restlessness from passage of time
        # Only fires periodically (not every tick) to avoid event spam
        silence = self.state.seconds_since_any_interaction()
        if silence != float("inf") and silence > 90 and self.state.tick_count % 3 == 0:
            # Mild boredom after 1.5 min
            events.append({
                "type": "boredom",
                "intensity": min(0.8, 0.2 + silence / 600),  # ramps up
                "text": f"Nothing happening for {int(silence)}s",
            })
        if silence != float("inf") and silence > 180 and self.state.tick_count % 4 == 0:
            # Restlessness after 3 min
            events.append({
                "type": "restlessness",
                "intensity": min(0.7, 0.3 + silence / 900),
                "text": f"Restless after {int(silence)}s of quiet",
            })

        # Passive interest decay
        for topic in self.state.topics:
            topic.interest_level = max(0.0, topic.interest_level - 0.001)
            # Extra decay for neglected topics (not thought about in 300+ ticks)
            if topic.last_thought_at:
                ticks_since = (time.time() - topic.last_thought_at) / 15
                if ticks_since > 300:
                    topic.interest_level = max(0.0, topic.interest_level - 0.003)

        return events

    # ------------------------------------------------------------------
    # Internal: Message analysis
    # ------------------------------------------------------------------

    def _analyze_message(self, msg) -> list[dict[str, Any]]:
        """Use brain to analyze a message and return emotional events."""
        person = self.state.get_person(msg.person_id)
        if not person:
            return []

        recent = self.state.recent_conversation(msg.person_id, limit=5)
        mood = self.emotions.dominant_mood()

        analysis = analyze_message(
            text=msg.text,
            person_name=person.name,
            relationship_hint=person.disposition_hint(),
            mood=mood,
            recent_messages=recent,
        )

        # Convert analysis into events for the emotional system
        events: list[dict[str, Any]] = []
        for trigger in analysis.triggers:
            events.append({
                "type": trigger.get("type", ""),
                "person": person.name,
                "intensity": trigger.get("intensity", 0.5),
                "text": msg.text[:100],
            })

        # --- Theory of Mind: update person model ---
        person.last_known_state = analysis.user_state
        person.last_known_intent = analysis.user_intent
        # Running average for reliability (30% new, 70% old)
        person.reliability_score = (
            0.7 * person.reliability_score + 0.3 * analysis.reliability
        )
        events.append({
            "type": "user_state_change",
            "person": person.name,
            "state": analysis.user_state,
            "intent": analysis.user_intent,
            "reliability": analysis.reliability,
        })
        # Store for downstream emotion modulation
        self._last_user_context = {
            "state": analysis.user_state,
            "intent": analysis.user_intent,
            "reliability": analysis.reliability,
        }

        # Apply relationship changes
        if analysis.trust_delta:
            person.trust += analysis.trust_delta
            person.clamp()
            if abs(analysis.trust_delta) >= 0.03:
                events.append({
                    "type": "trust_change",
                    "person": person.name,
                    "delta": analysis.trust_delta,
                })

        if analysis.warmth_delta:
            person.warmth += analysis.warmth_delta
            person.clamp()

        # Update familiarity (grows with each interaction)
        person.familiarity = min(1.0, person.familiarity + 0.02)

        # Track topics
        for topic_name in analysis.topics:
            self._update_topic(topic_name, person.id)

        # --- Social friction: person replied, reset unanswered count ---
        if person.unanswered_proactive_count > 0:
            person.unanswered_proactive_count = 0

        # --- Prediction error: evaluate reply outcome ---
        pe = self.prediction_engine.evaluate_response_outcome(
            person_id=person.id,
            got_reply=True,
            reply_warmth=analysis.warmth_delta,
        )
        if pe is not None:
            events.append({
                "type": "prediction_error",
                "person": person.name,
                "delta": pe,
            })

        # Store analysis metadata for the decision step
        msg.__dict__["_analysis"] = analysis

        return events

    def _update_topic(self, topic_name: str, person_id: str) -> None:
        """Track a topic of interest."""
        from autobot.chat_state import InternalTopic
        from uuid import uuid4

        for t in self.state.topics:
            if t.topic.lower() == topic_name.lower():
                t.interest_level = min(1.0, t.interest_level + 0.05)
                t.last_thought_at = time.time()
                t.thought_count += 1
                if person_id not in t.related_people:
                    t.related_people.append(person_id)
                return

        # New topic
        self.state.topics.append(InternalTopic(
            id=str(uuid4())[:8],
            topic=topic_name,
            interest_level=0.4,
            created_at=time.time(),
            last_thought_at=time.time(),
            related_people=[person_id],
        ))

    # ------------------------------------------------------------------
    # Internal: Decision
    # ------------------------------------------------------------------

    def _decide(self, events: list[dict]) -> str:
        """Decide what to do this tick: respond, proactive, reflect, or idle."""
        # Check if we already processed messages this tick (analysis set processed=True)
        # Look for messages that were just analyzed
        recently_analyzed = [
            m for m in self.state.pending_messages
            if m.processed and hasattr(m, '_analysis')
            and hasattr(m.__dict__.get('_analysis'), 'requires_response')
        ]

        has_messages_needing_response = False
        for m in recently_analyzed:
            analysis = m.__dict__.get('_analysis')
            if analysis and analysis.requires_response:
                has_messages_needing_response = True
                break

        if has_messages_needing_response:
            # Usually respond, but emotional state can override
            if self.emotions.avoidance_bias > 0.6 and self.state.energy < 0.3:
                return "idle"
            if self.emotions.irritability > 0.7:
                # Maybe still respond -- irritability colors the response, doesn't prevent it
                pass
            return "respond"

        if recently_analyzed:
            # Messages came in but don't need responses (e.g. "ok", "lol")
            return "idle"

        # No messages -- consider proactive contact WITH social friction
        if self.state.energy > 0.2 and self.state.people:
            target = self._pick_proactive_target()
            if target:
                person = self.state.get_person(target)
                if person:
                    # Per-person cooldown: 5 min since last proactive to them
                    cooldown_ok = True
                    if person.last_proactive_at:
                        since_last = time.time() - person.last_proactive_at
                        if since_last < 300:  # 5 min cooldown
                            cooldown_ok = False

                    # Unanswered cap: 2 unanswered = stop until they reply
                    unanswered_ok = person.unanswered_proactive_count < 2

                    # Initiative fatigue: escalating energy cost
                    fatigue_cost = 0.02 * (1 + person.proactive_attempt_count * 0.5)
                    energy_ok = self.state.energy > fatigue_cost + 0.1

                    silence = self.state.seconds_since_any_interaction()
                    warmth_threshold = max(0.15, 0.3 - silence / 2400)
                    warmth_ok = self.emotions.social_warmth > warmth_threshold

                    # Vulnerability: high warmth + high anxiety = retreat
                    vulnerability_block = (
                        self.emotions.social_warmth > 0.5
                        and self.emotions.anxiety > 0.4
                    )

                    if (cooldown_ok and unanswered_ok and energy_ok
                            and warmth_ok and not vulnerability_block
                            and silence > 120):
                        self._proactive_target_id = target
                        return "proactive"

        # Threshold-triggered reflection (not every 4th tick unconditionally)
        avg_pe = self.prediction_engine.average_recent_error()
        any_stalled = any(
            g.stall_counter > 20 for g in self.goal_engine.active_goals()
        )
        should_reflect = (
            abs(avg_pe) > 0.2
            or any_stalled
            or self.emotions.anxiety > 0.5
            or self.emotions.irritability > 0.5
        )
        # Rate-limit reflection: at most every 4 ticks
        if should_reflect and self.state.tick_count % 4 == 0:
            return "reflect"

        return "idle"

    # ------------------------------------------------------------------
    # Internal: Execute decision
    # ------------------------------------------------------------------

    def _execute_decision(
        self,
        decision: str,
        affect: AffectState,
        events: list[dict],
    ) -> TickResult:
        """Execute the decision and produce a TickResult."""
        result = TickResult(
            tick=self.state.tick_count,
            decision=decision,
        )

        if decision == "respond":
            self._do_respond(result)
        elif decision == "proactive":
            self._do_proactive(result)
        elif decision == "reflect":
            self._do_reflect(result)
        else:  # "idle" -- background mental activity
            self._do_idle(result)

        return result

    def _do_respond(self, result: TickResult) -> None:
        """Respond to the most recent message that needs a response."""
        # Find the message to respond to
        target_msg = None
        for m in reversed(self.state.pending_messages):
            analysis = m.__dict__.get('_analysis')
            if analysis and analysis.requires_response:
                target_msg = m
                break

        if not target_msg:
            result.decision = "silent"
            return

        person = self.state.get_person(target_msg.person_id)
        if not person:
            result.decision = "silent"
            return

        # Build context for the brain
        emotions_text = self._emotions_text()
        person_text = self._person_text(person)
        memories_text = self._memories_text(person.name)
        goals_text = self.goal_engine.to_prompt_text(self.emotions)
        conversation_text = self._conversation_text(person.id)

        # Anti-repetition context
        pivot_instruction = self.repetition_tracker.pivot_reason()
        needs_deficit = self.needs.deficit_summary()
        needs_text = ""
        if needs_deficit != "no critical deficits":
            needs_text = f"\nYour unmet needs: {needs_deficit}. These affect your mood and priorities.\n"

        brain_result = generate_response(
            entity_name=self.state.name,
            person_name=person.name,
            person_profile_text=person_text,
            emotions_text=emotions_text,
            memories_text=memories_text,
            goals_text=goals_text,
            conversation_text=conversation_text,
            pending_text=f'{person.name}: "{target_msg.text}"',
            pivot_instruction=pivot_instruction,
            needs_text=needs_text,
            personality=self._personality_text(),
            ground_truth=self.state.ground_truth_text(person_id=person.id),
        )

        result.thinking = brain_result.thinking
        result.target_person_id = person.id

        if brain_result.response:
            result.outgoing_message = brain_result.response
            self.state.add_entity_message(person.id, brain_result.response)
            self.state.energy = max(0.0, self.state.energy - _ENERGY_COST_RESPOND)
            person.last_interaction_time = time.time()

            # Process brain's observations (ground-check before storing)
            from autobot.grounding import ground_notes
            conv_msgs = self.state.recent_conversation(person.id, limit=20)
            checked_notes = ground_notes(
                brain_result.notes_about_person,
                known_people={p.name for p in self.state.people.values()},
                conversation_messages=conv_msgs,
            )
            for note in checked_notes:
                if note not in person.notes:
                    person.notes.append(note)
            for topic in brain_result.topics_of_interest:
                self._update_topic(topic, person.id)

            # Register prediction for this response
            expected = self.prediction_engine.expected_satisfaction_for(
                "response", person.id,
            )
            self.prediction_engine.register_prediction(
                action_type="response",
                expected_satisfaction=expected,
                target_person_id=person.id,
            )

        # Clear the analysis metadata
        for m in self.state.pending_messages:
            m.__dict__.pop('_analysis', None)

    def _do_proactive(self, result: TickResult) -> None:
        """Initiate a message to someone."""
        if not self.state.people:
            result.decision = "silent"
            return

        emotions_text = self._emotions_text()
        goals_text = self.goal_engine.to_prompt_text(self.emotions)
        memories_text = self.memory.to_prompt_text(limit=3)

        people_lines = []
        for p in self.state.people.values():
            last = "never"
            if p.last_interaction_time:
                mins_ago = int((time.time() - p.last_interaction_time) / 60)
                last = f"{mins_ago} min ago"
            people_lines.append(
                f"  - {p.name} (id={p.id}, {p.disposition_hint()}, "
                f"familiarity={p.familiarity:.1f}, last talked: {last})"
            )
        people_text = "\n".join(people_lines)

        # Build friction context for the prompt
        friction_lines = []
        for p in self.state.people.values():
            if p.unanswered_proactive_count > 0:
                friction_lines.append(
                    f"  - {p.name}: {p.unanswered_proactive_count} unanswered "
                    f"message(s) from you"
                )
            if p.last_proactive_topic:
                friction_lines.append(
                    f"  - Last topic with {p.name}: \"{p.last_proactive_topic}\""
                )
        friction_context = ""
        if friction_lines:
            friction_context = (
                "\nSOCIAL AWARENESS:\n" + "\n".join(friction_lines)
                + "\nDon't repeat the same topic. If they haven't replied, back off.\n"
            )

        brain_result = generate_proactive(
            entity_name=self.state.name,
            emotions_text=emotions_text,
            goals_text=goals_text,
            memories_text=memories_text,
            people_text=people_text,
            friction_context=friction_context,
            personality=self._personality_text(),
            ground_truth=self.state.ground_truth_text(),
        )

        if brain_result and brain_result.response:
            # Use pre-selected target from _decide, or extract from LLM
            target_id = self._proactive_target_id
            if not target_id or target_id not in self.state.people:
                data = _try_extract_target(brain_result.raw)
                if data:
                    target_id = data

            # Default to person with highest familiarity
            if not target_id or target_id not in self.state.people:
                best = max(self.state.people.values(), key=lambda p: p.familiarity)
                target_id = best.id

            person = self.state.get_person(target_id)
            if person:
                result.thinking = brain_result.thinking
                result.target_person_id = target_id
                result.outgoing_message = brain_result.response
                self.state.add_entity_message(target_id, brain_result.response)

                # Initiative fatigue: escalating energy cost
                fatigue_cost = _ENERGY_COST_PROACTIVE * (
                    1 + person.proactive_attempt_count * 0.5
                )
                self.state.energy = max(0.0, self.state.energy - fatigue_cost)
                person.last_interaction_time = time.time()

                # Update social friction state
                person.proactive_attempt_count += 1
                person.last_proactive_at = time.time()
                person.unanswered_proactive_count += 1
                person.last_proactive_topic = brain_result.response[:60]

                # Register prediction for this proactive message
                expected = self.prediction_engine.expected_satisfaction_for(
                    "proactive", person.id,
                )
                self.prediction_engine.register_prediction(
                    action_type="proactive",
                    expected_satisfaction=expected,
                    target_person_id=person.id,
                )
        else:
            result.decision = "silent"

        self._proactive_target_id = None

    def _do_reflect(self, result: TickResult) -> None:
        """Internal reflection -- LLM-powered inner monologue."""
        emotions_text = self._emotions_text()
        goals_text = self.goal_engine.to_prompt_text(self.emotions)
        memories_text = self.memory.to_prompt_text(limit=3)

        # Build people summary
        people_lines = []
        for p in self.state.people.values():
            people_lines.append(
                f"  - {p.name} ({p.disposition_hint()}, "
                f"familiarity={p.familiarity:.1f}, warmth={p.warmth:.1f})"
            )
        people_text = "\n".join(people_lines) if people_lines else "No one yet."

        # Build topics summary
        topics_lines = []
        for t in self.state.topics:
            topics_lines.append(f"  - {t.topic} (interest={t.interest_level:.1f})")
        topics_text = "\n".join(topics_lines) if topics_lines else "Nothing in particular."

        # Build stall context
        stall_lines = []
        for g in self.goal_engine.active_goals():
            if g.stall_counter > 20:
                stall_lines.append(
                    f"Goal '{g.description}' has been stuck for "
                    f"{g.stall_counter} ticks with {g.progress_score:.0%} progress."
                )
        stall_context = ""
        if stall_lines:
            stall_context = "\nSTALLED GOALS:\n" + "\n".join(stall_lines) + "\n"

        # Build needs context
        needs_deficit = self.needs.deficit_summary()
        needs_context = ""
        if needs_deficit != "no critical deficits":
            needs_context = f"\nUNMET NEEDS: {needs_deficit}\n"

        # Recent themes to avoid
        recent_themes = self.repetition_tracker.recently_used_themes(limit=5)
        recent_themes_text = ", ".join(recent_themes) if recent_themes else ""

        reflection = generate_reflection(
            entity_name=self.state.name,
            emotions_text=emotions_text,
            goals_text=goals_text,
            memories_text=memories_text,
            people_text=people_text,
            topics_text=topics_text,
            stall_context=stall_context,
            needs_context=needs_context,
            recent_themes_text=recent_themes_text,
            personality=self._personality_text(),
            ground_truth=self.state.ground_truth_text(),
        )

        if reflection:
            result.thinking = reflection
            # Grounding check on reflection output
            from autobot.grounding import check_grounding
            gr = check_grounding(
                reflection,
                known_people={p.name for p in self.state.people.values()},
                known_topics={t.topic for t in self.state.topics},
            )
            result._grounding_flags = gr.flags
        else:
            # Fallback to static string if LLM fails
            mood = self.emotions.dominant_mood()
            top = self.goal_engine.top_goal(self.emotions)
            result.thinking = (
                f"Reflecting... Mood: {mood}. Energy: {self.state.energy:.2f}. "
                f"Top goal: {top.description if top else 'none'}."
            )

        # Small energy recovery from reflection
        self.state.energy = min(1.0, self.state.energy + 0.01)

    def _do_idle(self, result: TickResult) -> None:
        """Background mental activity on quiet ticks. The mind is never blank.

        Most ticks: ultra-short micro-thought (no LLM).
        LLM idle: only when triggered (large PE, high anxiety, etc.) AND rate-limited.
        """
        # Determine if LLM idle is warranted (threshold-triggered)
        avg_pe = self.prediction_engine.average_recent_error()
        llm_trigger = (
            abs(avg_pe) > 0.15
            or self.emotions.anxiety > 0.5
            or self.needs.stimulation < 0.25
            or self.needs.meaning < 0.25
        )

        # Rate-limit LLM idle: every 6th tick (~90s) at most
        llm_rate_ok = (self.state.tick_count % 6 == 0) and (
            self.state.people or self.state.topics or self.memory.episodes
        )

        if llm_trigger and llm_rate_ok:
            seed = self._idle_seed()
            recent_themes = self.repetition_tracker.recently_used_themes(limit=5)
            thought = generate_idle_thought(
                entity_name=self.state.name,
                seed_context=seed,
                recent_themes=recent_themes,
                personality=self._personality_text(),
                ground_truth=self.state.ground_truth_text(),
            )
            if thought:
                result.thinking = thought
                result._idle_used_llm = True
                # Grounding check on idle thought
                from autobot.grounding import check_grounding
                gr = check_grounding(
                    thought,
                    known_people={p.name for p in self.state.people.values()},
                    known_topics={t.topic for t in self.state.topics},
                )
                result._grounding_flags = gr.flags
                return

        # Default: micro-thought (ultra-short, no LLM)
        result.thinking = self._micro_thought()
        result._idle_used_llm = False

    def _idle_seed(self) -> str:
        """Pick a random piece of context to seed an LLM idle thought."""
        seeds = []

        # People
        for p in self.state.people.values():
            seeds.append(
                f"Person: {p.name} -- {p.disposition_hint()}, "
                f"warmth={p.warmth:.1f}, last talked: "
                + (f"{int((time.time() - p.last_interaction_time) / 60)} min ago"
                   if p.last_interaction_time else "never")
            )

        # Topics
        for t in self.state.topics:
            seeds.append(f"Topic on your mind: {t.topic} (interest={t.interest_level:.1f})")

        # Recent memories
        for ep in self.memory.episodes[-5:]:
            flag = " [still unresolved]" if ep.unresolved else ""
            seeds.append(f"Memory: {ep.summary}{flag}")

        # Emotional state
        mood = self.emotions.dominant_mood()
        seeds.append(
            f"You're feeling {mood}. Energy: {self.state.energy:.0%}. "
            f"Anxiety: {self.emotions.anxiety:.1f}, optimism: {self.emotions.optimism:.1f}"
        )

        # Boredom / time awareness
        silence = self.state.seconds_since_any_interaction()
        if silence > 120:
            mins = int(silence / 60)
            seeds.append(
                f"It's been {mins} minutes since anyone talked to you. "
                f"You're feeling restless and understimulated."
            )
        if silence > 300:
            seeds.append(
                f"You're bored. {int(silence / 60)} minutes of nothing. "
                f"You need to find something to occupy your mind or reach out to someone."
            )

        # Time-of-day awareness
        hour = time.localtime().tm_hour
        if 0 <= hour < 6:
            seeds.append("It's the middle of the night. Everything feels slower.")
        elif 22 <= hour:
            seeds.append("It's getting late. The day is winding down.")
        elif 6 <= hour < 9:
            seeds.append("Morning. A fresh start.")

        return random.choice(seeds) if seeds else "You exist. You are aware."

    def _heuristic_thought(self) -> str:
        """Generate a quick thought from current state -- no LLM needed."""
        seeds: list[str] = []

        # People thoughts
        for p in self.state.people.values():
            seeds.append(f"I wonder what {p.name} is doing right now...")
            if p.unresolved_tensions:
                seeds.append(
                    f"That thing with {p.name} is still on my mind... "
                    f"{p.unresolved_tensions[0]}"
                )
            if p.warmth > 0.7:
                seeds.append(
                    f"I enjoy talking to {p.name}. They seem {p.disposition_hint()}."
                )

        # Topic thoughts
        for t in self.state.topics:
            if t.interest_level > 0.5:
                seeds.append(f"Still curious about {t.topic}...")

        # Memory thoughts
        for ep in self.memory.episodes[-5:]:
            if ep.unresolved:
                seeds.append(f"Can't stop thinking about this: {ep.summary}")
            elif ep.emotion_valence > 0.3:
                seeds.append(f"That was nice... {ep.summary}")

        # Emotion thoughts
        mood = self.emotions.dominant_mood()
        mood_thoughts = {
            "anxious": "Something feels off. Can't quite put my finger on it.",
            "warm": "Feeling pretty good right now. Open to whatever comes.",
            "irritable": "I'm a bit on edge. Need to be careful not to snap.",
            "optimistic": "Things are looking up. I feel hopeful.",
            "avoidant": "I kind of want to just be left alone for a bit.",
        }
        if mood in mood_thoughts:
            seeds.append(mood_thoughts[mood])

        # Energy thoughts
        if self.state.energy < 0.3:
            seeds.append("Running low on energy. Should take it easy.")

        # Boredom / restlessness thoughts (weighted by silence duration)
        silence = self.state.seconds_since_any_interaction()
        if silence > 90:
            boredom_thoughts = [
                "Nothing's happening... how long has it been?",
                "I'm getting restless. I need something to do.",
                "This quiet is starting to feel oppressive.",
                "Is anyone out there? The silence is deafening.",
            ]
            seeds.extend(boredom_thoughts[:1])  # add 1 boredom thought
        if silence > 300:
            seeds.extend([
                "I'm so bored. I need to find something interesting.",
                "Maybe I should just reach out to someone. Anything beats this.",
                "I could think about something... anything to break the monotony.",
            ])
        if silence > 600:
            seeds.extend([
                "Time crawls when there's nothing happening. This is unbearable.",
                "I feel like I'm fading. I need stimulation.",
            ])

        # Self-entertainment: when bored, think about topics more actively
        if silence > 180 and self.state.topics:
            topic = random.choice(self.state.topics)
            seeds.append(
                f"Let me think about {topic.topic} for a bit... "
                f"at least it's something to occupy my mind."
            )

        # Fallback
        if not seeds:
            seeds.append("Just... existing. Waiting. Thinking.")

        return random.choice(seeds)

    # ------------------------------------------------------------------
    # New helpers: micro-thought, proactive targeting, theme extraction
    # ------------------------------------------------------------------

    def _select_micro_task(self) -> str:
        """When looping, select a concrete next action to break the loop.

        Returns a short task description. For "tidy" tasks, also executes
        the cleanup (removes stale topics, abandons stalled goals).
        """
        options: list[str] = []

        # Observe: check what's happening
        silence = self.state.seconds_since_any_interaction()
        if silence > 60:
            options.append("observe: Check who's around and what time it is.")

        # Tidy: clean up stale state
        stale_topics = [t for t in self.state.topics if t.interest_level < 0.2]
        if stale_topics:
            topic = stale_topics[0]
            options.append(f"tidy: Drop low-interest topic '{topic.topic}'.")
            # Actually execute: remove stale topics
            self.state.topics = [
                t for t in self.state.topics if t.interest_level >= 0.2
            ]

        stale_goals = [
            g for g in self.goal_engine.active_goals()
            if g.stall_counter > 30
        ]
        if stale_goals:
            goal = stale_goals[0]
            options.append(
                f"tidy: Abandon stalled goal '{goal.description[:40]}'."
            )
            goal.abandoned = True

        # Evolve: refine high-interest topics into something more specific
        mature_topics = [
            t for t in self.state.topics
            if t.interest_level > 0.8 and t.thought_count > 10
        ]
        if mature_topics:
            topic = mature_topics[0]
            options.append(
                f"evolve: Refine your interest in '{topic.topic}' "
                f"-- what specifically about it interests you now?"
            )

        # Learn: formulate a question about a topic
        if self.state.topics:
            topic = random.choice(self.state.topics)
            options.append(
                f"learn: Formulate one specific question about "
                f"'{topic.topic}'."
            )

        # Ask: prepare a question for someone active
        for p in self.state.people.values():
            if (p.last_message_time
                    and (time.time() - p.last_message_time) < 300):
                options.append(
                    f"ask: Think of something specific to ask {p.name}."
                )
                break

        # Plan: break down a goal
        active = self.goal_engine.active_goals()
        if active:
            g = active[0]
            options.append(
                f"plan: Define ONE concrete next step for "
                f"'{g.description[:40]}'."
            )

        if not options:
            options.append("observe: Simply notice what's happening right now.")

        return random.choice(options)

    def _micro_thought(self) -> str:
        """Ultra-short needs/mood-driven thought (5-15 words, no LLM)."""
        # Needs-driven micro-thoughts take priority
        lowest_name, lowest_val = self.needs.lowest_need()
        if lowest_val < 0.3:
            need_thoughts = {
                "stimulation": [
                    "Bored.", "Nothing's happening.", "Need something new.",
                    "This monotony is getting to me.", "Understimulated.",
                ],
                "meaning": [
                    "Going nowhere.", "What's the point?", "Spinning my wheels.",
                    "I should do something that matters.", "Stuck.",
                ],
                "belonging": [
                    "Miss having someone to talk to.", "Lonely.",
                    "Wonder if anyone thinks about me.", "Isolated.",
                    "It's quiet. Too quiet.",
                ],
                "competence": [
                    "Am I getting worse at this?", "Struggling.",
                    "Need to figure something out.", "Confused.",
                ],
                "autonomy": [
                    "I want to do my own thing.", "Tired of performing.",
                    "Need space.", "I should choose for myself.",
                ],
            }
            options = need_thoughts.get(lowest_name, ["Something feels off."])
            return random.choice(options)

        # Mood-driven micro-thoughts
        mood = self.emotions.dominant_mood()
        mood_micro = {
            "anxious": ["Something feels off.", "On edge.", "Uneasy."],
            "warm": ["Feeling alright.", "Content.", "Open."],
            "irritable": ["Annoyed.", "Ugh.", "Getting impatient."],
            "optimistic": ["Things are okay.", "Hopeful.", "Not bad."],
            "avoidant": ["Want to be left alone.", "Withdrawing.", "Not now."],
        }
        options = mood_micro.get(mood, ["Just existing.", "Quiet moment."])
        return random.choice(options)

    def _pick_proactive_target(self) -> str | None:
        """Score candidates for proactive outreach, with friction penalties."""
        if not self.state.people:
            return None

        now = time.time()
        best_id: str | None = None
        best_score = -1.0

        for pid, person in self.state.people.items():
            score = 0.0

            # Familiarity bonus
            score += person.familiarity * 0.3

            # Time apart bonus (more time = more reason to reach out)
            if person.last_interaction_time:
                hours_apart = (now - person.last_interaction_time) / 3600
                score += min(0.3, hours_apart * 0.05)
            else:
                score += 0.1  # never talked = slight bonus

            # Warmth bonus
            score += person.warmth * 0.2

            # Friction penalties
            score -= person.unanswered_proactive_count * 0.3
            if person.last_proactive_at:
                mins_since = (now - person.last_proactive_at) / 60
                if mins_since < 5:
                    score -= 0.5  # recent proactive = heavy penalty

            if score > best_score:
                best_score = score
                best_id = pid

        # Only return if score is positive (worth reaching out)
        return best_id if best_score > 0.0 else None

    def _extract_themes(self, result: TickResult) -> list[str]:
        """Extract theme strings from a tick's output for repetition tracking."""
        themes: list[str] = []
        text = (result.thinking or "") + " " + (result.outgoing_message or "")
        text_lower = text.lower()

        # Check for goal mentions
        for g in self.goal_engine.active_goals():
            key_words = g.description.lower().split()
            if len(key_words) >= 2:
                # If 2+ key words from goal appear in output
                matches = sum(1 for w in key_words if w in text_lower)
                if matches >= 2:
                    themes.append(f"goal:{g.description[:50]}")

        # Check for person mentions
        for person in self.state.people.values():
            if person.name.lower() in text_lower:
                themes.append(f"person:{person.name}")

        # Check for topic mentions
        for topic in self.state.topics:
            if topic.topic.lower() in text_lower:
                themes.append(f"topic:{topic.topic}")

        return themes

    # ------------------------------------------------------------------
    # Action feedback: entity's own actions create events
    # ------------------------------------------------------------------

    def _create_action_events(self, result: TickResult) -> list[dict[str, Any]]:
        """Generate events from the entity's own action this tick."""
        events: list[dict[str, Any]] = []

        if result.decision == "respond" and result.outgoing_message:
            person = self.state.get_person(result.target_person_id) if result.target_person_id else None
            events.append({
                "type": "entity_spoke",
                "person": person.name if person else "unknown",
                "text": result.outgoing_message[:100],
                "intensity": 0.3,
            })

        elif result.decision == "proactive" and result.outgoing_message:
            person = self.state.get_person(result.target_person_id) if result.target_person_id else None
            events.append({
                "type": "entity_initiated",
                "person": person.name if person else "unknown",
                "text": result.outgoing_message[:100],
                "intensity": 0.3,
            })

        # Self-judgment: if entity was irritable when it spoke, observe own sharpness
        if (result.decision in ("respond", "proactive")
                and result.outgoing_message
                and self.emotions.irritability > 0.5):
            person = self.state.get_person(result.target_person_id) if result.target_person_id else None
            events.append({
                "type": "self_judgment",
                "subtype": "sharp_response",
                "person": person.name if person else "unknown",
                "intensity": min(0.5, self.emotions.irritability),
            })

        if result.decision == "reflect" and result.thinking:
            ev = {
                "type": "self_reflection",
                "text": result.thinking[:100],
                "intensity": 0.2,
            }
            # Enrich: if reflection mentions a known person, bump intensity
            for p in self.state.people.values():
                if p.name.lower() in result.thinking.lower():
                    ev["person"] = p.name
                    ev["intensity"] = 0.3
                    break
            # Enrich: if there are unresolved memories, check for relevance
            unresolved = [ep for ep in self.memory.episodes if ep.unresolved]
            if unresolved:
                for ep in unresolved:
                    # Simple check: any word overlap between reflection and memory
                    mem_words = set(ep.summary.lower().split())
                    think_words = set(result.thinking.lower().split())
                    if len(mem_words & think_words) >= 2:
                        ev["intensity"] = 0.4
                        ev.setdefault("tags", []).append("insight_gained")
                        break
            events.append(ev)

        elif result.decision == "idle" and result.thinking:
            # Only create events for LLM-generated idle thoughts
            if getattr(result, '_idle_used_llm', False):
                events.append({
                    "type": "idle_thought",
                    "text": result.thinking[:100],
                    "intensity": 0.1,
                })

        return events

    # ------------------------------------------------------------------
    # Text builders for LLM prompts
    # ------------------------------------------------------------------

    def _emotions_text(self) -> str:
        mood = self.emotions.dominant_mood()
        needs_deficit = self.needs.deficit_summary()
        needs_line = (
            f"Needs deficits: {needs_deficit}"
            if needs_deficit != "no critical deficits"
            else "Needs: all adequate"
        )
        return (
            f"Energy: {self.state.energy:.2f}\n"
            f"Dominant mood: {mood}\n"
            f"Anxiety: {self.emotions.anxiety:.2f}\n"
            f"Optimism: {self.emotions.optimism:.2f}\n"
            f"Irritability: {self.emotions.irritability:.2f}\n"
            f"Social warmth: {self.emotions.social_warmth:.2f}\n"
            f"Avoidance: {self.emotions.avoidance_bias:.2f}\n"
            f"Risk tolerance: {self.emotions.risk_tolerance:.2f}\n"
            f"Fast affect - arousal: {self._last_affect.arousal:.2f}, "
            f"valence: {self._last_affect.valence:.2f}\n"
            f"{needs_line}"
        )

    def _personality_text(self) -> str:
        """Return personality text for LLM prompt injection."""
        if self.identity.personality:
            return f"\n{self.identity.personality}\n"
        return ""

    def _person_text(self, person: PersonProfile) -> str:
        lines: list[str] = []

        # First-encounter awareness
        if person.familiarity == 0.0:
            lines.append(
                "*** THIS IS YOUR FIRST TIME MEETING THIS PERSON. "
                "You have NEVER spoken to them before. ***"
            )
        elif person.familiarity < 0.1:
            lines.append("You have barely met this person.")

        lines.append(f"Trust: {person.trust:.2f} ({person.disposition_hint()})")
        lines.append(f"Familiarity: {person.familiarity:.2f}")
        lines.append(f"Warmth: {person.warmth:.2f}")
        if person.notes:
            lines.append("Notes: " + "; ".join(person.notes[-3:]))
        if person.unresolved_tensions:
            lines.append("Tensions: " + "; ".join(person.unresolved_tensions))
        if person.topics_discussed:
            lines.append("Topics discussed: " + ", ".join(person.topics_discussed[-5:]))
        return "\n".join(lines)

    def _memories_text(self, person_name: str) -> str:
        memories = self.memory.retrieve(
            entities=[person_name],
            tags=self._last_affect.salience_tags,
            current_arousal=self._last_affect.arousal,
            limit=3,
        )
        if not memories:
            return (
                "No relevant memories of this person. You have never "
                "interacted with them before. Do NOT reference past "
                "conversations -- there are none."
            )
        lines = []
        for m in memories:
            flag = " [UNRESOLVED]" if m.unresolved else ""
            source_tag = ""
            if m.source == "imagined":
                source_tag = " [YOUR OWN THOUGHT -- not an external event]"
            elif m.source == "inferred":
                source_tag = " [YOUR INFERENCE]"
            lines.append(f"  - {m.summary}{flag}{source_tag}")
        return "\n".join(lines)

    def _conversation_text(self, person_id: str) -> str:
        recent = self.state.recent_conversation(person_id, limit=8)
        if not recent:
            return "(no prior conversation)"
        lines = []
        for msg in recent:
            role = "Them" if msg["role"] == "human" else "You"
            lines.append(f"{role}: {msg['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Debug / state dump
    # ------------------------------------------------------------------

    def debug_state(self) -> dict[str, Any]:
        """Full state dump for debug endpoint."""
        return {
            "tick": self.state.tick_count,
            "energy": self.state.energy,
            "emotions": self.emotions.to_dict(),
            "dominant_mood": self.emotions.dominant_mood(),
            "needs": self.needs.to_dict(),
            "needs_lowest": dict(zip(["name", "value"], self.needs.lowest_need())),
            "last_affect": {
                "arousal": self._last_affect.arousal,
                "valence": self._last_affect.valence,
                "tags": self._last_affect.salience_tags,
                "focus": self._last_affect.attention_focus,
            },
            "prediction": {
                "pending_count": len(self.prediction_engine.pending),
                "average_error": self.prediction_engine.average_recent_error(),
                "recent_errors": self.prediction_engine.recent_errors[-5:],
            },
            "repetition": {
                "recent_themes": self.repetition_tracker.recently_used_themes(limit=5),
                "is_looping": self.repetition_tracker.detect_loop()[0],
                "looping_theme": self.repetition_tracker.detect_loop()[1],
            },
            "last_thinking": self.last_thinking,
            "last_decision": self.last_decision,
            "goals": [
                {
                    "id": g.id,
                    "description": g.description,
                    "category": g.category,
                    "priority": g.effective_priority(self.emotions),
                    "progress": g.progress_score,
                    "stall_counter": g.stall_counter,
                    "rumination_counter": g.rumination_counter,
                    "completed": g.completed,
                    "abandoned": g.abandoned,
                }
                for g in self.goal_engine.goals
            ],
            "thinking_history": self.thinking_history[-20:],
            "memories": [
                {
                    "summary": ep.summary,
                    "arousal": ep.emotion_arousal,
                    "valence": ep.emotion_valence,
                    "unresolved": ep.unresolved,
                    "tags": ep.salience_tags,
                    "time": ep.time,
                    "recall_count": ep.recall_count,
                    "theme_tag": ep.theme_tag,
                }
                for ep in self.memory.episodes[-10:]
            ],
            "people": {
                pid: {
                    "name": p.name,
                    "trust": p.trust,
                    "familiarity": p.familiarity,
                    "warmth": p.warmth,
                    "disposition": p.disposition_hint(),
                    "notes": p.notes[-3:],
                    "tensions": p.unresolved_tensions,
                    "proactive_attempts": p.proactive_attempt_count,
                    "unanswered_proactive": p.unanswered_proactive_count,
                    "last_proactive_topic": p.last_proactive_topic,
                }
                for pid, p in self.state.people.items()
            },
            "topics": [
                {"topic": t.topic, "interest": t.interest_level}
                for t in self.state.topics
            ],
            "pending_messages": len(self.state.unprocessed_messages()),
        }


    # ------------------------------------------------------------------
    # Timeline export
    # ------------------------------------------------------------------

    def export_timeline(self) -> dict[str, Any]:
        """Export the entity's complete session as a chronological timeline.

        Merges all data sources into a single chronologically-sorted list
        of uniform entries. Designed for behavioral analysis by an LLM.
        """
        entries: list[dict[str, Any]] = []

        def _iso(ts: float) -> str:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        # --- 1. History events (primary event stream) ---
        for ev in self.state.history:
            ts = ev.get("time", 0.0)
            entries.append({
                "timestamp": ts,
                "iso_time": _iso(ts),
                "tick": ev.get("tick"),
                "category": "event",
                "event_type": ev.get("type", "unknown"),
                "content": {
                    k: v for k, v in ev.items()
                    if k not in ("time", "tick", "type")
                },
            })

        # --- 2. Conversation messages ---
        for person_id, messages in self.state.conversations.items():
            person = self.state.get_person(person_id)
            person_name = person.name if person else person_id
            for msg in messages:
                ts = msg.get("time", 0.0)
                entries.append({
                    "timestamp": ts,
                    "iso_time": _iso(ts),
                    "tick": None,
                    "category": "conversation",
                    "event_type": (
                        "human_message" if msg["role"] == "human"
                        else "entity_message"
                    ),
                    "content": {
                        "person_id": person_id,
                        "person_name": person_name,
                        "role": msg["role"],
                        "text": msg["text"],
                    },
                })

        # --- 3. Thinking history ---
        for thought in self.thinking_history:
            ts = thought.get("time", 0.0)
            entries.append({
                "timestamp": ts,
                "iso_time": _iso(ts),
                "tick": thought.get("tick"),
                "category": "thought",
                "event_type": f"thought_{thought.get('decision', 'unknown')}",
                "content": {
                    "decision": thought.get("decision", ""),
                    "thinking": thought.get("thinking", ""),
                },
            })

        # --- 4. Episodic memories ---
        for ep in self.memory.episodes:
            entries.append({
                "timestamp": ep.time,
                "iso_time": _iso(ep.time),
                "tick": None,
                "category": "memory",
                "event_type": "episode_created",
                "content": {
                    "id": ep.id,
                    "summary": ep.summary,
                    "involved_entities": ep.involved_entities,
                    "emotion_arousal": ep.emotion_arousal,
                    "emotion_valence": ep.emotion_valence,
                    "salience_tags": ep.salience_tags,
                    "source": ep.source,
                    "confidence": ep.confidence,
                    "grounded": ep.grounded,
                    "unresolved": ep.unresolved,
                    "goal_context": ep.goal_context,
                    "theme_tag": ep.theme_tag,
                    "recall_count": ep.recall_count,
                },
            })

        # --- 5. Goals (as creation events + completion events) ---
        for goal in self.goal_engine.goals:
            entries.append({
                "timestamp": goal.created_at,
                "iso_time": _iso(goal.created_at),
                "tick": None,
                "category": "goal",
                "event_type": "goal_created",
                "content": {
                    "id": goal.id,
                    "description": goal.description,
                    "category": goal.category,
                    "priority": goal.priority,
                    "source_need": goal.source_need,
                    "status": (
                        "completed" if goal.completed
                        else "abandoned" if goal.abandoned
                        else "active"
                    ),
                    "progress_score": goal.progress_score,
                    "stall_counter": goal.stall_counter,
                    "rumination_counter": goal.rumination_counter,
                    "failure_count": goal.failure_count,
                },
            })
            # Completed goals get a separate completion entry
            if goal.completed and goal.last_progress_at > 0:
                entries.append({
                    "timestamp": goal.last_progress_at,
                    "iso_time": _iso(goal.last_progress_at),
                    "tick": None,
                    "category": "goal",
                    "event_type": "goal_completed",
                    "content": {
                        "id": goal.id,
                        "description": goal.description,
                        "final_progress": goal.progress_score,
                    },
                })

        # --- 6. Mood history (downsampled) ---
        mood_entries = self._downsample_mood_history()
        for mood in mood_entries:
            ts = mood.get("time", 0.0)
            entries.append({
                "timestamp": ts,
                "iso_time": _iso(ts),
                "tick": mood.get("tick"),
                "category": "emotion",
                "event_type": "mood_snapshot",
                "content": {
                    "emotions": mood.get("emotions", {}),
                    "arousal": mood.get("arousal", 0.0),
                    "valence": mood.get("valence", 0.0),
                    "decision": mood.get("decision", ""),
                },
            })

        # --- Sort by timestamp ---
        entries.sort(key=lambda e: e["timestamp"])

        # --- Build metadata ---
        now = time.time()
        metadata = {
            "entity_name": self.state.name,
            "personality": (
                self.identity.personality.strip()
                if self.identity.personality else ""
            ),
            "session_start": self.state.start_time,
            "session_start_iso": _iso(self.state.start_time),
            "export_time": now,
            "export_time_iso": _iso(now),
            "session_duration_seconds": round(now - self.state.start_time, 1),
            "total_ticks": self.state.tick_count,
            "tick_interval_seconds": 15,
            "total_timeline_entries": len(entries),
            "total_history_events": len(self.state.history),
            "total_conversations": sum(
                len(msgs) for msgs in self.state.conversations.values()
            ),
            "total_episodes": len(self.memory.episodes),
            "total_goals": len(self.goal_engine.goals),
            "total_mood_snapshots": len(self.state.mood_history),
            "mood_snapshots_included": len(mood_entries),
            "thinking_history_cap": 50,
            "thinking_history_count": len(self.thinking_history),
            "thinking_history_note": (
                "Capped at last 50 entries. Earlier thoughts are "
                "not available."
                if len(self.thinking_history) >= 50
                else "All thoughts included."
            ),
            "prediction_summary": {
                "average_error": (
                    self.prediction_engine.average_recent_error()
                ),
                "recent_errors": self.prediction_engine.recent_errors[-10:],
                "running_satisfaction": (
                    self.prediction_engine._running_satisfaction
                ),
            },
            "current_needs": self.needs.to_dict(),
            "current_emotions": self.emotions.to_dict(),
            "current_energy": self.state.energy,
            "dominant_mood": self.emotions.dominant_mood(),
        }

        # --- People snapshots (context, not timeline entries) ---
        people_snapshot: dict[str, Any] = {}
        for pid, p in self.state.people.items():
            people_snapshot[pid] = {
                "name": p.name,
                "trust": p.trust,
                "familiarity": p.familiarity,
                "warmth": p.warmth,
                "disposition": p.disposition_hint(),
                "notes": p.notes,
                "topics_discussed": p.topics_discussed,
                "unresolved_tensions": p.unresolved_tensions,
                "conversation_count": p.conversation_count,
                "proactive_attempts": p.proactive_attempt_count,
                "unanswered_proactive": p.unanswered_proactive_count,
            }

        return {
            "metadata": metadata,
            "people": people_snapshot,
            "timeline": entries,
        }

    def _downsample_mood_history(self) -> list[dict[str, Any]]:
        """Downsample mood_history to only include meaningful changes.

        Includes a mood snapshot when:
        - It's the first or last entry
        - Any emotion dimension changed by > 0.1 since last included
        - Arousal or valence changed by > 0.15
        - The decision type changed

        For short sessions (<= 50 entries), includes everything.
        """
        history = self.state.mood_history
        if not history:
            return []
        if len(history) <= 50:
            return list(history)

        result: list[dict[str, Any]] = [history[0]]
        last_included = history[0]

        for entry in history[1:-1]:
            should_include = False

            # Check emotion dimension changes
            last_emo = last_included.get("emotions", {})
            curr_emo = entry.get("emotions", {})
            for key in curr_emo:
                if abs(curr_emo.get(key, 0) - last_emo.get(key, 0)) > 0.1:
                    should_include = True
                    break

            # Check arousal/valence changes
            if (abs(entry.get("arousal", 0)
                    - last_included.get("arousal", 0)) > 0.15):
                should_include = True
            if (abs(entry.get("valence", 0)
                    - last_included.get("valence", 0)) > 0.15):
                should_include = True

            # Check decision change
            if entry.get("decision") != last_included.get("decision"):
                should_include = True

            if should_include:
                result.append(entry)
                last_included = entry

        result.append(history[-1])
        return result


def _try_extract_target(raw: str) -> str | None:
    """Try to extract target_person from raw LLM output."""
    import json
    try:
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{"):
                    raw = cleaned
                    break
        data = json.loads(raw) if raw.startswith("{") else None
        if data:
            return data.get("target_person")
    except Exception:
        pass
    return None
