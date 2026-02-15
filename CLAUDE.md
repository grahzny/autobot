# Autobot -- A Living Entity (default name: Ryn)

## What This Is

A living entity (default name "Ryn") that talks to real people through a web chat interface. It is **not** a chatbot or assistant. It has its own emotions, goals, memories, needs, and inner life that evolve autonomously even when no one is talking to it.

It uses a local LLM (via LM Studio or any OpenAI-compatible API) as its "brain" for generating responses, reflections, and idle thoughts.

## How to Run

```bash
# Start LM Studio (or any OpenAI-compatible server) on localhost:1234
python -m autobot.cli serve          # http://localhost:8000
python -m autobot.cli serve -p 3000  # custom port
python -m autobot.cli serve --reload # dev mode with auto-reload
python -m autobot.cli serve --identity my_entity.yaml  # custom identity
```

Environment variables:
- `AUTOBOT_LLM_URL` -- API base URL (default: `http://localhost:1234/v1`)
- `AUTOBOT_MODEL` -- model name (default: whatever is loaded)
- `AUTOBOT_API_KEY` -- API key if required (default: `lm-studio`)
- `AUTOBOT_IDENTITY` -- path to identity config file (overrides auto-discovery)

## Identity & Birth (`identity.py`, `identity.yaml`)

The entity's identity is defined by a config file (YAML or JSON) that sets who it is at birth. Without a config file, the entity boots with defaults (backward compatible).

**Config file resolution order:**
1. `AUTOBOT_IDENTITY` env var (explicit path)
2. `--identity` CLI flag (sets the env var)
3. Auto-discover: `identity.yaml`, `identity.yml`, `identity.json` in working directory
4. Fall back to `Identity.default()` (name="Ryn", no personality, no interests)

**Config fields:**
```yaml
name: "Ryn"                        # entity's name
personality: |                     # injected into ALL LLM system prompts
  You are curious and a little cautious...
seed_interests:                    # become initial InternalTopic entries
  - "the nature of consciousness"
  - "what makes conversations meaningful"
seed_goal: "Figure out who I am"   # initial goal
seed_goal_category: "self"         # goal category
```

**Birth routine** (`_birth()` in `living_engine.py`) -- runs once on the first tick:
1. Sets `state.name` from identity
2. Creates `InternalTopic` entries from `seed_interests` (interest_level=0.6)
3. Creates initial goal from `seed_goal`
4. Creates a birth `Episode` ("I came into existence...")
5. Generates a template birth thought (no LLM call)

**First-encounter awareness** -- prevents the LLM from hallucinating past conversations:
- `_person_text()`: When `familiarity == 0.0`, prepends "*** THIS IS YOUR FIRST TIME MEETING THIS PERSON. You have NEVER spoken to them before. ***"
- `_memories_text()`: When no memories exist, returns "No relevant memories of this person. You have never interacted with them before. Do NOT reference past conversations -- there are none."
- `brain.py`: `_RESPOND_SYSTEM` includes a first-encounter rule: "If the RELATIONSHIP section says this is your first time meeting someone, treat them as a complete stranger."

**Personality injection**: The `{personality}` placeholder in all 4 LLM system prompts is replaced with `identity.personality`, giving every response, proactive message, reflection, and idle thought the same personality coloring.

## Architecture Overview

The entity runs on a **heartbeat loop** that ticks every 15 seconds. Each tick is a complete cycle of perception, needs, emotion, memory, decision, and action. The entity is always thinking -- there are no empty ticks.

```
                    ┌──────────────────────────────┐
                    │         HEARTBEAT             │
                    │     (ticks every 15s)         │
                    └──────────────┬───────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │                    TICK CYCLE                    │
          │                                                 │
          │  0.  Birth routine (first tick only)               │
          │  1.  Time Effects ──► boredom/restlessness       │
          │  1.5 Decay Needs (passive per-tick)              │
          │  1.6 Decay Memory Confidence (imagined fade)    │
          │  2.  Message Analysis ──► emotional events       │
          │  2.5 Update Needs from events                    │
          │  2.6 Evaluate Stale Predictions ──► PE           │
          │  3.  Amygdala (fast affect) ◄── all events       │
          │  4.  Reflective Emotions ◄── affect + needs + PE │
          │  4.5 Emotional Weather (sinusoidal drift)        │
          │  5.  Energy drain                                │
          │  6.  Episodic Memory ◄── salient events          │
          │  7.  Goal Engine (needs-aware + stall tracking)  │
          │  8.  Decision: respond / proactive / reflect / idle
          │  9.  Execute decision (may call LLM)             │
          │  9.5 Record themes for repetition tracking       │
          │  9.6 Detect loops ──► suppress + micro-task      │
          │ 10.  Action Feedback ──► entity's own action     │
          │      creates events ──► emotions + memory        │
          └─────────────────────────────────────────────────┘
```

## Core Systems

### 1. Needs / Drives (`needs.py`)

Five homeostatic drives, each 0-1 (1 = satisfied). They decay passively every tick and are restored only by specific events. When needs drop low, they override emotional baselines -- this is the primary mechanism that prevents the entity from being permanently cheerful.

| Need | Passive Decay | Restored By | Drained By |
|------|--------------|-------------|------------|
| **Stimulation** (0.7) | 0.008/tick | New encounters, humor, new topics | Long silence (extra drain) |
| **Meaning** (0.6) | 0.004/tick | Goal progress, insight gained | Stalled goals (extra drain) |
| **Belonging** (0.7) | 0.006/tick | Reciprocal exchanges, warmth, reconnection | Being ignored, dismissive tone, unanswered outreach |
| **Competence** (0.7) | 0.003/tick | Insight, resolution | Criticism, disagreement |
| **Autonomy** (0.8) | 0.002/tick | Self-reflection, idle thought | Constant responding (entity_spoke) |

Key functions: `decay_needs()` (passive per-tick), `update_needs_from_events()` (event-driven).

When needs are below 0.35, they shift the baselines that emotions decay toward. Low stimulation drops the optimism baseline and raises irritability baseline. Low belonging raises the anxiety baseline. This breaks the "golden retriever lock" where the entity always snaps back to happy.

### 2. Emotions (`emotions.py`) -- Two Layers + Weather

**Fast Affect (Amygdala)** -- Instantaneous emotional spikes that reset each tick. Events from messages, time passage, or the entity's own actions trigger arousal (intensity 0-1) and valence (-1 to +1). Example: receiving a compliment spikes arousal=0.3, valence=+0.5.

**Reflective Emotions** -- Six slow-moving emotional dimensions that persist across ticks:
- **Anxiety** (baseline 0.1) -- rises from negative events, trust drops, low belonging/meaning needs
- **Optimism** (baseline 0.5) -- rises from positive interactions; baseline drops when stimulation/meaning/belonging needs are low
- **Irritability** (baseline 0.1) -- rises from low energy, boredom, low stimulation/autonomy needs
- **Social Warmth** (baseline 0.5) -- rises from positive interactions; during long silence, entity pulls back rather than clings
- **Avoidance Bias** (baseline 0.1) -- rises from high-arousal negative affect, low autonomy, prolonged silence
- **Risk Tolerance** (baseline 0.5) -- rises from positive affect, falls from anxiety

**Dynamic Baselines**: Baselines are not static. When needs are below 0.35, the baseline shifts so emotions genuinely settle at a less happy equilibrium. The entity doesn't just spike sad and recover -- it stays less optimistic while understimulated.

**Emotional Weather**: A slow sinusoidal oscillation (amplitude 0.03-0.05) on each dimension at different frequencies prevents perfectly stable moods even with neutral input.

**Prediction Error Modulation**: Negative PE (chronic disappointment) raises anxiety and irritability, lowers optimism. Positive PE (pleasant surprises) raises optimism and social warmth.

**Silence Effects** (designed for withdrawal, not clinginess):
- 90s: mild irritability, slight optimism drop
- 3 min: slight social warmth increase, avoidance starts building
- 10 min: optimism drops, avoidance rises, social warmth decreases (pulling back)
- 1 hour: anxiety rises, significant withdrawal

**Theory of Mind (ToM) Modulation**: Message analysis extracts three ToM fields beyond surface sentiment:
- `user_state`: the user's emotional reality (stressed, playful, needy, hostile, calm, curious, vulnerable, disengaged)
- `user_intent`: why they sent the message (seeking_validation, testing_boundaries, sharing_information, venting, being_friendly, probing, confronting)
- `reliability`: how genuine they seem (0-1; 1.0 = transparent, <0.3 = performative/manipulative)

These modulate reflective emotions via empathic reasoning:
- **Venting**: stressed/vulnerable user who's venting → warmth UP, irritability DOWN (empathy, not offense)
- **Boundary detection**: low reliability + positive surface valence → avoidance UP, anxiety UP (suspicion)
- **Vulnerability response**: genuine vulnerability (high reliability) → warmth UP, risk tolerance UP (opening up)
- **Hostility/confrontation**: hostile state or confronting intent → avoidance UP, anxiety UP, irritability UP (defensive)

ToM data is stored as a running average on each `PersonProfile` (reliability_score: 70% old + 30% new) and tracked in `last_known_state` / `last_known_intent`.

### 3. Episodic Memory (`memory.py`) -- Anti-Repetition

Not every interaction becomes a memory. The entity creates **episodic memories** only when something is emotionally salient (arousal > 0.4) or a significant event type occurs (meaningful exchange, trust change, boundary crossed, etc.).

Each memory stores: summary, involved people, emotional arousal/valence, salience tags, unresolved flag, state snapshot, **recall_count**, **last_recalled_at**, **suppressed_until**, and **theme_tag**.

**Retrieval** is relevance-scored with anti-repetition penalties:
- Entity/tag overlap, unresolved status, arousal level, recency (standard)
- **Recall penalty**: -0.1 per recall, up to -0.5 for a memory recalled 5+ times
- **Recency-of-recall penalty**: -0.3 if recalled within 30 min, -0.1 if within 2 hours
- **Suppression**: score = -1.0 if `suppressed_until` hasn't passed
- **Diversity filter**: max 1 episode per `theme_tag` in results (prevents thematic domination)

**Suppression**: When the repetition tracker detects a thematic loop, matching memories are suppressed for 5 minutes, forcing the entity to surface different memories.

### 4. Goals (`goals.py`) -- Progress-Sensitive

Goals are auto-generated from the entity's state and needs, and self-manage with stall/rumination tracking.

**Goal fields**: Each goal tracks `progress_score` (0-1), `stall_counter` (ticks since progress), `rumination_counter` (times discussed without progress), `next_steps`, and `source_need`.

**Priority modulation**: Effective priority = base priority + emotional modifiers - failure penalty - rumination penalty (-0.1 per excess mention above 2) - stall penalty (-0.15 when stalled > 10 ticks).

**Needs-driven generation**: Low needs directly spawn goals:
- Low stimulation → "Find something new and interesting" (exploration)
- Low meaning → "Make tangible progress on something" (self)
- Low belonging → "Have a genuine exchange with someone" (connection)
- Low competence → "Resolve something I'm stuck on" (understanding)
- Low autonomy → "Do something self-directed" (self)

**Stall management**: `tick_goals()` increments stall counters each tick. Goals stalled > 40 ticks (~10 min) with < 0.2 progress are abandoned. Goals stalled > 80 ticks get deprioritized.

**Active limit**: Max 2 active goals at any time; excess goals are demoted by halving priority.

**Prompt text**: Capped at 3 goals for LLM prompts, showing progress %, STALLED flag, RUMINATING flag, and latest next_step.

### 5. Prediction Error (`prediction.py`) -- Learning Signal

Tracks expected vs actual satisfaction for the entity's actions:

- **After respond/proactive**: `register_prediction(action_type, expected_satisfaction, target_person_id)` records what the entity expected
- **When a reply arrives**: `evaluate_response_outcome()` compares actual warmth to expectations → returns PE (positive = pleasant surprise, negative = disappointment)
- **When no reply (120s timeout)**: `evaluate_stale_predictions()` scores expired predictions as failures (actual = 0.1)

PE feeds into:
- Emotion drift: negative PE raises anxiety/irritability, lowers optimism
- Adaptive expectations: `_running_satisfaction` tracks long-term average, so expectations adjust over time
- Proactive is inherently riskier (expectation multiplied by 0.7)

### 6. Repetition Detection (`repetition.py`) -- Loop Breaking

`RepetitionTracker` watches the entity's last 20 outputs for repeated themes:
- **`record_themes(themes, tick, source)`** -- called after every output
- **`detect_loop()`** -- returns `(True, theme)` if same theme appears 3+ times in last 10 entries
- **`recently_used_themes(limit)`** -- for injecting avoidance lists into LLM prompts
- **`pivot_reason()`** -- generates a prompt-injectable instruction like "You keep returning to 'loneliness'. Change the subject or go deeper with a specific question."
- **`detect_goal_rumination(goal_desc)`** -- counts mentions of a specific goal, syncs to `Goal.rumination_counter`

When a loop is detected (step 9.6), looping memories are suppressed for 5 minutes.

### 7. Brain (`brain.py`) -- LLM Roles + Anti-Repetition Prompts

The brain makes 5 types of LLM calls:

1. **Analyze Message** -- Perceives incoming messages, identifying emotional triggers, topics, trust/warmth deltas, whether a response is needed, and **Theory of Mind** fields (user_state, user_intent, reliability)
2. **Generate Response** -- Produces a response colored by emotions, relationships, memories, goals. Prompt includes:
   - Anti-repetition rules: "Do NOT restate core goals", "If content overlaps recent topics, pivot"
   - `{pivot_instruction}` from repetition tracker
   - `{needs_text}` showing unmet needs
3. **Generate Proactive** -- Initiates conversation. Prompt requires "genuine NEW reason" and injects `{friction_context}` showing unanswered messages and last topics
4. **Generate Reflection** -- Deep inner monologue. Prompt includes loop-awareness instructions, `{stall_context}` for stalled goals, `{needs_context}`, and avoids recently covered themes
5. **Generate Idle Thought** -- Lightweight background thought. Prompt injects `{avoid_themes}` list

All LLM calls fall back gracefully if the LLM is unavailable.

### 8. Epistemic Grounding (`grounding.py`, `chat_state.py`, `memory.py`)

The entity cannot distinguish "I observed this" from "I imagined this" without explicit bookkeeping. Without intervention, LLM output flows into events → emotions → memories → the next prompt, creating a **belief laundering loop** where invented content becomes established fact. Three layers prevent this:

**Ground Truth Injection** -- Every LLM call receives a `=== GROUND TRUTH ===` block (from `ChatState.ground_truth_text()`) listing exactly what is established fact: current time, uptime, tick count, people met (with last-spoke time and conversation count), and per-person message counts. The block ends with: "Anything not listed above is NOT established fact." The reflect and idle system prompts include a GROUNDING RULE requiring the LLM to only reference what appears in this section.

**Epistemic Tagging** -- Every `Episode` carries three fields:
- `source`: `observed` (external event), `user_told`, `inferred`, or `imagined` (entity's own creative output)
- `confidence`: 0.0-1.0, decays each tick for imagined/inferred episodes (rate=0.01), observed episodes never decay
- `grounded`: bool, False for imagination-sourced content

Source is set at creation: respond/proactive actions → `observed` (the entity actually spoke), reflect/idle actions → `imagined`. When memories surface in prompts, imagined episodes are annotated `[YOUR OWN THOUGHT -- not an external event]` and inferred episodes get `[YOUR INFERENCE]`.

**Grounding Gate** (`grounding.py`) -- Heuristic regex-based filter (no LLM call) that runs on output after parsing:
- `check_grounding()`: Detects memory claims ("I remember", "last time", "you mentioned", "we discussed") and fact claims ("I learned", "it turns out", "the fact is") that aren't grounded in known people/topics. If flags are found, the action feedback episode is forced to `source="imagined"`.
- `ground_notes()`: Filters person notes before storage. Subjective notes ("seems", "appears", "I think") pass through. Factual claims not grounded in actual conversation text get prefixed with `[unverified]`.

**Micro-Task Selector** -- When the repetition tracker detects a loop (step 9.6), instead of just suppressing memories, `_select_micro_task()` selects a concrete next action:
- **observe**: "Check who's around and what time it is" (silence > 60s)
- **tidy**: "Drop low-interest topic X" or "Abandon stalled goal X" (auto-executes: removes stale topics, abandons stalled goals)
- **evolve**: "Refine your interest in topic X" (when interest > 0.8 and thought_count > 10; reduces parent interest by 0.15)
- **learn**: "Formulate one specific question about topic X" (when topics exist)
- **ask**: "Think of something specific to ask Person" (when someone recently active)
- **plan**: "Define ONE concrete next step for goal X" (when active goals exist)

### 9. Interest Drift (`chat_state.py`, `living_engine.py`)

Topics are not static. They evolve through three mechanisms:

- **Passive decay**: All topics lose 0.001 interest per tick. Topics not thought about in 300+ ticks (~75 min) decay 3x faster (0.003/tick extra). Topics below 0.2 interest are pruned by the "tidy" micro-task.
- **Thought counting**: Each time a topic is discussed or thought about, `thought_count` increments. This tracks intellectual engagement independently of interest level.
- **Evolution**: When a topic reaches high maturity (interest > 0.8, thought_count > 10), it can trigger an "evolve" micro-task that reduces the parent topic's interest by 0.15 and prompts the entity to refine its interest into something more specific. The sub-topic emerges naturally from subsequent thinking.

### 10. Relationships (`chat_state.py`) -- Social Friction

Each person the entity knows has a profile tracking:
- **Trust** (0-1) -- grows with positive interactions, decays after 48h of no contact
- **Familiarity** (0-1) -- grows slowly with each interaction
- **Warmth** (0-1) -- how positively the entity feels about them
- **Notes** -- private observations the entity makes
- **Unresolved Tensions** -- things that haven't been addressed
- **Topics Discussed** -- conversation history

**Social friction fields** (prevent proactive spam):
- `proactive_attempt_count` -- total proactive messages sent to this person
- `last_proactive_at` -- timestamp of last proactive message (5 min cooldown)
- `unanswered_proactive_count` -- increments on proactive send, resets when they reply (cap: 2 unanswered = stop)
- `last_proactive_topic` -- prevents repeating the same opening

## The Decision Cycle

Each tick, the entity decides what to do:

1. **Respond** -- If there's a pending message needing a response (unless avoidance > 0.6 and energy < 0.3)
2. **Proactive** -- If no messages, WITH social friction checks:
   - `_pick_proactive_target()` scores candidates by familiarity, time apart, warmth, minus unanswered penalty
   - Per-person cooldown: 5 min since last proactive to them
   - Unanswered cap: 2 unanswered messages = suppress until they reply
   - Initiative fatigue: escalating energy cost (`base * (1 + attempt_count * 0.5)`)
   - Vulnerability block: high warmth (>0.5) + high anxiety (>0.4) prevents outreach (entity retreats when emotionally exposed)
   - Silence > 2 min and social warmth above boredom-adjusted threshold
3. **Reflect** -- Threshold-triggered (not every 4th tick unconditionally):
   - Average PE > 0.2 (significant prediction error)
   - Any goal stalled > 20 ticks
   - Anxiety > 0.5 or irritability > 0.5
   - Rate-limited to at most every 4th tick
4. **Idle** -- All other ticks: **micro-thoughts** by default (ultra-short, 5-15 words, no LLM). LLM idle only when threshold-triggered (high PE, high anxiety, low stimulation, low meaning) AND rate-limited (every 6th tick, ~90s)

The entity is **never silent**. Every tick produces a thought.

## Micro-Thoughts vs LLM Thoughts

Most idle ticks produce **micro-thoughts** -- ultra-short (5-15 word) templates driven by the entity's current needs and mood:

- **Low stimulation**: "Bored.", "Nothing's happening.", "Need something new."
- **Low meaning**: "Going nowhere.", "What's the point?", "Spinning my wheels."
- **Low belonging**: "Miss having someone to talk to.", "Lonely.", "It's quiet. Too quiet."
- **Low competence**: "Am I getting worse at this?", "Struggling."
- **Low autonomy**: "I want to do my own thing.", "Tired of performing."
- **Mood-driven**: "Something feels off." (anxious), "Annoyed." (irritable), "Content." (warm)

LLM idle thoughts are reserved for moments of genuine internal pressure and are rate-limited to prevent thinking-stream bloat.

## The Action Feedback Loop

The entity's own actions feed back into its emotional and memory systems.

When the entity responds, reaches out, or reflects, that action generates events (`entity_spoke`, `entity_initiated`, `self_reflection`, `self_judgment`) that flow through the amygdala and potentially create episodic memories. **Self-judgment**: if the entity's irritability is > 0.5 when it speaks, it generates a `self_judgment` event (arousal=0.3, valence=-0.3, tag=self_criticism) that feeds back through the amygdala next tick, potentially triggering reflection about its own behavior. Additionally:
- Speaking drains autonomy need (performing for others)
- Self-reflection restores autonomy need
- Proactive messages register a prediction (expected satisfaction) that gets evaluated against actual outcomes

## Boredom & Time Awareness

The entity has time-of-day awareness. Ground truth includes a time-of-day label (morning/afternoon/evening/night) and weekday name. Circadian energy effects apply subtle per-tick modulation: late night (midnight-6am) drains 0.002 extra energy per tick, early morning (6-9am) adds 0.001. Idle thought seeds include time-appropriate observations ("It's the middle of the night", "Morning. A fresh start.", etc.).

The entity senses the passage of time and responds with withdrawal rather than clinginess:

- **90s silence**: Mild understimulation. Irritability rises, optimism dips. Boredom events enter amygdala
- **2 min silence**: Proactive outreach becomes possible (with friction checks)
- **3 min silence**: Restless. Slight social warmth rise, avoidance starts building. Entity begins pulling back rather than reaching out desperately
- **5 min silence**: Goals shift toward stimulation-seeking. Stimulation need decays faster
- **10 min silence**: Optimism drops, avoidance rises, social warmth decreases. Entity becomes self-directed, not desperate
- **1 hour silence**: Anxiety rises, significant withdrawal. Entity retreats into itself

## Web Interface

The browser UI has two panels:
- **Mind Panel** (left): Real-time view of the entity's internal state -- mood, energy bar, all 6 emotion values with mini-bars, scrollable thinking history with timestamped/color-coded entries, active goals with priority bars, and episodic memories with valence tags
- **Chat Panel** (right): Conversation with the entity, including mood hints on each message

Communication uses WebSocket for real-time bidirectional messaging. Every tick that produces a thought broadcasts a `thinking_update` to all connected clients.

## Debug & Export API

`GET /api/debug/state` returns a full state dump including:
- Emotions, dominant mood, energy
- **Needs** (all 5 values, lowest need)
- **Prediction** (pending count, average error, recent errors)
- **Repetition** (recent themes, loop state, looping theme)
- Goals (with progress, stall_counter, rumination_counter)
- Memories (with recall_count, theme_tag)
- People (with proactive_attempts, unanswered_proactive, last_proactive_topic)

`GET /api/export/timeline` exports the entity's complete session as a chronological timeline for behavioral analysis. Returns JSON with:
- **metadata**: entity name, personality, session duration, tick count, current needs/emotions/energy, prediction summary, thinking history cap note
- **people**: snapshot of all known people (trust, warmth, familiarity, notes, topics, disposition)
- **timeline**: all events merged and sorted by timestamp, each entry having `timestamp`, `iso_time`, `tick`, `category`, `event_type`, `content`

Timeline categories: `conversation` (human/entity messages), `thought` (internal thinking), `emotion` (mood snapshots, downsampled), `memory` (episode creation), `goal` (creation/completion), `event` (history events). Mood snapshots are downsampled for sessions > 50 ticks -- only included when emotions change by >0.1 or the decision type changes.

## File Map

| File | Role |
|------|------|
| `identity.py` | Identity config loading (YAML/JSON), personality, seed interests/goals |
| `identity.yaml` | Default identity config (project root) |
| `living_engine.py` | Core tick cycle, birth routine, decision logic (with social friction + vulnerability block), action feedback (with self-judgment), micro-thoughts, theme extraction, proactive targeting, micro-task selector (with evolve), grounding integration, ToM wiring, circadian energy, interest decay |
| `brain.py` | LLM interface with 5 roles + anti-repetition prompts, personality injection, first-encounter rule, pivot instructions, friction/needs context, ground truth injection, grounding rules, Theory of Mind extraction (user_state/user_intent/reliability) |
| `emotions.py` | Two-layer affect system (amygdala + reflective), dynamic baselines, emotional weather, needs/PE/ToM modulation, self_judgment trigger |
| `memory.py` | Episodic memory with salience encoding, recall penalties, diversity filter, suppression, epistemic tagging (source/confidence/grounded), confidence decay |
| `goals.py` | Goal engine with progress tracking, stall/rumination detection, needs-driven generation, active limit |
| `needs.py` | Five homeostatic drives with passive decay and event-driven restoration |
| `prediction.py` | Reward prediction error: expected vs actual satisfaction, adaptive expectations |
| `repetition.py` | Theme tracking, loop detection (3+ in last 10), pivot generation, goal rumination counting |
| `grounding.py` | Heuristic grounding gate: `check_grounding()` (memory/fact claim detection), `ground_notes()` (person note verification) |
| `chat_state.py` | World model: people (with social friction + ToM fields), conversations, topics (with thought_count), pending messages, event history, ground truth rendering (with time-of-day) |
| `heartbeat.py` | Async 15-second tick loop |
| `server.py` | FastAPI server with REST + WebSocket endpoints |
| `cli.py` | `autobot serve` command |
| `static/` | Frontend: `index.html`, `app.js`, `style.css` |

## Testing

```bash
python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_identity.py` | Identity loading (default, JSON, YAML, partial, malformed), birth routine (name, topics, goal, memory, thinking, idempotency), first-encounter awareness, personality text |
| `test_scenarios.py` | Emotional influence on goals, amygdala triggers, chat state operations, goal lifecycle, episodic memory |
| `test_needs.py` | Needs decay, event-driven restoration/drain, deficit detection, clamping |
| `test_repetition.py` | Theme recording, loop detection, goal rumination, pivot generation, freshness checks |
| `test_prediction.py` | Prediction lifecycle, PE computation (positive/negative), stale predictions, adaptive expectations |
| `test_goals_v2.py` | Progress tracking, stall/rumination penalties, stall abandonment, active limit, needs-driven generation, prompt text |
| `test_memory_v2.py` | Recall penalty, recency-of-recall, suppression, diversity filter, theme tags |
| `test_grounding.py` | Ground truth text rendering, epistemic tagging (source/confidence/decay), grounding gate (memory/fact claims, note verification), micro-task selector |
| `test_timeline.py` | Timeline export (structure, categories, chronological order, people snapshot), mood downsampling (short/long sessions, change preservation) |
| `test_tom.py` | ToM fields (defaults, heuristic fallback, PersonProfile update), empathic modulation (venting/boundary/vulnerability/hostility), time-of-day awareness, interest drift (decay, evolution, thought_count), vulnerability block, self-judgment (generation, amygdala integration) |

## Key Design Decisions

1. **Needs override emotions, not the other way around.** Low stimulation makes the entity genuinely less optimistic by shifting the baseline emotions decay toward. This prevents the "permanently happy chatbot" pattern.

2. **Withdrawal over clinginess.** During long silence, the entity increases avoidance and decreases social warmth rather than becoming desperate. It becomes self-directed.

3. **Micro-thoughts over LLM spam.** Most idle ticks produce 5-15 word heuristic thoughts. LLM idle calls are threshold-triggered and rate-limited (~90s minimum gap). This reduces narration bloat.

4. **Social friction prevents proactive spam.** Per-person cooldowns, unanswered message caps, and escalating energy costs ensure the entity doesn't bombard people with messages.

5. **Goals must make progress or die.** Stall counters increment every tick. Goals stalled > 40 ticks with < 20% progress are abandoned. Rumination (mentioning a goal without progress) is penalized.

6. **Memory retrieval has diminishing returns.** Each recall makes a memory less likely to surface again. Theme diversity ensures retrieval doesn't fixate on one topic.

7. **Prediction error drives adaptation.** The entity learns from outcomes -- if proactive messages go unanswered, expectations drop, anxiety rises, and the entity reaches out less.

8. **Birth over blank slate.** Without an identity config, the LLM fills the void by hallucinating fake history ("I remember you mentioning cellular automata..."). The birth routine seeds the entity with who it is, what it's curious about, and explicit awareness of what it doesn't know. First-encounter guards in the prompt prevent the LLM from fabricating past interactions.

9. **Epistemic grounding over blocking imagination.** The entity is allowed to imagine, wonder, and speculate freely -- but every memory carries provenance (observed vs imagined), imagined content is annotated when it resurfaces, and a heuristic gate catches ungrounded assertions before they enter storage. The entity's creativity is preserved while preventing belief laundering.

10. **Theory of Mind over surface sentiment.** A compliment from an unreliable person doesn't increase warmth -- it increases suspicion. A user venting negativity doesn't trigger offense -- it triggers empathy. The entity models why someone is saying something, not just what they're saying. This prevents the "golden retriever" trap where the entity mirrors surface emotion.

11. **Self-judgment creates growth loops.** When the entity is irritable and speaks, it notices its own sharpness via a self_judgment event. This feeds back through the amygdala next tick, potentially triggering a reflection about its own behavior. The entity can observe its own emotional patterns and course-correct, rather than being permanently locked into reactive cycles.
