# Autobot -- Market Trading Entity (default name: Ryn)

## What This Is

A market-trading entity (default name "Ryn") that autonomously analyzes stock markets, forms trading theses, and executes simulated trades. It operates with real paid LLM API calls deducted from an Operating Capital budget ($1,000 starting). When capital reaches $0, the entity dies.

Chris (the user) is the entity's sole human contact -- a "Secondary Intelligence Source" who can provide market insights, ask about analysis, or give strategic direction through a web chat interface.

The entity has emotions, economic drives, episodic memory, and an inner life that evolve autonomously based on market events and trading outcomes.

## How to Run

```bash
# Start an OpenAI-compatible API server (LM Studio, or any provider)
python -m autobot.cli serve          # http://localhost:8000
python -m autobot.cli serve -p 3000  # custom port
python -m autobot.cli serve --identity identity.yaml
```

Environment variables:
- `AUTOBOT_LLM_URL` -- API base URL (default: `http://localhost:1234/v1`)
- `AUTOBOT_MODEL` -- model name (default: whatever is loaded)
- `AUTOBOT_API_KEY` -- API key if required (default: `lm-studio`)
- `AUTOBOT_IDENTITY` -- path to identity config file

## Identity & Configuration (`identity.py`, `identity.yaml`)

```yaml
name: "Ryn"
personality: |
  You are a Strategic Principal. Your primary drive is capital accumulation
  through disciplined stock market analysis.
seed_interests: ["ASX value stocks", "US tech sector momentum"]
seed_goal: "Build a profitable trading thesis and execute first trade"
seed_goal_category: "trading"
starting_capital: 1000.00
timezone: "Australia/Melbourne"
token_cost_input: 3.00       # $/1M input tokens
token_cost_output: 15.00     # $/1M output tokens
brokerage_fee: 6.00
slippage_pct: 0.0005         # 0.05%
max_position_pct: 0.20       # max 20% of portfolio per position
seed_watchlist: ["BHP.AX", "CBA.AX", "CSL.AX", "AAPL", "MSFT"]
```

## Architecture Overview

The entity runs on an **adaptive heartbeat loop** that ticks at intervals from 30s (market open, positions held) to 900s (capital critical). Each tick is a 15-step cycle:

```
                    ┌──────────────────────────────┐
                    │      MARKET HEARTBEAT         │
                    │   (adaptive 30s-900s)         │
                    └──────────────┬───────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │              MARKET TICK CYCLE (15 steps)        │
          │                                                 │
          │  0.  Birth routine (first tick only)             │
          │  1.  Starvation check (capital <= 0 → death)    │
          │  2.  Refresh daily budget                        │
          │  3.  Perceive:                                   │
          │      a. Fetch prices (yfinance, rate-limited)    │
          │      b. Check price alerts                       │
          │      c. Fetch news (rate-limited)                │
          │      d. Process Chris messages                   │
          │      e. Detect market session transitions        │
          │  4.  Decay economic state                        │
          │  5.  (events already in step 3)                  │
          │  6.  Amygdala (fast affect from market events)   │
          │  7.  Reflective emotions + emotional weather     │
          │  8.  Appraise:                                   │
          │      a. Portfolio mark-to-market                  │
          │      b. Starvation warning                       │
          │      c. Update economic state from metrics        │
          │  9.  Strategize (auto-generate objectives)       │
          │ 10.  Decide: respond_chris | analyze | monitor   │
          │                | synthesize | trade | idle       │
          │ 11.  Execute (LLM with budget check, or idle)    │
          │ 12.  Record themes for repetition tracking        │
          │ 13.  Post-mortem (if trade closed)                │
          │ 14.  Action feedback → emotions/memory            │
          └─────────────────────────────────────────────────┘
```

### Decision Priority

1. Always respond to Chris if message pending
2. Sunday → strategic synthesis
3. Capital < $5 → idle (survival mode)
4. Open positions + market open → monitor (check stop-loss/target)
5. Market open + budget → analyze watchlist ticker
6. Default → idle (micro-thought, no LLM cost)

### Adaptive Tick Interval

| Condition | Interval |
|-----------|----------|
| Chris message pending | 30s |
| Market open + positions | 30s |
| Market open, no positions | 60s |
| Sunday (synthesis) | 300s |
| Weekday, market closed | 300s |
| Saturday | 600s |
| Capital < $10 | 600s |
| Capital < $5 | 900s |

## Core Systems

### 1. Economic State (`needs.py`) -- Market-Driven Needs

Four economic indicators (0-1 each) replace the conversational drives:

| Indicator | Default | Decay | Restored By | Drained By |
|-----------|---------|-------|-------------|------------|
| **alpha** (Meaning) | 0.5 | 0.003/tick | Profitable trades, high-conviction research, thesis confirmation | Losses, thesis invalidation |
| **roi** (Competence) | 0.5 | 0.002/tick | Prediction accuracy, profitable trades, good Sharpe | Prediction errors, losses |
| **volatility** (Anxiety) | 0.3 | 0.005/tick (settling) | Recovery, positive news, profits | Drawdown, starvation, negative news |
| **cost_pressure** (Irritability) | 0.3 | 0.003/tick (settling) | Budget underspend | High token spend, starvation |

When economic state is poor, emotions shift and the entity becomes more cautious and cost-conscious.

### 2. Emotions (`emotions.py`) -- Market-Linked Two-Layer System

**Fast Affect (Amygdala)** -- Instantaneous spikes from market events:
- `trade_profit` → arousal + positive valence
- `trade_loss` → high arousal + negative valence
- `starvation_warning` → very high arousal + very negative valence
- `thesis_confirmed/invalidated`, `price_alert`, `news_positive/negative`, `drawdown`, `recovery`

**Reflective Emotions** -- Six slow-moving dimensions:
- **Anxiety** (baseline 0.1) -- rises from losses, low ROI, drawdown
- **Optimism** (baseline 0.5) -- rises from profits; baseline drops with low alpha
- **Irritability** (baseline 0.1) -- rises from cost pressure, low alpha
- **Conviction** (baseline 0.5) -- confidence in positions/theses (was `social_warmth`)
- **Caution** (baseline 0.2) -- tendency to wait (was `avoidance_bias`)
- **Risk Tolerance** (baseline 0.5) -- willingness to trade

**Dynamic Baselines**: Low alpha shifts optimism baseline down. High volatility shifts caution baseline up.

### 3. Accountant (`accountant.py`) -- Token Cost Tracking

Every LLM call deducts from Operating Capital. The Accountant enforces:
- **Daily budget** = operating_capital / 30 (survive for a month)
- **Role budgets**: analysis=40%, trading=15%, synthesis=15%, market_scan=15%, chat=10%, emergency=5%
- **Budget gating**: `should_use_llm(role, estimated_tokens)` checks affordability before each call
- **Starvation**: FATAL ($0), CRITICAL ($5), WARNING ($20)

### 4. Portfolio (`portfolio.py`) -- Simulated Trading

Simulated execution with realistic costs:
- **Brokerage**: $6 per trade (buy or sell)
- **Slippage**: 0.05% of trade value
- **Position limits**: max 20% of portfolio value per position
- **Performance**: Sharpe ratio, win rate, total P&L

### 5. Market Monitor (`market.py`) -- yfinance Integration

- **Watchlist**: tracked tickers with price history, fundamentals cache
- **Market hours**: ASX (10:00-16:00 AEST Mon-Fri), US (09:30-16:00 ET)
- **Price alerts**: one-shot threshold triggers
- **Rate limiting**: prices every 5 min, news every 30 min

### 6. Brain (`brain.py`) -- 4 LLM Roles

All calls track token usage via Accountant:

| Role | Function | Max Tokens | Budget |
|------|----------|-----------|--------|
| **MarketAnalyst** | `analyze_market_data()` -- ticker analysis → thesis, conviction, signals, risks | 256 | analysis (40%) |
| **TradeReasoner** | `reason_trade()` -- thesis + portfolio → action, shares, stop-loss, target | 512 | trading (15%) |
| **StrategicSynthesizer** | `strategic_synthesis()` -- Sunday review → watchlist changes, adjustments | 1024 | synthesis (15%) |
| **ChatResponder** | `respond_to_chris()` -- Chris messages with market context | 512 | chat (10%) |

Chris messages are analyzed with heuristics (no LLM cost) to save budget.

### 7. Strategy Engine (`strategy.py`) -- Cost-Aware Objectives

Generates trading objectives with priority-based scheduling:
- `respond_chris` (0.95) -- always top priority
- `review` (0.8) -- monitor open positions when market open
- `sunday` (0.7) -- periodic synthesis (cooldown: 40 ticks)
- `analyze` (0.6) -- deep ticker analysis when alpha low
- `scan` (0.5) -- market scanning (cooldown: 60 ticks)

Capital-aware: skips scan/analyze when capital < $5.

### 8. Research Ledger (`research.py`) -- Structured Research

- **ResearchNote**: ticker, thesis/observation/fundamental/technical, conviction, source
- **TradeRationale**: why a trade was entered (thesis, catalysts, risks, stop-loss, target)
- **PostMortem**: what went right/wrong, lessons learned
- **Weekly Reviews**: Sunday synthesis summaries

### 9. News Harvester (`news.py`) -- Free Market Intelligence

yfinance news fetching with URL deduplication, keeps last 100 items.

### 10. Trade Predictions (`prediction.py`) -- Accuracy Tracking

Tracks direction accuracy and prediction error across resolved trades. Also retains the original social prediction system for Chris interactions.

### 11. Episodic Memory (`memory.py`) -- Market Events

Creates episodes from emotionally salient events (trade outcomes, thesis confirmations, Chris insights). Retrieval with anti-repetition penalties.

### 12. Repetition Detection (`repetition.py`) -- Loop Breaking

Tracks themes (tickers, trade actions, decisions) to prevent fixation on single stocks or patterns.

### 13. Epistemic Grounding (`grounding.py`)

Market-specific grounding: catches ungrounded market predictions ("the stock will rally", "guaranteed return", "can't lose"). Price claim validation with 5% tolerance.

## MarketState (`chat_state.py`)

The entity's world view:
- **Chris**: single `PersonProfile` with Theory of Mind (state, intent, reliability)
- **Market tracking**: last prices, session transitions, thinking log
- **Ground truth**: renders market status, portfolio, capital for LLM prompts
- **Backward compat**: `ChatState` retained alongside `MarketState`

## Server & API (`server.py`)

### Chat Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/connect` | POST | Register Chris |
| `/api/message` | POST | Send message from Chris |
| `/api/poll/{person_id}` | GET | Poll for entity responses |
| `/ws/{person_id}` | WS | Real-time bidirectional messaging |

### Market Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market/status` | GET | Session, watchlist, prices |
| `/api/portfolio` | GET | Positions, P&L, Sharpe, win rate |
| `/api/capital` | GET | Operating capital, budget, is_alive |
| `/api/research` | GET | Notes, theses, post-mortems |

### General
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Quick status (mood, capital, tick) |
| `/api/debug/state` | GET | Full internal state dump |
| `/api/thinking` | GET | Recent thinking history |

## File Map

| File | Role |
|------|------|
| `identity.py` | Identity config with market parameters (capital, timezone, costs, watchlist) |
| `identity.yaml` | Market entity config (project root) |
| `living_engine.py` | `MarketEngine`: 15-step tick cycle, decision logic, trade execution, adaptive interval. `LivingEngine`: backward compat |
| `brain.py` | 4 LLM roles with cost tracking via Accountant |
| `emotions.py` | Market-linked two-layer affect (conviction, caution replace social_warmth, avoidance_bias) |
| `needs.py` | `EconomicState`: alpha, roi, volatility, cost_pressure (backward-compat aliases for old names) |
| `accountant.py` | Token cost tracking, daily budget, role budgets, starvation check |
| `market.py` | `MarketMonitor`: yfinance wrapper, ASX/US hours, watchlist, alerts |
| `portfolio.py` | Simulated trading with fees/slippage, P&L, Sharpe |
| `strategy.py` | `StrategyEngine`: trading objectives, cost-aware prioritization |
| `research.py` | `ResearchLedger`: notes, rationales, post-mortems, weekly reviews |
| `news.py` | `NewsHarvester`: yfinance news with deduplication |
| `prediction.py` | Trade prediction tracking + social prediction (Chris) |
| `memory.py` | Episodic memory with salience encoding |
| `goals.py` | Goal engine (backward compat, also used by market engine) |
| `repetition.py` | Theme tracking, loop detection |
| `grounding.py` | Epistemic grounding + market claim detection |
| `chat_state.py` | `MarketState` + `ChatState` (backward compat) |
| `heartbeat.py` | `market_heartbeat_loop` (adaptive) + `heartbeat_loop` (fixed, backward compat) |
| `server.py` | FastAPI server with market + chat endpoints |
| `cli.py` | `autobot serve` command |
| `static/` | Frontend files |

## Testing

```bash
python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_accountant.py` | Cost calc, budget allocation, starvation levels, usage recording |
| `test_market.py` | ASX/US market hours, session labels, watchlist, alerts |
| `test_portfolio.py` | Buy/sell with fees, P&L, Sharpe ratio, position limits |
| `test_economic_state.py` | Decay, event-driven updates, metrics, backward compat |
| `test_research.py` | Notes CRUD, thesis lifecycle, rationales, post-mortems |
| `test_news.py` | News fetch (mocked), deduplication, ticker filtering |
| `test_emotions_market.py` | Market triggers, economic modulation, backward compat |
| `test_market_state.py` | MarketState ground truth, session tracking, Chris management |
| `test_strategy.py` | Objective generation, cooldowns, cost-aware prioritization |
| `test_brain_market.py` | LLM roles (mocked), cost tracking, budget gating |
| `test_engine_market.py` | Tick cycle, decision logic, adaptive interval, starvation |
| `test_integration_market.py` | Trade lifecycle, starvation, Sunday protocol, Chris interaction |
| `test_market_grounding.py` | Market-specific grounding, price claim validation |
| `test_scenarios.py` | Core emotional/memory/goal systems |
| `test_tom.py` | Theory of Mind, empathy, interest drift, self-judgment |
| `test_identity.py` | Identity loading, birth routine |
| `test_needs.py` | Backward compat aliases |
| `test_goals_v2.py` | Goal progress, stall, needs-driven generation |
| `test_memory_v2.py` | Recall penalties, suppression, diversity |
| `test_prediction.py` | Prediction lifecycle, PE computation |
| `test_repetition.py` | Theme tracking, loop detection |
| `test_grounding.py` | Epistemic grounding, note verification |
| `test_timeline.py` | Timeline export |

## Cost Management Strategy

**Daily budget** = operating_capital / 30 (~$33/day at $1000)

**NEVER use LLM for:** price checks (yfinance), market hours (timezone math), P&L (arithmetic), stop-loss checks (comparison), micro-thoughts (templates)

**Cheap (256 tokens):** ticker analysis, quick news sentiment

**Medium (512 tokens):** trade reasoning, Chris interaction, post-mortems

**Expensive (1024 tokens, rare):** Sunday Strategic Synthesis

## Key Design Decisions

1. **Real economic stakes.** Every LLM call costs money from Operating Capital. The entity can die at $0. This drives genuine cost-consciousness.

2. **Market events drive emotions.** Trade P&L, drawdowns, and starvation warnings produce real emotional responses -- not simulated conversations.

3. **Budget gating prevents overspending.** Each LLM role has a daily budget allocation. The entity cannot analyze when its analysis budget is exhausted.

4. **Adaptive heartbeat.** Tick interval adjusts to market state: fast when positions are open, slow on weekends, very slow when capital is critical.

5. **Sunday protocol.** Weekly strategic synthesis reviews performance, adjusts watchlist, and records lessons learned.

6. **Chris is an intelligence source.** Not a customer to please -- a collaborator who provides market insights. The entity shares specific numbers and reasoning.

7. **Simulated trading with realistic friction.** $6 brokerage + 0.05% slippage on every trade. Position size limits prevent over-concentration.

8. **Backward compatibility preserved.** `LivingEngine`, `ChatState`, `NeedsState`, old field names all still work for existing tests and code paths.
