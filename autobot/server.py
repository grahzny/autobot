"""FastAPI web server for the Autobot simulation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autobot.emotions import AffectState
from autobot.engine import WorldEngine
from autobot.llm_agent import decide
from autobot.observation import build_observation
from autobot.scenarios import ALL_SCENARIOS

app = FastAPI(title="Autobot — AI World Model", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---- In-memory session store ----
_sessions: dict[str, dict[str, Any]] = {}


class StartRequest(BaseModel):
    scenario: str
    use_llm: bool = True


class ActionRequest(BaseModel):
    session_id: str
    action: dict[str, Any] | None = None


class StepRequest(BaseModel):
    session_id: str
    steps: int = 1
    use_llm: bool = True


# ---- Routes ----

@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Autobot</h1><p>Static files not found.</p>")


@app.get("/api/scenarios")
async def list_scenarios():
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    return {
        "scenarios": list(ALL_SCENARIOS.keys()),
        "llm_available": has_key,
    }


@app.post("/api/start")
async def start_simulation(req: StartRequest):
    if req.scenario not in ALL_SCENARIOS:
        raise HTTPException(400, f"Unknown scenario: {req.scenario}")

    engine = ALL_SCENARIOS[req.scenario]()
    session_id = f"session_{len(_sessions) + 1}"

    _sessions[session_id] = {
        "engine": engine,
        "use_llm": req.use_llm,
        "history": [],
    }

    obs = build_observation(engine.world, [])
    return {
        "session_id": session_id,
        "observation": obs,
        "emotions": engine.emotions.to_dict(),
        "goals": engine.goal_engine.to_prompt_text(engine.emotions),
        "memories": engine.memory.to_prompt_text(),
        "npcs": _npc_details(engine),
        "cycle": 0,
    }


@app.post("/api/step")
async def step_simulation(req: StepRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine: WorldEngine = session["engine"]
    use_llm = req.use_llm
    results = []

    for _ in range(req.steps):
        # Build the observation and agent prompt BEFORE ticking
        obs = build_observation(engine.world, [])
        prompt = engine._compose_prompt(obs, AffectState())

        # Agent thinks and decides
        decision = decide(
            ws=engine.world,
            emotions=engine.emotions,
            affect=AffectState(),
            goal_engine=engine.goal_engine,
            memory=engine.memory,
            observation=obs,
            agent_prompt=prompt,
            use_llm=use_llm,
        )

        # Tick the world with the agent's chosen action
        result = engine.tick(agent_action=decision.action)

        step_data = _build_step_data(engine, result, decision)
        results.append(step_data)
        session["history"].append(step_data)

    return {"steps": results}


@app.post("/api/action")
async def manual_action(req: ActionRequest):
    """Submit a manual action (override agent's choice)."""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine: WorldEngine = session["engine"]
    action = req.action
    if not action:
        raise HTTPException(400, "No action provided")

    result = engine.tick(agent_action=action)

    from autobot.llm_agent import AgentDecision
    decision = AgentDecision(
        thinking="[Manual action — player override]",
        action=action,
        used_llm=False,
    )

    step_data = _build_step_data(engine, result, decision)
    session["history"].append(step_data)
    return step_data


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    engine: WorldEngine = session["engine"]
    obs = build_observation(engine.world, [])
    return {
        "session_id": session_id,
        "cycle": engine.cycle_count,
        "observation": obs,
        "emotions": engine.emotions.to_dict(),
        "goals": engine.goal_engine.to_prompt_text(engine.emotions),
        "memories": engine.memory.to_prompt_text(),
        "npcs": _npc_details(engine),
        "history_length": len(session["history"]),
    }


# ---- Helpers ----

def _build_step_data(engine, result, decision):
    return {
        "cycle": engine.cycle_count,
        "time": result.observation["time"],
        "action": decision.action,
        "thinking": decision.thinking,
        "used_llm": decision.used_llm,
        "observation": result.observation,
        "events": [_clean_event(e) for e in result.events],
        "emotions": result.emotions.to_dict(),
        "affect": {
            "arousal": round(result.affect.arousal, 3),
            "valence": round(result.affect.valence, 3),
            "salience_tags": result.affect.salience_tags,
            "attention_focus": result.affect.attention_focus,
        },
        "goals": engine.goal_engine.to_prompt_text(engine.emotions),
        "goals_list": [
            {
                "id": g.id,
                "description": g.description,
                "category": g.category,
                "priority": g.effective_priority(engine.emotions),
                "failures": g.failure_count,
                "completed": g.completed,
                "abandoned": g.abandoned,
            }
            for g in engine.goal_engine.goals
        ],
        "memories": engine.memory.to_prompt_text(),
        "new_episode": result.new_episode,
        "npcs": _npc_details(engine),
        "agent_prompt": result.agent_prompt,
    }


def _npc_details(engine: WorldEngine) -> list[dict]:
    return [
        {
            "name": npc.name,
            "trust": round(npc.trust_in_agent, 3),
            "influence": round(npc.influence_weight, 2),
            "last_interaction": npc.last_interaction_time,
        }
        for npc in engine.world.npcs.values()
    ]


def _clean_event(ev: dict) -> dict:
    cleaned = dict(ev)
    for key in ("world_time", "day", "hour"):
        cleaned.pop(key, None)
    return cleaned
