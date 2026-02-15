"""FastAPI server for the Autobot market entity."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autobot.identity import Identity
from autobot.living_engine import MarketEngine, MarketTickResult
from autobot.heartbeat import market_heartbeat_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autobot.server")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Autobot -- Market Entity", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_identity() -> Identity:
    """Load identity from env var, well-known paths, or fall back to default."""
    env_path = os.environ.get("AUTOBOT_IDENTITY")
    if env_path:
        logger.info("Loading identity from AUTOBOT_IDENTITY=%s", env_path)
        return Identity.from_file(env_path)
    for path in [Path("identity.yaml"), Path("identity.yml"), Path("identity.json")]:
        if path.exists():
            logger.info("Loading identity from %s", path)
            return Identity.from_file(str(path))
    logger.info("No identity file found, using defaults.")
    return Identity.default()


# Global entity -- there is only one
engine = MarketEngine.from_identity(_load_identity())

# WebSocket connections (chris only, but keyed for compatibility)
ws_connections: dict[str, list[WebSocket]] = {}

# Outgoing message queue (for polling fallback)
message_queues: dict[str, list[dict[str, Any]]] = {}


# ======================================================================
# Startup
# ======================================================================

@app.on_event("startup")
async def startup():
    async def on_tick(result: MarketTickResult):
        """Push outgoing messages and thinking updates via WebSocket."""
        if result.outgoing_message and result.target_person_id:
            msg_data = {
                "type": "message",
                "from": "entity",
                "text": result.outgoing_message,
                "mood_hint": engine.emotions.dominant_mood(),
                "thinking": result.thinking,
            }

            # Push to WebSocket
            pid = result.target_person_id
            if pid in ws_connections:
                dead = []
                for ws in ws_connections[pid]:
                    try:
                        await ws.send_json(msg_data)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    ws_connections[pid].remove(ws)

            # Also queue for polling
            message_queues.setdefault(pid, []).append(msg_data)

        # Broadcast thinking/market update to ALL connected clients
        update_data = {
            "type": "thinking_update",
            "thinking": result.thinking,
            "decision": result.decision,
            "tick": result.tick,
            "mood": engine.emotions.dominant_mood(),
            "energy": round(engine.state.energy, 2),
            "capital": round(engine.accountant.operating_capital, 2),
            "starvation": result.starvation_level,
        }

        # Add trade info if relevant
        if result.trade_action:
            update_data["trade_action"] = result.trade_action
            update_data["trade_ticker"] = result.trade_ticker
            if result.trade_pnl is not None:
                update_data["trade_pnl"] = round(result.trade_pnl, 2)

        for pid, sockets in ws_connections.items():
            dead = []
            for ws_conn in sockets:
                try:
                    await ws_conn.send_json(update_data)
                except Exception:
                    dead.append(ws_conn)
            for ws_conn in dead:
                sockets.remove(ws_conn)

    async def on_death():
        """Broadcast death event to all connected clients."""
        death_data = {
            "type": "death",
            "message": "Entity has died. Capital exhausted.",
            "total_spent": round(engine.accountant.total_spent, 4),
        }
        for pid, sockets in ws_connections.items():
            for ws_conn in sockets:
                try:
                    await ws_conn.send_json(death_data)
                except Exception:
                    pass

    asyncio.create_task(market_heartbeat_loop(
        engine, on_tick=on_tick, on_death=on_death,
    ))
    logger.info(
        "Market entity is alive. Capital: $%.2f",
        engine.accountant.operating_capital,
    )


# ======================================================================
# REST Endpoints -- Chat (Chris)
# ======================================================================

class ConnectRequest(BaseModel):
    person_name: str

class MessageRequest(BaseModel):
    person_id: str
    text: str


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/connect")
async def connect(req: ConnectRequest):
    """Register Chris (the only human)."""
    person = engine.state.get_or_create_person("chris", req.person_name)
    return {
        "person_id": "chris",
        "entity_name": engine.state.name,
        "mood_hint": engine.emotions.dominant_mood(),
        "capital": round(engine.accountant.operating_capital, 2),
    }


@app.post("/api/message")
async def send_message(req: MessageRequest):
    """Send a message to the entity (from Chris)."""
    person = engine.state.chris
    if not person:
        engine.state.get_or_create_person("chris", "Chris")
        person = engine.state.chris

    engine.receive_message("chris", person.name, req.text)
    return {"received": True}


@app.get("/api/poll/{person_id}")
async def poll(person_id: str):
    """Poll for new messages from the entity."""
    messages = message_queues.pop(person_id, [])
    mood = engine.emotions.dominant_mood()
    energy = engine.state.energy

    energy_hint = "rested" if energy > 0.6 else "tired" if energy > 0.3 else "exhausted"

    return {
        "messages": messages,
        "entity_state": {
            "mood_hint": mood,
            "energy_hint": energy_hint,
            "capital": round(engine.accountant.operating_capital, 2),
        },
    }


# ======================================================================
# REST Endpoints -- Market Status
# ======================================================================

@app.get("/api/market/status")
async def market_status():
    """Market session, watchlist, prices."""
    return {
        "session": engine.monitor.market_session_label(),
        "is_market_open": engine.monitor.is_market_open(),
        "is_sunday": engine.monitor.is_sunday(),
        "watchlist": {
            ticker: {
                "price": entry.last_price,
                "exchange": entry.exchange,
                "name": entry.name,
                "sector": entry.sector,
            }
            for ticker, entry in engine.monitor.watchlist.items()
        },
    }


@app.get("/api/portfolio")
async def portfolio():
    """Portfolio positions, P&L, Sharpe, win rate."""
    prices = engine.state.last_prices
    return {
        **engine.portfolio.summary(prices),
        "positions": [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "current_price": prices.get(p.ticker, p.entry_price),
                "unrealized_pnl": round(
                    p.unrealized_pnl(prices.get(p.ticker, p.entry_price)), 2
                ),
            }
            for p in engine.portfolio.open_positions()
        ],
        "recent_trades": [
            {
                "ticker": t.ticker,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.realized_pnl or 0, 2),
            }
            for t in engine.portfolio.trade_history[-10:]
        ],
    }


@app.get("/api/capital")
async def capital():
    """Operating capital, total spent, daily budget, is_alive."""
    return engine.accountant.summary()


@app.get("/api/research")
async def research():
    """Notes, active theses, post-mortems."""
    return {
        **engine.research.summary(),
        "active_theses": [
            {
                "ticker": t.ticker,
                "content": t.content,
                "conviction": t.conviction,
                "timestamp": t.timestamp,
            }
            for t in engine.research.active_theses()[-10:]
        ],
        "recent_lessons": engine.research.recent_lessons(limit=5),
        "recent_notes": [
            n.to_dict() for n in engine.research.notes[-10:]
        ],
    }


# ======================================================================
# REST Endpoints -- General Status
# ======================================================================

@app.get("/api/status")
async def status():
    """Quick entity status."""
    return {
        "entity_name": engine.state.name,
        "mood": engine.emotions.dominant_mood(),
        "energy": round(engine.state.energy, 2),
        "tick": engine.state.tick_count,
        "capital": round(engine.accountant.operating_capital, 2),
        "is_alive": engine.accountant.is_alive(),
        "market_session": engine.monitor.market_session_label(),
        "open_positions": len(engine.portfolio.open_positions()),
        "total_trades": len(engine.portfolio.trade_history),
    }


@app.get("/api/debug/state")
async def debug_state():
    """Full internal state dump."""
    return engine.debug_state()


@app.get("/api/thinking")
async def thinking():
    """Recent thinking history."""
    return {
        "last_thinking": engine.last_thinking,
        "last_decision": engine.last_decision,
        "history": engine.thinking_history[-20:],
    }


# ======================================================================
# WebSocket
# ======================================================================

@app.websocket("/ws/{person_id}")
async def websocket_endpoint(websocket: WebSocket, person_id: str):
    # For market entity, always treat as Chris
    if not engine.state.chris:
        engine.state.get_or_create_person("chris", "Chris")

    await websocket.accept()
    ws_connections.setdefault(person_id, []).append(websocket)
    logger.info("WebSocket connected: %s", person_id)

    # Send current state on connect
    await websocket.send_json({
        "type": "connected",
        "entity_name": engine.state.name,
        "mood_hint": engine.emotions.dominant_mood(),
        "capital": round(engine.accountant.operating_capital, 2),
        "market_session": engine.monitor.market_session_label(),
    })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "message":
                text = data.get("text", "").strip()
                if text:
                    engine.receive_message(
                        "chris", engine.state.chris.name, text,
                    )
                    await websocket.send_json({"type": "received"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", person_id)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        if person_id in ws_connections:
            if websocket in ws_connections[person_id]:
                ws_connections[person_id].remove(websocket)


# ======================================================================
# Static files (must be last)
# ======================================================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
