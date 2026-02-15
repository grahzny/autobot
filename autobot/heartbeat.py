"""Heartbeat -- the entity's autonomous background loop.

Runs a tick every TICK_INTERVAL seconds, processing the entity's
emotional state, pending messages, and proactive behavior.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from autobot.living_engine import LivingEngine, TickResult

logger = logging.getLogger("autobot.heartbeat")

# Default tick interval in seconds
TICK_INTERVAL = 15


async def heartbeat_loop(
    engine: LivingEngine,
    on_tick: Callable[[TickResult], Coroutine[Any, Any, None]] | None = None,
    interval: float = TICK_INTERVAL,
) -> None:
    """
    The entity's autonomous heartbeat.

    Runs forever, ticking the engine and optionally calling on_tick
    with the result (used to push WebSocket messages).
    """
    logger.info("Heartbeat started (interval=%.1fs)", interval)

    while True:
        try:
            result = engine.tick()

            if result.decision not in ("idle",):
                logger.info(
                    "Tick %d: decision=%s target=%s msg=%s",
                    result.tick,
                    result.decision,
                    result.target_person_id,
                    (result.outgoing_message or "")[:50],
                )

            if on_tick:
                await on_tick(result)

        except Exception:
            logger.exception("Heartbeat tick failed")

        await asyncio.sleep(interval)
