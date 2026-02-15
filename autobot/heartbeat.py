"""Heartbeat -- the entity's autonomous background loop.

Two heartbeat functions:
  - heartbeat_loop: Original fixed-interval loop for LivingEngine (backward compat)
  - market_heartbeat_loop: Adaptive-interval loop for MarketEngine with starvation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from autobot.living_engine import LivingEngine, TickResult, MarketEngine, MarketTickResult

logger = logging.getLogger("autobot.heartbeat")

# Default tick interval in seconds
TICK_INTERVAL = 15


async def heartbeat_loop(
    engine: LivingEngine,
    on_tick: Callable[[TickResult], Coroutine[Any, Any, None]] | None = None,
    interval: float = TICK_INTERVAL,
) -> None:
    """
    The entity's autonomous heartbeat (original, fixed interval).

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


async def market_heartbeat_loop(
    engine: MarketEngine,
    on_tick: Callable[[MarketTickResult], Coroutine[Any, Any, None]] | None = None,
    on_death: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> None:
    """
    Adaptive market heartbeat with starvation termination.

    - Interval adapts based on market state (30s-900s)
    - Terminates when capital reaches $0 (starvation death)
    - Logs decision, capital, and interval each tick
    """
    logger.info(
        "Market heartbeat started (capital=$%.2f)",
        engine.accountant.operating_capital,
    )

    while True:
        try:
            result = engine.tick()

            # Log non-idle ticks
            if result.decision not in ("idle",):
                logger.info(
                    "Tick %d: decision=%s capital=$%.2f thinking=%s",
                    result.tick,
                    result.decision,
                    engine.accountant.operating_capital,
                    (result.thinking or "")[:80],
                )

            # Starvation check
            if result.is_dead:
                logger.warning(
                    "ENTITY DEATH: Capital exhausted at tick %d. "
                    "Total spent: $%.4f",
                    result.tick,
                    engine.accountant.total_spent,
                )
                if on_tick:
                    await on_tick(result)
                if on_death:
                    await on_death()
                break  # Exit the loop -- entity is dead

            if on_tick:
                await on_tick(result)

            # Log starvation warnings
            if result.starvation_level:
                logger.warning(
                    "Starvation %s: $%.2f remaining",
                    result.starvation_level,
                    engine.accountant.operating_capital,
                )

        except Exception:
            logger.exception("Market heartbeat tick failed")

        # Adaptive interval
        interval = engine.calculate_tick_interval()
        await asyncio.sleep(interval)
