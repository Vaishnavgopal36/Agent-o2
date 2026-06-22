"""ASGI entrypoint for the distributed agent WebSocket orchestrator."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI

from orchestrator.config import settings
from orchestrator.connection import ConnectionManager
from orchestrator.routes import register_routes
from orchestrator.state import RedisStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

connection_manager = ConnectionManager(
    heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
    heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
)
state_manager = RedisStateManager(
    redis_url=settings.redis_url,
    key_prefix=settings.redis_key_prefix,
    max_retries=settings.redis_max_retries,
    retry_base_delay_seconds=settings.redis_retry_base_delay_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize async resources without blocking Render's health checks."""

    try:
        await state_manager.connect()
        logger.info("redis connection established")
    except Exception as exc:
        logger.warning("redis unavailable on startup; will retry lazily: %s", exc)

    app.state.connection_manager = connection_manager
    app.state.state_manager = state_manager

    try:
        yield
    finally:
        await connection_manager.close_all()
        await state_manager.close()


app = FastAPI(
    title="Agentic Worker WebSocket Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)
register_routes(
    app,
    connection_manager=connection_manager,
    state_manager=state_manager,
)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
        ws_ping_interval=settings.heartbeat_interval_seconds,
        ws_ping_timeout=settings.heartbeat_timeout_seconds,
        ws_max_size=settings.websocket_max_message_bytes,
    )

