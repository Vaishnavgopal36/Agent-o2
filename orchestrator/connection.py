"""WebSocket connection management for distributed workers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from .schemas import HeartbeatPing

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerConnection:
    """Bookkeeping for a single connected worker."""

    worker_id: str
    websocket: WebSocket
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    heartbeat_task: asyncio.Task[None] | None = None


class ConnectionManager:
    """Tracks active worker sockets and serializes outbound messages per worker."""

    def __init__(
        self,
        *,
        heartbeat_interval_seconds: float = 20.0,
        heartbeat_timeout_seconds: float = 60.0,
    ) -> None:
        self._connections: dict[str, WorkerConnection] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._heartbeat_timeout_seconds = heartbeat_timeout_seconds

    async def connect(self, websocket: WebSocket, worker_id: str) -> None:
        """Accept and register a worker WebSocket connection."""

        await websocket.accept()
        await self.disconnect(worker_id, close_code=1012)

        connection = WorkerConnection(worker_id=worker_id, websocket=websocket)
        connection.heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(connection),
            name=f"heartbeat:{worker_id}",
        )

        async with self._lock:
            self._connections[worker_id] = connection

        logger.info("worker connected", extra={"worker_id": worker_id})

    async def disconnect(self, worker_id: str, *, close_code: int = 1000) -> None:
        """Remove a worker and close its socket if still connected."""

        async with self._lock:
            connection = self._connections.pop(worker_id, None)

        if connection is None:
            return

        if (
            connection.heartbeat_task is not None
            and connection.heartbeat_task is not asyncio.current_task()
        ):
            connection.heartbeat_task.cancel()

        if connection.websocket.application_state is WebSocketState.CONNECTED:
            try:
                await connection.websocket.close(code=close_code)
            except RuntimeError:
                logger.debug("worker socket already closed", extra={"worker_id": worker_id})

        logger.info("worker disconnected", extra={"worker_id": worker_id})

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all active workers, dropping stale connections."""

        async with self._lock:
            worker_ids = tuple(self._connections.keys())

        results = await asyncio.gather(
            *(self.send_task(worker_id, message) for worker_id in worker_ids),
            return_exceptions=True,
        )

        for worker_id, result in zip(worker_ids, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "broadcast failed for worker",
                    extra={"worker_id": worker_id, "error": str(result)},
                )

    async def send_task(self, worker_id: str, payload: dict[str, Any]) -> None:
        """Send one JSON payload to a worker.

        Raises:
            KeyError: The worker is not currently connected.
            ConnectionError: The socket send failed or the socket is no longer open.
        """

        connection = await self.get_connection(worker_id)
        if connection is None:
            raise KeyError(f"worker {worker_id!r} is not connected")

        try:
            async with connection.send_lock:
                await connection.websocket.send_json(payload)
        except Exception as exc:
            logger.warning(
                "send failed; disconnecting worker",
                extra={"worker_id": worker_id, "error": str(exc)},
            )
            await self.disconnect(worker_id, close_code=1011)
            raise ConnectionError(f"failed to send task to {worker_id!r}") from exc

    async def mark_seen(self, worker_id: str) -> None:
        """Record inbound activity from a worker for heartbeat liveness checks."""

        connection = await self.get_connection(worker_id)
        if connection is not None:
            connection.last_seen_at = datetime.now(timezone.utc)

    async def get_connection(self, worker_id: str) -> WorkerConnection | None:
        """Return the active connection for a worker, if present."""

        async with self._lock:
            return self._connections.get(worker_id)

    async def active_worker_ids(self) -> list[str]:
        """Return active worker ids for diagnostics or scheduling."""

        async with self._lock:
            return list(self._connections)

    async def close_all(self) -> None:
        """Disconnect all workers during application shutdown."""

        async with self._lock:
            worker_ids = tuple(self._connections.keys())

        await asyncio.gather(
            *(self.disconnect(worker_id, close_code=1001) for worker_id in worker_ids),
            return_exceptions=True,
        )

    async def _heartbeat_loop(self, connection: WorkerConnection) -> None:
        """Send app-level pings and evict workers that stop responding."""

        worker_id = connection.worker_id

        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_seconds)

                idle_seconds = (
                    datetime.now(timezone.utc) - connection.last_seen_at
                ).total_seconds()
                if idle_seconds > self._heartbeat_timeout_seconds:
                    logger.warning(
                        "worker heartbeat timed out",
                        extra={"worker_id": worker_id, "idle_seconds": idle_seconds},
                    )
                    await self.disconnect(worker_id, close_code=1001)
                    return

                ping = HeartbeatPing().model_dump(mode="json")
                async with connection.send_lock:
                    await connection.websocket.send_json(ping)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "heartbeat failed; disconnecting worker",
                extra={"worker_id": worker_id, "error": str(exc)},
            )
            await self.disconnect(worker_id, close_code=1011)
