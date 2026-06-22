"""FastAPI route registration for worker WebSocket orchestration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from .connection import ConnectionManager
from .schemas import (
    IncomingWorkerMessage,
    TaskRequest,
    TaskStatus,
    WorkerHeartbeat,
    WorkerReady,
    WorkerResponse,
    WorkerStderr,
)
from .state import RedisStateManager

logger = logging.getLogger(__name__)
incoming_worker_message_adapter = TypeAdapter(IncomingWorkerMessage)


def register_routes(
    app: FastAPI,
    *,
    connection_manager: ConnectionManager,
    state_manager: RedisStateManager,
) -> None:
    """Register HTTP and WebSocket routes on the provided FastAPI app."""

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Render health check endpoint; always responds with HTTP 200."""

        return {"status": "ok", "service": "agent-websocket-orchestrator"}

    @app.websocket("/ws/worker/{worker_id}")
    async def worker_socket(websocket: WebSocket, worker_id: str) -> None:
        """Maintain a worker WebSocket and dispatch inbound worker messages."""

        await connection_manager.connect(websocket, worker_id)

        try:
            while True:
                try:
                    raw_message = await websocket.receive_json()
                except WebSocketDisconnect:
                    logger.info("worker websocket disconnected", extra={"worker_id": worker_id})
                    break
                except RuntimeError as exc:
                    logger.warning(
                        "worker socket closed ungracefully",
                        extra={"worker_id": worker_id, "error": str(exc)},
                    )
                    break

                await connection_manager.mark_seen(worker_id)

                try:
                    message = parse_worker_message(raw_message)
                except ValidationError as exc:
                    await connection_manager.send_task(
                        worker_id,
                        {
                            "type": "error",
                            "error": "invalid_worker_message",
                            "details": exc.errors(),
                        },
                    )
                    continue

                await handle_worker_message(
                    worker_id=worker_id,
                    message=message,
                    connection_manager=connection_manager,
                    state_manager=state_manager,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "worker loop failed",
                extra={"worker_id": worker_id, "error": str(exc)},
            )
        finally:
            await connection_manager.disconnect(worker_id)


def parse_worker_message(raw_message: dict[str, Any]) -> IncomingWorkerMessage:
    """Parse incoming JSON into an explicit Pydantic worker schema."""

    return incoming_worker_message_adapter.validate_python(raw_message)


async def handle_worker_message(
    *,
    worker_id: str,
    message: IncomingWorkerMessage,
    connection_manager: ConnectionManager,
    state_manager: RedisStateManager,
) -> None:
    """Persist worker messages and trigger the next routing decision hook."""

    match message:
        case WorkerReady():
            logger.info(
                "worker ready",
                extra={
                    "worker_id": worker_id,
                    "capabilities": message.capabilities,
                    "max_concurrency": message.max_concurrency,
                },
            )
        case WorkerHeartbeat():
            logger.debug("worker heartbeat pong", extra={"worker_id": worker_id})
        case WorkerStderr():
            await state_manager.append_execution_log(
                graph_id=message.graph_id,
                task_id=message.task_id,
                line=message.line,
            )
            await trigger_next_groq_routing_decision(
                worker_id=worker_id,
                graph_id=message.graph_id,
                task_id=message.task_id,
                event="stderr",
                connection_manager=connection_manager,
                state_manager=state_manager,
            )
        case WorkerResponse():
            result = message.result or {}
            if message.error is not None:
                result = {**result, "error": message.error}

            await state_manager.update_node_state(
                task_id=message.task_id,
                status=message.status,
                result=result,
                graph_id=message.graph_id,
            )
            await trigger_next_groq_routing_decision(
                worker_id=worker_id,
                graph_id=message.graph_id,
                task_id=message.task_id,
                event=message.status.value,
                connection_manager=connection_manager,
                state_manager=state_manager,
            )


async def trigger_next_groq_routing_decision(
    *,
    worker_id: str,
    graph_id: str,
    task_id: str,
    event: str,
    connection_manager: ConnectionManager,
    state_manager: RedisStateManager,
) -> None:
    """Placeholder for Groq-only routing logic after worker state changes.

    The actual Groq API inference is intentionally omitted. This hook loads the
    current Redis-backed graph state and records a queue marker so a future
    Groq router can decide whether to dispatch a CodeAct, RAG, or other task.
    """

    graph_state = await state_manager.get_graph_state(graph_id)
    await state_manager.update_node_state(
        task_id=f"router:{task_id}",
        status=TaskStatus.QUEUED,
        result={
            "triggered_by_worker_id": worker_id,
            "triggered_by_task_id": task_id,
            "event": event,
            "known_node_count": len(graph_state["nodes"]),
            "next_step": "groq_routing_decision_pending",
        },
        graph_id=graph_id,
    )


async def dispatch_task(
    *,
    connection_manager: ConnectionManager,
    worker_id: str,
    task: TaskRequest,
) -> None:
    """Typed helper for future HTTP/queue based task dispatchers."""

    await connection_manager.send_task(worker_id, task.model_dump(mode="json"))
