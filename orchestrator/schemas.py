"""Pydantic message contracts for WebSocket worker communication."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class WorkerMessageType(str, Enum):
    """Message types accepted from worker nodes."""

    READY = "ready"
    RESULT = "result"
    STDERR = "stderr"
    HEARTBEAT = "heartbeat"


class TaskStatus(str, Enum):
    """Execution states persisted to Redis."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOG = "log"


class BaseWireMessage(BaseModel):
    """Base schema shared by all orchestrator wire messages."""

    model_config = ConfigDict(extra="forbid")

    type: str
    task_id: str | None = None
    graph_id: str | None = None


class WorkerReady(BaseWireMessage):
    """Worker announces capabilities after the WebSocket opens."""

    type: Literal[WorkerMessageType.READY]
    capabilities: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=1, ge=1, le=64)


class WorkerResponse(BaseWireMessage):
    """Worker reports a final or intermediate task result."""

    type: Literal[WorkerMessageType.RESULT]
    task_id: str
    graph_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
    error: str | None = None


class WorkerStderr(BaseWireMessage):
    """Worker streams execution logs or stderr output."""

    type: Literal[WorkerMessageType.STDERR]
    task_id: str
    graph_id: str
    line: str = Field(min_length=1, max_length=16_384)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkerHeartbeat(BaseWireMessage):
    """Application-level heartbeat response from a worker."""

    type: Literal[WorkerMessageType.HEARTBEAT]
    op: Literal["pong"]
    worker_time: datetime | None = None


class TaskRequest(BaseModel):
    """Command sent from the orchestrator to a worker."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["task"] = "task"
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    graph_id: str
    command: str
    agent_type: str = Field(
        default="generic",
        description="Extensible worker specialization, e.g. CodeAct or RAG Retriever.",
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)


class HeartbeatPing(BaseModel):
    """Application-level heartbeat request sent to workers."""

    type: Literal["heartbeat"] = "heartbeat"
    op: Literal["ping"] = "ping"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


IncomingWorkerMessage = WorkerReady | WorkerResponse | WorkerStderr | WorkerHeartbeat
