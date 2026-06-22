"""Redis-backed state synchronization for distributed DAG execution."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

from .schemas import TaskStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisStateManager:
    """Persists global graph state in an external asynchronous Redis instance."""

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "orchestrator",
        max_retries: int = 5,
        retry_base_delay_seconds: float = 0.2,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix.rstrip(":")
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._client: Redis | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize the Redis connection pool and verify connectivity."""

        await self._get_client()
        await self._execute(lambda client: client.ping())

    async def close(self) -> None:
        """Close the Redis connection pool on shutdown."""

        if self._client is None:
            return

        await self._client.aclose()
        self._client = None

    async def update_node_state(
        self,
        task_id: str,
        status: TaskStatus | str,
        result: dict[str, Any] | str | None,
        *,
        graph_id: str = "default",
    ) -> None:
        """Persist a task/node state transition inside a graph.

        The graph is stored as a Redis hash keyed by task id. Each field is a
        compact JSON document so updates remain atomic and memory-conscious.
        """

        normalized_status = status.value if isinstance(status, TaskStatus) else status
        payload = {
            "task_id": task_id,
            "graph_id": graph_id,
            "status": normalized_status,
            "result": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        graph_key = self._graph_key(graph_id)
        await self._execute(
            lambda client: client.hset(graph_key, task_id, json.dumps(payload))
        )

    async def append_execution_log(
        self,
        graph_id: str,
        task_id: str,
        line: str,
        *,
        max_entries: int = 1_000,
    ) -> None:
        """Append stderr or execution logs to a bounded Redis list."""

        log_key = self._task_log_key(graph_id, task_id)

        async def write_log(client: Redis) -> None:
            pipe = client.pipeline(transaction=True)
            entry = {
                "task_id": task_id,
                "graph_id": graph_id,
                "stream": "stderr",
                "line": line,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            pipe.lpush(log_key, json.dumps(entry))
            pipe.ltrim(log_key, 0, max_entries - 1)
            await pipe.execute()

        await self._execute(write_log)
        await self.update_node_state(
            task_id,
            TaskStatus.LOG,
            {"latest_stderr": line},
            graph_id=graph_id,
        )

    async def get_graph_state(self, graph_id: str) -> dict[str, Any]:
        """Fetch and deserialize the full graph state for a graph id."""

        graph_key = self._graph_key(graph_id)
        raw_nodes = await self._execute(lambda client: client.hgetall(graph_key))
        return {
            "graph_id": graph_id,
            "nodes": {
                task_id: json.loads(raw_state)
                for task_id, raw_state in raw_nodes.items()
                if raw_state
            },
        }

    async def _execute(self, operation: Callable[[Redis], Awaitable[T]]) -> T:
        """Run a Redis operation with reconnect and exponential backoff."""

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            client = await self._get_client()
            try:
                return await operation(client)
            except (RedisConnectionError, RedisTimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "redis connection failed; retrying",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                await self._reset_client()
                await asyncio.sleep(self._retry_delay(attempt))
            except RedisError:
                raise

        raise ConnectionError("redis operation failed after retries") from last_error

    async def _get_client(self) -> Redis:
        if self._client is None:
            async with self._connect_lock:
                if self._client is None:
                    self._client = redis.from_url(
                        self._redis_url,
                        decode_responses=True,
                        socket_connect_timeout=3,
                        socket_timeout=3,
                        health_check_interval=30,
                        retry_on_timeout=True,
                    )

        if self._client is None:
            raise ConnectionError("redis client is not initialized")

        return self._client

    async def _reset_client(self) -> None:
        async with self._connect_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    def _retry_delay(self, attempt: int) -> float:
        return min(self._retry_base_delay_seconds * (2 ** (attempt - 1)), 3.0)

    def _graph_key(self, graph_id: str) -> str:
        return f"{self._key_prefix}:graph:{graph_id}"

    def _task_log_key(self, graph_id: str, task_id: str) -> str:
        return f"{self._key_prefix}:graph:{graph_id}:task:{task_id}:stderr"
