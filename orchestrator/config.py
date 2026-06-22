"""Runtime configuration for the orchestrator service.

The settings object intentionally depends only on the standard library so the
service can boot in constrained Render containers before optional integrations
are fully reachable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Environment-backed application settings."""

    redis_url: str
    redis_key_prefix: str
    port: int
    host: str
    heartbeat_interval_seconds: float
    heartbeat_timeout_seconds: float
    redis_max_retries: int
    redis_retry_base_delay_seconds: float
    websocket_max_message_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with Render-safe defaults."""

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_key_prefix=os.getenv("REDIS_KEY_PREFIX", "orchestrator"),
            port=int(os.getenv("PORT", "8000")),
            host=os.getenv("HOST", "0.0.0.0"),
            heartbeat_interval_seconds=float(os.getenv("WS_HEARTBEAT_INTERVAL", "20")),
            heartbeat_timeout_seconds=float(os.getenv("WS_HEARTBEAT_TIMEOUT", "60")),
            redis_max_retries=int(os.getenv("REDIS_MAX_RETRIES", "5")),
            redis_retry_base_delay_seconds=float(os.getenv("REDIS_RETRY_BASE_DELAY", "0.2")),
            websocket_max_message_bytes=int(os.getenv("WS_MAX_MESSAGE_BYTES", "1048576")),
        )


settings = Settings.from_env()

