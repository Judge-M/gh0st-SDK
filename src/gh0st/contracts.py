"""Stable request and report contracts for the gh0st SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Ticket:
    """One task submitted to a fresh worker.

    The optional gateway fields are pass-through allocation metadata. The SDK
    does not reserve or account for provider spend; a gateway or the embedding
    application owns that policy.
    """

    prompt: str
    scope: str = "default"
    workspace_path: str | None = None
    source_commit: str | None = None
    permitted_local_capabilities: tuple[str, ...] = ()
    gateway_allowance: Mapping[str, Any] | None = None
    max_tokens_allocated: int | None = None
    max_cost_usd: float | None = None
    ticket_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Ticket prompt must not be empty")
        if not self.scope.strip():
            raise ValueError("Ticket scope must not be empty")
        if self.max_tokens_allocated is not None and self.max_tokens_allocated <= 0:
            raise ValueError("max_tokens_allocated must be positive")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be zero or greater")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


@dataclass(frozen=True)
class WorkerReport:
    ticket_id: str
    worker_name: str
    intent: str
    status: str
    output: str
    turns: int
    tool_calls: int
    usage: Usage
    memory_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
