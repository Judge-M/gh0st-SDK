"""Explicit, caller-registered tools available to an ephemeral worker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any] = field(repr=False, compare=False)
    capability: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name must not be empty")
        if not callable(self.handler):
            raise TypeError("Tool handler must be callable")

    @property
    def capability_name(self) -> str:
        return self.capability or self.name

    def model_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }

    def invoke(self, arguments: Mapping[str, Any]) -> str:
        result = self.handler(**dict(arguments))
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
