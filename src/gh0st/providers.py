"""Model-provider contracts and a direct OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Protocol, Sequence

from .contracts import ModelResponse, ToolCall, Usage


class CompletionProvider(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
    ) -> ModelResponse: ...


class ProviderError(RuntimeError):
    """Raised when a direct provider request cannot be completed."""


class OpenAICompatibleProvider:
    """Calls a provider's Chat Completions endpoint directly, without gh0st Gateway."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.extra_headers = dict(extra_headers or {})

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
    ) -> ModelResponse:
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_output_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Provider returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Provider request failed: {exc.__class__.__name__}") from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider response did not contain a chat choice") from exc

        parsed_calls: list[ToolCall] = []
        for call in message.get("tool_calls", ()):
            function = call.get("function", {})
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ProviderError("Provider returned invalid JSON tool arguments") from exc
            if not isinstance(arguments, dict):
                raise ProviderError("Provider tool arguments must be a JSON object")
            parsed_calls.append(
                ToolCall(
                    call_id=str(call.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )

        raw_usage = body.get("usage") or {}
        return ModelResponse(
            content=message.get("content"),
            tool_calls=tuple(parsed_calls),
            usage=Usage(
                input_tokens=int(raw_usage.get("prompt_tokens") or 0),
                output_tokens=int(raw_usage.get("completion_tokens") or 0),
            ),
            finish_reason=choice.get("finish_reason"),
        )
