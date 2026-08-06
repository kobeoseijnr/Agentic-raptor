"""Provider-neutral multimodal LLM interface + one real implementation.

The real provider speaks the OpenAI-compatible chat-completions protocol
(works with OpenAI, Azure-compatible gateways, and local Ollama via base URL)
using only the standard library — no SDK dependency.

Credentials come ONLY from environment variables at call time; they are never
stored, logged, or written to config files.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_raptor.utils.exceptions import GenerationError
from agentic_raptor.utils.logging import get_logger

logger = get_logger("agentic_raptor.provider")


class MultimodalTopologyModel(Protocol):
    """What generation and schematic parsing need from a multimodal model."""

    def generate_structured(
        self,
        prompt: str,
        images: list[bytes],
        structured_context: dict[str, Any],
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Return the model's response parsed as a JSON object."""
        ...


@dataclass
class ProviderSettings:
    provider: str = "openai_compatible"
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "AGENTIC_RAPTOR_LLM_BASE_URL"
    default_base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 60.0
    retries: int = 2

    def missing_credential(self) -> str | None:
        """Name of the missing env var, or None when configured."""
        if not os.environ.get(self.api_key_env):
            return self.api_key_env
        return None


@dataclass
class ProviderError(GenerationError):
    """Structured provider failure (also a GenerationError for retry hooks)."""

    kind: str = "provider_error"   # missing_credential | http_error | timeout | malformed_json | empty_response
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class OpenAICompatibleModel:
    """Real multimodal provider over the chat-completions protocol."""

    settings: ProviderSettings = field(default_factory=ProviderSettings)

    def generate_structured(
        self,
        prompt: str,
        images: list[bytes],
        structured_context: dict[str, Any],
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        missing = self.settings.missing_credential()
        if missing:
            raise ProviderError(
                kind="missing_credential",
                detail=f"environment variable {missing} is not set",
            )
        api_key = os.environ[self.settings.api_key_env]
        base_url = os.environ.get(self.settings.base_url_env) or self.settings.default_base_url

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if structured_context:
            content.append(
                {"type": "text", "text": "## Structured context\n" + json.dumps(structured_context, indent=2)}
            )
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )

        payload = {
            "model": self.settings.model_name,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_output_tokens,
            # Deterministic generation for reproducible experiments; retry
            # prompts differ textually, which is the intended variation channel.
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{base_url.rstrip('/')}/chat/completions"

        last_error: ProviderError | None = None
        for attempt in range(1 + max(0, self.settings.retries)):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = ProviderError(kind="http_error", detail=f"HTTP {exc.code}: {detail}")
                logger.warning("provider HTTP error (attempt %d): %s", attempt + 1, exc.code)
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = ProviderError(kind="timeout", detail=str(exc))
                logger.warning("provider transport error (attempt %d): %s", attempt + 1, exc)
                continue

            usage = raw.get("usage") or {}
            logger.info(
                "provider call ok: model=%s runtime=%.1fs prompt_tokens=%s completion_tokens=%s",
                self.settings.model_name,
                time.monotonic() - started,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            choices = raw.get("choices") or []
            if not choices:
                last_error = ProviderError(kind="empty_response", detail="no choices in response")
                continue
            text = ((choices[0].get("message") or {}).get("content")) or ""
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = ProviderError(kind="malformed_json", detail=f"{exc}; head={text[:200]!r}")
                continue
            if not isinstance(parsed, dict):
                last_error = ProviderError(kind="malformed_json", detail="top-level JSON is not an object")
                continue
            return parsed
        assert last_error is not None
        raise last_error


@dataclass
class MockMultimodalModel:
    """Deterministic provider stand-in for tests: replays canned responses."""

    responses: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_structured(
        self,
        prompt: str,
        images: list[bytes],
        structured_context: dict[str, Any],
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        self.calls.append(
            {"prompt_chars": len(prompt), "images": len(images), "context_keys": sorted(structured_context)}
        )
        if not self.responses:
            raise ProviderError(kind="empty_response", detail="mock has no responses queued")
        return self.responses.pop(0)


def build_provider(settings: ProviderSettings, mock: MockMultimodalModel | None = None) -> MultimodalTopologyModel:
    if settings.provider == "mock":
        return mock or MockMultimodalModel()
    if settings.provider == "openai_compatible":
        return OpenAICompatibleModel(settings)
    raise ProviderError(kind="provider_error", detail=f"unknown provider {settings.provider!r}")
