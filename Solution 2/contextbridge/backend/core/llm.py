"""Anthropic Claude client wrapper.

Every call returns a ``ClaudeResponse`` — API failures come back as
``response.error`` rather than raised exceptions, so no pipeline stage can crash
the request. Callers check ``response.ok``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend import config
from backend.core.models import ClaudeResponse, TokenUsage
from backend.utils.logger import logger

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.5


class AnthropicAPIError(RuntimeError):
    """Only raised by callers that explicitly want a hard failure."""


class ClaudeClient:
    """Thin async wrapper with retry, graceful degradation, and usage tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.ANTHROPIC_API_KEY
        self.model = model or config.CLAUDE_MODEL
        self.usage = TokenUsage()
        self._client = None
        self._unavailable_reason: str | None = None

        if not self.api_key.strip():
            self._unavailable_reason = (
                "ANTHROPIC_API_KEY is not set. LLM features are disabled; "
                "retrieval and chunking still work."
            )
            logger.warning(self._unavailable_reason)
            return

        try:
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            # Any Anthropic-compatible endpoint (e.g. a local model router).
            if config.ANTHROPIC_BASE_URL:
                kwargs["base_url"] = config.ANTHROPIC_BASE_URL
                logger.info(f"LLM endpoint: {config.ANTHROPIC_BASE_URL}")
            self._client = anthropic.AsyncAnthropic(**kwargs)
        except ImportError as exc:  # pragma: no cover - declared dependency
            self._unavailable_reason = f"anthropic SDK not installed: {exc}"
            logger.error(self._unavailable_reason)

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._client is not None

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> ClaudeResponse:
        """One Claude call. ``messages`` overrides ``prompt`` when supplied."""
        if not self.available:
            return ClaudeResponse(text="", error=self._unavailable_reason or "no client")

        payload_messages = messages or [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens or config.MAX_OUTPUT_TOKENS,
            "messages": payload_messages,
        }
        if system:
            kwargs["system"] = system

        last_error = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.messages.create(**kwargs)  # type: ignore[union-attr]
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                input_tokens = int(getattr(response.usage, "input_tokens", 0))
                output_tokens = int(getattr(response.usage, "output_tokens", 0))
                self.usage.add(input_tokens, output_tokens)

                stop_reason = str(getattr(response, "stop_reason", "") or "")
                if stop_reason == "refusal":
                    logger.warning("Claude declined the request (stop_reason=refusal)")
                    return ClaudeResponse(
                        text="",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        error="The model declined to answer this request.",
                    )

                return ClaudeResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stop_reason=stop_reason,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = _is_retryable(exc)
                logger.warning(
                    f"Claude call failed (attempt {attempt}/{_MAX_ATTEMPTS}): "
                    f"{last_error}"
                )
                if not retryable or attempt == _MAX_ATTEMPTS:
                    break
                await asyncio.sleep(_BACKOFF_SECONDS * attempt)

        logger.error(f"Claude call gave up: {last_error}")
        return ClaudeResponse(text="", error=last_error)

    async def complete_many(
        self,
        prompts: list[str],
        system: str | None = None,
        max_tokens: int | None = None,
        concurrency: int = config.SUMMARY_CONCURRENCY,
        model: str | None = None,
    ) -> list[ClaudeResponse]:
        """Run prompts concurrently under a semaphore. Order is preserved."""
        if not prompts:
            return []

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(prompt: str) -> ClaudeResponse:
            async with semaphore:
                return await self.complete(
                    prompt, system=system, max_tokens=max_tokens, model=model
                )

        return await asyncio.gather(*(_run(p) for p in prompts))

    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, overloads, timeouts and 5xx are worth retrying; 4xx are not."""
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    }:
        status = getattr(exc, "status_code", None)
        if name == "APIStatusError" and isinstance(status, int):
            return status == 429 or status >= 500
        return True
    return False


_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
