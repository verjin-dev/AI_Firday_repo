"""Token counting and budget management (tiktoken cl100k_base, cached)."""

from __future__ import annotations

from functools import lru_cache

from backend.utils.helpers import truncate_at_sentence
from backend.utils.logger import logger

_ROLE_OVERHEAD_TOKENS = 4  # per-message framing overhead
_FALLBACK_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoding():
    """cl100k_base encoder, or None when tiktoken is unavailable/offline."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning(f"tiktoken unavailable ({exc}); using char/4 approximation")
        return None


class TokenCounter:
    """Counts tokens with a per-instance cache for repeated strings."""

    def __init__(self) -> None:
        self._cache: dict[str, int] = {}
        self._enc = _encoding()

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        if self._enc is not None:
            try:
                total = len(self._enc.encode(text, disallowed_special=()))
            except Exception:  # pragma: no cover - defensive
                total = max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
        else:
            total = max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)

        # Only cache short strings — long documents would balloon memory.
        if len(text) <= 8192:
            self._cache[text] = total
        return total

    def count_messages(self, messages: list[dict]) -> int:
        """Token count for an OpenAI/Anthropic-style message list."""
        total = 0
        for message in messages or []:
            total += _ROLE_OVERHEAD_TOKENS
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.count(str(block.get("text", "")))
                    else:
                        total += self.count(str(block))
        return total

    def fits_in_budget(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget

    def truncate_to_budget(self, text: str, budget: int) -> tuple[str, int]:
        """Truncate to ``budget`` tokens at a sentence boundary. Returns (text, tokens)."""
        if budget <= 0:
            return "", 0

        current = self.count(text)
        if current <= budget:
            return text, current

        # Estimate the character window, then converge from above.
        ratio = budget / max(current, 1)
        approx_chars = max(1, int(len(text) * ratio))

        for _ in range(6):
            candidate = truncate_at_sentence(text, approx_chars)
            tokens = self.count(candidate)
            if tokens <= budget:
                return candidate, tokens
            approx_chars = int(approx_chars * 0.85)
            if approx_chars < 1:
                break

        hard = text[: budget * _FALLBACK_CHARS_PER_TOKEN]
        return hard, self.count(hard)

    def clear_cache(self) -> None:
        self._cache.clear()


# Module-level shared instance — the encoder is expensive to build.
token_counter = TokenCounter()
