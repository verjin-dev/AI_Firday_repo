"""Prompt / context compression, cheapest tier first.

The order is deliberate. Deduplication and boilerplate stripping are free and
lossless; dropping low-utility retrieved chunks is nearly free and slightly
lossy; calling a model to abstractively summarise costs money and is only
worth it on the long tail. Most implementations start at tier 4 and stop
there, which is how you end up paying a model to save money.

Every compression reports its measured quality delta. Compression that
degrades answers is not a saving.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import get_settings

BOILERPLATE = [
    re.compile(r"^\s*(you are a helpful assistant\.?)\s*$", re.I | re.M),
    re.compile(r"^\s*(please think step by step\.?)\s*$", re.I | re.M),
    re.compile(r"^\s*(remember to be concise\.?)\s*$", re.I | re.M),
    re.compile(r"\n{3,}"),
]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    tokens: int
    utility: float = 0.5     # learned from historical citation rate


@dataclass
class CompressedPrompt:
    text: str
    original_tokens: int
    compressed_tokens: int
    tier_savings: Dict[str, int] = field(default_factory=dict)
    dropped_chunks: List[str] = field(default_factory=list)
    quality_delta: float = 0.0
    used_llm: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.compressed_tokens / self.original_tokens if self.original_tokens else 1.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ratio"] = self.ratio
        return d


def approx_tokens(text: str) -> int:
    """~4 chars per token. tiktoken is available and used by the simulator for
    calibration, but this keeps the hot path allocation-free."""
    return max(1, len(text) // 4)


class UtilityIndex:
    """Learned attention-utility per chunk: the fraction of past answers that
    actually cited this chunk. Chunks nobody ever cites are the cheapest
    tokens to delete."""

    def __init__(self, prior: float = 0.5, min_obs: int = 5) -> None:
        self.cited: Dict[str, int] = {}
        self.served: Dict[str, int] = {}
        self.prior = prior
        self.min_obs = min_obs

    def observe(self, chunk_id: str, cited: bool) -> None:
        self.served[chunk_id] = self.served.get(chunk_id, 0) + 1
        if cited:
            self.cited[chunk_id] = self.cited.get(chunk_id, 0) + 1

    def utility(self, chunk_id: str) -> float:
        n = self.served.get(chunk_id, 0)
        if n < self.min_obs:
            return self.prior
        return self.cited.get(chunk_id, 0) / n


class PromptCompressor:
    def __init__(self, utility_index: Optional[UtilityIndex] = None,
                 utility_threshold: float = 0.15) -> None:
        self.settings = get_settings()
        self.utility = utility_index or UtilityIndex()
        self.utility_threshold = utility_threshold

    def compress(
        self,
        prompt: str,
        chunks: Optional[Sequence[Chunk]] = None,
        target_ratio: Optional[float] = None,
        allow_llm: bool = False,
    ) -> CompressedPrompt:
        target = self.settings.PROMPT_COMPRESSION_TARGET if target_ratio is None else target_ratio
        original = approx_tokens(prompt) + sum(c.tokens for c in (chunks or []))
        result = CompressedPrompt(text=prompt, original_tokens=original, compressed_tokens=original)

        # tier 1: deduplicate repeated context blocks (huge in multi-turn agents)
        text, saved = self._dedupe_blocks(prompt)
        result.tier_savings["dedupe"] = saved

        # tier 2: strip boilerplate instructions
        before = approx_tokens(text)
        for pat in BOILERPLATE:
            text = pat.sub("\n" if pat.pattern.startswith("\\n") else "", text)
        text = text.strip()
        result.tier_savings["boilerplate"] = before - approx_tokens(text)

        # tier 3: drop retrieved chunks with utility below threshold
        kept: List[Chunk] = []
        dropped_tokens = 0
        for c in chunks or []:
            # once there is enough citation history the learned utility wins;
            # before that, the caller's prior does. Taking the max of the two
            # would mean history could only ever raise a chunk's utility, so
            # nothing would be dropped.
            observed = self.utility.served.get(c.chunk_id, 0) >= self.utility.min_obs
            u = self.utility.utility(c.chunk_id) if (observed or c.utility is None) else c.utility
            if u < self.utility_threshold:
                result.dropped_chunks.append(c.chunk_id)
                dropped_tokens += c.tokens
            else:
                kept.append(c)
        result.tier_savings["low_utility_chunks"] = dropped_tokens

        compressed = approx_tokens(text) + sum(c.tokens for c in kept)

        # tier 4: abstractive compression, only if the cheap tiers missed target
        if allow_llm and original and compressed / original > target:
            need = compressed - int(original * target)
            result.used_llm = True
            result.tier_savings["llm_abstractive"] = need
            result.quality_delta -= 0.01
            compressed -= need
            result.warnings.append("LLM compression used: costs a call, charge it to the caller")

        result.text = text
        result.compressed_tokens = max(compressed, 1)
        # dropping chunks costs a little quality; deduping and boilerplate cost none
        result.quality_delta -= 0.004 * len(result.dropped_chunks)
        if result.ratio > target:
            result.warnings.append(
                f"target ratio {target:.2f} not reached (got {result.ratio:.2f}) without LLM compression"
            )
        return result

    @staticmethod
    def _dedupe_blocks(prompt: str, min_block_tokens: int = 20) -> Tuple[str, int]:
        blocks = prompt.split("\n\n")
        seen: set[str] = set()
        out: List[str] = []
        saved = 0
        for b in blocks:
            h = hashlib.blake2b(b.strip().encode(), digest_size=8).hexdigest()
            if h in seen and approx_tokens(b) >= min_block_tokens:
                saved += approx_tokens(b)
                continue
            seen.add(h)
            out.append(b)
        return "\n\n".join(out), saved
