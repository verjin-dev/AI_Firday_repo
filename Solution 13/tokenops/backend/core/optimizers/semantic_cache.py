"""Semantic cache, scoped to (tenant, workflow).

Two rules that most cache implementations get wrong:
  - a cached result whose recorded quality was below the floor is never
    served, no matter how well the query matches;
  - a cache hit is not free. It is billed at the cache-read tier, and the
    ledger records it that way.

The embedding is pluggable. Offline (the default) it is a deterministic
hashed bag-of-words vector, which is enough for near-duplicate detection and
keeps the demo runnable with no model download. Set `embedder=` to a
sentence-transformers encoder in production.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")
DIM = 384


def hashed_embedding(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic hashing embedding: unigrams + bigrams into `dim` buckets."""
    toks = _TOKEN.findall(text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


@dataclass
class CacheEntry:
    key: str
    text: str
    vector: np.ndarray
    result: Any
    quality: float
    cost_inr: float
    hits: int = 0
    created_epoch: float = 0.0


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    quality_blocked: int = 0
    expired: int = 0
    uncacheable: int = 0
    saved_inr: float = 0.0
    served_inr: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "quality_blocked": self.quality_blocked,
            "expired": self.expired,
            "uncacheable": self.uncacheable,
            "saved_inr": self.saved_inr,
            "cache_read_cost_inr": self.served_inr,
            "net_saved_inr": self.saved_inr - self.served_inr,
        }


class SemanticCache:
    def __init__(
        self,
        threshold: Optional[float] = None,
        quality_floor: Optional[float] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        max_entries_per_scope: int = 4000,
        ttl_seconds: float = 24 * 3600.0,
    ) -> None:
        s = get_settings()
        self.threshold = s.SEMANTIC_CACHE_THRESHOLD if threshold is None else threshold
        self.quality_floor = s.QUALITY_FLOOR if quality_floor is None else quality_floor
        self.embed = embedder or hashed_embedding
        self.max_entries = max_entries_per_scope
        self.ttl_seconds = ttl_seconds
        self.scopes: Dict[str, List[CacheEntry]] = {}
        self.stats = CacheStats()

    @staticmethod
    def _scope_key(tenant: str, workflow: str) -> str:
        return f"{tenant}::{workflow}"

    def lookup(self, text: str, tenant: str, workflow: str,
               threshold: Optional[float] = None,
               now_epoch: Optional[float] = None) -> Tuple[Optional[CacheEntry], float]:
        """A cached answer has a shelf life. Serving a policy answer from a
        cache populated three weeks ago is not a saving, it is an incident."""
        self.stats.lookups += 1
        entries = self.scopes.get(self._scope_key(tenant, workflow), [])
        if not entries:
            return None, 0.0
        thr = self.threshold if threshold is None else threshold
        q = self.embed(text)
        mat = np.vstack([e.vector for e in entries])
        sims = mat @ q
        idx = int(np.argmax(sims))
        best, score = entries[idx], float(sims[idx])
        if score < thr:
            return None, score
        if now_epoch is not None and self.ttl_seconds:
            if now_epoch - best.created_epoch > self.ttl_seconds:
                self.stats.expired += 1
                return None, score
        if best.quality < self.quality_floor:
            self.stats.quality_blocked += 1
            return None, score
        best.hits += 1
        self.stats.hits += 1
        return best, score

    def put(self, text: str, tenant: str, workflow: str, result: Any,
            quality: float, cost_inr: float, now_epoch: float = 0.0) -> CacheEntry:
        key = self._scope_key(tenant, workflow)
        entries = self.scopes.setdefault(key, [])
        entry = CacheEntry(
            key=hashlib.blake2b(text.encode(), digest_size=8).hexdigest(),
            text=text, vector=self.embed(text), result=result,
            quality=quality, cost_inr=cost_inr, created_epoch=now_epoch,
        )
        entries.append(entry)
        if len(entries) > self.max_entries:      # evict least-used
            entries.sort(key=lambda e: e.hits)
            del entries[: len(entries) - self.max_entries]
        return entry

    def record_saving(self, full_cost_inr: float, cache_read_cost_inr: float) -> None:
        self.stats.saved_inr += full_cost_inr - cache_read_cost_inr
        self.stats.served_inr += cache_read_cost_inr

    def size(self) -> int:
        return sum(len(v) for v in self.scopes.values())
