"""Single source of truth for turning tokens into money.

Cost is computed once, at record time, from a versioned price table. Nothing
downstream ever recomputes it - if the price table changes, historical rows
keep the price that was actually paid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from backend.config import PRICE_TABLE, get_settings


class UnknownModelError(KeyError):
    pass


@dataclass(frozen=True)
class Cost:
    usd: float
    inr: float

    def as_dict(self) -> Dict[str, float]:
        return {"cost_usd": round(self.usd, 8), "cost_inr": round(self.inr, 6)}


def price_of(model: str) -> Dict[str, float]:
    try:
        return PRICE_TABLE[model]
    except KeyError as exc:  # never silently price at zero
        raise UnknownModelError(f"no price entry for model {model!r}") from exc


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Cost:
    """Cache-read tokens are billed at the discounted tier, not free.

    Pretending cache hits are free is the most common way an LLM cost
    dashboard quietly lies to its owner.
    """
    p = price_of(model)
    usd = (
        max(input_tokens, 0) * p["in"]
        + max(output_tokens, 0) * p["out"]
        + max(cached_tokens, 0) * p.get("cache_read", p["in"] * 0.1)
        + max(cache_write_tokens, 0) * p.get("cache_write", p["in"] * 1.25)
    ) / 1_000_000.0
    return Cost(usd=usd, inr=usd * get_settings().USD_INR)


def to_inr(usd: float) -> float:
    return usd * get_settings().USD_INR


def fmt_inr(amount: float) -> str:
    """Indian-format currency: 1,84,000 not 184,000."""
    neg = amount < 0
    amount = abs(amount)
    if amount >= 10_000_000:
        body = f"{amount / 10_000_000:.2f} Cr"
    elif amount >= 100_000:
        body = f"{amount / 100_000:.2f} L"
    elif amount >= 1000:
        whole = int(round(amount))
        s = str(whole)
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        body = ",".join(parts) + "," + tail
    else:
        body = f"{amount:,.2f}"
    return ("-" if neg else "") + "₹" + body
