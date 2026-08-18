"""Cost guardrails: the part that turns a dashboard into insurance.

DEGRADE is the important decision. Blocking a workflow to save money is a
support ticket; degrading it - cheap route, half the context, verification
skipped - completes the work at a lower quality and a much lower cost, and
the user usually cannot tell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from backend.config import get_settings


class GuardAction(str, Enum):
    ALLOW = "ALLOW"
    DEGRADE = "DEGRADE"
    QUEUE = "QUEUE"
    BLOCK = "BLOCK"


@dataclass
class GuardDecision:
    action: GuardAction
    reason: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    remaining_pct: float = 100.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class LoopDetection:
    detected: bool
    session_id: Optional[str] = None
    signature: Optional[str] = None
    repeats: int = 0
    wasted_calls: int = 0
    wasted_inr: float = 0.0
    first_seen_epoch: Optional[float] = None
    detected_at_epoch: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostGuardrails:
    """Thresholds are expressed as remaining-budget percentage so the same
    policy works for a 2 lakh tenant and a 2 crore one."""

    DEGRADE_BELOW_PCT = 30.0
    QUEUE_BELOW_PCT = 10.0
    BLOCK_BELOW_PCT = 2.0

    def __init__(self) -> None:
        self.settings = get_settings()
        self.broken_scopes: Dict[str, str] = {}

    def check(self, request: Dict[str, Any], budget_state: Dict[str, Any]) -> GuardDecision:
        scope = budget_state.get("scope", "global")
        if scope in self.broken_scopes:
            return GuardDecision(GuardAction.BLOCK, f"circuit breaker open: {self.broken_scopes[scope]}",
                                 remaining_pct=budget_state.get("remaining_pct", 0.0))

        remaining = float(budget_state.get("remaining_pct", 100.0))
        interactive = bool(request.get("interactive", True))

        if remaining <= self.BLOCK_BELOW_PCT:
            return GuardDecision(GuardAction.BLOCK,
                                 f"budget exhausted ({remaining:.1f}% remaining)", remaining_pct=remaining)
        if remaining <= self.QUEUE_BELOW_PCT and not interactive:
            return GuardDecision(GuardAction.QUEUE,
                                 f"deferrable work queued to next period ({remaining:.1f}% remaining)",
                                 {"defer": True}, remaining)
        if remaining <= self.DEGRADE_BELOW_PCT:
            return GuardDecision(
                GuardAction.DEGRADE,
                f"low budget ({remaining:.1f}% remaining): cheap route, shallow context, "
                "verification skipped, cache threshold relaxed",
                {
                    "force_cheap": True,
                    "context_depth": "shallow",
                    "context_scale": 0.5,
                    "skip_verification": True,
                    "cache_threshold": max(0.86, self.settings.SEMANTIC_CACHE_THRESHOLD - 0.06),
                },
                remaining,
            )
        return GuardDecision(GuardAction.ALLOW, "within budget", {}, remaining)

    # ------------------------------------------------------------ loop guard
    def detect_loop(self, calls: Iterable[Dict[str, Any]],
                    threshold: Optional[int] = None) -> LoopDetection:
        """Repeated identical (step, prompt_hash) inside one session beyond
        threshold. Agent loops are the single largest source of surprise LLM
        bills, and they are trivially detectable - nobody looks."""
        threshold = threshold or self.settings.LOOP_DETECTION_REPEAT_THRESHOLD
        seen: Dict[str, List[Dict[str, Any]]] = {}
        for c in calls:
            sig = f"{c.get('step')}::{c.get('prompt_hash')}"
            seen.setdefault(sig, []).append(c)
            if len(seen[sig]) >= threshold:
                group = seen[sig]
                return LoopDetection(
                    detected=True,
                    session_id=c.get("session_id"),
                    signature=sig,
                    repeats=len(group),
                    wasted_calls=len(group) - 1,
                    wasted_inr=float(sum(x.get("cost_inr", 0.0) for x in group[1:])),
                    first_seen_epoch=group[0].get("ts_epoch"),
                    detected_at_epoch=c.get("ts_epoch"),
                )
        return LoopDetection(detected=False)

    # ------------------------------------------------------- circuit breaker
    def circuit_break(self, scope: str, reason: str) -> None:
        self.broken_scopes[scope] = reason

    def reset(self, scope: Optional[str] = None) -> None:
        if scope is None:
            self.broken_scopes.clear()
        else:
            self.broken_scopes.pop(scope, None)

    def should_break(self, observed_multiplier: float) -> bool:
        return observed_multiplier >= self.settings.CIRCUIT_BREAKER_MULTIPLIER


def degraded_request(request: Dict[str, Any], decision: GuardDecision) -> Dict[str, Any]:
    """Apply a DEGRADE decision to a request dict. Pure function so the demo
    can show the before/after side by side."""
    out = dict(request)
    if decision.action is not GuardAction.DEGRADE:
        return out
    o = decision.overrides
    out["force_cheap"] = True
    out["context_depth"] = o.get("context_depth", "shallow")
    out["input_tokens"] = int(out.get("input_tokens", 0) * o.get("context_scale", 0.5))
    out["skip_verification"] = True
    out["cache_threshold"] = o.get("cache_threshold")
    return out
