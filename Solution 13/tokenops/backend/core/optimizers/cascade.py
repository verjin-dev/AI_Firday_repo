"""Model cascade: cheap model first, escalate on low confidence.

The number that makes this credible is the break-even escalation rate. Above
it, the cascade costs more than going straight to the strong model, because
you have paid for the cheap attempt as well. Knowing your own break-even -
and reporting when you are near it - is the difference between an
optimisation and a superstition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.config import PRICE_TABLE, get_settings

CHEAP = "claude-haiku-4-5-20251001"
MID = "claude-sonnet-4-6"
STRONG = "claude-opus-5"
LADDER = [CHEAP, MID, STRONG]


@dataclass
class Attempt:
    model: str
    confidence: float
    quality: float
    cost_inr: float
    accepted: bool


@dataclass
class CascadeResult:
    final_model: str
    quality: float
    cost_inr: float
    escalated: bool
    attempts: List[Attempt] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attempts"] = [asdict(a) for a in self.attempts]
        return d


@dataclass
class CascadeStats:
    runs: int = 0
    escalations: int = 0
    cost_inr: float = 0.0
    counterfactual_strong_inr: float = 0.0
    quality_sum: float = 0.0
    counterfactual_quality_sum: float = 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.runs if self.runs else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runs": self.runs,
            "escalation_rate": self.escalation_rate,
            "cost_inr": self.cost_inr,
            "always_strong_cost_inr": self.counterfactual_strong_inr,
            "cost_saved_inr": self.counterfactual_strong_inr - self.cost_inr,
            "mean_quality": self.quality_sum / self.runs if self.runs else 0.0,
            "always_strong_mean_quality": self.counterfactual_quality_sum / self.runs if self.runs else 0.0,
            "quality_delta": (self.quality_sum - self.counterfactual_quality_sum) / self.runs if self.runs else 0.0,
        }


def break_even_escalation_rate(cheap_model: str = CHEAP, strong_model: str = MID,
                               in_tokens: int = 4000, out_tokens: int = 500) -> float:
    """Solve  c_cheap + r*(c_cheap + c_strong) ... more precisely:
    expected cascade cost = c_cheap + r * c_strong ; always-strong = c_strong.
    Break-even at r = 1 - c_cheap / c_strong."""
    def cost(m: str) -> float:
        p = PRICE_TABLE[m]
        return (in_tokens * p["in"] + out_tokens * p["out"]) / 1e6
    c_cheap, c_strong = cost(cheap_model), cost(strong_model)
    return max(0.0, 1.0 - c_cheap / c_strong)


class ModelCascade:
    """`run` is generic over an executor so the same class drives both the
    simulator and a live agent."""

    def __init__(self, ladder: Optional[List[str]] = None,
                 escalation_confidence: Optional[float] = None) -> None:
        s = get_settings()
        self.ladder = ladder or [CHEAP, MID]
        self.threshold = s.CASCADE_ESCALATION_CONFIDENCE if escalation_confidence is None else escalation_confidence
        self.stats = CascadeStats()

    def run(self, task: Dict[str, Any],
            executor: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> CascadeResult:
        """executor(model, task) -> {"quality": float, "confidence": float,
        "cost_inr": float}. Confidence is a composite signal: schema validity,
        retrieval support, and self-reported confidence."""
        attempts: List[Attempt] = []
        total = 0.0
        for i, model in enumerate(self.ladder):
            out = executor(model, task)
            conf = float(out["confidence"])
            total += float(out["cost_inr"])
            last = i == len(self.ladder) - 1
            accept = conf >= self.threshold or last
            attempts.append(Attempt(model, conf, float(out["quality"]), float(out["cost_inr"]), accept))
            if accept:
                result = CascadeResult(
                    final_model=model,
                    quality=float(out["quality"]),
                    cost_inr=total,
                    escalated=i > 0,
                    attempts=attempts,
                    reason=("confidence above threshold" if conf >= self.threshold
                            else "top of ladder reached; accepting"),
                )
                self._record(result, task, executor)
                return result
        raise RuntimeError("cascade ladder exhausted without a decision")

    def _record(self, result: CascadeResult, task: Dict[str, Any],
                executor: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        self.stats.runs += 1
        self.stats.escalations += int(result.escalated)
        self.stats.cost_inr += result.cost_inr
        self.stats.quality_sum += result.quality
        strong = executor(self.ladder[-1], {**task, "counterfactual": True})
        self.stats.counterfactual_strong_inr += float(strong["cost_inr"])
        self.stats.counterfactual_quality_sum += float(strong["quality"])

    def health(self) -> Dict[str, Any]:
        be = break_even_escalation_rate(self.ladder[0], self.ladder[-1])
        rate = self.stats.escalation_rate
        return {
            **self.stats.as_dict(),
            "break_even_escalation_rate": be,
            "headroom": be - rate,
            "verdict": (
                "cascade is paying for itself" if rate < be * 0.8
                else "approaching break-even - review the confidence threshold"
                if rate < be else "above break-even: go straight to the strong model"
            ),
        }
