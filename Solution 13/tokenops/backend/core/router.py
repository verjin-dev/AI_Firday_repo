"""The learning router - a constrained Thompson-sampling bandit over routes.

A static routing table is a snapshot of one engineer's beliefs on one Tuesday.
It rots as prompts, models and traffic change. This router learns the route
per task type from observed reward, under three constraints that make it
deployable rather than a research toy:

  1. a hard quality floor - an arm that has ever averaged below the floor
     (with enough samples to believe it) is excluded before sampling;
  2. an exploration budget cap - exploratory spend is capped at a fixed
     percentage of period spend, after which the router exploits only;
  3. warm-start priors from a heuristic table, so day one is sensible
     rather than random.

Everything is deliberately legible: `explain()` returns the full per-arm
table, because an operations team will not accept a black-box router.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config import MODEL_SHORT, get_settings

# ---------------------------------------------------------------- route space


@dataclass(frozen=True)
class Route:
    model: str
    prompt_variant: str      # "full" | "terse"
    context_depth: str       # "deep" | "shallow"
    cache_policy: str        # "standard" | "aggressive"

    @property
    def route_id(self) -> str:
        return f"{MODEL_SHORT.get(self.model, self.model)}|{self.prompt_variant}|{self.context_depth}|{self.cache_policy}"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["route_id"] = self.route_id
        return d


def default_route_space() -> List[Route]:
    """10 arms per task type: 3 models x prompt/context variants, plus two
    aggressive-cache variants. Wider spaces converge too slowly to be honest
    about within a 30-day window."""
    routes = [
        Route("claude-haiku-4-5-20251001", "terse", "shallow", "aggressive"),
        Route("claude-haiku-4-5-20251001", "terse", "deep", "standard"),
        Route("claude-haiku-4-5-20251001", "full", "deep", "standard"),
        Route("claude-sonnet-4-6", "terse", "shallow", "aggressive"),
        Route("claude-sonnet-4-6", "terse", "deep", "standard"),
        Route("claude-sonnet-4-6", "full", "shallow", "standard"),
        Route("claude-sonnet-4-6", "full", "deep", "standard"),
        Route("claude-opus-5", "terse", "shallow", "aggressive"),
        Route("claude-opus-5", "full", "deep", "standard"),
        Route("claude-opus-5", "full", "shallow", "standard"),
    ]
    return routes


# Warm-start beliefs: what a cost-aware engineer would guess on day one, and
# no more. Deliberately weak and nearly flat - if the prior already encoded
# the answer, the bandit would be decoration. The only thing it really asserts
# is "the strong model is probably not worth it for the easy tasks".
HEURISTIC_PRIOR = {
    "classification": {"claude-haiku-4-5-20251001": 0.58, "claude-sonnet-4-6": 0.56, "claude-opus-5": 0.50},
    "retrieval": {"claude-haiku-4-5-20251001": 0.55, "claude-sonnet-4-6": 0.57, "claude-opus-5": 0.51},
    "generation": {"claude-haiku-4-5-20251001": 0.52, "claude-sonnet-4-6": 0.58, "claude-opus-5": 0.55},
    "verification": {"claude-haiku-4-5-20251001": 0.57, "claude-sonnet-4-6": 0.56, "claude-opus-5": 0.50},
}
PRIOR_STRENGTH = 3.0     # in pseudo-observations; overruled within an hour of traffic


@dataclass
class ArmState:
    route: Route
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    explore_pulls: int = 0
    quality_sum: float = 0.0
    cost_sum: float = 0.0
    reward_sum: float = 0.0
    excluded: bool = False
    exclusion_reason: Optional[str] = None

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / self.pulls if self.pulls else float("nan")

    @property
    def mean_cost_inr(self) -> float:
        return self.cost_sum / self.pulls if self.pulls else float("nan")

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else float("nan")

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class RouterDecision:
    route: Route
    task_type: str
    exploring: bool
    reason: str
    candidates: int
    exploration_spend_pct: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.as_dict(),
            "task_type": self.task_type,
            "exploring": self.exploring,
            "reason": self.reason,
            "candidates": self.candidates,
            "exploration_spend_pct": self.exploration_spend_pct,
        }


class LearningRouter:
    """One bandit per task type. State is a few hundred Beta distributions -
    it fits in memory at any realistic scale."""

    def __init__(
        self,
        routes: Optional[List[Route]] = None,
        lam: Optional[float] = None,
        seed: int = 7,
        cost_reference_inr: float = 3.0,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.lam = s.BANDIT_LAMBDA if lam is None else lam
        self.routes = routes or default_route_space()
        self.rng = np.random.default_rng(seed)
        self.cost_reference_inr = cost_reference_inr
        self.arms: Dict[str, Dict[str, ArmState]] = {}
        self.exploration_spend: Dict[str, float] = {}
        self.total_spend: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- internals
    def _task_arms(self, task_type: str) -> Dict[str, ArmState]:
        if task_type not in self.arms:
            prior = HEURISTIC_PRIOR.get(task_type, {})
            arms: Dict[str, ArmState] = {}
            for r in self.routes:
                p = prior.get(r.model, 0.5)
                # cheap/shallow variants get a small prior nudge: they are the
                # ones a cost-aware engineer would try first.
                if r.context_depth == "shallow":
                    p += 0.02
                if r.cache_policy == "aggressive":
                    p += 0.01
                p = float(np.clip(p, 0.05, 0.95))
                arms[r.route_id] = ArmState(route=r, alpha=1.0 + PRIOR_STRENGTH * p,
                                            beta=1.0 + PRIOR_STRENGTH * (1.0 - p))
            self.arms[task_type] = arms
            self.exploration_spend.setdefault(task_type, 0.0)
            self.total_spend.setdefault(task_type, 0.0)
        return self.arms[task_type]

    def _normalised_cost(self, cost_inr: float) -> float:
        return float(np.clip(cost_inr / max(self.cost_reference_inr, 1e-9), 0.0, 1.0))

    def reward(self, quality: float, cost_inr: float) -> float:
        """reward = quality - lambda * normalised_cost, squashed into [0, 1]
        so it can drive a Beta posterior."""
        raw = quality - self.lam * self._normalised_cost(cost_inr)
        return float(np.clip((raw + self.lam) / (1.0 + self.lam), 0.0, 1.0))

    def exploration_spend_pct(self, task_type: str) -> float:
        total = self.total_spend.get(task_type, 0.0)
        return (self.exploration_spend.get(task_type, 0.0) / total * 100.0) if total > 0 else 0.0

    def _best_known(self, arms: Dict[str, ArmState]) -> ArmState:
        seen = [a for a in arms.values() if not a.excluded and a.pulls > 0]
        pool = seen or [a for a in arms.values() if not a.excluded] or list(arms.values())
        return max(pool, key=lambda a: a.posterior_mean)

    # ---------------------------------------------------------------- select
    def select(self, task_type: str, context: Optional[Dict[str, Any]] = None) -> RouterDecision:
        context = context or {}
        arms = self._task_arms(task_type)
        s = self.settings

        candidates = {rid: a for rid, a in arms.items() if not a.excluded}
        # Budget-aware degradation is a hard constraint, not a preference.
        if context.get("force_cheap"):
            cheap = {rid: a for rid, a in candidates.items()
                     if a.route.model == "claude-haiku-4-5-20251001" and a.route.context_depth == "shallow"}
            if cheap:
                best = max(cheap.values(), key=lambda a: a.posterior_mean)
                return RouterDecision(best.route, task_type, False, "degraded: budget guardrail forced cheap route",
                                      len(cheap), self.exploration_spend_pct(task_type))
        if not candidates:
            best = self._best_known(arms)
            return RouterDecision(best.route, task_type, False, "all arms excluded by quality floor; "
                                  "falling back to best observed", 0, self.exploration_spend_pct(task_type))

        exploit_only = self.exploration_spend_pct(task_type) >= s.BANDIT_EXPLORATION_BUDGET_PCT
        greedy = max(candidates.values(), key=lambda a: a.posterior_mean)

        if exploit_only:
            return RouterDecision(greedy.route, task_type, False,
                                  "exploration budget exhausted; exploiting best posterior",
                                  len(candidates), self.exploration_spend_pct(task_type))

        draws = {rid: float(self.rng.beta(a.alpha, a.beta)) for rid, a in candidates.items()}
        pick_id = max(draws, key=draws.get)
        picked = candidates[pick_id]
        exploring = pick_id != greedy.route.route_id
        reason = "thompson draw (exploring)" if exploring else "thompson draw (agrees with greedy)"
        return RouterDecision(picked.route, task_type, exploring, reason, len(candidates),
                              self.exploration_spend_pct(task_type))

    # ---------------------------------------------------------------- update
    def update(self, task_type: str, route_id: str, quality: float, cost_inr: float,
               exploring: bool = False, day: Optional[int] = None) -> float:
        arms = self._task_arms(task_type)
        arm = arms.get(route_id)
        if arm is None:
            return 0.0
        r = self.reward(quality, cost_inr)
        arm.alpha += r
        arm.beta += 1.0 - r
        arm.pulls += 1
        arm.quality_sum += quality
        arm.cost_sum += cost_inr
        arm.reward_sum += r
        if exploring:
            arm.explore_pulls += 1
            self.exploration_spend[task_type] = self.exploration_spend.get(task_type, 0.0) + cost_inr
        self.total_spend[task_type] = self.total_spend.get(task_type, 0.0) + cost_inr
        self._apply_quality_floor(arm)
        return r

    def _apply_quality_floor(self, arm: ArmState) -> None:
        """Exclude only once there is enough evidence: excluding on one bad
        sample would make the router superstitious."""
        floor = self.settings.QUALITY_FLOOR
        min_n = max(8, self.settings.BANDIT_MIN_SAMPLES_PER_ARM // 4)
        if arm.pulls >= min_n and arm.mean_quality < floor:
            arm.excluded = True
            arm.exclusion_reason = f"mean quality {arm.mean_quality:.3f} < floor {floor:.2f} over {arm.pulls} pulls"

    def reset_exploration_budget(self) -> None:
        """Called at the start of each budget period."""
        for k in list(self.exploration_spend):
            self.exploration_spend[k] = 0.0
            self.total_spend[k] = 0.0

    # --------------------------------------------------------------- explain
    def explain(self, task_type: str, ci_draws: int = 4000) -> Dict[str, Any]:
        arms = self._task_arms(task_type)
        live = {rid: a for rid, a in arms.items() if not a.excluded}
        # selection probability by Monte-Carlo over the posteriors
        probs: Dict[str, float] = {rid: 0.0 for rid in arms}
        if live:
            names = list(live)
            samples = np.column_stack([self.rng.beta(live[n].alpha, live[n].beta, ci_draws) for n in names])
            winners = np.argmax(samples, axis=1)
            for i, n in enumerate(names):
                probs[n] = float(np.mean(winners == i))

        rows: List[Dict[str, Any]] = []
        for rid, a in arms.items():
            draws = self.rng.beta(a.alpha, a.beta, ci_draws)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append(
                {
                    "route_id": rid,
                    "model": a.route.model,
                    "model_short": MODEL_SHORT.get(a.route.model, a.route.model),
                    "prompt_variant": a.route.prompt_variant,
                    "context_depth": a.route.context_depth,
                    "cache_policy": a.route.cache_policy,
                    "pulls": a.pulls,
                    "explore_pulls": a.explore_pulls,
                    "mean_quality": None if a.pulls == 0 else round(a.mean_quality, 4),
                    "mean_cost_inr": None if a.pulls == 0 else round(a.mean_cost_inr, 4),
                    "mean_reward": None if a.pulls == 0 else round(a.mean_reward, 4),
                    "posterior_mean": round(a.posterior_mean, 4),
                    "ci_low": round(float(lo), 4),
                    "ci_high": round(float(hi), 4),
                    "selection_prob": round(probs.get(rid, 0.0), 4),
                    "excluded": a.excluded,
                    "exclusion_reason": a.exclusion_reason,
                }
            )
        rows.sort(key=lambda r: (-r["selection_prob"], -(r["posterior_mean"])))
        total_pulls = sum(r["pulls"] for r in rows)
        return {
            "task_type": task_type,
            "lambda": self.lam,
            "quality_floor": self.settings.QUALITY_FLOOR,
            "exploration_budget_pct": self.settings.BANDIT_EXPLORATION_BUDGET_PCT,
            "exploration_spend_pct": round(self.exploration_spend_pct(task_type), 3),
            "total_pulls": total_pulls,
            "arms": rows,
            "status": "success",
        }

    def snapshot(self, day: int, task_type: str) -> List[Dict[str, Any]]:
        """Row-per-arm snapshot for the convergence chart."""
        arms = self._task_arms(task_type)
        total = sum(a.pulls for a in arms.values()) or 1
        out = []
        for rid, a in arms.items():
            out.append(
                {
                    "day": day,
                    "task_type": task_type,
                    "route_id": rid,
                    "model": a.route.model,
                    "pulls": a.pulls,
                    "share": a.pulls / total,
                    "mean_quality": None if a.pulls == 0 else float(a.mean_quality),
                    "mean_cost_inr": None if a.pulls == 0 else float(a.mean_cost_inr),
                    "mean_reward": None if a.pulls == 0 else float(a.mean_reward),
                    "alpha": a.alpha,
                    "beta": a.beta,
                    "exploration": a.explore_pulls > 0,
                }
            )
        return out

    def policy(self, task_type: str) -> Route:
        arms = self._task_arms(task_type)
        return self._best_known(arms).route

    def convergence_day(self, snapshots: List[Dict[str, Any]], task_type: str,
                        stable_days: int = 3) -> Optional[int]:
        """First day after which the top-share model stops changing."""
        by_day: Dict[int, Dict[str, float]] = {}
        for s in snapshots:
            if s["task_type"] != task_type:
                continue
            by_day.setdefault(s["day"], {})
            by_day[s["day"]][s["model"]] = by_day[s["day"]].get(s["model"], 0.0) + s["share"]
        days = sorted(by_day)
        leaders = [max(by_day[d], key=by_day[d].get) if by_day[d] else None for d in days]
        for i in range(len(days) - stable_days):
            window = leaders[i : i + stable_days + 1]
            if window and all(w == window[0] for w in window) and all(l == window[0] for l in leaders[i:]):
                return days[i]
        return None
