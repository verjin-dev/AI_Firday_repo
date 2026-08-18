"""Router endpoints: inspect the bandit, and ask it for a route.

The live router is rehydrated from the last day of `router_state`, so the
decisions this endpoint returns reflect what the system actually learned
rather than a fresh, ignorant bandit.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from backend.api.schemas import RouteRequest, RouteResponse
from backend.core.guardrails import CostGuardrails, GuardAction
from backend.core.optimizers.cascade import break_even_escalation_rate
from backend.core.router import LearningRouter
from backend.storage.db import query_df

router = APIRouter(tags=["router"])


@lru_cache
def live_router() -> LearningRouter:
    lr = LearningRouter()
    df = query_df("SELECT * FROM router_state WHERE day = (SELECT MAX(day) FROM router_state)")
    for _, r in df.iterrows():
        arms = lr._task_arms(str(r["task_type"]))          # noqa: SLF001 - intentional rehydrate
        arm = arms.get(str(r["route_id"]))
        if arm is None:
            continue
        arm.alpha = float(r["alpha"])
        arm.beta = float(r["beta"])
        arm.pulls = int(r["pulls"] or 0)
        if r["mean_quality"] is not None and arm.pulls:
            arm.quality_sum = float(r["mean_quality"]) * arm.pulls
            arm.cost_sum = float(r["mean_cost_inr"] or 0.0) * arm.pulls
            arm.reward_sum = float(r["mean_reward"] or 0.0) * arm.pulls
            lr._apply_quality_floor(arm)                    # noqa: SLF001
    return lr


@router.get("/router/{task_type}")
def explain(task_type: str) -> Dict[str, Any]:
    out = live_router().explain(task_type)
    out["break_even_escalation_rate"] = break_even_escalation_rate()
    return out


@router.get("/router/{task_type}/convergence")
def convergence(task_type: str) -> Dict[str, Any]:
    df = query_df(
        "SELECT day, model, SUM(pulls) AS pulls FROM router_state "
        "WHERE task_type = :t GROUP BY day, model ORDER BY day",
        {"t": task_type},
    )
    if df.empty:
        return {"status": "failed", "message": f"no router state for {task_type}"}
    piv = df.pivot(index="day", columns="model", values="pulls").fillna(0)
    daily = piv.diff().fillna(piv)
    share = daily.div(daily.sum(axis=1).replace(0, 1), axis=0)
    return {
        "status": "success",
        "task_type": task_type,
        "days": [int(d) for d in share.index],
        "share": {str(c): [float(v) for v in share[c]] for c in share.columns},
    }


@router.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    """What an agent calls before every step. The guardrail runs first: a
    budget decision outranks a bandit preference."""
    guard = CostGuardrails()
    decision = guard.check(
        {"interactive": req.interactive},
        {"scope": f"tenant:{req.tenant}", "remaining_pct": req.remaining_budget_pct},
    )
    context: Dict[str, Any] = {}
    if decision.action is GuardAction.DEGRADE:
        context["force_cheap"] = True
    rd = live_router().select(req.task_type, context)
    return RouteResponse(
        route=rd.route.as_dict(),
        guard=decision.as_dict(),
        exploring=rd.exploring,
        reason=rd.reason,
    )


@router.get("/router-explain-file")
def explain_file() -> Dict[str, Any]:
    """The explanation snapshot written by the simulator - used by the UI when
    the API is not running."""
    p = Path("data/samples/router_explain.json")
    if not p.exists():
        return {"status": "failed", "message": "run scripts/simulate_workload.py first"}
    return {"status": "success", "data": json.loads(p.read_text(encoding="utf-8"))}
