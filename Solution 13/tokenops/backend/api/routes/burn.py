"""Burn-rate, guardrail and forecast endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Query

from backend.api.schemas import WhatIfRequest
from backend.config import get_settings
from backend.core.burn_rate import Budget, BurnRateMonitor, summarise_alerts
from backend.core.forecaster import CostForecaster
from backend.core.guardrails import CostGuardrails, degraded_request
from backend.core.ledger import CostLedger
from backend.storage.db import query_df

router = APIRouter(tags=["burn"])


def _budget(days: int = 30) -> Budget:
    """Prefer the budget the simulator derived from baseline spend; fall back
    to the configured one."""
    p = Path("data/samples/simulation_notes.json")
    monthly = get_settings().MONTHLY_BUDGET_INR
    if p.exists():
        try:
            monthly = float(json.loads(p.read_text(encoding="utf-8")).get("monthly_budget_inr", monthly))
        except Exception:
            pass
    return Budget("global", "all-tenants", monthly, days)


@router.get("/burn")
def burn(arm: str = Query("tokenops"), dedup_hours: int = 6) -> Dict[str, Any]:
    ledger = CostLedger(arm)
    hourly = ledger.hourly_series()
    if hourly.empty:
        return {"status": "failed", "message": "no data; run the simulator"}
    days = int(hourly["day"].max()) + 1
    monitor = BurnRateMonitor(_budget(days))
    alerts = monitor.scan(hourly, dedup_hours=dedup_hours)
    return {
        "status": "success",
        "arm": arm,
        "budget": monitor.budget_state(hourly),
        "alerts": [a.as_dict() for a in alerts],
        "summary": summarise_alerts(alerts),
        "windows": monitor.windows,
    }


@router.get("/burn/incident")
def incident(day: int = 18) -> Dict[str, Any]:
    """Minute-level replay of one day for both arms - the incident timeline."""
    out: Dict[str, Any] = {"status": "success", "day": day, "arms": {}}
    for arm in ("baseline", "tokenops"):
        df = query_df(
            "SELECT ts_epoch, cost_inr, incident FROM llm_calls WHERE arm = :a AND day = :d",
            {"a": arm, "d": day},
        )
        if df.empty:
            continue
        start = df["ts_epoch"].min()
        df["minute"] = ((df["ts_epoch"] - start) // 60).astype(int)
        per_min = df.groupby("minute")["cost_inr"].sum().cumsum()
        incident_cost = float(df[df["incident"].notna()]["cost_inr"].sum())
        out["arms"][arm] = {
            "minutes": [int(m) for m in per_min.index],
            "cumulative_inr": [float(v) for v in per_min.values],
            "incident_cost_inr": incident_cost,
            "incident_calls": int(df["incident"].notna().sum()),
        }
    p = Path("data/samples/simulation_notes.json")
    if p.exists():
        try:
            notes = json.loads(p.read_text(encoding="utf-8"))
            out["detection"] = notes.get("incident_detection", {})
            out["tokenops_notes"] = notes.get("tokenops_notes", {})
            out["baseline_notes"] = notes.get("baseline_notes", {})
        except Exception:
            pass
    return out


@router.get("/guardrail")
def guardrail(remaining_pct: float = 25.0, interactive: bool = True,
              input_tokens: int = 8400) -> Dict[str, Any]:
    """Show the decision AND the request it would produce. The demo needs the
    before/after, not a verdict string."""
    guard = CostGuardrails()
    request = {"interactive": interactive, "input_tokens": input_tokens}
    decision = guard.check(request, {"scope": "tenant:demo", "remaining_pct": remaining_pct})
    return {
        "status": "success",
        "decision": decision.as_dict(),
        "request_before": request,
        "request_after": degraded_request(request, decision),
    }


@router.get("/forecast")
def forecast(arm: str = Query("tokenops"), horizon_days: int = 30) -> Dict[str, Any]:
    ledger = CostLedger(arm)
    daily = ledger.daily_series()
    if daily.empty:
        return {"status": "failed", "message": "no data; run the simulator"}
    unit = ledger.daily_unit_cost()
    incident_days = query_df(
        "SELECT DISTINCT day FROM llm_calls WHERE arm = :a AND incident IS NOT NULL", {"a": arm}
    )["day"].tolist()
    f = CostForecaster().forecast(
        daily["cost_inr"].tolist(), horizon_days=horizon_days, drivers=unit,
        exclude_incident_days=incident_days,
    )
    out = f.as_dict()
    out["history"] = daily.to_dict(orient="records")
    return out


@router.post("/whatif")
def whatif(req: WhatIfRequest) -> Dict[str, Any]:
    ledger = CostLedger(req.arm)
    daily = ledger.daily_series()
    if daily.empty:
        return {"status": "failed", "message": "no data; run the simulator"}
    days = len(daily)
    monthly = float(daily["cost_inr"].sum()) / days * 30.0
    stats = ledger.optimiser_stats()
    current = {
        "cache_hit_rate": stats["cache_hit_rate"],
        "cheap_model_share": _cheap_share(req.arm),
        "context_reduction": 0.0,
    }
    changes = {k: v for k, v in {
        "cache_hit_rate": req.cache_hit_rate,
        "cheap_model_share": req.cheap_model_share,
        "context_reduction": req.context_reduction,
        "volume_growth": req.volume_growth,
    }.items() if v is not None}
    result = CostForecaster.what_if(monthly, changes, current)
    result["current"] = current
    return result


def _cheap_share(arm: str) -> float:
    df = query_df(
        "SELECT model, COUNT(*) n FROM llm_calls WHERE arm = :a AND is_overhead = 0 GROUP BY model",
        {"a": arm},
    )
    if df.empty:
        return 0.0
    total = float(df["n"].sum())
    cheap = float(df[df["model"].str.contains("haiku")]["n"].sum())
    return cheap / total if total else 0.0
