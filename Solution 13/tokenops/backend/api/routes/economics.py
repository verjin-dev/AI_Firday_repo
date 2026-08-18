"""Unit economics, attribution, waste, showback - the CFO-facing surface."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from backend.api.schemas import TraceRequest, TraceResponse, UnitEconomicsResponse
from backend.core.ledger import CostLedger, LLMCall, OutcomeTags
from backend.storage.db import query_df

router = APIRouter(tags=["economics"])


@router.post("/trace", response_model=TraceResponse)
def ingest_trace(req: TraceRequest) -> TraceResponse:
    """The outcome-tagging middleware endpoint. An untagged call is a call you
    cannot attribute to anything, so tags are required, not optional."""
    ledger = CostLedger(req.arm)
    call = LLMCall(
        model=req.model, input_tokens=req.input_tokens, output_tokens=req.output_tokens,
        cached_tokens=req.cached_tokens, latency_ms=req.latency_ms, quality=req.quality,
        cache_hit=req.cache_hit, escalated=req.escalated, status=req.status,
        route_id=req.route_id, prompt_hash=req.prompt_hash,
    )
    tags = OutcomeTags(
        tenant=req.tenant, team=req.team, agent=req.agent, workflow=req.workflow,
        step=req.step, outcome_id=req.outcome_id, outcome_type=req.outcome_type,
        session_id=req.session_id, task_type=req.task_type,
    )
    row = ledger.record(call, tags)
    return TraceResponse(call_id=row["call_id"], cost_inr=row["cost_inr"], cost_usd=row["cost_usd"])


@router.get("/unit-economics", response_model=UnitEconomicsResponse)
def unit_economics(
    arm: str = Query("tokenops"),
    outcome_type: str = Query("ticket_resolved"),
    group_by: Optional[str] = Query(None, description="tenant | team | agent | step"),
    day_from: Optional[int] = None,
    day_to: Optional[int] = None,
) -> UnitEconomicsResponse:
    ue = CostLedger(arm).unit_economics(
        outcome_type=outcome_type, day_from=day_from, day_to=day_to, group_by=group_by
    )
    return UnitEconomicsResponse(**ue.as_dict())


@router.get("/attribution")
def attribution(arm: str = Query("tokenops"), day_from: Optional[int] = None,
                day_to: Optional[int] = None) -> Dict[str, Any]:
    return CostLedger(arm).attribution(day_from=day_from, day_to=day_to)


@router.get("/waste")
def waste(arm: str = Query("tokenops")) -> Dict[str, Any]:
    return CostLedger(arm).waste_report()


@router.get("/showback")
def showback(arm: str = Query("tokenops")) -> Dict[str, Any]:
    df = CostLedger(arm).showback()
    return {"status": "success", "arm": arm, "rows": df.to_dict(orient="records")}


@router.get("/optimizers")
def optimizers(arm: str = Query("tokenops")) -> Dict[str, Any]:
    return CostLedger(arm).optimiser_stats()


@router.get("/series/daily")
def daily(arm: str = Query("tokenops"), outcome_type: str = Query("ticket_resolved")) -> Dict[str, Any]:
    ledger = CostLedger(arm)
    return {
        "status": "success",
        "spend": ledger.daily_series().to_dict(orient="records"),
        "unit_cost": ledger.daily_unit_cost(outcome_type=outcome_type).to_dict(orient="records"),
    }


@router.get("/alerts")
def alerts(arm: str = Query("tokenops")) -> Dict[str, Any]:
    df = query_df("SELECT * FROM alerts WHERE arm = :a ORDER BY ts_epoch", {"a": arm})
    return {"status": "success", "arm": arm, "rows": df.to_dict(orient="records")}
