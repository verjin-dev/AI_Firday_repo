"""Live-ops endpoints: drive the real-time engine from the API.

The Streamlit page talks to the same process-wide engine, so a route decision
made here shows up there and vice versa.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.core.live import INCIDENTS, get_engine

router = APIRouter(tags=["live"])


class SpeedRequest(BaseModel):
    minutes_per_second: float = Field(1.0, gt=0, le=10)


class BudgetRequest(BaseModel):
    monthly_inr: float = Field(..., gt=0)


class InjectRequest(BaseModel):
    kind: str = Field(..., description=f"one of {list(INCIDENTS)}")


@router.post("/live/start")
def start() -> Dict[str, Any]:
    e = get_engine()
    e.start()
    return {"status": "success", "running": True, "sim_minute": e.sim_minute}


@router.post("/live/stop")
def stop() -> Dict[str, Any]:
    e = get_engine()
    e.stop()
    return {"status": "success", "running": False, "sim_minute": e.sim_minute}


@router.post("/live/reset")
def reset() -> Dict[str, Any]:
    e = get_engine()
    e.stop()
    e.reset()
    return {"status": "success"}


@router.get("/live/state")
def state(minutes: int = 120) -> Dict[str, Any]:
    return get_engine().snapshot(minutes=minutes)


@router.get("/live/incidents")
def incidents() -> Dict[str, Any]:
    return {"status": "success", "available": INCIDENTS,
            "active": get_engine().snapshot()["active_incidents"]}


@router.post("/live/inject")
def inject(req: InjectRequest) -> Dict[str, Any]:
    return get_engine().inject(req.kind)


@router.post("/live/speed")
def speed(req: SpeedRequest) -> Dict[str, Any]:
    get_engine().set_speed(req.minutes_per_second)
    return {"status": "success", "minutes_per_second": req.minutes_per_second}


@router.post("/live/budget")
def budget(req: BudgetRequest) -> Dict[str, Any]:
    get_engine().set_budget(req.monthly_inr)
    return {"status": "success", "monthly_inr": req.monthly_inr}


@router.post("/live/clear-breakers")
def clear_breakers() -> Dict[str, Any]:
    get_engine().clear_breakers()
    return {"status": "success"}
