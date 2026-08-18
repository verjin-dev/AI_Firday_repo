"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "success"
    service: str = "tokenops"
    version: str = "1.0.0"
    price_table_version: str
    tables: Dict[str, int]
    has_data: bool
    offline: bool


class TraceRequest(BaseModel):
    """Outcome-tagging middleware payload: an LLM call plus its business tags."""

    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    latency_ms: int = 0
    quality: Optional[float] = None
    cache_hit: bool = False
    escalated: bool = False
    status: str = "success"
    route_id: str = "manual"
    prompt_hash: str = ""

    tenant: str
    team: str
    agent: str
    workflow: str
    step: str
    session_id: str
    outcome_id: str
    outcome_type: str
    task_type: str = "generic"
    arm: str = "tokenops"


class TraceResponse(BaseModel):
    status: str = "success"
    call_id: str
    cost_inr: float
    cost_usd: float


class UnitEconomicsResponse(BaseModel):
    status: str
    arm: str
    outcome_type: str
    cost_per_outcome_inr: float
    total_cost_inr: float
    successful_outcomes: int
    attempted_outcomes: int
    calls_per_outcome: float
    tokens_per_outcome: float
    p50_cost_inr: float
    p95_cost_inr: float
    mean_quality: float
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []


class RouteRequest(BaseModel):
    task_type: str
    tenant: str = "acme-bank"
    remaining_budget_pct: float = 100.0
    interactive: bool = True


class RouteResponse(BaseModel):
    status: str = "success"
    route: Dict[str, Any]
    guard: Dict[str, Any]
    exploring: bool
    reason: str


class WhatIfRequest(BaseModel):
    cache_hit_rate: Optional[float] = None
    cheap_model_share: Optional[float] = None
    context_reduction: Optional[float] = None
    volume_growth: float = 0.0
    arm: str = "tokenops"
