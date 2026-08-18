"""ORM models. The ledger is append-only and columnar-friendly: one row per
LLM call, denormalised with its attribution tags so every aggregation is a
single scan with no joins."""
from __future__ import annotations

from sqlalchemy import Boolean, Column, Float, Index, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class LLMCallRow(Base):
    """The cost ledger. One row per call, in BOTH benchmark arms."""

    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String, index=True)
    arm = Column(String, index=True)            # "baseline" | "tokenops"
    ts = Column(String)                          # UTC ISO-8601
    ts_epoch = Column(Float, index=True)
    day = Column(Integer, index=True)            # 0-based simulation day
    hour = Column(Integer)

    # attribution tags
    tenant = Column(String, index=True)
    team = Column(String)
    agent = Column(String, index=True)
    workflow = Column(String)
    step = Column(String)
    session_id = Column(String, index=True)
    outcome_id = Column(String, index=True)
    outcome_type = Column(String, index=True)
    task_type = Column(String, index=True)

    # routing
    model = Column(String, index=True)
    route_id = Column(String)
    prompt_hash = Column(String, index=True)

    # tokens & money
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cached_tokens = Column(Integer, default=0)
    cost_usd = Column(Float)
    cost_inr = Column(Float)
    latency_ms = Column(Integer)

    # quality & flags
    quality = Column(Float, nullable=True)
    cache_hit = Column(Boolean, default=False)
    escalated = Column(Boolean, default=False)
    compressed = Column(Boolean, default=False)
    is_overhead = Column(Boolean, default=False)   # TokenOps' own metering cost
    status = Column(String, default="success")
    waste_tag = Column(String, nullable=True, index=True)
    incident = Column(String, nullable=True, index=True)


Index("ix_calls_arm_day", LLMCallRow.arm, LLMCallRow.day)
Index("ix_calls_arm_outcome", LLMCallRow.arm, LLMCallRow.outcome_type)


class OutcomeRow(Base):
    """One row per business outcome attempt. `success` decides whether it
    enters the denominator of cost-per-outcome."""

    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    outcome_id = Column(String, index=True)
    arm = Column(String, index=True)
    outcome_type = Column(String, index=True)
    tenant = Column(String, index=True)
    team = Column(String)
    session_id = Column(String, index=True)
    day = Column(Integer, index=True)
    ts_epoch = Column(Float, index=True)
    success = Column(Boolean, index=True)
    quality = Column(Float)
    degraded = Column(Boolean, default=False)


class AlertRow(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    arm = Column(String, index=True)
    ts = Column(String)
    ts_epoch = Column(Float, index=True)
    kind = Column(String)                # burn_rate | loop | circuit_breaker
    severity = Column(String)            # info | warning | critical
    scope = Column(String)
    window_hours = Column(Integer, nullable=True)
    observed_multiplier = Column(Float, nullable=True)
    message = Column(String)


class RouterStateRow(Base):
    """Daily snapshot of the bandit so the convergence chart is real data,
    not a redraw."""

    __tablename__ = "router_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    day = Column(Integer, index=True)
    task_type = Column(String, index=True)
    route_id = Column(String, index=True)
    model = Column(String)
    pulls = Column(Integer)
    share = Column(Float)
    mean_quality = Column(Float)
    mean_cost_inr = Column(Float)
    mean_reward = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)
    exploration = Column(Boolean, default=False)
