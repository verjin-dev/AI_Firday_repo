"""The cost ledger - every analytic in TokenOps reads from here.

The design decision that matters for the demo: FAILED outcomes contribute to
the numerator of cost-per-outcome but never to the denominator. Wasted spend
must appear in the unit cost, or you optimise the wrong thing.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import get_settings
from backend.core.pricing import compute_cost
from backend.storage.db import bulk_insert, query_df
from backend.utils.errors import NoDataError

ARMS = ("baseline", "tokenops")

WASTE_OWNERS = {
    "duplicate_calls": "Platform / caching",
    "retry_waste": "Reliability (retry policy)",
    "abandoned_sessions": "Product (session UX)",
    "over_retrieval": "Retrieval team",
    "verbose_output": "Prompt owners",
    "loop_waste": "Agent orchestration",
}


@dataclass
class OutcomeTags:
    tenant: str
    team: str
    agent: str
    workflow: str
    step: str
    outcome_id: str
    outcome_type: str
    session_id: str
    task_type: str = "generic"


@dataclass
class LLMCall:
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    latency_ms: int = 0
    quality: Optional[float] = None
    cache_hit: bool = False
    escalated: bool = False
    compressed: bool = False
    is_overhead: bool = False
    status: str = "success"
    route_id: str = "default"
    prompt_hash: str = ""
    waste_tag: Optional[str] = None
    incident: Optional[str] = None
    ts_epoch: Optional[float] = None
    day: int = 0
    hour: int = 0


@dataclass
class UnitEconomics:
    arm: str
    outcome_type: str
    group_by: Optional[str] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)
    total_cost_inr: float = 0.0
    successful_outcomes: int = 0
    attempted_outcomes: int = 0
    cost_per_outcome_inr: float = 0.0
    calls_per_outcome: float = 0.0
    tokens_per_outcome: float = 0.0
    p50_cost_inr: float = 0.0
    p95_cost_inr: float = 0.0
    mean_quality: float = 0.0
    status: str = "success"
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostLedger:
    """Append-only. `record` is the live-tracing path; the simulator uses
    `record_many` for bulk load."""

    def __init__(self, arm: str = "tokenops") -> None:
        self.arm = arm
        self.settings = get_settings()

    # ---------------------------------------------------------------- write --
    def to_row(self, call: LLMCall, tags: OutcomeTags, arm: Optional[str] = None) -> Dict[str, Any]:
        cost = compute_cost(
            call.model,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            cached_tokens=call.cached_tokens,
        )
        ts_epoch = call.ts_epoch if call.ts_epoch is not None else datetime.now(timezone.utc).timestamp()
        return {
            "call_id": f"call_{uuid.uuid4().hex[:16]}",
            "arm": arm or self.arm,
            "ts": datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat(),
            "ts_epoch": ts_epoch,
            "day": call.day,
            "hour": call.hour,
            "tenant": tags.tenant,
            "team": tags.team,
            "agent": tags.agent,
            "workflow": tags.workflow,
            "step": tags.step,
            "session_id": tags.session_id,
            "outcome_id": tags.outcome_id,
            "outcome_type": tags.outcome_type,
            "task_type": tags.task_type,
            "model": call.model,
            "route_id": call.route_id,
            "prompt_hash": call.prompt_hash,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cached_tokens": call.cached_tokens,
            "cost_usd": cost.usd,
            "cost_inr": cost.inr,
            "latency_ms": call.latency_ms,
            "quality": call.quality,
            "cache_hit": call.cache_hit,
            "escalated": call.escalated,
            "compressed": call.compressed,
            "is_overhead": call.is_overhead,
            "status": call.status,
            "waste_tag": call.waste_tag,
            "incident": call.incident,
        }

    def record(self, call: LLMCall, tags: OutcomeTags) -> Dict[str, Any]:
        row = self.to_row(call, tags)
        bulk_insert("llm_calls", [row])
        return row

    def record_many(self, rows: List[Dict[str, Any]]) -> int:
        return bulk_insert("llm_calls", rows)

    # ------------------------------------------------------- unit economics --
    def unit_economics(
        self,
        outcome_type: str = "ticket_resolved",
        arm: Optional[str] = None,
        day_from: Optional[int] = None,
        day_to: Optional[int] = None,
        group_by: Optional[str] = None,
    ) -> UnitEconomics:
        arm = arm or self.arm
        where = ["arm = :arm", "outcome_type = :ot"]
        params: Dict[str, Any] = {"arm": arm, "ot": outcome_type}
        if day_from is not None:
            where.append("day >= :df")
            params["df"] = day_from
        if day_to is not None:
            where.append("day <= :dt")
            params["dt"] = day_to
        w = " AND ".join(where)

        calls = query_df(
            "SELECT outcome_id, cost_inr, input_tokens, output_tokens, tenant, team, "
            "agent, step, quality FROM llm_calls WHERE " + w,
            params,
        )
        outs = query_df("SELECT * FROM outcomes WHERE " + w, params)
        if calls.empty:
            raise NoDataError(
                f"no ledger rows for arm={arm} outcome_type={outcome_type}",
                {"hint": "run: python scripts/simulate_workload.py"},
            )

        total_cost = float(calls["cost_inr"].sum())
        successful = outs[outs["success"] == 1] if not outs.empty else outs
        n_success = int(len(successful))
        n_attempt = int(len(outs))
        per_outcome = calls.groupby("outcome_id")["cost_inr"].sum()
        succ_ids = set(successful["outcome_id"]) if n_success else set()
        succ_costs = per_outcome[per_outcome.index.isin(succ_ids)] if succ_ids else per_outcome.iloc[0:0]

        ue = UnitEconomics(
            arm=arm,
            outcome_type=outcome_type,
            group_by=group_by,
            total_cost_inr=total_cost,
            successful_outcomes=n_success,
            attempted_outcomes=n_attempt,
            cost_per_outcome_inr=(total_cost / n_success) if n_success else 0.0,
            calls_per_outcome=(len(calls) / n_success) if n_success else 0.0,
            tokens_per_outcome=(
                float(calls["input_tokens"].sum() + calls["output_tokens"].sum()) / n_success
            )
            if n_success
            else 0.0,
            p50_cost_inr=float(succ_costs.median()) if len(succ_costs) else 0.0,
            p95_cost_inr=float(succ_costs.quantile(0.95)) if len(succ_costs) else 0.0,
            mean_quality=float(successful["quality"].mean()) if n_success else 0.0,
        )
        if n_attempt and n_success < n_attempt:
            failed = n_attempt - n_success
            ue.warnings.append(
                f"{failed} failed outcomes ({failed / n_attempt:.1%}) counted in spend, "
                "excluded from the denominator"
            )
        if group_by:
            ue.rows = self._grouped(calls, outs, group_by)
        return ue

    @staticmethod
    def _grouped(calls: pd.DataFrame, outs: pd.DataFrame, group_by: str) -> List[Dict[str, Any]]:
        if group_by not in calls.columns:
            return []
        cost = calls.groupby(group_by)["cost_inr"].sum()
        total = float(cost.sum())
        if group_by in outs.columns and not outs.empty:
            ok = outs[outs["success"] == 1]
            succ = ok.groupby(group_by)["outcome_id"].nunique()
            qual = ok.groupby(group_by)["quality"].mean()
        else:
            succ = pd.Series(dtype=float)
            qual = pd.Series(dtype=float)
        rows = []
        for key, c in cost.sort_values(ascending=False).items():
            n = int(succ.get(key, 0))
            rows.append(
                {
                    group_by: str(key),
                    "cost_inr": float(c),
                    "outcomes": n,
                    "cost_per_outcome_inr": float(c / n) if n else None,
                    "mean_quality": float(qual.get(key)) if n and key in qual.index else None,
                    "share_pct": float(c / total * 100.0) if total else 0.0,
                }
            )
        return rows

    # -------------------------------------------------------------- series --
    def daily_series(self, arm: Optional[str] = None) -> pd.DataFrame:
        return query_df(
            "SELECT day, SUM(cost_inr) AS cost_inr, COUNT(*) AS calls, "
            "SUM(input_tokens + output_tokens) AS tokens "
            "FROM llm_calls WHERE arm = :arm GROUP BY day ORDER BY day",
            {"arm": arm or self.arm},
        )

    def hourly_series(self, arm: Optional[str] = None) -> pd.DataFrame:
        return query_df(
            "SELECT day, hour, SUM(cost_inr) AS cost_inr, COUNT(*) AS calls, "
            "MIN(ts_epoch) AS ts_epoch FROM llm_calls WHERE arm = :arm "
            "GROUP BY day, hour ORDER BY day, hour",
            {"arm": arm or self.arm},
        )

    def daily_unit_cost(self, arm: Optional[str] = None,
                        outcome_type: str = "ticket_resolved") -> pd.DataFrame:
        """Volume vs unit-cost decomposition: the CFO view needs both curves."""
        arm = arm or self.arm
        cost = query_df(
            "SELECT day, SUM(cost_inr) AS cost_inr FROM llm_calls "
            "WHERE arm = :arm AND outcome_type = :ot GROUP BY day ORDER BY day",
            {"arm": arm, "ot": outcome_type},
        )
        outs = query_df(
            "SELECT day, COUNT(*) AS outcomes FROM outcomes "
            "WHERE arm = :arm AND outcome_type = :ot AND success = 1 "
            "GROUP BY day ORDER BY day",
            {"arm": arm, "ot": outcome_type},
        )
        df = cost.merge(outs, on="day", how="left").fillna({"outcomes": 0})
        df["cost_per_outcome_inr"] = df.apply(
            lambda r: r["cost_inr"] / r["outcomes"] if r["outcomes"] else None, axis=1
        )
        return df

    # --------------------------------------------------------- attribution --
    def flat_attribution(self, arm: Optional[str] = None) -> pd.DataFrame:
        return query_df(
            "SELECT tenant, team, agent, workflow, step, SUM(cost_inr) AS cost_inr, "
            "COUNT(*) AS calls, AVG(quality) AS quality FROM llm_calls "
            "WHERE arm = :arm GROUP BY tenant, team, agent, workflow, step",
            {"arm": arm or self.arm},
        )

    def attribution(
        self,
        arm: Optional[str] = None,
        levels: Optional[List[str]] = None,
        day_from: Optional[int] = None,
        day_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Drill-down tree tenant -> team -> agent -> workflow -> step.

        Every node carries cost, share of parent, and week-over-week delta.
        """
        arm = arm or self.arm
        levels = levels or ["tenant", "team", "agent", "workflow", "step"]
        where = ["arm = :arm"]
        params: Dict[str, Any] = {"arm": arm}
        if day_from is not None:
            where.append("day >= :df")
            params["df"] = day_from
        if day_to is not None:
            where.append("day <= :dt")
            params["dt"] = day_to
        df = query_df(
            f"SELECT {', '.join(levels)}, day, cost_inr FROM llm_calls WHERE {' AND '.join(where)}",
            params,
        )
        if df.empty:
            raise NoDataError(f"no ledger rows for arm={arm}")

        max_day = int(df["day"].max())
        total = float(df["cost_inr"].sum())

        def build(subset: pd.DataFrame, prefix: List[str], depth: int) -> List[Dict[str, Any]]:
            if depth >= len(levels):
                return []
            level = levels[depth]
            parent_total = float(subset["cost_inr"].sum())
            nodes: List[Dict[str, Any]] = []
            for key, grp in subset.groupby(level):
                cost = float(grp["cost_inr"].sum())
                tw = float(grp[grp["day"] > max_day - 7]["cost_inr"].sum())
                pw = float(grp[(grp["day"] <= max_day - 7) & (grp["day"] > max_day - 14)]["cost_inr"].sum())
                nodes.append(
                    {
                        "level": level,
                        "name": str(key),
                        "path": "/".join(prefix + [str(key)]),
                        "cost_inr": cost,
                        "share_pct": cost / total * 100.0 if total else 0.0,
                        "share_of_parent_pct": cost / parent_total * 100.0 if parent_total else 0.0,
                        "wow_delta_pct": ((tw - pw) / pw * 100.0) if pw > 0 else None,
                        "children": build(grp, prefix + [str(key)], depth + 1),
                    }
                )
            return sorted(nodes, key=lambda n: -n["cost_inr"])

        return {
            "arm": arm,
            "total_cost_inr": total,
            "levels": levels,
            "tree": build(df, [], 0),
            "status": "success",
        }

    # ---------------------------------------------------------- waste report --
    def waste_report(self, arm: Optional[str] = None) -> Dict[str, Any]:
        """Named waste categories with rupee figures and an owner each.

        `abandoned_sessions` is derived, not tagged: it is all spend on
        sessions whose outcome never succeeded.
        """
        arm = arm or self.arm
        tagged = query_df(
            "SELECT waste_tag, SUM(cost_inr) AS cost_inr, COUNT(*) AS calls "
            "FROM llm_calls WHERE arm = :arm AND waste_tag IS NOT NULL "
            "GROUP BY waste_tag",
            {"arm": arm},
        )
        abandoned = query_df(
            "SELECT SUM(c.cost_inr) AS cost_inr, COUNT(*) AS calls FROM llm_calls c "
            "JOIN outcomes o ON o.outcome_id = c.outcome_id AND o.arm = c.arm "
            "WHERE c.arm = :arm AND o.success = 0",
            {"arm": arm},
        )
        total = float(
            query_df("SELECT SUM(cost_inr) AS c FROM llm_calls WHERE arm = :arm", {"arm": arm})["c"].iloc[0]
            or 0.0
        )

        items: List[Dict[str, Any]] = []
        for _, r in tagged.iterrows():
            items.append(
                {
                    "category": r["waste_tag"],
                    "cost_inr": float(r["cost_inr"]),
                    "calls": int(r["calls"]),
                    "owner": WASTE_OWNERS.get(r["waste_tag"], "unassigned"),
                }
            )
        ab_cost = float(abandoned["cost_inr"].iloc[0] or 0.0)
        if ab_cost > 0:
            items.append(
                {
                    "category": "abandoned_sessions",
                    "cost_inr": ab_cost,
                    "calls": int(abandoned["calls"].iloc[0] or 0),
                    "owner": WASTE_OWNERS["abandoned_sessions"],
                }
            )
        items.sort(key=lambda x: -x["cost_inr"])
        waste_total = sum(i["cost_inr"] for i in items)
        days = int(query_df("SELECT MAX(day) AS d FROM llm_calls WHERE arm = :arm", {"arm": arm})["d"].iloc[0] or 0) + 1
        return {
            "arm": arm,
            "items": items,
            "waste_total_inr": waste_total,
            "monthly_waste_inr": waste_total / days * 30.0 if days else 0.0,
            "total_spend_inr": total,
            "waste_share_pct": waste_total / total * 100.0 if total else 0.0,
            "window_days": days,
            "status": "success",
            "warnings": [
                "categories can overlap (a looped call may also be a duplicate); "
                "the total is an upper bound on recoverable spend"
            ],
        }

    # --------------------------------------------------------------- extras --
    def optimiser_stats(self, arm: Optional[str] = None) -> Dict[str, Any]:
        arm = arm or self.arm
        df = query_df(
            "SELECT COUNT(*) AS calls, SUM(cache_hit) AS hits, SUM(escalated) AS esc, "
            "SUM(compressed) AS comp, SUM(is_overhead) AS overhead_calls, "
            "SUM(CASE WHEN is_overhead = 1 THEN cost_inr ELSE 0 END) AS overhead_inr, "
            "SUM(cost_inr) AS cost_inr FROM llm_calls WHERE arm = :arm",
            {"arm": arm},
        )
        med = query_df(
            "SELECT input_tokens FROM llm_calls WHERE arm = :arm AND cache_hit = 0",
            {"arm": arm},
        )
        r = df.iloc[0]
        calls = int(r["calls"] or 0)
        return {
            "arm": arm,
            "calls": calls,
            "cache_hit_rate": float(r["hits"] or 0) / calls if calls else 0.0,
            "escalation_rate": float(r["esc"] or 0) / calls if calls else 0.0,
            "compression_rate": float(r["comp"] or 0) / calls if calls else 0.0,
            "median_context_tokens": float(med["input_tokens"].median()) if not med.empty else 0.0,
            "overhead_inr": float(r["overhead_inr"] or 0.0),
            "overhead_pct": float(r["overhead_inr"] or 0.0) / float(r["cost_inr"] or 1.0) * 100.0,
            "total_cost_inr": float(r["cost_inr"] or 0.0),
        }

    def showback(self, arm: Optional[str] = None) -> pd.DataFrame:
        arm = arm or self.arm
        return query_df(
            "SELECT c.tenant, c.team, SUM(c.cost_inr) AS cost_inr, COUNT(*) AS calls, "
            "COUNT(DISTINCT c.outcome_id) AS outcomes FROM llm_calls c "
            "WHERE c.arm = :arm GROUP BY c.tenant, c.team ORDER BY cost_inr DESC",
            {"arm": arm},
        )
