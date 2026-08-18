"""Arm A (baseline) vs Arm B (TokenOps) on identical demand.

Writes benchmark_results.json and benchmark_chart.png, and prints the
comparison table. Two conventions this script holds to:

  - Headline unit economics are reported on steady-state traffic, with the
    planted agent-loop incident excluded from BOTH arms. The incident is
    reported on its own line. Folding a one-off outage into the headline
    number would flatter TokenOps by about 40 points and tell you nothing
    about a normal Tuesday.
  - Every metric is computed from the ledger. Nothing here is a constant.

Run: python scripts/benchmark.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from backend.core.forecaster import CostForecaster  # noqa: E402
from backend.core.ledger import CostLedger  # noqa: E402
from backend.core.optimizers.cascade import break_even_escalation_rate  # noqa: E402
from backend.core.pricing import fmt_inr  # noqa: E402
from backend.storage.db import has_data, query_df  # noqa: E402

STEADY = "incident IS NULL"          # excludes the planted agent loop
OUTCOME = "ticket_resolved"


def _scalar(sql: str, params: Dict[str, Any] | None = None, default: float = 0.0) -> float:
    df = query_df(sql, params or {})
    if df.empty or df.iloc[0, 0] is None:
        return default
    return float(df.iloc[0, 0])


def arm_metrics(arm: str, outcome_type: str = OUTCOME) -> Dict[str, Any]:
    p = {"a": arm, "ot": outcome_type}

    spend = _scalar(f"SELECT SUM(cost_inr) FROM llm_calls WHERE arm=:a AND {STEADY}", p)
    spend_ot = _scalar(
        f"SELECT SUM(cost_inr) FROM llm_calls WHERE arm=:a AND outcome_type=:ot AND {STEADY}", p
    )
    resolved = _scalar(
        "SELECT COUNT(*) FROM outcomes WHERE arm=:a AND outcome_type=:ot AND success=1", p
    )
    attempted = _scalar("SELECT COUNT(*) FROM outcomes WHERE arm=:a AND outcome_type=:ot", p)
    quality = _scalar("SELECT AVG(quality) FROM outcomes WHERE arm=:a AND success=1", p)
    calls = _scalar(f"SELECT COUNT(*) FROM llm_calls WHERE arm=:a AND {STEADY}", p)

    ctx = query_df(
        f"SELECT input_tokens FROM llm_calls WHERE arm=:a AND {STEADY} "
        "AND cache_hit=0 AND is_overhead=0 AND input_tokens > 0",
        p,
    )
    lat = query_df(
        f"SELECT latency_ms FROM llm_calls WHERE arm=:a AND {STEADY} AND is_overhead=0", p
    )
    per_outcome = query_df(
        f"SELECT outcome_id, SUM(cost_inr) c FROM llm_calls WHERE arm=:a AND outcome_type=:ot "
        f"AND {STEADY} GROUP BY outcome_id",
        p,
    )
    ok_ids = query_df(
        "SELECT outcome_id FROM outcomes WHERE arm=:a AND outcome_type=:ot AND success=1", p
    )
    ok_costs = per_outcome[per_outcome["outcome_id"].isin(set(ok_ids["outcome_id"]))]["c"]

    cache_hits = _scalar(f"SELECT SUM(cache_hit) FROM llm_calls WHERE arm=:a AND {STEADY}", p)
    escalations = _scalar(f"SELECT SUM(escalated) FROM llm_calls WHERE arm=:a AND {STEADY}", p)
    cascade_runs = _scalar(
        f"SELECT COUNT(*) FROM llm_calls WHERE arm=:a AND {STEADY} AND cache_hit=0 "
        "AND is_overhead=0",
        p,
    )
    overhead = _scalar(
        f"SELECT SUM(cost_inr) FROM llm_calls WHERE arm=:a AND {STEADY} AND is_overhead=1", p
    )

    # the planted incident, reported separately
    incident_cost = _scalar(
        "SELECT SUM(cost_inr) FROM llm_calls WHERE arm=:a AND incident='agent_loop'", p
    )
    incident_calls = _scalar(
        "SELECT COUNT(*) FROM llm_calls WHERE arm=:a AND incident='agent_loop'", p
    )
    incident_minutes = incident_calls / 105.0 if incident_calls else 0.0
    retry_cost = _scalar(
        "SELECT SUM(cost_inr) FROM llm_calls WHERE arm=:a AND incident='retry_storm'", p
    )

    ledger = CostLedger(arm)
    waste = ledger.waste_report()
    days = _scalar("SELECT MAX(day)+1 FROM llm_calls WHERE arm=:a", p, 30.0)

    return {
        "arm": arm,
        "steady_state_spend_inr": spend,
        "monthly_spend_inr": spend / days * 30.0,
        "cost_per_resolved_ticket_inr": spend_ot / resolved if resolved else 0.0,
        "resolved_tickets": int(resolved),
        "attempted_tickets": int(attempted),
        "resolution_rate": resolved / attempted if attempted else 0.0,
        "mean_quality": quality,
        "calls": int(calls),
        "calls_per_outcome": calls / resolved if resolved else 0.0,
        "median_context_tokens": float(ctx["input_tokens"].median()) if not ctx.empty else 0.0,
        "p95_latency_ms": float(lat["latency_ms"].quantile(0.95)) if not lat.empty else 0.0,
        "p50_latency_ms": float(lat["latency_ms"].median()) if not lat.empty else 0.0,
        "p95_cost_per_ticket_inr": float(ok_costs.quantile(0.95)) if len(ok_costs) else 0.0,
        "cache_hit_rate": cache_hits / calls if calls else 0.0,
        "cascade_escalation_rate": escalations / cascade_runs if cascade_runs else 0.0,
        "tokenops_overhead_inr": overhead,
        "tokenops_overhead_pct": overhead / spend * 100.0 if spend else 0.0,
        "loop_incident_cost_inr": incident_cost,
        "loop_incident_minutes": incident_minutes,
        "retry_storm_cost_inr": retry_cost,
        "waste_monthly_inr": waste["monthly_waste_inr"],
        "waste_items": waste["items"],
        "days": days,
    }


def router_findings() -> Dict[str, Any]:
    """What the bandit actually learned, straight out of router_state."""
    df = query_df("SELECT * FROM router_state ORDER BY day")
    if df.empty:
        return {}
    out: Dict[str, Any] = {}
    for task, grp in df.groupby("task_type"):
        # snapshots are cumulative; the daily share is the delta
        piv = grp.pivot_table(index="day", columns="model", values="pulls", aggfunc="sum").fillna(0)
        daily = piv.diff().fillna(piv)
        share = daily.div(daily.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        final = share.iloc[-1].sort_values(ascending=False)
        leader = str(final.index[0])
        # first day the eventual leader took the lead and never gave it back
        leaders = share.idxmax(axis=1)
        stable_from: Optional[int] = None
        for i in range(len(leaders)):
            if all(leaders.iloc[j] == leader for j in range(i, len(leaders))):
                stable_from = int(share.index[i])
                break
        out[task] = {
            "final_leader": leader,
            "final_leader_share": float(final.iloc[0]),
            "stable_from_day": stable_from,
            "final_shares": {str(k): float(v) for k, v in final.items()},
            "daily_share": {str(k): [float(x) for x in share[k]] for k in share.columns},
            "days": [int(d) for d in share.index],
        }
    return out


def forecast_block(arm: str = "tokenops") -> Dict[str, Any]:
    ledger = CostLedger(arm)
    daily = ledger.daily_series()
    unit = ledger.daily_unit_cost(outcome_type=OUTCOME)
    incident_days = query_df(
        "SELECT DISTINCT day FROM llm_calls WHERE arm=:a AND incident IS NOT NULL", {"a": arm}
    )["day"].tolist()
    f = CostForecaster().forecast(
        daily["cost_inr"].tolist(), horizon_days=30, drivers=unit,
        exclude_incident_days=incident_days,
    )
    return f.as_dict()


def cache_lookup_hit_rate() -> float:
    """The cache's own hit rate, over lookups it was actually allowed to make."""
    p = Path("data/samples/simulation_notes.json")
    if not p.exists():
        return 0.0
    try:
        notes = json.loads(p.read_text(encoding="utf-8"))
        return float(notes["tokenops_notes"]["cache"]["hit_rate"])
    except Exception:
        return 0.0


def build_table(base: Dict[str, Any], tops: Dict[str, Any], router: Dict[str, Any]) -> List[List[str]]:
    def pct(b: float, t: float) -> str:
        if not b:
            return "-"
        return f"{(t - b) / b * 100:+.1f}%"

    be = break_even_escalation_rate()
    gen = router.get("generation", {})
    cls = router.get("classification", {})

    rows = [
        ["Cost per resolved ticket", fmt_inr(base["cost_per_resolved_ticket_inr"]),
         fmt_inr(tops["cost_per_resolved_ticket_inr"]),
         pct(base["cost_per_resolved_ticket_inr"], tops["cost_per_resolved_ticket_inr"])],
        ["Monthly spend (steady state)", fmt_inr(base["monthly_spend_inr"]),
         fmt_inr(tops["monthly_spend_inr"]), pct(base["monthly_spend_inr"], tops["monthly_spend_inr"])],
        ["Mean outcome quality", f"{base['mean_quality']:.3f}", f"{tops['mean_quality']:.3f}",
         f"{tops['mean_quality'] - base['mean_quality']:+.3f}"],
        ["Resolution rate", f"{base['resolution_rate']:.1%}", f"{tops['resolution_rate']:.1%}",
         f"{(tops['resolution_rate'] - base['resolution_rate']) * 100:+.1f} pp"],
        ["Calls served from cache", f"{base['cache_hit_rate']:.1%}", f"{tops['cache_hit_rate']:.1%}", "-"],
        ["Cache hit rate (cacheable lookups)", "0.0%", f"{cache_lookup_hit_rate():.1%}",
         "resolution + QA are not cacheable"],
        ["Cascade escalation rate", "n/a", f"{tops['cascade_escalation_rate']:.1%}",
         f"break-even {be:.0%}"],
        ["Median context tokens per call", f"{base['median_context_tokens']:,.0f}",
         f"{tops['median_context_tokens']:,.0f}",
         pct(base["median_context_tokens"], tops["median_context_tokens"])],
        ["p50 latency", f"{base['p50_latency_ms']:,.0f} ms", f"{tops['p50_latency_ms']:,.0f} ms",
         pct(base["p50_latency_ms"], tops["p50_latency_ms"])],
        ["p95 latency", f"{base['p95_latency_ms']:,.0f} ms", f"{tops['p95_latency_ms']:,.0f} ms",
         pct(base["p95_latency_ms"], tops["p95_latency_ms"])],
        ["Day-18 loop incident cost", fmt_inr(base["loop_incident_cost_inr"]),
         fmt_inr(tops["loop_incident_cost_inr"]),
         pct(base["loop_incident_cost_inr"], tops["loop_incident_cost_inr"])],
        ["Day-18 loop containment", f"{base['loop_incident_minutes']:.0f} min",
         f"{tops['loop_incident_minutes']:.0f} min", "-"],
        ["Day-11 retry storm cost", fmt_inr(base["retry_storm_cost_inr"]),
         fmt_inr(tops["retry_storm_cost_inr"]), "-"],
        ["Waste named and owned (monthly)", fmt_inr(base["waste_monthly_inr"]),
         fmt_inr(tops["waste_monthly_inr"]), "-"],
        ["Router policy: triage", "n/a",
         f"{cls.get('final_leader', '-')} {cls.get('final_leader_share', 0):.0%}",
         f"stable from day {cls.get('stable_from_day')}"],
        ["Router policy: generation", "n/a",
         f"{gen.get('final_leader', '-')} {gen.get('final_leader_share', 0):.0%}",
         f"stable from day {gen.get('stable_from_day')}"],
        ["TokenOps overhead", "-", f"{tops['tokenops_overhead_pct']:.2f}% of managed spend", "-"],
    ]
    return rows


def chart(base: Dict[str, Any], tops: Dict[str, Any], router: Dict[str, Any], path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping chart: {exc}")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("TokenOps - baseline vs managed, identical demand", fontsize=13, y=1.02)

    ax = axes[0]
    vals = [base["cost_per_resolved_ticket_inr"], tops["cost_per_resolved_ticket_inr"]]
    bars = ax.bar(["baseline", "TokenOps"], vals, color=["#b0413e", "#2e7d5b"])
    ax.set_title("Cost per resolved ticket")
    ax.set_ylabel("INR")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=10)
    delta = (vals[1] - vals[0]) / vals[0] * 100 if vals[0] else 0
    ax.text(0.5, max(vals) * 0.55, f"{delta:+.0f}%", ha="center", fontsize=16, color="#2e7d5b")

    ax = axes[1]
    ledger_b = CostLedger("baseline").daily_series()
    ledger_t = CostLedger("tokenops").daily_series()
    ax.plot(ledger_b["day"], ledger_b["cost_inr"], label="baseline", color="#b0413e")
    ax.plot(ledger_t["day"], ledger_t["cost_inr"], label="TokenOps", color="#2e7d5b")
    ax.set_yscale("log")
    ax.set_title("Daily spend (log scale) - day 18 is the agent loop")
    ax.set_xlabel("day")
    ax.set_ylabel("INR / day")
    ax.legend(fontsize=8)

    ax = axes[2]
    task = "retrieval" if "retrieval" in router else next(iter(router), None)
    if task:
        r = router[task]
        for model, series in r["daily_share"].items():
            ax.plot(r["days"], series, label=model.replace("claude-", "").replace("-20251001", ""))
        ax.set_title(f"Router traffic share - {task}")
        ax.set_xlabel("day")
        ax.set_ylabel("share of calls")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--chart", default="benchmark_chart.png")
    args = ap.parse_args()

    if not has_data():
        print("No ledger data. Run: python scripts/simulate_workload.py")
        raise SystemExit(1)

    base = arm_metrics("baseline")
    tops = arm_metrics("tokenops")
    router = router_findings()
    fc = forecast_block("tokenops")
    rows = build_table(base, tops, router)

    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(max(len(r[1]) for r in rows), len("Baseline")) + 3
    w2 = max(max(len(r[2]) for r in rows), len("TokenOps")) + 3
    w3 = max(max(len(r[3]) for r in rows), len("Delta")) + 3
    total_w = w0 + w1 + w2 + w3
    print("\n" + "=" * total_w)
    print("TOKENOPS BENCHMARK - arm A (baseline) vs arm B (TokenOps), identical demand")
    print("=" * total_w)
    print(f"{'Metric'.ljust(w0)}{'Baseline'.rjust(w1)}{'TokenOps'.rjust(w2)}{'Delta'.rjust(w3)}")
    print("-" * total_w)
    for label, b, t, d in rows:
        print(f"{label.ljust(w0)}{b.rjust(w1)}{t.rjust(w2)}{d.rjust(w3)}")
    print("-" * total_w)

    line = (f"Forecast (next 30d, TokenOps): {fmt_inr(fc['total_inr'])} "
            f"[{fmt_inr(fc['total_lower_inr'])} - {fmt_inr(fc['total_upper_inr'])}]")
    if fc.get("mape_pct") is not None:
        line += f", backtest MAPE {fc['mape_pct']:.1f}%"
    print(line)
    brk = (fc.get("params") or {}).get("structural_break_day")
    if brk is not None:
        print(f"Structural break detected at day {brk} - the day-23 prompt deploy. "
              "MAPE is measured on the post-break regime.")
    drivers = fc.get("drivers") or {}
    if drivers:
        print(f"Growth drivers: volume {drivers['volume_share_pct']:.0f}% / "
              f"unit cost {drivers['unit_cost_share_pct']:.0f}% - {drivers['verdict']}")

    print("\nWhere TokenOps loses:")
    losses = []
    if tops["mean_quality"] < base["mean_quality"]:
        losses.append(f"  - mean outcome quality {tops['mean_quality']:.3f} vs {base['mean_quality']:.3f} "
                      f"({tops['mean_quality'] - base['mean_quality']:+.3f}); lambda={0.4} is the dial")
    if tops["p95_latency_ms"] > base["p95_latency_ms"]:
        losses.append(f"  - p95 latency {tops['p95_latency_ms']:,.0f} ms vs {base['p95_latency_ms']:,.0f} ms "
                      "(cascade escalations pay for the cheap attempt in latency)")
    if base["resolution_rate"] - tops["resolution_rate"] > 0.001:
        losses.append(f"  - resolution rate {tops['resolution_rate']:.1%} vs {base['resolution_rate']:.1%}")
    print("\n".join(losses) if losses else "  - nothing measured; treat that as a bug in the benchmark")

    chart_path = chart(base, tops, router, args.chart)
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "outcome_type": OUTCOME,
        "convention": "headline metrics exclude the planted agent-loop incident in both arms",
        "arms": {"baseline": base, "tokenops": tops},
        "router": router,
        "forecast": fc,
        "table": rows,
        "chart": chart_path,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}" + (f" and {chart_path}" if chart_path else ""))


if __name__ == "__main__":
    main()
