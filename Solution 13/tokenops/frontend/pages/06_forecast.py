"""Forecast, driver decomposition, and what-if."""
from __future__ import annotations

import sys
from pathlib import Path


import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    GREEN, evidence, fmt_inr, optimiser_stats, page_setup, require_data, sidebar, sql,
)
from backend.core.forecaster import CostForecaster  # noqa: E402
from backend.core.ledger import CostLedger  # noqa: E402

page_setup("Forecast", icon="🔮")
arm = sidebar("forecast")
st.title("Forecast and scenarios")

if not require_data():
    st.stop()

ledger = CostLedger(arm)
daily = ledger.daily_series()
unit = ledger.daily_unit_cost()
incident_days = sql(
    "SELECT DISTINCT day FROM llm_calls WHERE arm = :a AND incident IS NOT NULL", {"a": arm}
)["day"].tolist()

horizon = st.slider("Horizon (days)", 7, 60, 30, 7)
f = CostForecaster().forecast(daily["cost_inr"].tolist(), horizon_days=horizon,
                              drivers=unit, exclude_incident_days=incident_days)

c1, c2, c3 = st.columns(3)
c1.metric(f"Next {horizon} days", fmt_inr(f.total_inr),
          f"{fmt_inr(f.total_lower_inr)} – {fmt_inr(f.total_upper_inr)}")
c2.metric("Backtest MAPE", f"{f.mape_pct:.1f}%" if f.mape_pct is not None else "n/a",
          help="Fit on all but the last 7 days, predict those 7, measure the error.")
brk = f.params.get("structural_break_day")
c3.metric("Structural break", f"day {brk}" if brk is not None else "none detected",
          help="A deploy or model swap. No forecaster sees one coming; the honest response "
               "is to find it and measure accuracy on the current regime.")

hist_x = list(range(len(daily)))
fut_x = list(range(len(daily), len(daily) + horizon))
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_x, y=daily["cost_inr"], name="actual", line=dict(color="#666")))
fig.add_trace(go.Scatter(x=fut_x, y=f.upper, name="upper", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=fut_x, y=f.lower, name="95% interval", fill="tonexty",
                         fillcolor="rgba(46,125,91,0.18)", line=dict(width=0)))
fig.add_trace(go.Scatter(x=fut_x, y=f.point, name="forecast", line=dict(color=GREEN, width=3)))
fig.update_layout(height=380, xaxis_title="day", yaxis_title="INR / day",
                  legend=dict(orientation="h"), margin=dict(t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

for w in f.warnings:
    st.info(w)

st.divider()

# ------------------------------------------------------------------ drivers
st.markdown("#### Growth drivers")
d = f.drivers
if d:
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume", f"{d['volume_delta_pct']:+.1f}%",
              f"{d['volume_share_pct']:.0f}% of growth")
    c2.metric("Unit cost", f"{d['unit_cost_delta_pct']:+.1f}%",
              f"{d['unit_cost_share_pct']:.0f}% of growth")
    c3.metric("Period-over-period growth", fmt_inr(d["growth_inr"]))
    (st.success if d["unit_cost_share_pct"] < 25 else st.warning)(d["verdict"])
    st.caption(
        "ΔC = u₀·ΔV + V₀·Δu + ΔV·Δu — volume contribution, unit-cost contribution, and the "
        "interaction term. Volume growth is a good problem; unit-cost growth is a regression."
    )
else:
    st.caption("Not enough history for a driver decomposition.")

st.divider()

# ------------------------------------------------------------------ what-if
st.markdown("#### What-if")
stats = optimiser_stats(arm)
cheap = sql(
    "SELECT model, COUNT(*) n FROM llm_calls WHERE arm = :a AND is_overhead = 0 GROUP BY model",
    {"a": arm},
)
cheap_share = (float(cheap[cheap["model"].str.contains("haiku")]["n"].sum()) /
               float(cheap["n"].sum())) if not cheap.empty else 0.0

monthly = float(daily["cost_inr"].sum()) / max(len(daily), 1) * 30.0
c1, c2, c3, c4 = st.columns(4)
hit = c1.slider("Cache hit rate", 0.0, 0.8, float(round(stats["cache_hit_rate"], 2)), 0.01)
cs = c2.slider("Cheap-model share", 0.0, 1.0, float(round(cheap_share, 2)), 0.01)
ctx = c3.slider("Further context reduction", 0.0, 0.6, 0.0, 0.05)
vol = c4.slider("Volume growth", -0.3, 1.0, 0.0, 0.05)

res = CostForecaster.what_if(
    monthly,
    {"cache_hit_rate": hit, "cheap_model_share": cs, "context_reduction": ctx, "volume_growth": vol},
    {"cache_hit_rate": stats["cache_hit_rate"], "cheap_model_share": cheap_share,
     "context_reduction": 0.0},
)
c1, c2 = st.columns(2)
c1.metric("Current monthly run-rate", fmt_inr(res["baseline_monthly_inr"]))
c2.metric("Scenario", fmt_inr(res["scenario_monthly_inr"]), f"{res['delta_pct']:+.1f}%",
          delta_color="inverse")
if res["steps"]:
    st.dataframe(res["steps"], use_container_width=True, hide_index=True)
for w in res.get("warnings", []):
    st.caption(w)

evidence("daily spend series", daily, "The series the forecaster is fitted on.")
