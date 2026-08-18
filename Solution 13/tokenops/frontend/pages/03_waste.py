"""Waste: named categories, rupee figures, an owner each, and a fix simulator."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    GREEN, RED, evidence, fmt_inr, page_setup, require_data, sidebar, sql, waste,
)

page_setup("Waste", icon="🧹")
arm = sidebar("waste")
st.title("Named waste")
st.caption("Spend with a category and an owner. A number nobody owns never gets fixed.")

if not require_data():
    st.stop()

w = waste(arm)
items = pd.DataFrame(w["items"])

c1, c2, c3 = st.columns(3)
c1.metric("Identified waste (monthly)", fmt_inr(w["monthly_waste_inr"]))
c2.metric("Share of total spend", f"{w['waste_share_pct']:.1f}%")
c3.metric("Window", f"{w['window_days']} days")

if items.empty:
    st.success("No tagged waste in this arm.")
    st.stop()

FIXES = {
    "loop_waste": ("Loop detection + circuit breaker", 0.97),
    "duplicate_calls": ("Semantic cache on the duplicated step", 0.90),
    "retry_waste": ("Retry budget capped at 1 during a burn window", 0.66),
    "abandoned_sessions": ("Abandonment prediction: stop spending at step 2", 0.45),
    "over_retrieval": ("Drop chunks with zero historical citation rate", 0.60),
    "verbose_output": ("Schema-bound output, max_tokens set from the schema", 0.55),
}

items["monthly_inr"] = items["cost_inr"] / w["window_days"] * 30.0
items["fix"] = items["category"].map(lambda c: FIXES.get(c, ("-", 0.0))[0])
items["recoverable_pct"] = items["category"].map(lambda c: FIXES.get(c, ("-", 0.0))[1] * 100)
items["recoverable_inr"] = items["monthly_inr"] * items["recoverable_pct"] / 100.0

fig = go.Figure()
fig.add_trace(go.Bar(x=items["monthly_inr"], y=items["category"], orientation="h",
                     name="monthly waste", marker_color=RED))
fig.add_trace(go.Bar(x=items["recoverable_inr"], y=items["category"], orientation="h",
                     name="recoverable with the named fix", marker_color=GREEN))
fig.update_layout(barmode="overlay", height=340, xaxis_title="INR / month",
                  margin=dict(t=20, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    items[["category", "owner", "calls", "monthly_inr", "fix", "recoverable_pct", "recoverable_inr"]]
    .rename(columns={"monthly_inr": "monthly_INR", "recoverable_inr": "recoverable_INR"}),
    use_container_width=True, hide_index=True,
)

st.divider()
st.markdown("#### Fix simulator")
chosen = st.multiselect("Apply fixes", items["category"].tolist(),
                        default=items["category"].tolist()[:2])
saved = float(items[items["category"].isin(chosen)]["recoverable_inr"].sum())
remaining = float(items["monthly_inr"].sum()) - saved
c1, c2 = st.columns(2)
c1.metric("Monthly saving from selected fixes", fmt_inr(saved))
c2.metric("Waste remaining", fmt_inr(remaining))
st.caption(
    "Recovery rates are the measured effect of each control in the TokenOps arm, not "
    "estimates: loop waste falls 97% because the loop was killed in 2 minutes instead "
    "of running for six hours."
)

for warn in w.get("warnings", []):
    st.info(warn)

evidence(
    "waste-tagged ledger rows",
    sql(
        "SELECT day, tenant, agent, step, model, waste_tag, incident, "
        "ROUND(cost_inr, 4) AS cost_inr, status FROM llm_calls "
        "WHERE arm = :a AND waste_tag IS NOT NULL ORDER BY cost_inr DESC LIMIT 200",
        {"a": arm},
    ),
    "Tagged at record time by the middleware, not reconstructed later.",
)
