"""The CFO view: cost per outcome, its trend, and its decomposition."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    GREEN, RED, daily_unit, evidence, fmt_inr, outcome_types, page_setup,
    require_data, sidebar, sql, unit_economics,
)

page_setup("Economics", icon="📉")
arm = sidebar("econ")
st.title("Unit economics")

if not require_data():
    st.stop()

ot = st.selectbox("Outcome type", outcome_types(), index=0)

ue = unit_economics(arm, ot)
other = unit_economics("baseline" if arm == "tokenops" else "tokenops", ot)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Cost per {ot.replace('_', ' ')}", fmt_inr(ue["cost_per_outcome_inr"]),
          f"{(ue['cost_per_outcome_inr'] - other['cost_per_outcome_inr']) / other['cost_per_outcome_inr'] * 100:+.0f}% vs other arm"
          if other["cost_per_outcome_inr"] else None, delta_color="inverse")
c2.metric("Successful outcomes", f"{ue['successful_outcomes']:,}",
          f"{ue['successful_outcomes'] / ue['attempted_outcomes']:.1%} of attempts"
          if ue["attempted_outcomes"] else None)
c3.metric("Calls per outcome", f"{ue['calls_per_outcome']:.2f}")
c4.metric("Mean quality", f"{ue['mean_quality']:.3f}")

st.caption(
    "Failed outcomes are counted in the numerator and excluded from the denominator. "
    "Wasted spend has to show up in the unit cost, or you optimise the wrong thing."
)
for w in ue.get("warnings", []):
    st.warning(w)

st.divider()

# ---------------------------------------------------------------- trend + split
tab1, tab2, tab3 = st.tabs(["Trend", "Volume vs unit cost", "Showback"])

with tab1:
    du_a = daily_unit(arm, ot)
    du_b = daily_unit("baseline" if arm == "tokenops" else "tokenops", ot)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=du_b["day"], y=du_b["cost_per_outcome_inr"],
                             name="other arm", line=dict(color=RED, width=2)))
    fig.add_trace(go.Scatter(x=du_a["day"], y=du_a["cost_per_outcome_inr"],
                             name=arm, line=dict(color=GREEN, width=3)))
    fig.add_vline(x=18, line_dash="dot", line_color="#888",
                  annotation_text="day 18: agent loop")
    fig.add_vline(x=23, line_dash="dot", line_color="#888",
                  annotation_text="day 23: prompt deploy")
    fig.update_layout(height=380, yaxis_title="INR per outcome", xaxis_title="day",
                      margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Two visible events. Day 18 is a runaway agent. Day 23 is a prompt change that "
        "tripled context size - nobody filed a ticket for either."
    )

with tab2:
    du = daily_unit(arm, ot)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=du["day"], y=du["outcomes"], name="outcomes (volume)",
                         marker_color="#9dbfd4", yaxis="y"))
    fig.add_trace(go.Scatter(x=du["day"], y=du["cost_per_outcome_inr"],
                             name="cost per outcome", line=dict(color=GREEN, width=3), yaxis="y2"))
    fig.update_layout(
        height=380, xaxis_title="day",
        yaxis=dict(title="outcomes"), yaxis2=dict(title="INR / outcome", overlaying="y", side="right"),
        legend=dict(orientation="h"), margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Volume growth and unit-cost growth need opposite responses. Volume growth is a "
        "good problem. Unit-cost growth is a regression, and it has an owner."
    )

with tab3:
    rows = unit_economics(arm, ot, group_by="tenant")["rows"]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Per-tenant showback. Cost, share of spend, unit cost, and the quality it bought.")

st.divider()
evidence(
    "the ledger rows behind this number",
    sql(
        "SELECT day, tenant, agent, step, model, input_tokens, output_tokens, "
        "ROUND(cost_inr, 4) AS cost_inr, quality, cache_hit, escalated "
        "FROM llm_calls WHERE arm = :a AND outcome_type = :o ORDER BY ts_epoch LIMIT 200",
        {"a": arm, "o": ot},
    ),
    "Every figure on this page is an aggregation over rows like these. "
    "Cost is computed at record time from a versioned price table.",
)
