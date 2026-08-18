"""Attribution: find the money in two clicks."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    evidence, flat_attribution, fmt_inr, page_setup, require_data, sidebar, sql,
)

page_setup("Attribution", icon="🧭")
arm = sidebar("attr")
st.title("Attribution")
st.caption("tenant → team → agent → workflow → step. Click a wedge to drill in.")

if not require_data():
    st.stop()

df = flat_attribution(arm)
total = float(df["cost_inr"].sum())

levels = st.multiselect(
    "Drill-down path", ["tenant", "team", "agent", "workflow", "step"],
    default=["tenant", "agent", "step"],
)
if not levels:
    levels = ["tenant", "agent", "step"]

fig = px.sunburst(df, path=levels, values="cost_inr", color="cost_inr",
                  color_continuous_scale="Teal", height=520)
fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- the finding the demo is built around --------------------------------
step_cost = df.groupby("step")["cost_inr"].sum().sort_values(ascending=False)
qa_share = float(step_cost.get("qa_verify", 0.0)) / total * 100.0 if total else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total attributed spend", fmt_inr(total))
c2.metric("Top step", str(step_cost.index[0]), f"{step_cost.iloc[0] / total:.1%} of spend")
c3.metric("QA verification step", f"{qa_share:.1f}% of spend",
          help="Re-verifies answers the resolution agent already scored above the floor.")

st.markdown("#### Cost by step")
step_df = step_cost.reset_index()
step_df.columns = ["step", "cost_inr"]
step_df["share_pct"] = step_df["cost_inr"] / total * 100.0
st.bar_chart(step_df.set_index("step")["cost_inr"], height=260)

redundant = sql(
    """SELECT COUNT(*) AS redundant_qa_calls, SUM(c.cost_inr) AS cost_inr
       FROM llm_calls c
       JOIN outcomes o ON o.outcome_id = c.outcome_id AND o.arm = c.arm
       WHERE c.arm = :a AND c.step = 'qa_verify' AND o.quality >= 0.90""",
    {"a": arm},
)
if not redundant.empty and redundant["cost_inr"].iloc[0]:
    n = int(redundant["redundant_qa_calls"].iloc[0])
    c = float(redundant["cost_inr"].iloc[0])
    st.warning(
        f"**{n:,} QA verification calls** ({fmt_inr(c)}, {c / total:.1%} of spend) ran on "
        "sessions the resolution agent had already scored at or above 0.90. Nobody designed "
        "that. It accreted."
    )

st.divider()

st.markdown("#### Week-over-week movers")
wow = sql(
    """SELECT agent, step,
              SUM(CASE WHEN day > (SELECT MAX(day) FROM llm_calls) - 7 THEN cost_inr ELSE 0 END) AS this_week,
              SUM(CASE WHEN day <= (SELECT MAX(day) FROM llm_calls) - 7
                        AND day > (SELECT MAX(day) FROM llm_calls) - 14 THEN cost_inr ELSE 0 END) AS prev_week
       FROM llm_calls WHERE arm = :a GROUP BY agent, step""",
    {"a": arm},
)
if not wow.empty:
    wow["delta_pct"] = (wow["this_week"] - wow["prev_week"]) / wow["prev_week"].replace(0, pd.NA) * 100
    wow = wow.sort_values("delta_pct", ascending=False)
    st.dataframe(wow, use_container_width=True, hide_index=True)
    st.caption("A step whose spend jumped week-over-week without a matching volume jump is a deploy, not demand.")

evidence("attribution table", df.sort_values("cost_inr", ascending=False),
         "Grouped straight off the ledger. No sampling, no estimation.")
