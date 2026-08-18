"""TokenOps - Streamlit entry point.

Run: streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.common import (  # noqa: E402
    benchmark, fmt_inr, notes, page_setup, require_data, sidebar, sql,
)

page_setup("Overview", icon="📉")
arm = sidebar("home")

st.title("TokenOps")
st.subheader("Cost per business outcome, not cost per token")

st.markdown(
    "Cost per token is a vanity metric. **\"₹4.20 per resolved support ticket, down from "
    "₹11.80, at the same CSAT\"** is a sentence a CFO can act on. Everything in this "
    "project exists to produce that sentence - and to make sure a runaway agent never "
    "gets to write a different one."
)

if not require_data():
    st.stop()

bench = benchmark()
n = notes()

if bench:
    b = bench["arms"]["baseline"]
    t = bench["arms"]["tokenops"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Cost per resolved ticket",
        fmt_inr(t["cost_per_resolved_ticket_inr"]),
        f"{(t['cost_per_resolved_ticket_inr'] - b['cost_per_resolved_ticket_inr']) / b['cost_per_resolved_ticket_inr'] * 100:+.0f}% vs baseline",
        delta_color="inverse",
    )
    c2.metric(
        "Outcome quality",
        f"{t['mean_quality']:.3f}",
        f"{t['mean_quality'] - b['mean_quality']:+.3f} vs baseline",
    )
    c3.metric(
        "Day-18 loop incident",
        fmt_inr(t["loop_incident_cost_inr"]),
        f"vs {fmt_inr(b['loop_incident_cost_inr'])} unmanaged",
        delta_color="inverse",
    )
    c4.metric(
        "Containment time",
        f"{t['loop_incident_minutes']:.0f} min",
        f"vs {b['loop_incident_minutes']:.0f} min",
        delta_color="inverse",
    )

st.divider()

left, right = st.columns([3, 2])
with left:
    st.markdown("#### What this is")
    st.markdown(
        """
- **A cost ledger** where every LLM call carries the business outcome it belongs to,
  so spend can be divided by *resolved tickets* rather than by *calls*.
- **A learning router** - a constrained Thompson-sampling bandit that discovers, per
  task type, the cheapest route that still clears a quality floor.
- **Burn-rate alerting** borrowed from SRE error budgets, so a runaway agent is caught
  in minutes instead of at month-end.
- **Guardrails** that degrade a workflow rather than stopping it when budget runs low.
- **A forecaster** that separates volume growth from unit-cost growth, because those
  two need opposite responses.
        """
    )
with right:
    st.markdown("#### Where to click")
    st.markdown(
        """
1. **Economics** - the CFO view and the headline number
2. **Attribution** - find the money in two clicks
3. **Waste** - ₹ figures with an owner each
4. **Router** - watch the bandit learn, and read why it chose
5. **Burn & incidents** - the day-18 agent loop
6. **Forecast** - next month, decomposed, with what-if sliders
        """
    )
    if n.get("monthly_budget_inr"):
        st.info(
            f"Monthly budget: **{fmt_inr(n['monthly_budget_inr'])}** - derived from baseline "
            "steady-state spend plus 10% headroom, the way a finance team would set it."
        )

st.divider()
st.markdown("#### Simulated estate")
c = sql(
    "SELECT arm, COUNT(*) AS calls, COUNT(DISTINCT session_id) AS sessions, "
    "COUNT(DISTINCT tenant) AS tenants, MAX(day)+1 AS days, SUM(cost_inr) AS spend_inr "
    "FROM llm_calls GROUP BY arm"
)
st.dataframe(c, use_container_width=True, hide_index=True)
st.caption(
    "Both arms run against the same generated demand - the same sessions, the same "
    "questions, the same arrival times. Only the control plane differs."
)
