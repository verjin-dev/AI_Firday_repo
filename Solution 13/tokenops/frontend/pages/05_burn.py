"""Burn-rate alerting and the day-18 incident. This is the insurance page."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    AMBER, GREEN, RED, evidence, fmt_inr, notes, page_setup, require_data, sidebar, sql,
)
from backend.core.burn_rate import Budget, BurnRateMonitor  # noqa: E402
from backend.core.guardrails import CostGuardrails, degraded_request  # noqa: E402
from backend.core.ledger import CostLedger  # noqa: E402

page_setup("Burn & incidents", icon="🚨")
sidebar("burn")
st.title("Burn rate and incidents")

if not require_data():
    st.stop()

n = notes()
budget_inr = float(n.get("monthly_budget_inr", 0) or 0)
days = int(n.get("days", 30) or 30)

# ------------------------------------------------------------ the incident
st.markdown("#### Day 18, 02:14 — a resolution agent starts looping")

frames = {}
for arm in ("baseline", "tokenops"):
    df = sql(
        "SELECT ts_epoch, cost_inr, incident FROM llm_calls WHERE arm = :a AND day = 18",
        {"a": arm},
    )
    if df.empty:
        continue
    start = df["ts_epoch"].min()
    df["minute"] = ((df["ts_epoch"] - start) // 60).astype(int)
    frames[arm] = df.groupby("minute")["cost_inr"].sum().cumsum()

fig = go.Figure()
if "baseline" in frames:
    fig.add_trace(go.Scatter(x=frames["baseline"].index, y=frames["baseline"].values,
                             name="baseline (unmanaged)", line=dict(color=RED, width=3)))
if "tokenops" in frames:
    fig.add_trace(go.Scatter(x=frames["tokenops"].index, y=frames["tokenops"].values,
                             name="TokenOps", line=dict(color=GREEN, width=3)))
fig.add_vline(x=134, line_dash="dot", line_color=AMBER, annotation_text="02:14 loop starts")
fig.add_vline(x=136, line_dash="dot", line_color=GREEN, annotation_text="02:16 killed")
fig.add_vline(x=480, line_dash="dot", line_color=RED, annotation_text="08:00 human arrives")
fig.update_layout(height=400, xaxis_title="minute of day 18",
                  yaxis_title="cumulative spend (INR)", legend=dict(orientation="h"),
                  margin=dict(t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

loop = sql(
    "SELECT arm, COUNT(*) AS calls, SUM(cost_inr) AS cost_inr "
    "FROM llm_calls WHERE incident = 'agent_loop' GROUP BY arm"
)
if not loop.empty:
    lb = loop[loop["arm"] == "baseline"]
    lt = loop[loop["arm"] == "tokenops"]
    c1, c2, c3, c4 = st.columns(4)
    if not lb.empty:
        c1.metric("Unmanaged incident cost", fmt_inr(float(lb["cost_inr"].iloc[0])),
                  f"{int(lb['calls'].iloc[0]):,} calls")
    if not lt.empty:
        c2.metric("TokenOps incident cost", fmt_inr(float(lt["cost_inr"].iloc[0])),
                  f"{int(lt['calls'].iloc[0]):,} calls", delta_color="off")
    tn = n.get("tokenops_notes", {})
    c3.metric("Containment", f"{tn.get('loop_kill_min', '?')} min",
              "loop detector: identical (step, prompt_hash) x4")
    det = n.get("incident_detection", {})
    if det.get("burn_alert_after_min"):
        c4.metric("Burn alert would have fired", f"{det['burn_alert_after_min']} min",
                  "even with no loop signature")

st.markdown(
    "Two independent mechanisms caught this. The **loop detector** matched a repeated "
    "`(step, prompt_hash)` signature inside one session and killed it — that is fast because "
    "it needs no budget knowledge. The **burn-rate monitor** would have caught it anyway from "
    "spend alone, which is what matters for the next incident, because the next incident will "
    "not be a loop."
)

st.divider()

# ------------------------------------------------------------- burn windows
st.markdown("#### Multi-window burn-rate alerts")
arm_sel = st.radio("Arm", ["baseline", "tokenops"], horizontal=True)
ledger = CostLedger(arm_sel)
hourly = ledger.hourly_series()
if budget_inr and not hourly.empty:
    monitor = BurnRateMonitor(Budget("global", "all-tenants", budget_inr, days))
    state = monitor.budget_state(hourly)
    alerts = monitor.scan(hourly, dedup_hours=6)

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly budget", fmt_inr(budget_inr))
    c2.metric("Spent", fmt_inr(state["spent_inr"]), f"{100 - state['remaining_pct']:.0f}% consumed")
    tte = state["time_to_exhaustion_hours"]
    c3.metric("Time to exhaustion", f"{tte / 24:.1f} days" if tte else "n/a")

    if alerts:
        adf = pd.DataFrame([a.as_dict() for a in alerts])
        st.dataframe(
            adf[["day", "hour", "window_hours", "threshold_multiplier", "observed_multiplier",
                 "severity", "message"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.success(f"No burn-rate alerts fired for **{arm_sel}** over {days} days.")
    st.caption(
        "Windows: " + ", ".join(f"{h}h at {m}×" for h, m in monitor.windows)
        + ". Each long window is paired with a short window one twelfth its length; both must "
          "breach. That is what makes the alert fast without making it noisy."
    )

st.divider()

# ------------------------------------------------------------- degradation
st.markdown("#### Graceful degradation")
st.caption("Blocking a workflow to save money is a support ticket. Degrading it is a saving.")
remaining_pct = st.slider("Remaining budget for this tenant (%)", 0, 100, 20, 5)
guard = CostGuardrails()
request = {"interactive": True, "input_tokens": 8400, "context_depth": "deep",
           "skip_verification": False}
decision = guard.check(request, {"scope": "tenant:vertex-insurance", "remaining_pct": remaining_pct})
after = degraded_request(request, decision)

c1, c2 = st.columns(2)
c1.markdown("**Request as submitted**")
c1.json(request)
c2.markdown(f"**After the guardrail — `{decision.action.value}`**")
c2.json(after)
st.info(decision.reason)

evidence(
    "alert log",
    sql("SELECT arm, ts, kind, severity, scope, window_hours, observed_multiplier, message "
        "FROM alerts ORDER BY ts_epoch"),
    "Written by the simulator's live guardrails and by the post-hoc burn scan.",
)
