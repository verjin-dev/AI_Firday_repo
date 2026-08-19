"""Live ops - traffic arriving now, and an incident you inject by hand.

This is the page to put on the projector. Everything else in TokenOps is a
30-day replay; here the router is choosing routes in real time, the savings
counter is ticking, and the red buttons cause a real incident that the real
guardrails have to contain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import GREEN, RED, page_setup, short_model  # noqa: E402
from backend.core.live import INCIDENTS, get_engine  # noqa: E402
from backend.core.pricing import fmt_inr  # noqa: E402

page_setup("Live ops", icon="📉")

SEV_COLOR = {"critical": RED, "warning": "#c98a1b", "success": GREEN, "info": "#7a7a7a"}
SEV_ICON = {"critical": "🔴", "warning": "🟠", "success": "🟢", "info": "⚪"}


@st.cache_resource
def engine():
    return get_engine()


eng = engine()

st.title("Live ops")
st.caption(
    "Real traffic on a compressed clock. The router, cache, compressor, burn monitor "
    "and guardrails driving this page are the same objects the benchmark used - "
    "nothing here is a re-enactment."
)

# ----------------------------------------------------------------- controls
with st.sidebar:
    st.markdown("### Live controls")
    speed = st.select_slider(
        "Clock speed", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0,
        format_func=lambda v: f"{v:g} sim-min / sec  ({v * 60:g}×)",
    )
    eng.set_speed(speed)

    budget = st.number_input(
        "Monthly budget (INR)", min_value=10_000.0, max_value=50_000_000.0,
        value=float(eng.monthly_budget_inr), step=50_000.0,
        help="Drag this down while traffic runs to watch the guardrail move from "
             "ALLOW to DEGRADE to BLOCK.",
    )
    if abs(budget - eng.monthly_budget_inr) > 1:
        eng.set_budget(budget)

    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True, type="primary"):
        eng.start()
        st.rerun()
    if c2.button("⏸ Stop", use_container_width=True):
        eng.stop()
        st.rerun()
    if st.button("↺ Reset run", use_container_width=True):
        eng.stop()
        eng.reset()
        st.rerun()
    if st.button("Clear circuit breakers", use_container_width=True):
        eng.clear_breakers()

    st.divider()
    st.markdown("### Inject an incident")
    for kind, description in INCIDENTS.items():
        if st.button(f"💥 {kind.replace('_', ' ')}", use_container_width=True,
                     help=description, key=f"inject_{kind}"):
            eng.inject(kind)
            st.rerun()


@st.fragment(run_every="1s")
def live_view() -> None:
    s = eng.snapshot(minutes=180)
    t = s["totals"]

    status = "🟢 running" if s["running"] else "⏸ stopped"
    st.markdown(
        f"**{status}** · simulated clock **{s['sim_clock']}** "
        f"(minute {s['sim_minute']}) · {s['minutes_per_second']:g} sim-min per real second"
    )
    if not s["running"] and s["sim_minute"] == 0:
        st.info("Press **▶ Start** in the sidebar. Give it ~30 seconds of traffic before "
                "injecting an incident, so the router has something to learn from.")
    elif s["saved_inr"] < 0 and t["calls"] < 60:
        st.caption(
            "The savings counter is negative because exploration is charged before it "
            "pays: the bandit is buying information, and each cascade escalation pays "
            "for the cheap attempt too. It crosses over within the first minute or two."
        )

    # ------------------------------------------------------------ KPI row --
    c1, c2, c3, c4, c5 = st.columns(5)
    warming = t["calls"] < 60
    c1.metric(
        "Saved so far" if not warming else "Saved so far (warming up)",
        fmt_inr(s["saved_inr"]), f"{s['saved_pct']:.0f}% vs unmanaged",
        help="Exploration costs money before it saves any: for the first minute the "
             "bandit is paying to find out which routes work, and a cascade "
             "escalation charges for the cheap attempt as well as the good one. "
             "This counter is expected to start negative and cross over quickly.",
    )
    c2.metric("Cost per outcome", fmt_inr(s["cost_per_outcome_inr"]),
              f"unmanaged {fmt_inr(s['shadow_cost_per_outcome_inr'])}", delta_color="off")
    c3.metric("Outcomes", f"{t['outcomes_ok']:,}", f"{t['calls']:,} calls")
    c4.metric("Quality", f"{s['mean_quality']:.3f}",
              f"{s['mean_quality'] - s['shadow_mean_quality']:+.3f} vs unmanaged")
    c5.metric("Budget remaining", f"{s['budget']['remaining_pct']:.0f}%",
              f"worst burn {s['burn']['worst_multiplier']:.1f}×",
              delta_color="off")

    if s["circuit_breakers"]:
        st.error("**Circuit breaker open:** " + "; ".join(
            f"`{k}` — {v}" for k, v in s["circuit_breakers"].items()))
    if s["active_incidents"]:
        for kind, inc in s["active_incidents"].items():
            st.warning(
                f"**Incident running — {kind}**: {inc['calls']:,} calls, "
                f"{fmt_inr(inc['cost_inr'])} burned since minute {inc['started_minute']}"
            )

    # -------------------------------------------------------------- charts --
    left, right = st.columns([3, 2])

    with left:
        managed = s["minute_costs"]
        shadow = s["minute_shadow"]
        start = s["sim_minute"] - len(managed) + 1
        x = list(range(start, s["sim_minute"] + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=shadow, name="unmanaged (shadow)",
                                 line=dict(color=RED, width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=x, y=managed, name="TokenOps", fill="tozeroy",
                                 line=dict(color=GREEN, width=2),
                                 fillcolor="rgba(46,125,91,0.18)"))
        budget_line = s["burn"]["budget_inr_per_min"]
        fig.add_hline(y=budget_line, line_dash="dash", line_color="#888",
                      annotation_text="budgeted rate")
        fig.update_layout(height=300, margin=dict(t=24, b=8, l=8, r=8),
                          xaxis_title="simulated minute", yaxis_title="INR / min",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True, key=f"spend_{s['sim_minute']}")

        st.markdown("**Burn-rate windows**")
        bw = pd.DataFrame(s["burn"]["windows"])
        bw["status"] = bw["breaching"].map({True: "🔴 BREACHING", False: "🟢 ok"})
        st.dataframe(
            bw[["window_hours", "short_window_minutes", "threshold_multiplier",
                "observed_multiplier", "status"]].rename(columns={
                    "window_hours": "long window (h)",
                    "short_window_minutes": "short window (min)",
                    "threshold_multiplier": "threshold ×",
                    "observed_multiplier": "observed ×",
                }),
            use_container_width=True, hide_index=True,
        )

    with right:
        st.markdown("**Event feed**")
        events = s["events"][:14]
        if not events:
            st.caption("no events yet")
        for e in events:
            st.markdown(
                f"{SEV_ICON.get(e['severity'], '⚪')} `{e['ts']}` "
                f"<span style='color:{SEV_COLOR.get(e['severity'], '#777')}'>{e['message']}</span>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------- routing + optimisers --
    st.divider()
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("**Routing decisions, most recent first**")
        routes = s["recent_routes"][:12]
        if routes:
            df = pd.DataFrame(routes)
            df["model"] = df["model"].map(short_model)
            df["mode"] = df.apply(
                lambda r: "degraded" if r["degraded"] else ("exploring" if r["exploring"] else "exploit"),
                axis=1,
            )
            st.dataframe(
                df[["sim_minute", "task_type", "step", "route", "model", "mode",
                    "quality", "cost_inr", "saved_inr"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("waiting for traffic")

    with c2:
        st.markdown("**Live policy and optimisers**")
        pol = pd.DataFrame(
            [{"task type": k, "current best route": v} for k, v in s["policy"].items()]
        )
        st.dataframe(pol, use_container_width=True, hide_index=True)
        m1, m2 = st.columns(2)
        m1.metric("Cache hit rate", f"{s['cache_hit_rate']:.0%}",
                  help="Over cacheable lookups in this live run only. The 30-day "
                       "steady-state figure in the benchmark is 20.7%, because a "
                       "24-hour TTL expires most entries over a month.")
        m2.metric("Escalation rate", f"{s['escalation_rate']:.0%}",
                  help="Cascade escalations as a share of cascade runs. Falls as the "
                       "router learns which task types the cheap model can hold.")
        st.caption(
            f"degraded calls: {t['degraded_calls']:,} · blocked: {t['blocked_calls']:,} · "
            f"incident spend: {fmt_inr(t['incident_inr'])}"
        )


live_view()

st.divider()
with st.expander("How to run this on stage"):
    st.markdown(
        """
1. **Start** and let it run for ~30 seconds. Point out the *Saved so far* counter
   climbing and the two curves separating — same traffic, two control planes.
2. Watch the **routing table**: early decisions are marked `exploring`. Within a
   minute or two the router settles, and the quality floor removes any route that
   cannot hold the bar — that shows up in the event feed as a `QUALITY FLOOR` line.
3. Press **💥 agent loop**. The unmanaged curve is irrelevant here — watch the
   event feed. The loop detector matches four identical `(step, prompt_hash)` calls
   in one session, kills it, and opens the circuit breaker. Read the rupee figure
   it stopped at, then compare it to the ₹2.70 L the same loop cost in the
   unmanaged 30-day run.
4. Press **💥 prompt bloat**. Context per call triples. Nothing breaks, the spend
   curve steps up, and the burn windows start climbing — this is the incident
   *without* a signature, which is what burn-rate alerting is for.
5. Drag the **monthly budget** down in the sidebar. Guardrail decisions move
   ALLOW → DEGRADE: routing forces the cheap model, context halves, verification
   is skipped. Sessions keep completing. That is the point.
        """
    )
