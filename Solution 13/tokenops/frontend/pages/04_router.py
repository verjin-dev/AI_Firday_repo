"""The learning router: what it chose, and exactly why."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.common import (  # noqa: E402
    MODEL_COLORS, ROOT, evidence, page_setup, require_data,
    short_model, sidebar, sql,
)
from backend.core.optimizers.cascade import break_even_escalation_rate  # noqa: E402
from backend.core.router import LearningRouter  # noqa: E402

page_setup("Router", icon="🎛️")
sidebar("router")
st.title("Learning router")
st.caption(
    "A constrained Thompson-sampling bandit over (model × prompt variant × context depth "
    "× cache policy). It is a bandit, not a black box - every arm's evidence is below."
)

if not require_data():
    st.stop()

tasks = sql("SELECT DISTINCT task_type FROM router_state ORDER BY task_type")["task_type"].tolist()
task = st.selectbox("Task type", tasks, index=tasks.index("classification") if "classification" in tasks else 0)

# ------------------------------------------------------------- convergence
raw = sql(
    "SELECT day, model, SUM(pulls) AS pulls FROM router_state WHERE task_type = :t "
    "GROUP BY day, model ORDER BY day",
    {"t": task},
)
piv = raw.pivot(index="day", columns="model", values="pulls").fillna(0)
daily = piv.diff().fillna(piv)
share = daily.div(daily.sum(axis=1).replace(0, 1), axis=0)

fig = go.Figure()
for model in share.columns:
    fig.add_trace(go.Scatter(
        x=share.index, y=share[model], name=short_model(model), stackgroup="one",
        line=dict(width=0.5, color=MODEL_COLORS.get(model)),
    ))
fig.update_layout(height=340, yaxis_title="share of calls that day", xaxis_title="day",
                  yaxis_range=[0, 1], legend=dict(orientation="h"), margin=dict(t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

final = share.iloc[-1].sort_values(ascending=False)
leader = str(final.index[0])
leaders = share.idxmax(axis=1)
stable_from = next((int(share.index[i]) for i in range(len(leaders))
                    if all(leaders.iloc[j] == leader for j in range(i, len(leaders)))), None)

c1, c2, c3 = st.columns(3)
c1.metric("Settled on", short_model(leader), f"{final.iloc[0]:.0%} of traffic")
c2.metric("Stable from", f"day {stable_from}" if stable_from is not None else "not stable")
c3.metric("Cascade break-even escalation rate", f"{break_even_escalation_rate():.0%}",
          help="Above this, paying for a cheap attempt first costs more than going straight "
               "to the strong model.")

st.info(
    f"Nobody would have written **{short_model(leader)} at {final.iloc[0]:.0%}** into a static "
    f"routing table for **{task}** on day one. The router found it from {int(piv.iloc[-1].sum()):,} "
    "observed calls, and it keeps adapting when the next model ships."
)

st.divider()

# --------------------------------------------------------------- arm table
st.markdown("#### Per-arm evidence")

explain_path = ROOT / "data" / "samples" / "router_explain.json"
arms_df = None
if explain_path.exists():
    try:
        data = json.loads(explain_path.read_text(encoding="utf-8"))
        if task in data:
            arms_df = pd.DataFrame(data[task]["arms"])
            meta = data[task]
    except Exception:
        arms_df = None

if arms_df is not None and not arms_df.empty:
    show = arms_df[[
        "route_id", "model_short", "prompt_variant", "context_depth", "cache_policy",
        "pulls", "mean_quality", "mean_cost_inr", "mean_reward", "posterior_mean",
        "ci_low", "ci_high", "selection_prob", "excluded", "exclusion_reason",
    ]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Quality floor", f"{meta['quality_floor']:.2f}")
    m2.metric("Exploration budget", f"{meta['exploration_budget_pct']:.0f}% of period spend",
              f"used {meta['exploration_spend_pct']:.2f}%")
    m3.metric("λ (cost weight in reward)", f"{meta['lambda']}")

    excluded = arms_df[arms_df["excluded"] == True]  # noqa: E712
    if not excluded.empty:
        st.error(
            "Excluded by the quality floor: "
            + "; ".join(f"`{r.route_id}` - {r.exclusion_reason}" for r in excluded.itertuples())
        )
    st.caption(
        "reward = quality − λ·normalised cost, squashed to [0,1] to drive a Beta posterior. "
        "Arms below the quality floor are removed **before** sampling, so exploration can "
        "never cost quality below the floor."
    )
else:
    st.warning("Run `python scripts/simulate_workload.py` to generate the router explanation snapshot.")

st.divider()

# ------------------------------------------------------------ live decision
st.markdown("#### Ask the router")
c1, c2 = st.columns([1, 2])
with c1:
    remaining = st.slider("Remaining budget (%)", 0, 100, 100, 5)
    force = remaining <= 30
    st.caption("Below 30% the guardrail overrides the bandit and forces the cheap route.")
with c2:
    lr = LearningRouter()
    state = sql("SELECT * FROM router_state WHERE day = (SELECT MAX(day) FROM router_state) "
                "AND task_type = :t", {"t": task})
    arms = lr._task_arms(task)  # noqa: SLF001
    for _, r in state.iterrows():
        a = arms.get(str(r["route_id"]))
        if a is not None:
            a.alpha, a.beta, a.pulls = float(r["alpha"]), float(r["beta"]), int(r["pulls"] or 0)
            if r["mean_quality"] is not None and a.pulls:
                a.quality_sum = float(r["mean_quality"]) * a.pulls
                a.cost_sum = float(r["mean_cost_inr"] or 0) * a.pulls
                lr._apply_quality_floor(a)  # noqa: SLF001
    d = lr.select(task, {"force_cheap": force})
    st.json({
        "route": d.route.as_dict(),
        "exploring": d.exploring,
        "reason": d.reason,
        "candidate_arms": d.candidates,
    })

evidence(
    "daily router snapshots",
    raw,
    "Cumulative pulls per model per day, written by the simulator. The chart above is "
    "the day-over-day delta of this table.",
)
