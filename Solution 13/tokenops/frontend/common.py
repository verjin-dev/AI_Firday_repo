"""Shared helpers for the Streamlit app: data access, formatting, theming.

The UI reads the ledger directly rather than through the API. That is a
deliberate demo-safety choice: one less process to fail on stage. The API
exists and is documented, and every page here maps to an endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.ledger import CostLedger  # noqa: E402
from backend.core.pricing import fmt_inr  # noqa: E402
from backend.storage.db import has_data, query_df, table_counts  # noqa: E402

GREEN = "#2e7d5b"
RED = "#b0413e"
AMBER = "#c98a1b"
GREY = "#8a8a8a"
MODEL_COLORS = {
    "claude-haiku-4-5-20251001": "#2e7d5b",
    "claude-sonnet-4-6": "#2f6f9f",
    "claude-opus-5": "#8a5bb5",
}

ARM_LABEL = {"tokenops": "TokenOps", "baseline": "Baseline (no FinOps)"}


def short_model(m: str) -> str:
    return m.replace("claude-", "").replace("-20251001", "")


def page_setup(title: str, icon: str = "₹") -> None:
    st.set_page_config(page_title=f"TokenOps - {title}", page_icon=icon, layout="wide")


@st.cache_data(show_spinner=False)
def notes() -> Dict[str, Any]:
    p = ROOT / "data" / "samples" / "simulation_notes.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def benchmark() -> Dict[str, Any]:
    p = ROOT / "benchmark_results.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def unit_economics(arm: str, outcome_type: str, group_by: Optional[str] = None) -> Dict[str, Any]:
    return CostLedger(arm).unit_economics(outcome_type=outcome_type, group_by=group_by).as_dict()


@st.cache_data(show_spinner=False)
def daily(arm: str) -> pd.DataFrame:
    return CostLedger(arm).daily_series()


@st.cache_data(show_spinner=False)
def daily_unit(arm: str, outcome_type: str) -> pd.DataFrame:
    return CostLedger(arm).daily_unit_cost(outcome_type=outcome_type)


@st.cache_data(show_spinner=False)
def flat_attribution(arm: str) -> pd.DataFrame:
    return CostLedger(arm).flat_attribution()


@st.cache_data(show_spinner=False)
def waste(arm: str) -> Dict[str, Any]:
    return CostLedger(arm).waste_report()


@st.cache_data(show_spinner=False)
def optimiser_stats(arm: str) -> Dict[str, Any]:
    return CostLedger(arm).optimiser_stats()


@st.cache_data(show_spinner=False)
def sql(q: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return query_df(q, params or {})


def outcome_types() -> List[str]:
    df = sql("SELECT DISTINCT outcome_type FROM outcomes ORDER BY outcome_type")
    return df["outcome_type"].tolist() if not df.empty else ["ticket_resolved"]


def require_data() -> bool:
    if has_data():
        return True
    st.error("The ledger is empty.")
    st.code("python scripts/simulate_workload.py\npython scripts/benchmark.py", language="bash")
    st.caption("Then reload this page.")
    return False


def sidebar(active_arm_key: str = "arm") -> str:
    """Standard sidebar. Returns the selected arm."""
    with st.sidebar:
        st.markdown("### TokenOps")
        st.caption("Cost per business outcome, not cost per token.")
        st.divider()

        baseline_mode = st.toggle(
            "Baseline mode", value=False, key=f"{active_arm_key}_toggle",
            help="Flip to the naive approach - one strong model, no cache, no guardrails - "
                 "computed on identical demand.",
        )
        arm = "baseline" if baseline_mode else "tokenops"
        st.caption(f"Viewing: **{ARM_LABEL[arm]}**")
        st.divider()

        totals = sql(
            "SELECT SUM(cost_inr) c, SUM(input_tokens + output_tokens) t, COUNT(*) n "
            "FROM llm_calls WHERE arm = :a", {"a": arm},
        )
        if not totals.empty and totals["c"].iloc[0]:
            st.metric("Ledger spend", fmt_inr(float(totals["c"].iloc[0])))
            st.caption(
                f"{int(totals['n'].iloc[0]):,} calls · {float(totals['t'].iloc[0]) / 1e6:,.1f}M tokens"
            )
        b = notes()
        if b.get("monthly_budget_inr"):
            st.caption(f"Monthly budget: {fmt_inr(b['monthly_budget_inr'])}")

        st.divider()
        if st.button("Reset demo (clear caches)", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"tables: {table_counts()}")
    return arm


def kpi(col, label: str, value: str, delta: Optional[str] = None,
        help_text: Optional[str] = None, color: Optional[str] = None) -> None:
    with col:
        st.metric(label, value, delta=delta, help=help_text,
                  delta_color="normal" if color is None else color)


def comparison(label: str, baseline_value: str, solution_value: str,
               note: Optional[str] = None) -> None:
    c1, c2, c3 = st.columns([2, 2, 3])
    c1.markdown(f"**Baseline**\n\n{baseline_value}")
    c2.markdown(f"**TokenOps**\n\n{solution_value}")
    c3.markdown(f"**{label}**\n\n{note or ''}")


def evidence(title: str, df: pd.DataFrame, caption: str = "") -> None:
    """Nothing is asserted without a clickable source."""
    with st.expander(f"Evidence - {title}"):
        if caption:
            st.caption(caption)
        st.dataframe(df, use_container_width=True, hide_index=True)
