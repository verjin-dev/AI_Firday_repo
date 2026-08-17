"""🔍 Extract & Analyze"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_client import BackendError, document_selector, extract  # noqa: E402
from components.citation_viewer import render_evidence  # noqa: E402
from components.confidence_display import (  # noqa: E402
    render_risk_gauge,
    render_severity_badge,
)

st.set_page_config(page_title="Extract · ContextBridge", page_icon="🔍", layout="wide")
st.title("🔍 Extract & Analyze")

TYPES = {
    "Everything": "all",
    "Fraud indicators": "fraud",
    "Contract clauses": "clauses",
    "Risk assessment": "risk",
    "Entities": "entities",
}

controls = st.columns([3, 2, 1])
with controls[0]:
    doc_id = document_selector("Document", key="extract_doc")
with controls[1]:
    label = st.selectbox("Extraction type", list(TYPES))
with controls[2]:
    st.write("")
    run = st.button("Analyze", type="primary", use_container_width=True)

with st.expander("Client profile (optional — feeds the risk score)"):
    profile_cols = st.columns(3)
    profile = {
        "tenure_years": profile_cols[0].number_input("Client tenure (years)", 0, 60, 3),
        "prior_claims": profile_cols[1].number_input("Prior claims on file", 0, 50, 0),
        "segment": profile_cols[2].selectbox(
            "Segment", ["commercial", "retail", "high-net-worth"]
        ),
    }

if run and doc_id:
    with st.spinner("Analyzing — this runs several retrieval passes and LLM calls…"):
        try:
            st.session_state["extract_result"] = extract(
                doc_id, TYPES[label], client_profile=profile
            )
        except BackendError as exc:
            st.error(str(exc))

result = st.session_state.get("extract_result")

if result and (not doc_id or result["doc_id"] == doc_id):
    st.divider()

    if result.get("warnings"):
        with st.expander(f"⚠️ Warnings ({len(result['warnings'])})"):
            for warning in result["warnings"]:
                st.caption(f"• {warning}")

    # --- fraud ---------------------------------------------------------
    fraud = result.get("fraud")
    if fraud:
        st.subheader("🚩 Fraud indicators")
        head = st.columns([1, 1, 3])
        head[0].metric("Indicators", len(fraud["indicators"]))
        head[1].metric("Likelihood", fraud["fraud_likelihood"])
        head[2].caption(fraud.get("overall_assessment", ""))

        if not fraud["indicators"]:
            st.success("No fraud indicators identified.")
        for indicator in fraud["indicators"]:
            with st.container(border=True):
                title = st.columns([4, 1])
                title[0].markdown(f"**{indicator['type']}**")
                title[1].markdown(
                    render_severity_badge(indicator["severity"]),
                    unsafe_allow_html=True,
                )
                if indicator.get("explanation"):
                    st.markdown(indicator["explanation"])
                render_evidence(
                    indicator.get("evidence", ""),
                    indicator.get("page", 0),
                    indicator.get("section", ""),
                )

    # --- clauses --------------------------------------------------------
    clauses = result.get("clauses")
    if clauses:
        st.subheader("📜 Contract clauses")
        if clauses.get("summary"):
            st.caption(clauses["summary"])
        if not clauses["clauses"]:
            st.info("No contract clauses extracted from this document.")
        for clause in clauses["clauses"]:
            location = (
                f" · {clause['section']}" if clause.get("section") else ""
            )
            header = (
                f"{clause['risk_rating']} — {clause['clause_type']} "
                f"(page {clause.get('page', 0)}{location})"
            )
            with st.expander(header, expanded=clause["risk_rating"] == "HIGH"):
                st.markdown(
                    render_severity_badge(clause["risk_rating"]),
                    unsafe_allow_html=True,
                )
                if clause.get("explanation"):
                    st.markdown(f"**Plain English:** {clause['explanation']}")
                render_evidence(clause.get("text", ""), clause.get("page", 0))

    # --- risk -----------------------------------------------------------
    risk = result.get("risk")
    if risk:
        st.subheader("⚖️ Risk assessment")
        gauge_col, factors_col = st.columns([1, 2])
        with gauge_col:
            render_risk_gauge(risk["risk_score"], risk["risk_band"])
        with factors_col:
            st.markdown(risk.get("summary", ""))
            for factor in risk.get("top_factors", []):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{factor['factor']}**")
                cols[1].markdown(
                    render_severity_badge(factor["severity"]), unsafe_allow_html=True
                )
                if factor.get("evidence"):
                    st.caption(f"Evidence: {factor['evidence']}")
                if factor.get("recommendation"):
                    st.caption(f"→ {factor['recommendation']}")

    # --- entities -------------------------------------------------------
    entities = result.get("entities")
    if entities:
        st.subheader("🏷️ Entities")
        populated = {k: v for k, v in entities.items() if v}
        if not populated:
            st.info("No entities extracted.")
        for key, values in populated.items():
            st.markdown(f"**{key.replace('_', ' ').title()}** ({len(values)})")
            st.markdown(
                " ".join(
                    f"<span style='background:rgba(88,166,255,0.15);border-radius:4px;"
                    f"padding:2px 8px;margin:2px;font-size:0.8rem;"
                    f"display:inline-block;'>{value}</span>"
                    for value in values[:40]
                ),
                unsafe_allow_html=True,
            )

    st.divider()
    st.download_button(
        "Export analysis as JSON",
        json.dumps(result, indent=2),
        file_name=f"{result['doc_id']}_analysis.json",
        mime="application/json",
    )
elif not result:
    st.info("Select a document and press **Analyze**.")
