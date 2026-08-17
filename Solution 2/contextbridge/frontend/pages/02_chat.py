"""💬 Chat with Document"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_client import BackendError, chat, document_selector, get_session  # noqa: E402
from components.citation_viewer import (  # noqa: E402
    render_citations,
    render_truncation_risks,
)
from components.confidence_display import (  # noqa: E402
    render_completeness,
    render_confidence,
)
from components.token_meter import render_token_meter  # noqa: E402

st.set_page_config(page_title="Chat · ContextBridge", page_icon="💬", layout="wide")
st.title("💬 Chat with Document")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:12]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------------------------------------
controls = st.columns([3, 1, 1])
with controls[0]:
    doc_id = document_selector("Document to query", key="chat_doc", allow_none=True)
with controls[1]:
    mode = st.selectbox(
        "Mode",
        ["auto", "rag", "summary"],
        help=(
            "auto = detect summary-style questions; rag = always retrieve; "
            "summary = lead with the cached document summary"
        ),
    )
with controls[2]:
    top_k = st.number_input("Chunks retrieved", 1, 30, 8)

with st.expander("🧠 Session memory state"):
    try:
        state = get_session(st.session_state.session_id)
        m1, m2 = st.columns(2)
        m1.metric("Total exchanges", state["total_exchanges"])
        m2.metric("Verbatim in buffer", state["short_term_count"])
        if state.get("mid_term_summary"):
            st.markdown("**Rolling summary (tier 2)**")
            st.caption(state["mid_term_summary"])
        entities = {k: v for k, v in (state.get("entity_store") or {}).items() if v}
        if entities:
            st.markdown("**Entity store (tier 3)**")
            for key, values in entities.items():
                st.caption(f"**{key}**: {', '.join(values[:12])}")
        if not state.get("mid_term_summary") and not entities:
            st.caption(
                "Tiers 2 and 3 populate once the conversation exceeds the verbatim "
                "buffer — keep asking questions."
            )
    except BackendError as exc:
        st.caption(f"Session state unavailable: {exc}")

st.divider()

# ----------------------------------------------------------------------
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("meta"):
            meta = entry["meta"]
            render_confidence(meta.get("confidence", "MEDIUM"))
            render_completeness(meta.get("completeness", {}))
            render_citations(meta.get("citations", []))
            render_truncation_risks(meta.get("truncation_risks", []))
            usage = meta.get("token_usage", {})
            render_token_meter(
                usage.get("context_tokens", 0),
                usage.get("token_budget", 1),
                usage.get("breakdown"),
            )

question = st.chat_input("Ask something about the document…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, packing context, and asking Claude…"):
            try:
                response = chat(
                    st.session_state.session_id,
                    question,
                    doc_id=doc_id,
                    mode=mode,
                    top_k=int(top_k),
                )
            except BackendError as exc:
                st.error(str(exc))
                st.session_state.chat_history.pop()
                st.stop()

        st.markdown(response["answer"])
        render_confidence(response.get("confidence", "MEDIUM"))
        render_completeness(response.get("completeness", {}))
        render_citations(response.get("citations", []))
        render_truncation_risks(response.get("truncation_risks", []))

        usage = response.get("token_usage", {})
        render_token_meter(
            usage.get("context_tokens", 0),
            usage.get("token_budget", 1),
            usage.get("breakdown"),
        )
        st.session_state["last_token_usage"] = usage

        for warning in response.get("warnings", []):
            st.caption(f"⚠️ {warning}")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "meta": response,
        }
    )

# ----------------------------------------------------------------------
if not st.session_state.chat_history:
    st.info(
        "**Try these on `sample_insurance_claim.txt`:**\n\n"
        "- Has this claimant filed any similar claims before?\n"
        "- Are there any anomalies or fraud indicators in this claim?\n"
        "- What was the exact policy number from the prior claim?\n"
        "- Summarize the business interruption analysis."
    )
