"""ContextBridge — Streamlit entry point.

    streamlit run frontend/app.py

Uses Streamlit's `pages/` directory convention rather than `st.navigation()`,
which only exists on Streamlit >= 1.36; this runs on any 1.3x release.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import BackendError, cached_documents, clear_session, health  # noqa: E402
from components.token_meter import render_sidebar_meter  # noqa: E402

st.set_page_config(
    page_title="ContextBridge",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_session() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess-{uuid.uuid4().hex[:12]}"
    return st.session_state.session_id


def render_sidebar() -> None:
    session_id = ensure_session()

    with st.sidebar:
        st.markdown("## 🌉 ContextBridge")
        st.caption("Overcoming LLM context window limitations")
        st.divider()

        # --- backend status -------------------------------------------
        try:
            status = health()
            if status["llm_available"]:
                st.success(f"Backend online · {status['llm_model']}")
            else:
                st.warning("Backend online · LLM disabled (no API key)")
            st.caption(
                f"Embeddings: `{status['embedding_backend']}` "
                f"({status['embedding_dimension']}d)"
            )
            if status.get("warnings"):
                with st.expander("Warnings"):
                    for warning in status["warnings"]:
                        st.caption(f"• {warning}")
        except BackendError as exc:
            st.error(str(exc))
            st.stop()

        st.divider()

        # --- documents -------------------------------------------------
        st.markdown("### 📚 Indexed documents")
        documents = cached_documents()
        if documents:
            for document in documents:
                st.markdown(
                    f"**{document['file_name']}**  \n"
                    f"<span style='font-size:0.78rem;opacity:0.7;'>"
                    f"{document['chunk_count']} chunks · "
                    f"{document['total_tokens']:,} tokens · "
                    f"{document['doc_type']}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("None yet — start on the Upload page.")

        st.divider()

        # --- session ---------------------------------------------------
        st.markdown("### 🧠 Session")
        st.code(session_id, language=None)

        last = st.session_state.get("last_token_usage")
        if last:
            render_sidebar_meter(
                last.get("context_tokens", 0), last.get("token_budget", 1)
            )

        if st.button("Clear session memory", use_container_width=True):
            try:
                clear_session(session_id)
            except BackendError as exc:
                st.error(str(exc))
            st.session_state.pop("chat_history", None)
            st.session_state.pop("last_token_usage", None)
            st.session_state.session_id = f"sess-{uuid.uuid4().hex[:12]}"
            st.rerun()


render_sidebar()

# ----------------------------------------------------------------------
st.title("ContextBridge")
st.markdown(
    "**Large documents don't fit in a context window. ContextBridge makes that "
    "stop mattering.**"
)

st.markdown(
    """
A 100-page insurance claim is roughly 45,000 tokens. Feed it to a model with an
8K window and the last 80% of the document silently disappears — including,
reliably, the part that mattered. ContextBridge combines four layers so nothing
is lost and every answer is traceable:

| Layer | What it does |
| --- | --- |
| **Hierarchical summarization** | Map-Reduce over every chunk, keeping all intermediate levels so buried facts survive compression |
| **RAG retrieval** | Hybrid semantic + BM25 search with section-aware reranking |
| **Multi-tier memory** | Verbatim recent turns, a rolling summary, and a structured entity store |
| **Completeness auditing** | Every answer states what *didn't* fit in context, instead of quietly dropping it |
"""
)

st.divider()

left, right = st.columns(2)
with left:
    st.markdown(
        """
### Getting started
1. **📤 Upload** — ingest a document (or load a bundled sample)
2. **💬 Chat** — ask questions and get cited answers
3. **📋 Summarize** — inspect the compression hierarchy
4. **🔍 Extract** — fraud flags, clauses, entities, risk score
5. **⚖️ Compare** — two documents side by side
"""
    )

with right:
    st.markdown(
        """
### Demo path for judges
Upload `sample_insurance_claim.txt`, then ask:

> *Has this claimant filed any similar claims before?*

The answer is planted on **page 31** of 47 — beyond an 8K context window.
ContextBridge finds it, cites the chunk, and flags it as a fraud indicator.
Then ask a follow-up about the prior policy number to see the entity store
answer from memory rather than re-retrieving.
"""
    )

st.divider()
st.caption(
    "AI Friday National Finals · Context Window Hackathon · "
    "Banking & Insurance demo scenario"
)
