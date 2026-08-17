"""📋 Summarization"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_client import BackendError, document_selector, summarize  # noqa: E402
from components.token_meter import render_compression  # noqa: E402

st.set_page_config(
    page_title="Summarize · ContextBridge", page_icon="📋", layout="wide"
)
st.title("📋 Hierarchical Summarization")
st.caption(
    "Map-Reduce over every chunk. All intermediate levels are retained — that is "
    "what stops a fact buried on page 31 from being compressed away."
)

controls = st.columns([3, 2, 1, 1])
with controls[0]:
    doc_id = document_selector("Document", key="summ_doc")
with controls[1]:
    focus = st.text_input("Focus topic (optional)", placeholder="e.g. prior claims")
with controls[2]:
    refresh = st.toggle("Recompute", value=False)
with controls[3]:
    st.write("")
    run = st.button("Summarize", type="primary", use_container_width=True)

if run and doc_id:
    with st.spinner("Running Map-Reduce summarization…"):
        try:
            st.session_state["summary_result"] = summarize(
                doc_id, level="all", focus=focus or None, refresh=refresh
            )
        except BackendError as exc:
            st.error(str(exc))

result = st.session_state.get("summary_result")

if result and (not doc_id or result["doc_id"] == doc_id):
    st.divider()

    savings = result.get("token_savings", {})
    metrics = st.columns(5)
    metrics[0].metric("Chunks", f"{result['chunk_count']:,}")
    metrics[1].metric("Hierarchy levels", result["levels_used"])
    metrics[2].metric("Completeness", f"{result['completeness_score']:.0%}")
    metrics[3].metric("Original tokens", f"{savings.get('original_tokens', 0):,}")
    metrics[4].metric(
        "Time", f"{result.get('processing_time_seconds', 0):.1f}s"
    )

    if result.get("cached"):
        st.caption("Served from the cached summary — tick *Recompute* to rebuild.")

    render_compression(
        savings.get("original_tokens", 0), savings.get("summary_tokens", 0)
    )

    if result.get("warnings"):
        with st.expander(f"⚠️ Warnings ({len(result['warnings'])})"):
            for warning in result["warnings"]:
                st.caption(f"• {warning}")

    tab_master, tab_sections, tab_chunks = st.tabs(
        ["Master", f"Sections ({len(result.get('section_summaries', []))})",
         f"Chunks ({len(result.get('chunk_summaries', []))})"]
    )

    with tab_master:
        master = result.get("master_summary", "")
        if master:
            st.markdown(master)
            st.download_button(
                "Export master summary (.txt)",
                master,
                file_name=f"{result['doc_id']}_master_summary.txt",
                mime="text/plain",
            )
        else:
            st.info("No master summary available.")

    with tab_sections:
        sections = result.get("section_summaries", [])
        if not sections:
            st.info("No section-level summaries at this hierarchy depth.")
        for index, section in enumerate(sections, start=1):
            with st.expander(f"Section summary {index}", expanded=index == 1):
                st.markdown(section)
        if sections:
            st.download_button(
                "Export section summaries (.txt)",
                "\n\n---\n\n".join(sections),
                file_name=f"{result['doc_id']}_sections.txt",
                mime="text/plain",
            )

    with tab_chunks:
        chunks = result.get("chunk_summaries", [])
        if not chunks:
            st.info("No chunk-level summaries.")
        else:
            st.caption(
                "Level 1 of the hierarchy — one summary per source chunk, in "
                "document order."
            )
            page_size = 25
            page = st.number_input(
                "Page",
                1,
                max(1, (len(chunks) + page_size - 1) // page_size),
                1,
            )
            start = (int(page) - 1) * page_size
            for offset, chunk in enumerate(chunks[start : start + page_size]):
                with st.expander(f"Chunk {start + offset:04d}"):
                    st.markdown(chunk)
            st.download_button(
                "Export chunk summaries (.txt)",
                "\n\n---\n\n".join(chunks),
                file_name=f"{result['doc_id']}_chunks.txt",
                mime="text/plain",
            )
elif not result:
    st.info("Select a document and press **Summarize**.")
