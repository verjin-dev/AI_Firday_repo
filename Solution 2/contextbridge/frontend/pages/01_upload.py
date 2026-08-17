"""📤 Upload Document"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_client import (  # noqa: E402
    BackendError,
    cached_documents,
    delete_document,
    upload_document,
)

st.set_page_config(page_title="Upload · ContextBridge", page_icon="📤", layout="wide")
st.title("📤 Upload Document")

DOC_TYPES = {
    "General document": "general",
    "Insurance claim": "insurance_claim",
    "Contract / agreement": "contract",
    "Standard operating procedure": "sop",
    "Transaction history": "transaction_history",
}

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"


def run_ingest(name: str, data: bytes, doc_type: str, summarize: bool) -> None:
    progress = st.progress(0.0, text="Uploading…")
    try:
        progress.progress(0.15, text="Parsing and chunking…")
        if summarize:
            progress.progress(
                0.35,
                text="Running hierarchical summarization (this is the slow part)…",
            )
        result = upload_document(name, data, doc_type, summarize)
        progress.progress(1.0, text="Done")
    except BackendError as exc:
        progress.empty()
        st.error(str(exc))
        return

    progress.empty()
    st.session_state["last_upload"] = result
    cached_documents.clear()

    if result["status"] == "failed":
        st.error(f"Ingestion failed for **{result['file_name']}**")
    elif result["status"] == "partial":
        st.warning(f"Ingested **{result['file_name']}** with warnings")
    else:
        st.success(f"Ingested **{result['file_name']}**")


# ----------------------------------------------------------------------
tab_upload, tab_samples, tab_manage = st.tabs(
    ["Upload a file", "Load a sample", "Manage indexed documents"]
)

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a document", type=["pdf", "docx", "txt", "md", "csv"]
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        doc_type_label = st.selectbox("Document type", list(DOC_TYPES))
    with col_b:
        summarize = st.toggle(
            "Run summarization on upload",
            value=True,
            help=(
                "Builds the full Map-Reduce summary hierarchy. Costs one LLM call "
                "per chunk — turn off for a fast index-only ingest."
            ),
        )

    if uploaded is not None and st.button("Ingest document", type="primary"):
        run_ingest(uploaded.name, uploaded.getvalue(), DOC_TYPES[doc_type_label], summarize)

with tab_samples:
    st.caption(
        "Pre-generated demo documents with planted signals. "
        "Run `python scripts/generate_sample_docs.py` if this list is empty."
    )
    samples = sorted(SAMPLES_DIR.glob("*.txt")) if SAMPLES_DIR.exists() else []
    if not samples:
        st.info(f"No samples found in `{SAMPLES_DIR}`.")
    else:
        sample_types = {
            "sample_insurance_claim.txt": "insurance_claim",
            "sample_contract.txt": "contract",
            "sample_fraud_case.txt": "transaction_history",
        }
        for sample in samples:
            words = len(sample.read_text(encoding="utf-8").split())
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**{sample.name}**  \n`{words:,} words`")
            col2.caption(sample_types.get(sample.name, "general"))
            if col3.button("Ingest", key=f"sample-{sample.name}"):
                run_ingest(
                    sample.name,
                    sample.read_bytes(),
                    sample_types.get(sample.name, "general"),
                    True,
                )

with tab_manage:
    documents = cached_documents()
    if not documents:
        st.info("No documents indexed.")
    for document in documents:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.markdown(f"**{document['file_name']}**  \n`{document['doc_id']}`")
        col2.metric("Chunks", f"{document['chunk_count']:,}")
        col3.metric("Tokens", f"{document['total_tokens']:,}")
        if col4.button("Delete", key=f"del-{document['doc_id']}"):
            try:
                delete_document(document["doc_id"])
                cached_documents.clear()
                st.rerun()
            except BackendError as exc:
                st.error(str(exc))

# ----------------------------------------------------------------------
result = st.session_state.get("last_upload")
if result:
    st.divider()
    st.subheader("Ingestion result")

    overflow = result.get("context_overflow_factor", 0.0)
    if overflow >= 1:
        st.error(
            f"### Without ContextBridge, this document would exceed a standard "
            f"8K context window by **{overflow}×**",
            icon="🚨",
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pages", f"{result['total_pages']:,}")
    c2.metric("Chunks", f"{result['total_chunks']:,}")
    c3.metric("Tokens", f"{result['total_tokens']:,}")
    c4.metric("Stored", f"{result['chunks_stored']:,}")
    c5.metric("Time", f"{result['ingestion_time_seconds']:.1f}s")

    summary = result.get("summary")
    if summary and summary.get("master_summary"):
        st.markdown("#### Summary preview")
        st.info(summary["master_summary"][:1200])
        s1, s2, s3 = st.columns(3)
        s1.metric("Hierarchy levels", summary["levels"])
        s2.metric("Chunks summarized", f"{summary['total_chunks_processed']:,}")
        s3.metric("Completeness", f"{summary['completeness_score']:.0%}")

    entities = result.get("entities") or {}
    populated = {k: v for k, v in entities.items() if v}
    if populated:
        st.markdown("#### Entities found")
        for key, values in populated.items():
            st.markdown(
                f"**{key.replace('_', ' ').title()}** — "
                + " ".join(
                    f"<span style='background:rgba(88,166,255,0.15);"
                    f"border-radius:4px;padding:1px 7px;margin:2px;"
                    f"font-size:0.8rem;display:inline-block;'>{v}</span>"
                    for v in values[:14]
                ),
                unsafe_allow_html=True,
            )

    if result.get("warnings"):
        with st.expander(f"⚠️ Warnings ({len(result['warnings'])})"):
            for warning in result["warnings"]:
                st.caption(f"• {warning}")
