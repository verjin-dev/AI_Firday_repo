"""Shows the source chunk behind each answer."""

from __future__ import annotations

import html

import streamlit as st


def render_citations(citations: list[dict], key_prefix: str = "") -> None:
    """Expandable 📎 Sources panel with page numbers and the cited text."""
    if not citations:
        st.caption("No source sections cited for this answer.")
        return

    label = f"📎 Sources ({len(citations)})"
    with st.expander(label, expanded=False):
        for index, citation in enumerate(citations):
            page = citation.get("page", 0)
            section = citation.get("section") or "—"
            score = citation.get("score", 0.0)
            chunk_id = citation.get("chunk_id", "")

            location = f"page {page}" if page else "page n/a"
            st.markdown(
                f"**{index + 1}. {section}** · {location} · "
                f"relevance `{score:.3f}`"
            )
            st.caption(f"`{chunk_id}`")
            st.markdown(
                f"<div style='border-left:3px solid rgba(88,166,255,0.6);"
                f"padding:0.4rem 0.75rem;margin:0.2rem 0 0.8rem 0;"
                f"background:rgba(88,166,255,0.06);font-size:0.86rem;"
                f"white-space:pre-wrap;'>{html.escape(citation.get('text', ''))}</div>",
                unsafe_allow_html=True,
            )


def render_evidence(evidence: str, page: int = 0, section: str = "") -> None:
    """Single quoted passage — used by the fraud and clause views."""
    if not evidence:
        return
    location = " · ".join(
        part for part in (section, f"page {page}" if page else "") if part
    )
    st.markdown(
        f"<div style='border-left:3px solid rgba(218,54,51,0.6);"
        f"padding:0.4rem 0.75rem;margin:0.3rem 0;background:rgba(218,54,51,0.06);"
        f"font-size:0.86rem;font-style:italic;white-space:pre-wrap;'>"
        f"“{html.escape(evidence)}”</div>"
        + (
            f"<div style='font-size:0.75rem;opacity:0.7;margin-bottom:0.6rem;'>"
            f"{html.escape(location)}</div>"
            if location
            else ""
        ),
        unsafe_allow_html=True,
    )


def render_truncation_risks(risks: list[str]) -> None:
    if not risks:
        return
    with st.expander(f"⚠️ Truncation risks detected ({len(risks)})"):
        for risk in risks:
            st.markdown(f"- {risk}")
