"""Live token budget visualiser."""

from __future__ import annotations

import streamlit as st

GREEN = "#2ea043"
YELLOW = "#d29922"
RED = "#da3633"

_LABELS = {
    "system_scaffold": "System prompt",
    "entity_store": "Entity store",
    "mid_term_summary": "Memory summary",
    "retrieved_chunks": "Document chunks",
    "short_term_buffer": "Recent turns",
    "query": "Your question",
}


def budget_color(percent: float) -> str:
    if percent < 70:
        return GREEN
    if percent < 90:
        return YELLOW
    return RED


def render_token_meter(
    used: int,
    budget: int,
    breakdown: dict[str, int] | None = None,
    compact: bool = False,
) -> None:
    """Horizontal budget bar: green <70%, yellow 70-90%, red >90%."""
    budget = max(1, budget)
    percent = min(100.0, 100.0 * used / budget)
    color = budget_color(percent)

    st.markdown(
        f"""
        <div style="margin:0.35rem 0 0.5rem 0;">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.8rem;opacity:0.85;margin-bottom:0.2rem;">
            <span>Context budget</span>
            <span><b>{used:,}</b> / {budget:,} tokens ({percent:.0f}%)</span>
          </div>
          <div style="background:rgba(128,128,128,0.22);border-radius:6px;
                      height:12px;width:100%;overflow:hidden;">
            <div style="width:{percent:.2f}%;background:{color};height:100%;
                        border-radius:6px;transition:width .3s;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if compact or not breakdown:
        return

    rows = [(_LABELS.get(k, k), v) for k, v in breakdown.items() if v]
    if not rows:
        return

    rows.sort(key=lambda pair: pair[1], reverse=True)
    total = sum(v for _, v in rows) or 1
    lines = "".join(
        f"<div style='display:flex;justify-content:space-between;font-size:0.78rem;"
        f"opacity:0.75;padding:1px 0;'><span>{name}</span>"
        f"<span>{value:,} ({100 * value / total:.0f}%)</span></div>"
        for name, value in rows
    )
    st.markdown(
        f"<div style='padding:0.2rem 0 0.5rem 0;'>{lines}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_meter(used: int, budget: int) -> None:
    render_token_meter(used, budget, compact=True)


def render_compression(original: int, summarized: int) -> None:
    """`200 pages -> 3 paragraphs (98.5% compression)` style callout."""
    if original <= 0:
        return
    ratio = 100.0 * (1 - summarized / original)
    st.markdown(
        f"""
        <div style="border:1px solid rgba(46,160,67,0.4);border-radius:8px;
                    padding:0.7rem 0.9rem;background:rgba(46,160,67,0.08);">
          <div style="font-size:1.5rem;font-weight:700;color:{GREEN};">
            {ratio:.1f}% compression
          </div>
          <div style="font-size:0.85rem;opacity:0.85;">
            {original:,} tokens &rarr; {summarized:,} tokens
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
