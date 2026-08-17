"""Confidence badges and dropped-context warnings."""

from __future__ import annotations

import streamlit as st

BADGES = {
    "HIGH": ("🟢", "#2ea043", "High confidence"),
    "MEDIUM": ("🟡", "#d29922", "Medium confidence"),
    "LOW": ("🔴", "#da3633", "Low confidence"),
}

SEVERITY_COLORS = {
    "HIGH": "#da3633",
    "MEDIUM": "#d29922",
    "LOW": "#c9a227",
}


def render_confidence(confidence: str) -> None:
    emoji, color, label = BADGES.get(confidence.upper(), BADGES["MEDIUM"])
    st.markdown(
        f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
        f"border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:600;'>"
        f"{emoji} {label}</span>",
        unsafe_allow_html=True,
    )


def render_completeness(completeness: dict) -> None:
    """The ⚠️ banner when sections were dropped from context."""
    dropped = completeness.get("dropped_chunk_count", 0)
    message = completeness.get("message", "")
    sections = completeness.get("dropped_sections", [])
    notes = completeness.get("notes", [])

    if not dropped and not notes:
        if message:
            st.caption(f"✓ {message}")
        return

    st.warning(f"⚠️ {message}")
    if sections:
        with st.expander(f"Sections that didn't fit ({len(sections)})"):
            for section in sections:
                st.markdown(f"- {section}")
    for note in notes:
        st.caption(f"↳ {note}")


def render_severity_badge(severity: str) -> str:
    """Inline HTML badge — returned rather than rendered, for use inside tables."""
    color = SEVERITY_COLORS.get(severity.upper(), "#888")
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
        f"border-radius:4px;padding:1px 7px;font-size:0.75rem;font-weight:600;'>"
        f"{severity.upper()}</span>"
    )


def render_risk_gauge(score: int, band: str) -> None:
    """0-100 risk gauge. Falls back to a bar when plotly is unavailable."""
    try:
        import plotly.graph_objects as go

        figure = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": " / 100"},
                title={"text": f"Risk score — {band}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _band_color(score)},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(46,160,67,0.25)"},
                        {"range": [25, 50], "color": "rgba(201,162,39,0.25)"},
                        {"range": [50, 75], "color": "rgba(210,153,34,0.3)"},
                        {"range": [75, 100], "color": "rgba(218,54,51,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
            )
        )
        figure.update_layout(height=260, margin={"t": 60, "b": 10, "l": 30, "r": 30})
        st.plotly_chart(figure, use_container_width=True)
    except ImportError:
        st.metric("Risk score", f"{score}/100", band)
        st.progress(min(100, max(0, score)) / 100)


def _band_color(score: int) -> str:
    if score >= 75:
        return SEVERITY_COLORS["HIGH"]
    if score >= 50:
        return SEVERITY_COLORS["MEDIUM"]
    if score >= 25:
        return SEVERITY_COLORS["LOW"]
    return "#2ea043"
