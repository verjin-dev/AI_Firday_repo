"""⚖️ Compare Documents"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api_client import BackendError, cached_documents, extract, summarize  # noqa: E402

st.set_page_config(page_title="Compare · ContextBridge", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Documents")
st.caption(
    "Summaries and entity sets side by side — useful for spotting the same party, "
    "property or amount appearing across two files."
)

documents = cached_documents()
if len(documents) < 2:
    st.info("Index at least two documents to use this page.")
    st.stop()

options = {f"{d['file_name']} ({d['chunk_count']} chunks)": d["doc_id"] for d in documents}
labels = list(options)

left_col, right_col = st.columns(2)
with left_col:
    left_label = st.selectbox("Left document", labels, index=0, key="cmp_left")
with right_col:
    right_label = st.selectbox(
        "Right document", labels, index=min(1, len(labels) - 1), key="cmp_right"
    )

left_id, right_id = options[left_label], options[right_label]

if left_id == right_id:
    st.warning("Select two different documents.")
    st.stop()

if st.button("Compare", type="primary"):
    with st.spinner("Fetching summaries and entities for both documents…"):
        try:
            st.session_state["cmp"] = {
                "left": {
                    "summary": summarize(left_id, level="master"),
                    "entities": extract(left_id, "entities").get("entities") or {},
                    "label": left_label,
                },
                "right": {
                    "summary": summarize(right_id, level="master"),
                    "entities": extract(right_id, "entities").get("entities") or {},
                    "label": right_label,
                },
            }
        except BackendError as exc:
            st.error(str(exc))

comparison = st.session_state.get("cmp")
if not comparison:
    st.info("Press **Compare** to load both documents.")
    st.stop()

left, right = comparison["left"], comparison["right"]

st.divider()
st.subheader("Summaries side by side")
col_a, col_b = st.columns(2)
for column, side in ((col_a, left), (col_b, right)):
    with column:
        st.markdown(f"**{side['label']}**")
        savings = side["summary"].get("token_savings", {})
        st.caption(
            f"{savings.get('original_tokens', 0):,} tokens → "
            f"{savings.get('summary_tokens', 0):,} "
            f"({savings.get('compression_ratio', 0):.1%} compression)"
        )
        st.markdown(side["summary"].get("master_summary") or "_No summary available._")

# ----------------------------------------------------------------------
st.divider()
st.subheader("Shared entities")


def flatten(entities: dict[str, list[str]]) -> dict[str, set[str]]:
    return {key: {v.strip().lower() for v in values if v.strip()}
            for key, values in entities.items()}


left_sets = flatten(left["entities"])
right_sets = flatten(right["entities"])
all_keys = sorted(set(left_sets) | set(right_sets))

shared_rows = []
left_only_rows = []
right_only_rows = []

for key in all_keys:
    left_values = left_sets.get(key, set())
    right_values = right_sets.get(key, set())
    for value in sorted(left_values & right_values):
        shared_rows.append({"Type": key, "Value": value})
    for value in sorted(left_values - right_values):
        left_only_rows.append({"Type": key, "Value": value})
    for value in sorted(right_values - left_values):
        right_only_rows.append({"Type": key, "Value": value})

if shared_rows:
    st.success(
        f"{len(shared_rows)} entity value(s) appear in both documents — "
        "shared parties or amounts across files are worth a second look."
    )
    st.dataframe(shared_rows, use_container_width=True, hide_index=True)
else:
    st.info("No overlapping entity values found.")

st.subheader("Differences")
diff_a, diff_b = st.columns(2)
with diff_a:
    st.markdown(f"**Only in {left['label']}** ({len(left_only_rows)})")
    if left_only_rows:
        st.dataframe(left_only_rows, use_container_width=True, hide_index=True, height=380)
with diff_b:
    st.markdown(f"**Only in {right['label']}** ({len(right_only_rows)})")
    if right_only_rows:
        st.dataframe(right_only_rows, use_container_width=True, hide_index=True, height=380)
