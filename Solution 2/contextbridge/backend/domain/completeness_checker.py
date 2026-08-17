"""Audits every answer for what the context window couldn't hold.

This is the honesty layer: a RAG system that silently drops sections is
indistinguishable from one that hallucinates, from the user's side. Every chat
response carries a completeness report naming what was left out.
"""

from __future__ import annotations

import re

from backend.core.models import ChunkResult, CompletenessReport, SearchResult
from backend.core.retriever import tokenize
from backend.utils.helpers import dedupe_preserving_order

# "as mentioned in section 4", "see Appendix B", "refer to clause 12"
_CROSS_REF_RE = re.compile(
    r"\b(?:as (?:described|mentioned|stated|set out|noted) (?:in|above|below)|"
    r"see|refer to|pursuant to|per)\s+"
    r"(?:section|clause|appendix|exhibit|schedule|annex|part|page)\s+"
    r"[A-Z0-9][\w.\-]*",
    re.IGNORECASE,
)

_MID_SENTENCE_START = re.compile(r"^[a-z,;)\]]")
_MID_SENTENCE_END = re.compile(r"[A-Za-z0-9,;:\-]$")

_NOT_FOUND_MARKERS = (
    "not found in provided context",
    "i don't have enough information",
    "cannot find",
    "no information",
)


class CompletenessChecker:
    """Grades answer completeness and flags truncation artefacts."""

    def check_response_completeness(
        self,
        query: str,
        response: str,
        dropped_chunks: list[SearchResult],
        included_chunks: list[SearchResult] | None = None,
    ) -> CompletenessReport:
        included_chunks = included_chunks or []
        notes: list[str] = []

        if not dropped_chunks:
            confidence = "HIGH"
            message = "All retrieved sections fit within the context budget."
            if _looks_unanswered(response):
                confidence = "MEDIUM"
                message = (
                    "Nothing was dropped from context, but the answer did not find "
                    "the information — try rephrasing or widening the search."
                )
            if not included_chunks:
                confidence = "LOW"
                message = (
                    "No document sections were retrieved for this query. The answer "
                    "is not grounded in the document."
                )
            return CompletenessReport(
                confidence=confidence, message=message, notes=notes
            )

        # How relevant was what we dropped? Overlap between query terms and text.
        query_terms = set(tokenize(query))
        relevant_dropped = [
            result
            for result in dropped_chunks
            if query_terms & set(tokenize(result.chunk.text))
        ]

        sections = dedupe_preserving_order(
            _describe(result) for result in dropped_chunks
        )

        best_included = max((r.score for r in included_chunks), default=0.0)
        best_dropped = max((r.score for r in dropped_chunks), default=0.0)

        if relevant_dropped and best_dropped >= best_included * 0.9:
            confidence = "LOW"
        elif relevant_dropped:
            confidence = "MEDIUM"
        else:
            confidence = "MEDIUM" if len(dropped_chunks) > 3 else "HIGH"

        if _looks_unanswered(response) and dropped_chunks:
            confidence = "LOW"
            notes.append(
                "The model reported not finding the answer while sections were "
                "dropped from context — those sections are the likely location."
            )

        message = (
            f"Note: {len(dropped_chunks)} document section(s) couldn't fit in "
            "context. They may contain relevant information."
        )
        if relevant_dropped:
            message += (
                f" {len(relevant_dropped)} of them share terms with your question."
            )

        return CompletenessReport(
            confidence=confidence,
            message=message,
            dropped_sections=sections[:12],
            dropped_chunk_count=len(dropped_chunks),
            notes=notes,
        )

    # ------------------------------------------------------------------
    def detect_truncation_risks(
        self, chunks: list[ChunkResult], total_chunks: int | None = None
    ) -> list[str]:
        """Flag mid-sentence boundaries and unresolved cross-references."""
        risks: list[str] = []
        total_chunks = total_chunks or (chunks[0].total_chunks if chunks else 0)
        present = {c.chunk_index for c in chunks}

        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue

            if _MID_SENTENCE_START.match(text) and chunk.chunk_index > 0:
                if chunk.chunk_index - 1 not in present:
                    risks.append(
                        f"Chunk {chunk.chunk_index} (page {chunk.page}) starts "
                        "mid-sentence and its preceding chunk is not in context."
                    )

            if (
                _MID_SENTENCE_END.search(text)
                and chunk.chunk_index + 1 < (total_chunks or 0)
                and chunk.chunk_index + 1 not in present
            ):
                risks.append(
                    f"Chunk {chunk.chunk_index} (page {chunk.page}) ends "
                    "mid-sentence and its following chunk is not in context."
                )

            for match in _CROSS_REF_RE.findall(text):
                risks.append(
                    f"Chunk {chunk.chunk_index} (page {chunk.page}) references "
                    f'"{match.strip()}" — that target may not be in context.'
                )

        return dedupe_preserving_order(risks)[:15]


def _describe(result: SearchResult) -> str:
    chunk = result.chunk
    if chunk.section_name:
        return f"{chunk.section_name} (page {chunk.page})"
    return f"Page {chunk.page}, chunk {chunk.chunk_index}"


def _looks_unanswered(response: str) -> bool:
    lowered = (response or "").lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)
