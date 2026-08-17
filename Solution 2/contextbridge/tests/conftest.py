"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.models import ChunkResult, SearchResult  # noqa: E402


@pytest.fixture
def sample_text() -> str:
    """A small document with headings and page markers."""
    return (
        "\n\n--- PAGE 1 ---\n\n"
        "SECTION 1. INTRODUCTION\n\n"
        + ("This is the opening section of the document. " * 40)
        + "\n\n--- PAGE 2 ---\n\n"
        "SECTION 2. FINDINGS\n\n"
        + ("The investigation produced several findings of note. " * 40)
        + "\n\n--- PAGE 3 ---\n\n"
        "SECTION 3. PRIOR CLAIMS HISTORY\n\n"
        "A prior claim CLM-2024-778341 was filed under policy POL-CG-88213-B "
        "for the same property on 12 September 2024, settling for $412,500. "
        + ("Supporting detail follows in this section. " * 30)
    )


@pytest.fixture
def claim_document() -> str:
    """The generated insurance claim, skipped if it hasn't been built yet."""
    path = ROOT / "data" / "sample_docs" / "sample_insurance_claim.txt"
    if not path.exists():
        pytest.skip("Run scripts/generate_sample_docs.py first")
    return path.read_text(encoding="utf-8")


def make_chunk(
    index: int,
    text: str,
    doc_id: str = "doc-test",
    section: str = "",
    page: int = 1,
    total: int = 10,
) -> ChunkResult:
    return ChunkResult(
        chunk_id=f"{doc_id}_chunk_{index:04d}",
        text=text,
        token_count=max(1, len(text) // 4),
        char_start=index * 100,
        char_end=index * 100 + len(text),
        chunk_index=index,
        total_chunks=total,
        metadata={
            "doc_id": doc_id,
            "section_name": section,
            "page": page,
            "chunk_index": index,
        },
    )


def make_result(chunk: ChunkResult, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk=chunk, score=score, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id
    )


@pytest.fixture
def chunk_factory():
    return make_chunk


@pytest.fixture
def result_factory():
    return make_result
