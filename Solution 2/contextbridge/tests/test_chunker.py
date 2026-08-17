"""Chunker tests."""

from __future__ import annotations

import pytest

from backend import config
from backend.core.chunker import DocumentChunker, EmptyChunkError
from backend.core.token_counter import token_counter


def test_chunk_text_produces_chunks(sample_text):
    chunks = DocumentChunker().chunk_text(sample_text, "doc-1")
    assert chunks, "expected at least one chunk"
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_sequential_and_formatted(sample_text):
    chunks = DocumentChunker().chunk_text(sample_text, "doc-1")
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc-1_chunk_{index:04d}"
        assert chunk.chunk_index == index


def test_total_chunks_is_set_on_every_chunk(sample_text):
    chunks = DocumentChunker().chunk_text(sample_text, "doc-1")
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_chunks_respect_min_size(sample_text):
    chunker = DocumentChunker(min_chunk_size=50)
    chunks = chunker.chunk_text(sample_text, "doc-1")
    assert all(c.token_count >= 50 for c in chunks)


def test_chunks_do_not_greatly_exceed_chunk_size(sample_text):
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=10)
    chunks = chunker.chunk_text(sample_text, "doc-1")
    # The splitter works in characters, so allow generous headroom over the
    # token target — but a chunk 3x the budget means the sizing is broken.
    assert all(c.token_count <= 200 * 3 for c in chunks)


def test_empty_text_raises():
    with pytest.raises(EmptyChunkError):
        DocumentChunker().chunk_text("   ", "doc-1")


def test_metadata_is_inherited_and_extended(sample_text):
    chunks = DocumentChunker().chunk_text(
        sample_text, "doc-1", metadata={"file_name": "x.txt", "doc_type": "contract"}
    )
    first = chunks[0]
    assert first.metadata["file_name"] == "x.txt"
    assert first.metadata["doc_type"] == "contract"
    assert first.metadata["doc_id"] == "doc-1"
    assert "page" in first.metadata


def test_chunk_by_section_captures_section_names(sample_text):
    chunks = DocumentChunker(min_chunk_size=20).chunk_by_section(sample_text, "doc-2")
    sections = {c.section_name for c in chunks}
    assert any("PRIOR CLAIMS HISTORY" in s.upper() for s in sections)


def test_chunk_by_section_assigns_pages(sample_text):
    chunks = DocumentChunker(min_chunk_size=20).chunk_by_section(sample_text, "doc-2")
    pages = {c.page for c in chunks}
    assert pages != {1}, "page markers should produce more than one page number"
    assert max(pages) <= 3


def test_section_chunking_keeps_the_planted_fact_locatable(sample_text):
    chunks = DocumentChunker(min_chunk_size=20).chunk_by_section(sample_text, "doc-2")
    hits = [c for c in chunks if "CLM-2024-778341" in c.text]
    assert hits, "the planted claim number must survive chunking"
    assert hits[0].page == 3


def test_large_document_chunks(claim_document):
    chunks = DocumentChunker().chunk_by_section(claim_document, "claim-1")
    assert len(chunks) > 40, f"expected many chunks, got {len(chunks)}"
    assert sum(1 for c in chunks if "CLM-2024-778341" in c.text) == 1


def test_token_counts_match_the_counter(sample_text):
    chunks = DocumentChunker().chunk_text(sample_text, "doc-1")
    for chunk in chunks[:5]:
        assert chunk.token_count == token_counter.count(chunk.text)


def test_overlap_config_is_smaller_than_chunk_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
