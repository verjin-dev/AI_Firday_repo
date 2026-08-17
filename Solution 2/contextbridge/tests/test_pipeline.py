"""End-to-end ingestion pipeline tests against a temporary ChromaDB."""

from __future__ import annotations

import pytest

from backend.core.models import ClaudeResponse
from backend.domain.completeness_checker import CompletenessChecker
from backend.ingestion.text_parser import TextParser
from tests.conftest import make_chunk, make_result

chromadb = pytest.importorskip("chromadb")


class StubClient:
    available = True

    def unavailable_reason(self):
        return None

    async def complete(
        self, prompt, system=None, max_tokens=None, messages=None, model=None
    ):
        if "JSON" in (system or ""):
            return ClaudeResponse(
                text='{"people": [], "organizations": [], "dates": [], '
                '"amounts": ["$412,500"], "locations": [], "decisions": [], '
                '"risks": [], "claim_ids": ["CLM-2024-778341"]}',
                input_tokens=20,
                output_tokens=10,
            )
        return ClaudeResponse(text="Concise summary.", input_tokens=40, output_tokens=15)

    async def complete_many(
        self, prompts, system=None, max_tokens=None, concurrency=1, model=None
    ):
        return [await self.complete(p, system, max_tokens, model=model) for p in prompts]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A pipeline wired to an isolated on-disk store and a stubbed LLM."""
    from backend.core import vector_store as vs_module
    from backend.core.vector_store import VectorStore

    store = VectorStore(
        collection_name="test_collection", persist_dir=str(tmp_path / "chroma")
    )
    monkeypatch.setattr(vs_module, "_store", store)

    from backend.ingestion.pipeline import IngestionPipeline

    instance = IngestionPipeline()
    instance.vector_store = store
    instance.summarizer.client = StubClient()
    instance.entity_extractor.client = StubClient()
    return instance


@pytest.fixture
def claim_file(tmp_path):
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "sample_insurance_claim.txt"
    if not source.exists():
        pytest.skip("Run scripts/generate_sample_docs.py first")
    target = tmp_path / "claim.txt"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(target)


# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ingest_succeeds(pipeline, claim_file):
    result = await pipeline.ingest(claim_file, doc_type="insurance_claim")
    assert result.status in {"success", "partial"}
    assert result.chunks_stored > 0


@pytest.mark.asyncio
async def test_ingest_reports_document_statistics(pipeline, claim_file):
    result = await pipeline.ingest(claim_file, doc_type="insurance_claim")
    assert result.total_chars > 50_000
    assert result.total_tokens > 10_000
    assert result.total_pages > 20
    assert result.total_chunks > 20


@pytest.mark.asyncio
async def test_chunks_are_queryable_after_ingest(pipeline, claim_file):
    result = await pipeline.ingest(claim_file)
    stored = pipeline.vector_store.get_document_chunks(result.doc_id)
    assert len(stored) == result.chunks_stored


@pytest.mark.asyncio
async def test_planted_fact_survives_the_whole_pipeline(pipeline, claim_file):
    """The fraud indicator must be retrievable end to end — this is the demo."""
    result = await pipeline.ingest(claim_file, doc_type="insurance_claim")
    stored = pipeline.vector_store.get_document_chunks(result.doc_id)
    hits = [c for c in stored if "CLM-2024-778341" in c.text]
    assert hits, "the planted prior-claim number must survive ingestion"
    assert hits[0].page >= 25, "it should sit deep in the document, not near the front"


@pytest.mark.asyncio
async def test_retrieval_finds_the_planted_fact(pipeline, claim_file):
    from backend.core.retriever import IntelligentRetriever

    result = await pipeline.ingest(claim_file, doc_type="insurance_claim")
    retriever = IntelligentRetriever(
        vector_store=pipeline.vector_store, embedder=pipeline.embedder
    )
    hits = await retriever.retrieve(
        "Has this claimant filed any similar claims before?",
        doc_id=result.doc_id,
        top_k=8,
    )
    assert hits
    assert any("CLM-2024-778341" in h.chunk.text for h in hits), (
        "hybrid retrieval must surface the buried prior claim"
    )


@pytest.mark.asyncio
async def test_summary_is_produced(pipeline, claim_file):
    result = await pipeline.ingest(claim_file, run_summarization=True)
    assert result.summary is not None
    assert result.summary.master_summary


@pytest.mark.asyncio
async def test_summarization_can_be_skipped(pipeline, claim_file):
    result = await pipeline.ingest(claim_file, run_summarization=False)
    assert result.summary is None
    assert result.chunks_stored > 0


@pytest.mark.asyncio
async def test_entities_are_extracted(pipeline, claim_file):
    result = await pipeline.ingest(claim_file)
    assert result.entities is not None
    assert result.entities.get("claim_ids")


@pytest.mark.asyncio
async def test_unsupported_file_type_fails_cleanly(pipeline, tmp_path):
    bad = tmp_path / "image.png"
    bad.write_bytes(b"\x89PNG\r\n")
    result = await pipeline.ingest(str(bad))
    assert result.status == "failed"
    assert result.warnings


@pytest.mark.asyncio
async def test_missing_file_fails_cleanly(pipeline):
    result = await pipeline.ingest("does/not/exist.txt")
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_delete_removes_all_chunks(pipeline, claim_file):
    result = await pipeline.ingest(claim_file)
    deleted = pipeline.vector_store.delete_document(result.doc_id)
    assert deleted == result.chunks_stored
    assert pipeline.vector_store.get_document_chunks(result.doc_id) == []


def test_context_overflow_factor(pipeline):
    assert pipeline.context_overflow_factor(48_000) > 1


# ----------------------------------------------------------------------
def test_text_parser_adds_page_markers(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("Paragraph one.\n\n" * 400, encoding="utf-8")
    parsed = TextParser().parse(str(path))
    assert "--- PAGE 1 ---" in parsed.text
    assert parsed.page_count > 1
    assert parsed.warnings


def test_text_parser_preserves_existing_markers(tmp_path):
    path = tmp_path / "paged.txt"
    path.write_text(
        "\n\n--- PAGE 1 ---\n\nfirst\n\n--- PAGE 2 ---\n\nsecond", encoding="utf-8"
    )
    parsed = TextParser().parse(str(path))
    assert parsed.page_count == 2
    assert not any("synthetic" in w for w in parsed.warnings)


# ----------------------------------------------------------------------
def test_completeness_flags_dropped_sections():
    checker = CompletenessChecker()
    dropped = [
        make_result(make_chunk(i, "prior claim history text", section="Prior Claims"))
        for i in range(3)
    ]
    report = checker.check_response_completeness(
        "prior claim history", "Not found in provided context sections.", dropped, []
    )
    assert report.confidence == "LOW"
    assert report.dropped_chunk_count == 3
    assert report.dropped_sections


def test_completeness_is_high_when_nothing_dropped():
    checker = CompletenessChecker()
    included = [make_result(make_chunk(0, "relevant answer text"))]
    report = checker.check_response_completeness("q", "Here is the answer.", [], included)
    assert report.confidence == "HIGH"
    assert report.dropped_chunk_count == 0


def test_truncation_risk_detects_cross_reference():
    checker = CompletenessChecker()
    chunk = make_chunk(1, "The obligations are as described in section 12 above.")
    risks = checker.detect_truncation_risks([chunk], total_chunks=10)
    assert any("section 12" in r for r in risks)
