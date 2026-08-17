"""Retriever tests — the vector store and embedder are stubbed."""

from __future__ import annotations

import pytest

from backend.core.models import SearchResult
from backend.core.retriever import IntelligentRetriever, tokenize
from tests.conftest import make_chunk


class StubEmbedder:
    backend = "stub"
    dimension = 8

    def embed_query(self, query: str) -> list[float]:
        vector = [0.0] * self.dimension
        for i, token in enumerate(tokenize(query)):
            vector[hash(token) % self.dimension] += 1.0
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class StubStore:
    """Returns semantic hits by keyword presence, so results are predictable."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.search_calls = 0

    def search(self, query_embedding, doc_id=None, top_k=8, threshold=0.0):
        self.search_calls += 1
        pool = [c for c in self._chunks if not doc_id or c.doc_id == doc_id]
        return [
            SearchResult(
                chunk=chunk,
                score=0.9 - i * 0.05,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                match_type="semantic",
            )
            for i, chunk in enumerate(pool[:top_k])
        ]

    def get_document_chunks(self, doc_id):
        return [c for c in self._chunks if c.doc_id == doc_id]

    def get_chunk_by_id(self, chunk_id):
        return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    def list_documents(self):
        from backend.core.models import DocumentInfo

        ids = {c.doc_id for c in self._chunks}
        return [DocumentInfo(d, d, "general", 0) for d in ids]


@pytest.fixture
def corpus():
    texts = [
        "The policyholder reported a fire loss at the industrial parkway premises.",
        "Prior claim CLM-2024-778341 was filed under policy POL-CG-88213-B in 2024.",
        "Business interruption analysis covers payroll and continuing expenses.",
        "Structural engineering assessment of the roof and building envelope.",
        "Witness statement from the night security guard on duty.",
        "Debris removal scope and salvage assessment for damaged stock.",
    ]
    return [
        make_chunk(i, text * 4, section=f"Section {i}", page=i + 1, total=len(texts))
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def retriever(corpus):
    return IntelligentRetriever(vector_store=StubStore(corpus), embedder=StubEmbedder())


# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_semantic_mode_returns_results(retriever):
    results = await retriever.retrieve("fire loss", retrieval_mode="semantic", top_k=3)
    assert results
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_keyword_mode_finds_exact_identifier(retriever):
    results = await retriever.retrieve(
        "CLM-2024-778341", doc_id="doc-test", retrieval_mode="keyword", top_k=3
    )
    assert results, "BM25 must match a literal identifier"
    assert "CLM-2024-778341" in results[0].chunk.text


@pytest.mark.asyncio
async def test_hybrid_mode_merges_both_signals(retriever):
    results = await retriever.retrieve(
        "prior claim policy number", doc_id="doc-test", retrieval_mode="hybrid", top_k=6
    )
    assert results
    assert {r.match_type for r in results} & {"semantic", "keyword", "hybrid"}


@pytest.mark.asyncio
async def test_results_are_sorted_by_score(retriever):
    results = await retriever.retrieve("claim", doc_id="doc-test", top_k=6)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_top_k_is_respected(retriever):
    results = await retriever.retrieve("claim", doc_id="doc-test", top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_empty_query_returns_nothing(retriever):
    assert await retriever.retrieve("   ") == []


@pytest.mark.asyncio
async def test_no_duplicate_chunk_ids(retriever):
    results = await retriever.retrieve(
        "claim policy fire", doc_id="doc-test", retrieval_mode="hybrid", top_k=10
    )
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_section_rerank_boosts_referenced_sections(retriever, corpus):
    plain = await retriever.retrieve(
        "witness statement", doc_id="doc-test", top_k=6
    )
    boosted = await retriever.retrieve(
        "witness statement",
        doc_id="doc-test",
        top_k=6,
        referenced_sections={"Section 4"},
    )
    plain_scores = {r.chunk_id: r.score for r in plain}
    boosted_scores = {r.chunk_id: r.score for r in boosted}
    target = "doc-test_chunk_0004"
    if target in plain_scores and target in boosted_scores:
        assert boosted_scores[target] > plain_scores[target]


@pytest.mark.asyncio
async def test_semantic_failure_falls_back_to_keyword(corpus):
    class BrokenEmbedder(StubEmbedder):
        def embed_query(self, query):
            raise RuntimeError("embedding backend down")

    retriever = IntelligentRetriever(
        vector_store=StubStore(corpus), embedder=BrokenEmbedder()
    )
    results = await retriever.retrieve(
        "CLM-2024-778341", doc_id="doc-test", retrieval_mode="semantic", top_k=3
    )
    assert results, "a dead embedder must not mean zero results"


def test_get_neighboring_chunks(retriever):
    neighbours = retriever.get_neighboring_chunks("doc-test_chunk_0002", window=1)
    indices = sorted(c.chunk_index for c in neighbours)
    assert indices == [1, 2, 3]


def test_neighbors_clamp_at_document_start(retriever):
    neighbours = retriever.get_neighboring_chunks("doc-test_chunk_0000", window=2)
    assert min(c.chunk_index for c in neighbours) == 0


def test_tokenize_strips_stopwords():
    tokens = tokenize("What is the policy number for the claim?")
    assert "the" not in tokens
    assert "policy" in tokens
