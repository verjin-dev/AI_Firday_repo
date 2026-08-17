"""Hybrid retrieval: semantic search + BM25-style keyword scoring + reranking."""

from __future__ import annotations

import math
import re
from collections import Counter

from backend import config
from backend.core.embedder import EmbeddingEngine, get_embedder
from backend.core.models import ChunkResult, SearchResult
from backend.core.vector_store import VectorStore, get_vector_store
from backend.utils.logger import logger

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-_/.]*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "any", "been", "if", "into", "no", "not", "you", "your", "we", "us",
}

# BM25 parameters
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _WORD_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


class IntelligentRetriever:
    """Retrieves the chunks most likely to answer a query."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingEngine | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        # doc_id -> chunks, so keyword scoring doesn't re-read Chroma per query.
        self._corpus_cache: dict[str, list[ChunkResult]] = {}

    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int = config.TOP_K_CHUNKS,
        retrieval_mode: str = "hybrid",
        referenced_sections: set[str] | None = None,
    ) -> list[SearchResult]:
        """Semantic, keyword, or hybrid retrieval with section-aware reranking."""
        if not query.strip():
            return []

        semantic: list[SearchResult] = []
        keyword: list[SearchResult] = []

        if retrieval_mode in {"semantic", "hybrid"}:
            semantic = self._semantic(query, doc_id, top_k * 2)

        if retrieval_mode in {"keyword", "hybrid"} or (
            retrieval_mode == "semantic" and not semantic
        ):
            keyword = self._keyword(query, doc_id, top_k * 2)

        merged = self._merge(semantic, keyword, retrieval_mode)
        reranked = self._rerank(merged, referenced_sections or set())
        results = reranked[:top_k]

        logger.info(
            f"Retrieved {len(results)}/{len(merged)} chunks "
            f"(mode={retrieval_mode}, doc={doc_id or 'all'})"
        )
        return results

    # ------------------------------------------------------------------
    def _semantic(
        self, query: str, doc_id: str | None, limit: int
    ) -> list[SearchResult]:
        try:
            embedding = self.embedder.embed_query(query)
        except Exception as exc:
            logger.error(f"Query embedding failed ({exc}); keyword-only retrieval")
            return []

        try:
            return self.vector_store.search(
                embedding,
                doc_id=doc_id,
                top_k=limit,
                threshold=config.SIMILARITY_THRESHOLD,
            )
        except Exception as exc:
            logger.error(f"Vector search failed ({exc}); keyword-only retrieval")
            return []

    def _keyword(
        self, query: str, doc_id: str | None, limit: int
    ) -> list[SearchResult]:
        """BM25 over the candidate corpus."""
        terms = tokenize(query)
        if not terms:
            return []

        corpus = self._corpus(doc_id)
        if not corpus:
            return []

        doc_lengths = [len(tokenize(c.text)) for c in corpus]
        avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        n_docs = len(corpus)

        term_counts: list[Counter] = [Counter(tokenize(c.text)) for c in corpus]
        doc_freq = {
            term: sum(1 for counts in term_counts if counts.get(term)) for term in terms
        }

        scored: list[tuple[float, ChunkResult]] = []
        for i, chunk in enumerate(corpus):
            counts = term_counts[i]
            length = doc_lengths[i] or 1
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
                )
                denominator = frequency + _K1 * (1 - _B + _B * length / avg_length)
                score += idf * (frequency * (_K1 + 1)) / denominator
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:limit]
        max_score = top[0][0] or 1.0

        return [
            SearchResult(
                chunk=chunk,
                score=score / max_score,  # normalise into 0..1 to blend with cosine
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                match_type="keyword",
            )
            for score, chunk in top
        ]

    def _corpus(self, doc_id: str | None) -> list[ChunkResult]:
        key = doc_id or "__all__"
        cached = self._corpus_cache.get(key)
        if cached is not None:
            return cached

        try:
            if doc_id:
                chunks = self.vector_store.get_document_chunks(doc_id)
            else:
                chunks = []
                for info in self.vector_store.list_documents():
                    chunks.extend(self.vector_store.get_document_chunks(info.doc_id))
        except Exception as exc:
            logger.error(f"Corpus load failed for keyword search: {exc}")
            return []

        self._corpus_cache[key] = chunks
        return chunks

    def invalidate_cache(self, doc_id: str | None = None) -> None:
        """Called after ingest/delete so keyword search sees the new corpus."""
        if doc_id:
            self._corpus_cache.pop(doc_id, None)
        self._corpus_cache.pop("__all__", None)
        if doc_id is None:
            self._corpus_cache.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(
        semantic: list[SearchResult],
        keyword: list[SearchResult],
        mode: str,
    ) -> list[SearchResult]:
        if mode == "semantic" and semantic:
            return semantic
        if mode == "keyword":
            return keyword

        merged: dict[str, SearchResult] = {}
        for result in semantic:
            clone = SearchResult(
                chunk=result.chunk,
                score=result.score * config.SEMANTIC_WEIGHT,
                doc_id=result.doc_id,
                chunk_id=result.chunk_id,
                match_type="semantic",
            )
            merged[result.chunk_id] = clone

        for result in keyword:
            existing = merged.get(result.chunk_id)
            weighted = result.score * config.KEYWORD_WEIGHT
            if existing:
                existing.score += weighted
                existing.match_type = "hybrid"
            else:
                merged[result.chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=weighted,
                    doc_id=result.doc_id,
                    chunk_id=result.chunk_id,
                    match_type="keyword",
                )

        return list(merged.values())

    @staticmethod
    def _rerank(
        results: list[SearchResult], referenced_sections: set[str]
    ) -> list[SearchResult]:
        """Boost chunks from sections the conversation already touched."""
        if referenced_sections:
            lowered = {s.lower() for s in referenced_sections}
            for result in results:
                section = result.chunk.section_name.lower()
                if section and section in lowered:
                    result.score += config.RERANK_SECTION_BOOST
        return sorted(results, key=lambda r: r.score, reverse=True)

    # ------------------------------------------------------------------
    async def retrieve_for_summary(
        self, topic: str, doc_id: str, top_k: int = config.TOP_K_CHUNKS * 3
    ) -> list[SearchResult]:
        """Topic-scoped retrieval across a whole document, for focused summarization."""
        return await self.retrieve(
            topic, doc_id=doc_id, top_k=top_k, retrieval_mode="hybrid"
        )

    def get_neighboring_chunks(
        self, chunk_id: str, window: int = 1
    ) -> list[ChunkResult]:
        """Chunks at index ± ``window`` — used to expand context around a hit."""
        anchor = self.vector_store.get_chunk_by_id(chunk_id)
        if anchor is None:
            return []

        doc_id = anchor.doc_id
        if not doc_id:
            return [anchor]

        wanted = {
            anchor.chunk_index + offset
            for offset in range(-window, window + 1)
            if anchor.chunk_index + offset >= 0
        }
        return [
            chunk
            for chunk in self.vector_store.get_document_chunks(doc_id)
            if chunk.chunk_index in wanted
        ]


_retriever: IntelligentRetriever | None = None


def get_retriever() -> IntelligentRetriever:
    global _retriever
    if _retriever is None:
        _retriever = IntelligentRetriever()
    return _retriever
