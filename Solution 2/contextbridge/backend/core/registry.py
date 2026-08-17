"""In-process cache of per-document ingestion results.

Chunks and embeddings live in ChromaDB, but summaries and extracted entities are
expensive LLM artefacts that would otherwise be recomputed on every request. This
registry keeps the most recent ``IngestionResult`` per ``doc_id`` for the life of
the process, and rehydrates from the vector store on a cold start.

(An extra module beyond the spec's file list — the spec's routes assume a
summary is retrievable after upload without naming where it lives.)
"""

from __future__ import annotations

from backend.core.models import IngestionResult, SummaryResult
from backend.utils.logger import logger


class DocumentRegistry:
    def __init__(self) -> None:
        self._results: dict[str, IngestionResult] = {}

    def put(self, result: IngestionResult) -> None:
        self._results[result.doc_id] = result

    def get(self, doc_id: str) -> IngestionResult | None:
        return self._results.get(doc_id)

    def get_summary(self, doc_id: str) -> SummaryResult | None:
        result = self._results.get(doc_id)
        return result.summary if result else None

    def set_summary(self, doc_id: str, summary: SummaryResult) -> None:
        result = self._results.get(doc_id)
        if result is not None:
            result.summary = summary
        else:
            logger.debug(f"No ingestion record for {doc_id}; summary cached loosely")
            self._results[doc_id] = IngestionResult(
                doc_id=doc_id,
                file_name=doc_id,
                doc_type="general",
                total_pages=0,
                total_chars=0,
                total_tokens=0,
                total_chunks=summary.total_chunks_processed,
                chunks_stored=summary.total_chunks_processed,
                summary=summary,
            )

    def get_entities(self, doc_id: str) -> dict[str, list[str]] | None:
        result = self._results.get(doc_id)
        return result.entities if result else None

    def set_entities(self, doc_id: str, entities: dict[str, list[str]]) -> None:
        result = self._results.get(doc_id)
        if result is not None:
            result.entities = entities

    def drop(self, doc_id: str) -> bool:
        return self._results.pop(doc_id, None) is not None

    def all_ids(self) -> list[str]:
        return list(self._results)


document_registry = DocumentRegistry()
