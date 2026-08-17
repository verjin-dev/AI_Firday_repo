"""ChromaDB wrapper: persistent cosine-similarity storage for document chunks."""

from __future__ import annotations

from typing import Any

from backend import config
from backend.core.models import ChunkResult, DocumentInfo, SearchResult
from backend.utils.helpers import sanitize_metadata
from backend.utils.logger import logger


class VectorStoreError(RuntimeError):
    """ChromaDB failed after a reconnect attempt."""


class VectorStore:
    """Chunk storage + semantic search. One collection holds every document."""

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._connect()

    # ------------------------------------------------------------------
    def _connect(self) -> None:
        import chromadb
        from chromadb.config import Settings

        from pathlib import Path

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready: '{self.collection_name}' "
            f"({self._collection.count()} chunks) at {self.persist_dir}"
        )

    def _reconnect(self) -> None:
        logger.warning("ChromaDB error — attempting one reconnect")
        self._connect()

    def _with_retry(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"ChromaDB operation failed: {exc}")
            try:
                self._reconnect()
                return operation(*args, **kwargs)
            except Exception as exc2:
                raise VectorStoreError(str(exc2)) from exc2

    # ------------------------------------------------------------------
    def add_chunks(
        self, chunks: list[ChunkResult], embeddings: list[list[float]]
    ) -> int:
        """Upsert chunks with their vectors. Returns the number stored."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            sanitize_metadata(
                {
                    **c.metadata,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "token_count": c.token_count,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
            )
            for c in chunks
        ]

        def _upsert() -> None:
            self._collection.upsert(  # type: ignore[union-attr]
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        try:
            self._with_retry(_upsert)
        except VectorStoreError as exc:
            # Switching embedding backends changes vector width; an existing
            # collection built with the old width rejects the new vectors.
            if "dimension" in str(exc).lower():
                raise VectorStoreError(
                    f"Embedding dimension mismatch ({len(embeddings[0])}d vectors) "
                    f"against collection '{self.collection_name}'. This happens when "
                    "the embedding backend changes. Delete the persisted store "
                    f"({self.persist_dir}) and re-ingest, or set EMBEDDING_BASE_URL "
                    "back to its previous value."
                ) from exc
            raise
        logger.info(f"Stored {len(ids)} chunks in ChromaDB")
        return len(ids)

    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        doc_id: str | None = None,
        top_k: int = config.TOP_K_CHUNKS,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> list[SearchResult]:
        """Cosine search, optionally scoped to one document."""
        where = {"doc_id": doc_id} if doc_id else None

        def _query():
            return self._collection.query(  # type: ignore[union-attr]
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        raw = self._with_retry(_query)
        return self._to_results(raw, threshold)

    def _to_results(self, raw: dict, threshold: float) -> list[SearchResult]:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for i, chunk_id in enumerate(ids):
            metadata = dict(metadatas[i] or {})
            # Chroma cosine distance is 1 - cosine_similarity.
            score = 1.0 - float(distances[i])
            if score < threshold:
                continue
            results.append(
                SearchResult(
                    chunk=self._to_chunk(chunk_id, documents[i], metadata),
                    score=score,
                    doc_id=str(metadata.get("doc_id", "")),
                    chunk_id=str(chunk_id),
                    match_type="semantic",
                )
            )
        return results

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> ChunkResult:
        def _int(key: str, default: int = 0) -> int:
            try:
                return int(metadata.get(key, default))
            except (TypeError, ValueError):
                return default

        return ChunkResult(
            chunk_id=str(chunk_id),
            text=text or "",
            token_count=_int("token_count"),
            char_start=_int("char_start"),
            char_end=_int("char_end"),
            chunk_index=_int("chunk_index"),
            total_chunks=_int("total_chunks"),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk for ``doc_id``. Returns the count removed."""

        def _count() -> int:
            existing = self._collection.get(where={"doc_id": doc_id}, include=[])  # type: ignore[union-attr]
            return len(existing.get("ids") or [])

        count = self._with_retry(_count)
        if count:
            self._with_retry(
                lambda: self._collection.delete(where={"doc_id": doc_id})  # type: ignore[union-attr]
            )
        logger.info(f"Deleted {count} chunks for {doc_id}")
        return count

    def list_documents(self) -> list[DocumentInfo]:
        """One entry per unique ``doc_id`` with aggregate counts."""

        def _all():
            return self._collection.get(include=["metadatas"])  # type: ignore[union-attr]

        raw = self._with_retry(_all)
        metadatas = raw.get("metadatas") or []

        grouped: dict[str, DocumentInfo] = {}
        pages: dict[str, int] = {}
        for metadata in metadatas:
            metadata = metadata or {}
            doc_id = str(metadata.get("doc_id", "")).strip()
            if not doc_id:
                continue
            info = grouped.get(doc_id)
            if info is None:
                info = DocumentInfo(
                    doc_id=doc_id,
                    file_name=str(metadata.get("file_name", doc_id)),
                    doc_type=str(metadata.get("doc_type", "general")),
                    chunk_count=0,
                )
                grouped[doc_id] = info
            info.chunk_count += 1
            try:
                info.total_tokens += int(metadata.get("token_count", 0))
            except (TypeError, ValueError):
                pass
            try:
                pages[doc_id] = max(pages.get(doc_id, 0), int(metadata.get("page", 1)))
            except (TypeError, ValueError):
                pass

        for doc_id, info in grouped.items():
            info.total_pages = pages.get(doc_id, 0)
        return sorted(grouped.values(), key=lambda d: d.file_name.lower())

    def get_chunk_by_id(self, chunk_id: str) -> ChunkResult | None:
        """Direct lookup, used for citation display."""

        def _get():
            return self._collection.get(  # type: ignore[union-attr]
                ids=[chunk_id], include=["documents", "metadatas"]
            )

        raw = self._with_retry(_get)
        ids = raw.get("ids") or []
        if not ids:
            return None
        documents = raw.get("documents") or [""]
        metadatas = raw.get("metadatas") or [{}]
        return self._to_chunk(ids[0], documents[0], dict(metadatas[0] or {}))

    def get_document_chunks(self, doc_id: str) -> list[ChunkResult]:
        """Every chunk for a document, ordered by ``chunk_index``."""

        def _get():
            return self._collection.get(  # type: ignore[union-attr]
                where={"doc_id": doc_id}, include=["documents", "metadatas"]
            )

        raw = self._with_retry(_get)
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        chunks = [
            self._to_chunk(ids[i], documents[i], dict(metadatas[i] or {}))
            for i in range(len(ids))
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def count(self) -> int:
        return int(self._with_retry(lambda: self._collection.count()))  # type: ignore[union-attr]

    def reset(self) -> None:
        """Drop and recreate the collection (used by tests)."""
        try:
            self._client.delete_collection(self.collection_name)  # type: ignore[union-attr]
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
