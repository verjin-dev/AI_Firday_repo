"""GET /api/health"""

from __future__ import annotations

from fastapi import APIRouter

from backend import config
from backend.api.schemas import HealthResponse
from backend.core.embedder import get_embedder
from backend.core.llm import get_claude_client
from backend.core.vector_store import get_vector_store
from backend.utils.logger import logger

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    warnings: list[str] = []

    client = get_claude_client()
    if not client.available:
        warnings.append(client.unavailable_reason() or "LLM unavailable")

    embedder = get_embedder()
    if embedder.backend == "hash":
        warnings.append(
            "Embeddings are using the hashing fallback — semantic retrieval quality "
            "is degraded. Install sentence-transformers or allow ONNX model download."
        )

    chunks = 0
    documents = 0
    try:
        store = get_vector_store()
        chunks = store.count()
        documents = len(store.list_documents())
    except Exception as exc:
        logger.error(f"Health check: vector store unreachable: {exc}")
        warnings.append(f"Vector store unreachable: {exc}")

    return HealthResponse(
        status="degraded" if warnings else "ok",
        version=VERSION,
        llm_available=client.available,
        llm_model=config.CLAUDE_MODEL,
        embedding_backend=embedder.backend,
        embedding_dimension=embedder.dimension,
        indexed_chunks=chunks,
        indexed_documents=documents,
        warnings=warnings,
    )
