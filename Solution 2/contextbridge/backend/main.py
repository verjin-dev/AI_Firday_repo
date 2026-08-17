"""FastAPI entry point.

    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.api.routes import chat, extract, health, summarize, upload
from backend.utils.logger import configure, logger


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure(config.LOG_LEVEL)
    config.ensure_dirs()
    logger.info("ContextBridge starting")

    # Warm the heavy singletons so the first request isn't a cold start.
    from backend.core.embedder import get_embedder
    from backend.core.llm import get_claude_client
    from backend.core.vector_store import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()
    client = get_claude_client()

    logger.info(
        f"Ready | embeddings={embedder.backend} | chunks={store.count()} | "
        f"llm={'on' if client.available else 'OFF'} model={config.CLAUDE_MODEL}"
    )
    if not client.available:
        logger.warning(
            "No ANTHROPIC_API_KEY — ingestion and retrieval work, but chat, "
            "summarization and extraction will return errors."
        )

    yield
    logger.info("ContextBridge shutting down")


app = FastAPI(
    title="ContextBridge",
    description=(
        "Overcoming LLM context window limitations for large Banking & Insurance "
        "documents via hierarchical summarization, RAG, and multi-tier memory."
    ),
    version=health.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(summarize.router, prefix="/api")
app.include_router(extract.router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a raw traceback to the frontend."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": f"{type(exc).__name__}: {exc}",
            "warnings": [],
        },
    )


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "name": "ContextBridge",
        "version": health.VERSION,
        "docs": "/docs",
        "endpoints": [
            "GET  /api/health",
            "POST /api/upload",
            "GET  /api/documents",
            "DELETE /api/documents/{doc_id}",
            "POST /api/chat",
            "GET  /api/session/{session_id}",
            "DELETE /api/session/{session_id}",
            "POST /api/summarize",
            "POST /api/extract",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
