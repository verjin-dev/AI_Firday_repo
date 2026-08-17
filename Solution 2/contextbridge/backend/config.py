"""Central configuration. Every value is env-overridable with a sensible default."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the directory containing backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# override=True: the project's .env is authoritative. Without it an ambient
# ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY in the developer's shell silently wins
# and requests go to the wrong endpoint.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


def _strip_v1(url: str) -> str:
    """The Anthropic SDK appends its own `/v1`; a configured `/v1` would double it."""
    url = url.rstrip("/")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


def _with_v1(url: str) -> str:
    """OpenAI-compatible routes are called as `{base}/embeddings`, so keep `/v1`."""
    url = url.rstrip("/")
    return url if not url or url.endswith("/v1") else f"{url}/v1"


# Point at any Anthropic-compatible endpoint (e.g. a local router exposing
# /v1/messages). Either `http://host:port` or `http://host:port/v1` works.
# Leave unset to use api.anthropic.com directly.
ANTHROPIC_BASE_URL: str = _strip_v1(os.getenv("ANTHROPIC_BASE_URL", ""))

CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Summarization fires one call per chunk. Routing those to a cheaper/faster model
# than the chat model is the single biggest cost lever in the pipeline.
SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL", "") or CLAUDE_MODEL

MAX_OUTPUT_TOKENS: int = _int("MAX_OUTPUT_TOKENS", 4096)

# Smaller cap for the many short map-phase calls — keeps summarisation fast/cheap.
SUMMARY_MAX_OUTPUT_TOKENS: int = _int("SUMMARY_MAX_OUTPUT_TOKENS", 1024)

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE: int = _int("CHUNK_SIZE", 800)          # tokens per chunk
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 80)     # 10% overlap
MIN_CHUNK_SIZE: int = _int("MIN_CHUNK_SIZE", 100)  # discard smaller chunks

# RecursiveCharacterTextSplitter works in characters; ~4 chars/token for English.
CHARS_PER_TOKEN: int = _int("CHARS_PER_TOKEN", 4)

# --------------------------------------------------------------------------
# Context window budget
# --------------------------------------------------------------------------
CONTEXT_BUDGET_TOKENS: int = _int("CONTEXT_BUDGET_TOKENS", 150_000)
RESPONSE_RESERVE_TOKENS: int = _int("RESPONSE_RESERVE_TOKENS", 4096)
USABLE_CONTEXT_TOKENS: int = CONTEXT_BUDGET_TOKENS - RESPONSE_RESERVE_TOKENS

# Baseline used by the demo/benchmark to represent a "small context window" model.
BASELINE_CONTEXT_TOKENS: int = _int("BASELINE_CONTEXT_TOKENS", 8000)

# --------------------------------------------------------------------------
# Memory tiers
# --------------------------------------------------------------------------
SHORT_TERM_EXCHANGES: int = _int("SHORT_TERM_EXCHANGES", 5)
MID_TERM_SUMMARY_EXCHANGES: int = _int("MID_TERM_SUMMARY_EXCHANGES", 20)
ENTITY_STORE_MAX_ENTRIES: int = _int("ENTITY_STORE_MAX_ENTRIES", 100)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K_CHUNKS: int = _int("TOP_K_CHUNKS", 8)
SIMILARITY_THRESHOLD: float = _float("SIMILARITY_THRESHOLD", 0.35)
KEYWORD_WEIGHT: float = _float("KEYWORD_WEIGHT", 0.35)   # hybrid blend
SEMANTIC_WEIGHT: float = _float("SEMANTIC_WEIGHT", 0.65)
RERANK_SECTION_BOOST: float = _float("RERANK_SECTION_BOOST", 0.05)

# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------
SUMMARY_CHUNK_BATCH_SIZE: int = _int("SUMMARY_CHUNK_BATCH_SIZE", 5)
MAX_SUMMARY_LEVELS: int = _int("MAX_SUMMARY_LEVELS", 3)
SUMMARY_CONCURRENCY: int = _int("SUMMARY_CONCURRENCY", 6)
REDUCE_BATCH_TOKEN_BUDGET: int = _int("REDUCE_BATCH_TOKEN_BUDGET", 12_000)

# --------------------------------------------------------------------------
# ChromaDB
# --------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db")
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "contextbridge_docs")

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = _int("EMBEDDING_DIMENSION", 384)
EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 32)

# Optional OpenAI-compatible embeddings endpoint (POST {base}/embeddings).
# When set it takes priority over the local backends — no model download needed.
EMBEDDING_BASE_URL: str = _with_v1(os.getenv("EMBEDDING_BASE_URL", ""))
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "") or ANTHROPIC_API_KEY
EMBEDDING_REMOTE_MODEL: str = os.getenv("EMBEDDING_REMOTE_MODEL", "auto")

# --------------------------------------------------------------------------
# Paths / misc
# --------------------------------------------------------------------------
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "data" / "uploads"))
SAMPLE_DOCS_DIR: str = str(PROJECT_ROOT / "data" / "sample_docs")
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]


def has_api_key() -> bool:
    """True when an Anthropic key is configured. Callers degrade gracefully if not."""
    return bool(ANTHROPIC_API_KEY.strip())


def ensure_dirs() -> None:
    for path in (CHROMA_PERSIST_DIR, UPLOAD_DIR, SAMPLE_DOCS_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
