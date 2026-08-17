"""Internal data structures shared across core / ingestion / domain modules.

(The spec's file list doesn't name this module; the types it describes — ChunkResult,
SearchResult, SummaryResult, ContextPayload — need a single home that both the core
and the API layer can import without a cycle. API request/response models stay in
``backend/api/schemas.py``.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkResult:
    chunk_id: str
    text: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def section_name(self) -> str:
        return str(self.metadata.get("section_name", ""))

    @property
    def page(self) -> int:
        try:
            return int(self.metadata.get("page", 1))
        except (TypeError, ValueError):
            return 1

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    chunk: ChunkResult
    score: float
    doc_id: str
    chunk_id: str
    match_type: str = "semantic"  # semantic | keyword | hybrid

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "match_type": self.match_type,
            "text": self.chunk.text,
            "page": self.chunk.page,
            "section_name": self.chunk.section_name,
            "chunk_index": self.chunk.chunk_index,
        }


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    def merge(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class SummaryResult:
    doc_id: str
    master_summary: str
    section_summaries: list[str] = field(default_factory=list)
    chunk_summaries: list[str] = field(default_factory=list)
    total_chunks_processed: int = 0
    levels: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    processing_time_seconds: float = 0.0
    completeness_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    # chunk_index -> summary, so a chunk summary can be traced back to its source.
    chunk_summary_map: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "master_summary": self.master_summary,
            "section_summaries": self.section_summaries,
            "chunk_summaries": self.chunk_summaries,
            "total_chunks_processed": self.total_chunks_processed,
            "levels": self.levels,
            "token_usage": self.token_usage.to_dict(),
            "processing_time_seconds": self.processing_time_seconds,
            "completeness_score": round(self.completeness_score, 4),
            "warnings": self.warnings,
        }


@dataclass
class ContextPayload:
    system_prompt: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    included_chunks: list[SearchResult] = field(default_factory=list)
    dropped_chunks: list[SearchResult] = field(default_factory=list)
    total_tokens_used: int = 0
    token_budget: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def utilization_percent(self) -> float:
        if self.token_budget <= 0:
            return 0.0
        return round(100.0 * self.total_tokens_used / self.token_budget, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens_used": self.total_tokens_used,
            "token_budget": self.token_budget,
            "utilization_percent": self.utilization_percent,
            "included_chunks": len(self.included_chunks),
            "dropped_chunks": len(self.dropped_chunks),
            "breakdown": self.breakdown,
        }


@dataclass
class ConversationExchange:
    user_message: str
    assistant_response: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    session_id: str
    short_term_count: int
    mid_term_summary: str
    entity_store: dict[str, list[str]]
    total_exchanges: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "short_term_count": self.short_term_count,
            "mid_term_summary": self.mid_term_summary,
            "entity_store": self.entity_store,
            "total_exchanges": self.total_exchanges,
        }


@dataclass
class DocumentInfo:
    doc_id: str
    file_name: str
    doc_type: str
    chunk_count: int
    total_tokens: int = 0
    total_pages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
            "total_pages": self.total_pages,
        }


@dataclass
class TableResult:
    page: int
    markdown: str
    rows: int
    cols: int


@dataclass
class ParsedDocument:
    text: str
    file_name: str
    doc_type: str = "general"
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: list[TableResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return len(self.text)


@dataclass
class ClaudeResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class CompletenessReport:
    confidence: str = "HIGH"  # HIGH | MEDIUM | LOW
    message: str = ""
    dropped_sections: list[str] = field(default_factory=list)
    dropped_chunk_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "message": self.message,
            "dropped_sections": self.dropped_sections,
            "dropped_chunk_count": self.dropped_chunk_count,
            "notes": self.notes,
        }


@dataclass
class IngestionResult:
    doc_id: str
    file_name: str
    doc_type: str
    total_pages: int
    total_chars: int
    total_tokens: int
    total_chunks: int
    chunks_stored: int
    summary: SummaryResult | None = None
    entities: dict[str, list[str]] | None = None
    ingestion_time_seconds: float = 0.0
    status: str = "success"  # success | partial | failed
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "total_pages": self.total_pages,
            "total_chars": self.total_chars,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "chunks_stored": self.chunks_stored,
            "summary": self.summary.to_dict() if self.summary else None,
            "entities": self.entities,
            "ingestion_time_seconds": self.ingestion_time_seconds,
            "status": self.status,
            "warnings": self.warnings,
        }
