"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal["general", "insurance_claim", "contract", "sop", "transaction_history"]
ChatMode = Literal["rag", "summary", "auto"]
SummaryLevel = Literal["chunk", "section", "master", "all"]
ExtractionType = Literal["fraud", "clauses", "risk", "entities", "all"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


# ----------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    warnings: list[str] = Field(default_factory=list)


class TokenUsageModel(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class CompletenessModel(BaseModel):
    confidence: Confidence = "HIGH"
    message: str = ""
    dropped_sections: list[str] = Field(default_factory=list)
    dropped_chunk_count: int = 0
    notes: list[str] = Field(default_factory=list)


class CitationModel(BaseModel):
    chunk_id: str
    text: str
    page: int = 0
    section: str = ""
    score: float = 0.0


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    llm_model: str
    embedding_backend: str
    embedding_dimension: int
    indexed_chunks: int
    indexed_documents: int
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------
class SummaryModel(BaseModel):
    doc_id: str
    master_summary: str = ""
    section_summaries: list[str] = Field(default_factory=list)
    chunk_summaries: list[str] = Field(default_factory=list)
    total_chunks_processed: int = 0
    levels: int = 0
    token_usage: TokenUsageModel = Field(default_factory=TokenUsageModel)
    processing_time_seconds: float = 0.0
    completeness_score: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class IngestionResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    total_pages: int
    total_chars: int
    total_tokens: int
    total_chunks: int
    chunks_stored: int
    summary: SummaryModel | None = None
    entities: dict[str, list[str]] | None = None
    ingestion_time_seconds: float
    status: str
    warnings: list[str] = Field(default_factory=list)
    context_overflow_factor: float = Field(
        default=0.0,
        description="How many times this document exceeds a baseline context window.",
    )


class DocumentInfoModel(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    chunk_count: int
    total_tokens: int = 0
    total_pages: int = 0


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfoModel] = Field(default_factory=list)
    total_chunks: int = 0


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    doc_id: str | None = None
    message: str = Field(..., min_length=1)
    mode: ChatMode = "auto"
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatTokenUsage(BaseModel):
    context_tokens: int = 0
    response_tokens: int = 0
    budget_utilization: float = 0.0
    token_budget: int = 0
    breakdown: dict[str, int] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: Confidence = "HIGH"
    completeness: CompletenessModel = Field(default_factory=CompletenessModel)
    token_usage: ChatTokenUsage = Field(default_factory=ChatTokenUsage)
    dropped_sections: list[str] = Field(default_factory=list)
    session_memory_summary: str = ""
    truncation_risks: list[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    warnings: list[str] = Field(default_factory=list)


class SessionStateResponse(BaseModel):
    session_id: str
    short_term_count: int
    mid_term_summary: str
    entity_store: dict[str, list[str]] = Field(default_factory=dict)
    total_exchanges: int


# ----------------------------------------------------------------------
# Summarize
# ----------------------------------------------------------------------
class SummarizeRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    level: SummaryLevel = "master"
    focus: str | None = None
    refresh: bool = Field(
        default=False, description="Recompute even when a cached summary exists."
    )


class TokenSavings(BaseModel):
    original_tokens: int = 0
    summary_tokens: int = 0
    compression_ratio: float = 0.0


class SummarizeResponse(BaseModel):
    doc_id: str
    master_summary: str = ""
    section_summaries: list[str] = Field(default_factory=list)
    chunk_summaries: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    levels_used: int = 0
    completeness_score: float = 0.0
    token_savings: TokenSavings = Field(default_factory=TokenSavings)
    processing_time_seconds: float = 0.0
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Extract
# ----------------------------------------------------------------------
class ExtractRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    extraction_type: ExtractionType = "all"
    client_profile: dict[str, Any] = Field(default_factory=dict)


class FraudIndicatorModel(BaseModel):
    type: str
    severity: Confidence
    evidence: str = ""
    page: int = 0
    section: str = ""
    explanation: str = ""


class FraudResultModel(BaseModel):
    doc_id: str
    indicators: list[FraudIndicatorModel] = Field(default_factory=list)
    overall_assessment: str = ""
    fraud_likelihood: Confidence = "LOW"
    chunks_analyzed: int = 0
    warnings: list[str] = Field(default_factory=list)


class ClauseModel(BaseModel):
    clause_type: str
    text: str = ""
    page: int = 0
    section: str = ""
    risk_rating: Confidence = "LOW"
    explanation: str = ""


class ClauseResultModel(BaseModel):
    doc_id: str
    clauses: list[ClauseModel] = Field(default_factory=list)
    summary: str = ""
    chunks_analyzed: int = 0
    warnings: list[str] = Field(default_factory=list)


class RiskFactorModel(BaseModel):
    factor: str
    severity: Confidence
    evidence: str = ""
    recommendation: str = ""


class RiskResultModel(BaseModel):
    doc_id: str
    risk_score: int = 0
    risk_band: str = "LOW"
    top_factors: list[RiskFactorModel] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    doc_id: str
    extraction_type: str
    fraud: FraudResultModel | None = None
    clauses: ClauseResultModel | None = None
    risk: RiskResultModel | None = None
    entities: dict[str, list[str]] | None = None
    warnings: list[str] = Field(default_factory=list)
