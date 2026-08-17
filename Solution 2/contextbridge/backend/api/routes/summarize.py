"""POST /api/summarize — hierarchical summary of an indexed document."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import SummarizeRequest, SummarizeResponse, TokenSavings
from backend.core.registry import document_registry
from backend.core.retriever import get_retriever
from backend.core.summarizer import HierarchicalSummarizer
from backend.core.token_counter import token_counter
from backend.core.vector_store import get_vector_store
from backend.utils.logger import logger

router = APIRouter(tags=["summarize"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    store = get_vector_store()

    try:
        chunks = store.get_document_chunks(request.doc_id)
    except Exception as exc:
        logger.error(f"Chunk load failed for {request.doc_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if not chunks:
        raise HTTPException(
            status_code=404, detail=f"No indexed chunks for doc_id '{request.doc_id}'"
        )

    cached = document_registry.get_summary(request.doc_id)
    reuse = (
        cached is not None
        and not request.refresh
        and not request.focus
        and cached.master_summary
    )

    if reuse:
        summary = cached  # type: ignore[assignment]
        was_cached = True
    else:
        was_cached = False
        record = document_registry.get(request.doc_id)
        doc_type = record.doc_type if record else "general"

        if request.focus:
            # Focused summary: narrow to chunks relevant to the topic first.
            try:
                hits = await get_retriever().retrieve_for_summary(
                    request.focus, request.doc_id
                )
            except Exception as exc:
                logger.warning(f"Focused retrieval failed ({exc}); using all chunks")
                hits = []
            if hits:
                wanted = {h.chunk_id for h in hits}
                focused = [c for c in chunks if c.chunk_id in wanted]
                if focused:
                    chunks = focused

        summary = await HierarchicalSummarizer().summarize_document(
            chunks, doc_type=doc_type, focus=request.focus
        )
        if not request.focus:
            document_registry.set_summary(request.doc_id, summary)

    original_tokens = sum(c.token_count for c in chunks)
    summary_tokens = token_counter.count(summary.master_summary)
    ratio = (
        1.0 - (summary_tokens / original_tokens) if original_tokens else 0.0
    )

    level = request.level
    return SummarizeResponse(
        doc_id=request.doc_id,
        master_summary=summary.master_summary if level in {"master", "all"} else "",
        section_summaries=(
            summary.section_summaries if level in {"section", "all"} else []
        ),
        chunk_summaries=summary.chunk_summaries if level in {"chunk", "all"} else [],
        chunk_count=len(chunks),
        levels_used=summary.levels,
        completeness_score=summary.completeness_score,
        token_savings=TokenSavings(
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            compression_ratio=round(max(0.0, ratio), 4),
        ),
        processing_time_seconds=summary.processing_time_seconds,
        cached=was_cached,
        warnings=summary.warnings,
    )
