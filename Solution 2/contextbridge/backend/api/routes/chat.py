"""POST /api/chat — conversational Q&A grounded in an indexed document."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from backend import config
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTokenUsage,
    CitationModel,
    CompletenessModel,
    SessionStateResponse,
)
from backend.core.llm import get_claude_client
from backend.core.memory_manager import session_store
from backend.core.models import SearchResult
from backend.core.registry import document_registry
from backend.core.retriever import get_retriever
from backend.domain.completeness_checker import CompletenessChecker
from backend.utils.logger import logger

router = APIRouter(tags=["chat"])

_CITATION_RE = re.compile(r"\[CHUNK:\s*([A-Za-z0-9_\-]+)\s*\]")

# Queries that are about the document as a whole rather than a specific passage.
_SUMMARY_TRIGGERS = (
    "summar", "overview", "what is this document", "main points", "key points",
    "tl;dr", "gist", "high level", "in general", "overall",
)

checker = CompletenessChecker()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    client = get_claude_client()
    memory = session_store.get(request.session_id)
    retriever = get_retriever()
    warnings: list[str] = []

    mode = _resolve_mode(request.mode, request.message)
    top_k = request.top_k or config.TOP_K_CHUNKS

    # 1-2. retrieve ----------------------------------------------------
    try:
        retrieved = await retriever.retrieve(
            request.message,
            doc_id=request.doc_id,
            top_k=top_k,
            retrieval_mode="hybrid",
            referenced_sections=memory.referenced_sections,
        )
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        retrieved = []
        warnings.append(f"Retrieval failed: {exc}")

    # In summary mode, prepend the cached master summary as pseudo-context.
    if mode == "summary" and request.doc_id:
        summary = document_registry.get_summary(request.doc_id)
        if summary and summary.master_summary:
            retrieved = [_summary_as_result(request.doc_id, summary.master_summary)] + retrieved
        else:
            warnings.append(
                "No cached summary for this document — answering from retrieval only."
            )

    # 3. build the context payload within budget -----------------------
    payload = memory.build_context_payload(
        request.message, retrieved, token_budget=config.USABLE_CONTEXT_TOKENS
    )

    # 4. completeness audit --------------------------------------------
    # (Run before the LLM call so the report exists even if generation fails.)
    truncation_risks = checker.detect_truncation_risks(
        [r.chunk for r in payload.included_chunks]
    )

    # 5. call Claude ---------------------------------------------------
    if not client.available:
        reason = client.unavailable_reason() or "LLM unavailable"
        raise HTTPException(
            status_code=503,
            detail=(
                f"{reason} Retrieval still works — "
                f"{len(payload.included_chunks)} relevant chunks were found."
            ),
        )

    messages = payload.conversation_history + [
        {"role": "user", "content": request.message}
    ]
    response = await client.complete(
        prompt=request.message,
        system=payload.system_prompt,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        messages=messages,
    )

    if not response.ok:
        raise HTTPException(
            status_code=502, detail=f"Claude call failed: {response.error}"
        )

    answer = response.text.strip()

    # 6. parse citations ------------------------------------------------
    citations = _extract_citations(answer, payload.included_chunks)

    # 4b. completeness report on the finished answer --------------------
    report = checker.check_response_completeness(
        request.message,
        answer,
        payload.dropped_chunks,
        payload.included_chunks,
    )

    # 7. update memory ---------------------------------------------------
    try:
        await memory.add_exchange(request.message, answer, payload.included_chunks)
    except Exception as exc:
        logger.error(f"Memory update failed: {exc}")
        warnings.append(f"Memory update failed: {exc}")

    # 8. respond ---------------------------------------------------------
    return ChatResponse(
        answer=answer,
        citations=citations,
        confidence=report.confidence,
        completeness=CompletenessModel(**report.to_dict()),
        token_usage=ChatTokenUsage(
            context_tokens=payload.total_tokens_used,
            response_tokens=response.output_tokens,
            budget_utilization=payload.utilization_percent,
            token_budget=payload.token_budget,
            breakdown=payload.breakdown,
        ),
        dropped_sections=report.dropped_sections,
        session_memory_summary=memory.mid_term_summary,
        truncation_risks=truncation_risks,
        retrieval_mode=mode,
        warnings=warnings,
    )


@router.get("/session/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str) -> SessionStateResponse:
    state = session_store.get(session_id).get_session_summary()
    return SessionStateResponse(**state.to_dict())


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, object]:
    existed = session_store.drop(session_id)
    return {"session_id": session_id, "cleared": existed}


# ----------------------------------------------------------------------
def _resolve_mode(mode: str, message: str) -> str:
    if mode != "auto":
        return mode
    lowered = message.lower()
    return "summary" if any(t in lowered for t in _SUMMARY_TRIGGERS) else "rag"


def _summary_as_result(doc_id: str, summary_text: str) -> SearchResult:
    from backend.core.models import ChunkResult
    from backend.core.token_counter import token_counter

    chunk = ChunkResult(
        chunk_id=f"{doc_id}_master_summary",
        text=summary_text,
        token_count=token_counter.count(summary_text),
        char_start=0,
        char_end=len(summary_text),
        chunk_index=-1,
        total_chunks=0,
        metadata={
            "doc_id": doc_id,
            "section_name": "Document master summary",
            "page": 0,
        },
    )
    return SearchResult(
        chunk=chunk,
        score=1.0,
        doc_id=doc_id,
        chunk_id=chunk.chunk_id,
        match_type="summary",
    )


def _extract_citations(
    answer: str, included: list[SearchResult]
) -> list[CitationModel]:
    """Match [CHUNK: id] markers back to the chunks actually placed in context."""
    by_id = {r.chunk_id: r for r in included}
    cited_ids = []
    for match in _CITATION_RE.findall(answer):
        if match not in cited_ids:
            cited_ids.append(match)

    citations = [
        CitationModel(
            chunk_id=result.chunk_id,
            text=result.chunk.text[:1500],
            page=result.chunk.page,
            section=result.chunk.section_name,
            score=round(result.score, 4),
        )
        for chunk_id in cited_ids
        if (result := by_id.get(chunk_id)) is not None
    ]

    # If the model didn't cite anything, surface the top sources anyway so the
    # user can still verify the answer.
    if not citations:
        citations = [
            CitationModel(
                chunk_id=result.chunk_id,
                text=result.chunk.text[:1500],
                page=result.chunk.page,
                section=result.chunk.section_name,
                score=round(result.score, 4),
            )
            for result in included[:3]
        ]
    return citations
