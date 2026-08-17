"""POST /api/upload — ingest a document. Also lists and deletes documents."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend import config
from backend.api.schemas import (
    DeleteResponse,
    DocumentInfoModel,
    DocumentListResponse,
    IngestionResponse,
    SummaryModel,
    TokenUsageModel,
)
from backend.core.models import IngestionResult
from backend.core.registry import document_registry
from backend.core.retriever import get_retriever
from backend.core.vector_store import get_vector_store
from backend.ingestion.pipeline import get_pipeline
from backend.utils.helpers import make_doc_id
from backend.utils.logger import logger

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def to_response(result: IngestionResult) -> IngestionResponse:
    summary = None
    if result.summary:
        summary = SummaryModel(
            doc_id=result.summary.doc_id,
            master_summary=result.summary.master_summary,
            section_summaries=result.summary.section_summaries,
            chunk_summaries=result.summary.chunk_summaries,
            total_chunks_processed=result.summary.total_chunks_processed,
            levels=result.summary.levels,
            token_usage=TokenUsageModel(**result.summary.token_usage.to_dict()),
            processing_time_seconds=result.summary.processing_time_seconds,
            completeness_score=result.summary.completeness_score,
            warnings=result.summary.warnings,
        )

    return IngestionResponse(
        doc_id=result.doc_id,
        file_name=result.file_name,
        doc_type=result.doc_type,
        total_pages=result.total_pages,
        total_chars=result.total_chars,
        total_tokens=result.total_tokens,
        total_chunks=result.total_chunks,
        chunks_stored=result.chunks_stored,
        summary=summary,
        entities=result.entities,
        ingestion_time_seconds=result.ingestion_time_seconds,
        status=result.status,
        warnings=result.warnings,
        context_overflow_factor=get_pipeline().context_overflow_factor(
            result.total_tokens
        ),
    )


@router.post("/upload", response_model=IngestionResponse)
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    run_summarization: bool = Form(True),
) -> IngestionResponse:
    """Ingest synchronously — the demo needs the result in the same response."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename supplied.")

    pipeline = get_pipeline()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in pipeline.supported_extensions():
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. Supported: "
                f"{', '.join(pipeline.supported_extensions())}"
            ),
        )

    upload_dir = Path(config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = make_doc_id(file.filename)
    # Never trust the client-supplied path — use our own generated stem.
    destination = upload_dir / f"{doc_id}{suffix}"

    try:
        size = 0
        with destination.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to save upload {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}")
    finally:
        await file.close()

    try:
        result = await pipeline.ingest(
            str(destination),
            doc_type=doc_type,
            run_summarization=run_summarization,
            doc_id=doc_id,
            original_name=file.filename,
        )
    except Exception as exc:
        logger.exception(f"Ingestion crashed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    document_registry.put(result)

    if result.status == "failed":
        logger.error(f"Ingestion failed for {file.filename}: {result.warnings}")

    return to_response(result)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    store = get_vector_store()
    try:
        documents = store.list_documents()
        total = store.count()
    except Exception as exc:
        logger.error(f"Listing documents failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return DocumentListResponse(
        documents=[DocumentInfoModel(**d.to_dict()) for d in documents],
        total_chunks=total,
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str) -> DeleteResponse:
    store = get_vector_store()
    try:
        deleted = store.delete_document(doc_id)
    except Exception as exc:
        logger.error(f"Delete failed for {doc_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    document_registry.drop(doc_id)
    get_retriever().invalidate_cache(doc_id)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id}")
    return DeleteResponse(doc_id=doc_id, chunks_deleted=deleted)
