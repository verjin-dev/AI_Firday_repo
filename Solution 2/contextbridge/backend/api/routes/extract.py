"""POST /api/extract — fraud flags, contract clauses, risk score, entities."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    ClauseModel,
    ClauseResultModel,
    ExtractRequest,
    ExtractResponse,
    FraudIndicatorModel,
    FraudResultModel,
    RiskFactorModel,
    RiskResultModel,
)
from backend.core.registry import document_registry
from backend.core.token_counter import token_counter
from backend.core.vector_store import get_vector_store
from backend.domain.banking_extractor import BankingExtractor
from backend.domain.entity_extractor import EntityExtractor
from backend.utils.logger import logger

router = APIRouter(tags=["extract"])


@router.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
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

    extractor = BankingExtractor()
    summary = document_registry.get_summary(request.doc_id)
    wanted = request.extraction_type
    warnings: list[str] = []

    fraud = clauses = risk = None
    entities = None

    if wanted in {"fraud", "risk", "all"}:
        fraud = await extractor.extract_fraud_indicators(request.doc_id, summary)
        warnings.extend(fraud.warnings)

    if wanted in {"clauses", "risk", "all"}:
        clauses = await extractor.extract_contract_clauses(request.doc_id)
        warnings.extend(clauses.warnings)

    if wanted in {"risk", "all"}:
        risk = await extractor.assess_risk(
            request.doc_id,
            client_profile=request.client_profile,
            fraud=fraud,
            clauses=clauses,
        )
        warnings.extend(risk.warnings)

    if wanted in {"entities", "all"}:
        entities = document_registry.get_entities(request.doc_id)
        if entities is None:
            source = (
                summary.master_summary
                if summary and summary.master_summary
                else "\n\n".join(c.text for c in chunks)
            )
            budgeted, _ = token_counter.truncate_to_budget(source, 20_000)
            entities = await EntityExtractor().extract(budgeted)
            document_registry.set_entities(request.doc_id, entities)

    return ExtractResponse(
        doc_id=request.doc_id,
        extraction_type=wanted,
        fraud=(
            FraudResultModel(
                doc_id=fraud.doc_id,
                indicators=[FraudIndicatorModel(**i.to_dict()) for i in fraud.indicators],
                overall_assessment=fraud.overall_assessment,
                fraud_likelihood=fraud.fraud_likelihood,
                chunks_analyzed=fraud.chunks_analyzed,
                warnings=fraud.warnings,
            )
            if fraud
            else None
        ),
        clauses=(
            ClauseResultModel(
                doc_id=clauses.doc_id,
                clauses=[ClauseModel(**c.to_dict()) for c in clauses.clauses],
                summary=clauses.summary,
                chunks_analyzed=clauses.chunks_analyzed,
                warnings=clauses.warnings,
            )
            if clauses
            else None
        ),
        risk=(
            RiskResultModel(
                doc_id=risk.doc_id,
                risk_score=risk.risk_score,
                risk_band=risk.risk_band,
                top_factors=[RiskFactorModel(**f.to_dict()) for f in risk.top_factors],
                summary=risk.summary,
                warnings=risk.warnings,
            )
            if risk
            else None
        ),
        entities=entities,
        warnings=warnings,
    )
