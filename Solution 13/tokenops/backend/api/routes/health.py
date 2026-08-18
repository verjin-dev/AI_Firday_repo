from __future__ import annotations

from fastapi import APIRouter

from backend.api.schemas import HealthResponse
from backend.config import get_settings
from backend.storage.db import has_data, table_counts

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        price_table_version=s.PRICE_TABLE_VERSION,
        tables=table_counts(),
        has_data=has_data(),
        offline=s.OFFLINE,
    )
