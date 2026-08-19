"""FastAPI entry point."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import burn, economics, health, live, router_api
from backend.config import get_settings
from backend.storage.db import get_engine, table_counts
from backend.utils.errors import AppError
from backend.utils.logger import log

app = FastAPI(
    title="TokenOps",
    description="FinOps for agentic AI: cost per business outcome, a learning "
                "router, and burn-rate alerting that catches a runaway agent in minutes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(economics.router, prefix="/api")
app.include_router(router_api.router, prefix="/api")
app.include_router(burn.router, prefix="/api")
app.include_router(live.router, prefix="/api")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """No exception reaches the client raw."""
    log.warning(f"{exc.code} on {request.url.path}: {exc.message}")
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception(f"unhandled error on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"status": "failed", "code": "internal_error",
                 "message": "internal error", "details": {"path": str(request.url.path)}},
    )


@app.on_event("startup")
def startup() -> None:
    get_engine()
    counts = table_counts()
    s = get_settings()
    log.info(f"TokenOps API up | price table {s.PRICE_TABLE_VERSION} | tables {counts}")
    if not counts.get("llm_calls"):
        log.warning("ledger is empty - run: python scripts/simulate_workload.py")


@app.get("/")
def root() -> dict:
    return {"status": "success", "service": "tokenops", "docs": "/docs", "health": "/api/health"}
