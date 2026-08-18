"""Central configuration. Everything price- or policy-related lives here so it
is versioned in one place (§5 of the blueprint)."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM (optional: TokenOps runs fully offline on simulated telemetry) ----
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_FAST_MODEL: str = "claude-haiku-4-5-20251001"
    MAX_OUTPUT_TOKENS: int = 4096
    TEMPERATURE: float = 0.0
    OFFLINE: bool = True

    # ---- storage ----
    SQLITE_URL: str = "sqlite:///./data/db/tokenops.db"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8013
    LOG_LEVEL: str = "INFO"

    # ---- pricing: USD per 1M tokens. Versioned; see PRICE_TABLE_VERSION ----
    PRICE_TABLE_VERSION: str = "2026-08-01"
    USD_INR: float = 87.0

    # ---- unit economics ----
    OUTCOME_TYPES: List[str] = [
        "ticket_resolved",
        "claim_adjudicated",
        "document_processed",
        "lead_qualified",
    ]

    # ---- learning router ----
    BANDIT_LAMBDA: float = 0.4
    BANDIT_EXPLORATION_BUDGET_PCT: float = 5.0
    BANDIT_MIN_SAMPLES_PER_ARM: int = 30
    QUALITY_FLOOR: float = 0.80

    # ---- optimisers ----
    SEMANTIC_CACHE_THRESHOLD: float = 0.94
    PROMPT_COMPRESSION_TARGET: float = 0.55
    CASCADE_ESCALATION_CONFIDENCE: float = 0.75

    # ---- burn rate / guardrails ----
    BURN_RATE_WINDOWS: List[Tuple[int, float]] = [(1, 14.4), (6, 6.0), (24, 3.0)]
    CIRCUIT_BREAKER_MULTIPLIER: float = 25.0
    LOOP_DETECTION_REPEAT_THRESHOLD: int = 4

    # ---- simulation ----
    SIM_DAYS: int = 30
    SIM_SESSIONS_PER_DAY: int = 1200
    SIM_SEED: int = 13
    MONTHLY_BUDGET_INR: float = 1_800_000.0


# Price table is a module constant, not env-configurable: it must be identical
# for both benchmark arms or the comparison is meaningless.
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "claude-opus-5": {"in": 15.00, "out": 75.00, "cache_read": 1.50, "cache_write": 18.75},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5-20251001": {"in": 0.80, "out": 4.00, "cache_read": 0.08, "cache_write": 1.00},
}

MODEL_SHORT = {
    "claude-opus-5": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
