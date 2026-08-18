"""Exception hierarchy (§9 of the common foundation)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    code = "app_error"
    http_status = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "failed",
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class LLMError(AppError):
    code = "llm_error"


class StructuredOutputError(LLMError):
    code = "structured_output_error"


class BudgetExceededError(AppError):
    code = "budget_exceeded"
    http_status = 429


class LedgerError(AppError):
    code = "ledger_error"
    http_status = 400


class PolicyViolationError(AppError):
    code = "policy_violation"
    http_status = 403


class NoDataError(AppError):
    code = "no_data"
    http_status = 404
