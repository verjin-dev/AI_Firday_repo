"""Domain-agnostic entity extraction.

LLM-based extraction with a regex pre-pass. The regex pass is not a fallback bolted
on for failure — it runs always, so amounts, dates and identifiers are never lost to
a model omission, and it keeps extraction useful with no API key at all.
"""

from __future__ import annotations

import re

from backend.core.llm import ClaudeClient, get_claude_client
from backend.core.memory_manager import ENTITY_KEYS
from backend.core.token_counter import token_counter
from backend.utils.helpers import dedupe_preserving_order, extract_json
from backend.utils.logger import logger

EXTRACTION_PROMPT = """Extract structured entities from the document text below.

Return ONLY a JSON object with exactly these keys, each mapping to an array of
strings (empty array when nothing applies):
{{
  "people": [], "organizations": [], "dates": [], "amounts": [],
  "locations": [], "decisions": [], "risks": [], "claim_ids": []
}}

Rules:
- Copy values verbatim from the text; never paraphrase identifiers or figures.
- "decisions" = concrete determinations made (approvals, denials, settlements).
- "risks" = stated risks, exposures, anomalies or red flags.
- "claim_ids" = claim numbers, policy numbers, case references, contract numbers.

TEXT:
{text}

JSON:"""

_AMOUNT_RE = re.compile(r"(?:USD\s*)?[$€£]\s?\d[\d,]*(?:\.\d{2})?(?:\s?(?:million|bn|billion|k))?", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4})\b"
)
_ID_RE = re.compile(
    r"\b(?:POL|CLM|CL|CASE|REF|CTR|ACCT|TXN)[-–/]?[A-Z0-9]{3,}\b", re.IGNORECASE
)


class EntityExtractor:
    """Extracts named entities, figures, decisions and risks from text."""

    def __init__(self, client: ClaudeClient | None = None) -> None:
        self.client = client or get_claude_client()

    async def extract(self, text: str, max_input_tokens: int = 12_000) -> dict[str, list[str]]:
        """Regex pass merged with an LLM pass. Never raises."""
        entities: dict[str, list[str]] = {key: [] for key in ENTITY_KEYS}
        if not text or not text.strip():
            return entities

        entities = self._merge(entities, self.extract_regex(text))

        if not self.client.available:
            logger.info("Entity extraction: regex only (no API key configured)")
            return entities

        budgeted, _ = token_counter.truncate_to_budget(text, max_input_tokens)
        response = await self.client.complete(
            EXTRACTION_PROMPT.format(text=budgeted),
            system="You extract structured data and return only valid JSON.",
            # Entity lists on a long document run long; too small a cap truncates
            # the JSON mid-write and costs us the tail of every category.
            max_tokens=4000,
        )
        if not response.ok:
            logger.warning(f"LLM entity extraction failed: {response.error}")
            return entities

        parsed = extract_json(response.text)
        if not isinstance(parsed, dict):
            logger.warning("LLM entity extraction returned unparseable output")
            return entities

        llm_entities = {
            key: [str(v).strip() for v in parsed.get(key, []) if str(v).strip()]
            if isinstance(parsed.get(key), list)
            else []
            for key in ENTITY_KEYS
        }
        return self._merge(entities, llm_entities)

    # ------------------------------------------------------------------
    @staticmethod
    def extract_regex(text: str) -> dict[str, list[str]]:
        """Deterministic extraction of amounts, dates and identifiers."""
        found: dict[str, list[str]] = {key: [] for key in ENTITY_KEYS}
        found["amounts"] = dedupe_preserving_order(_AMOUNT_RE.findall(text))[:60]
        found["dates"] = dedupe_preserving_order(_DATE_RE.findall(text))[:60]
        found["claim_ids"] = dedupe_preserving_order(_ID_RE.findall(text))[:60]
        return found

    @staticmethod
    def _merge(
        base: dict[str, list[str]], extra: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        return {
            key: dedupe_preserving_order(base.get(key, []) + extra.get(key, []))
            for key in ENTITY_KEYS
        }
