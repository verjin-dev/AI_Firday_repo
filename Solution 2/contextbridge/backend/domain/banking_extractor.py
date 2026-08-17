"""Banking & Insurance domain analysis: fraud flags, contract clauses, risk scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend import config
from backend.core.llm import ClaudeClient, get_claude_client
from backend.core.models import SummaryResult
from backend.core.retriever import IntelligentRetriever, get_retriever
from backend.core.token_counter import token_counter
from backend.utils.helpers import extract_json
from backend.utils.logger import logger

SEVERITIES = ("HIGH", "MEDIUM", "LOW")

FRAUD_PROMPT = """You are a fraud analyst reviewing an insurance/banking document.

Identify fraud indicators. Look specifically for:
- inconsistent or contradictory dates
- duplicate or previously-filed claims for the same loss
- inflated, rounded, or unsupported amounts
- suspicious third parties (repeat contractors, related-party vendors)
- timeline anomalies (loss reported before it occurred, gaps, back-dating)
- policy violations (coverage lapses, undisclosed material facts)
- structuring: repeated transactions just below a reporting threshold

Return ONLY JSON:
{{
  "indicators": [
    {{
      "type": "short label",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "evidence": "verbatim quote from the document",
      "page": <integer page number, or 0 if unknown>,
      "section": "section name if identifiable, else empty string",
      "explanation": "one or two sentences on why this is suspicious"
    }}
  ],
  "overall_assessment": "two or three sentences",
  "fraud_likelihood": "HIGH" | "MEDIUM" | "LOW"
}}

If you find no indicators, return an empty "indicators" array and say so in the
assessment. Do not invent evidence — every quote must appear in the source.

DOCUMENT CONTENT:
{content}

JSON:"""

CLAUSE_PROMPT = """You are a contracts lawyer reviewing a commercial agreement.

Extract these clause types where present: indemnification, liability caps,
termination, payment terms, jurisdiction / governing law, force majeure,
intellectual property rights, confidentiality, service levels (SLA).

Return ONLY JSON:
{{
  "clauses": [
    {{
      "clause_type": "one of the types above",
      "text": "verbatim quote of the operative language",
      "page": <integer page number, or 0 if unknown>,
      "section": "section name/number if identifiable, else empty string",
      "risk_rating": "HIGH" | "MEDIUM" | "LOW",
      "explanation": "plain-English explanation of what it means and why the risk rating"
    }}
  ],
  "summary": "two or three sentences on the contract's overall risk posture"
}}

Rate risk HIGH for uncapped/unlimited liability, one-sided indemnities, unusual
or inconvenient jurisdictions, unilateral termination, or perpetual obligations.
Do not invent clause text.

DOCUMENT CONTENT:
{content}

JSON:"""

RISK_PROMPT = """You are a risk officer producing an overall assessment.

Combine the fraud indicators, contract clause analysis and client profile below
into a single risk view.

Return ONLY JSON:
{{
  "risk_score": <integer 0-100, where 100 is maximum risk>,
  "risk_band": "LOW" | "MODERATE" | "ELEVATED" | "SEVERE",
  "top_factors": [
    {{
      "factor": "short label",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "evidence": "specific supporting detail from the inputs",
      "recommendation": "concrete next action"
    }}
  ],
  "summary": "three or four sentences"
}}

Return at most 5 factors, ordered most severe first.

FRAUD INDICATORS:
{fraud}

CONTRACT CLAUSES:
{clauses}

CLIENT PROFILE:
{profile}

JSON:"""

_FRAUD_QUERIES = [
    "prior claims history previous claim filed before",
    "duplicate claim same property different policy",
    "inconsistent dates timeline discrepancy",
    "suspicious contractor third party vendor",
    "amount discrepancy inflated estimate valuation",
    "policy violation coverage lapse non-disclosure",
    "transactions below reporting threshold structuring",
]

_CLAUSE_QUERIES = [
    "indemnification hold harmless",
    "limitation of liability cap damages",
    "termination for convenience notice period",
    "payment terms invoice net days late fee",
    "governing law jurisdiction venue dispute",
    "force majeure act of god",
    "intellectual property ownership license rights",
    "confidentiality non-disclosure",
    "service level agreement uptime credits",
]


@dataclass
class FraudIndicator:
    type: str
    severity: str
    evidence: str
    page: int = 0
    section: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "evidence": self.evidence,
            "page": self.page,
            "section": self.section,
            "explanation": self.explanation,
        }


@dataclass
class FraudAnalysisResult:
    doc_id: str
    indicators: list[FraudIndicator] = field(default_factory=list)
    overall_assessment: str = ""
    fraud_likelihood: str = "LOW"
    chunks_analyzed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "indicators": [i.to_dict() for i in self.indicators],
            "overall_assessment": self.overall_assessment,
            "fraud_likelihood": self.fraud_likelihood,
            "chunks_analyzed": self.chunks_analyzed,
            "warnings": self.warnings,
        }


@dataclass
class ContractClause:
    clause_type: str
    text: str
    page: int = 0
    section: str = ""
    risk_rating: str = "LOW"
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_type": self.clause_type,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "risk_rating": self.risk_rating,
            "explanation": self.explanation,
        }


@dataclass
class ContractClauseResult:
    doc_id: str
    clauses: list[ContractClause] = field(default_factory=list)
    summary: str = ""
    chunks_analyzed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "clauses": [c.to_dict() for c in self.clauses],
            "summary": self.summary,
            "chunks_analyzed": self.chunks_analyzed,
            "warnings": self.warnings,
        }


@dataclass
class RiskFactor:
    factor: str
    severity: str
    evidence: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "severity": self.severity,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class RiskAssessmentResult:
    doc_id: str
    risk_score: int = 0
    risk_band: str = "LOW"
    top_factors: list[RiskFactor] = field(default_factory=list)
    summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "top_factors": [f.to_dict() for f in self.top_factors],
            "summary": self.summary,
            "warnings": self.warnings,
        }


class BankingExtractor:
    """Domain analysis over indexed documents."""

    def __init__(
        self,
        client: ClaudeClient | None = None,
        retriever: IntelligentRetriever | None = None,
    ) -> None:
        self.client = client or get_claude_client()
        self.retriever = retriever or get_retriever()

    # ------------------------------------------------------------------
    async def extract_fraud_indicators(
        self,
        doc_id: str,
        summary: SummaryResult | None = None,
        max_content_tokens: int | None = None,
    ) -> FraudAnalysisResult:
        """Analyse the hierarchical summary plus targeted retrieval for red flags.

        Retrieval matters here: a fraud indicator planted late in a long document
        is exactly what a summary can flatten, so we pull the highest-signal raw
        chunks alongside the summary rather than trusting compression alone.
        """
        max_content_tokens = max_content_tokens or config.DOMAIN_ANALYSIS_CONTENT_TOKENS
        warnings: list[str] = []
        if not self.client.available:
            return FraudAnalysisResult(
                doc_id=doc_id,
                overall_assessment="LLM unavailable — configure ANTHROPIC_API_KEY.",
                warnings=[self.client.unavailable_reason() or "LLM unavailable"],
            )

        chunks = await self._gather(doc_id, _FRAUD_QUERIES, per_query=4)
        content = self._compose(summary, chunks, max_content_tokens)
        if not content.strip():
            return FraudAnalysisResult(
                doc_id=doc_id,
                overall_assessment="No indexed content found for this document.",
                warnings=["Document has no retrievable content."],
            )

        response = await self.client.complete(
            FRAUD_PROMPT.format(content=content),
            system="You are a meticulous fraud analyst. Return only valid JSON.",
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
        if not response.ok:
            return FraudAnalysisResult(
                doc_id=doc_id,
                overall_assessment="Fraud analysis could not be completed.",
                chunks_analyzed=len(chunks),
                warnings=[response.error or "LLM call failed"],
            )

        parsed = extract_json(response.text)
        if not isinstance(parsed, dict):
            warnings.append("Model returned unparseable JSON; showing raw assessment.")
            return FraudAnalysisResult(
                doc_id=doc_id,
                overall_assessment=response.text.strip()[:2000],
                chunks_analyzed=len(chunks),
                warnings=warnings,
            )

        indicators = [
            FraudIndicator(
                type=str(item.get("type", "Unspecified")),
                severity=_normalize_severity(item.get("severity")),
                evidence=str(item.get("evidence", "")),
                page=_safe_int(item.get("page")),
                section=str(item.get("section", "")),
                explanation=str(item.get("explanation", "")),
            )
            for item in parsed.get("indicators", [])
            if isinstance(item, dict)
        ]
        indicators.sort(key=lambda i: SEVERITIES.index(i.severity))

        return FraudAnalysisResult(
            doc_id=doc_id,
            indicators=indicators,
            overall_assessment=str(parsed.get("overall_assessment", "")),
            fraud_likelihood=_normalize_severity(parsed.get("fraud_likelihood")),
            chunks_analyzed=len(chunks),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    async def extract_contract_clauses(
        self, doc_id: str, max_content_tokens: int | None = None
    ) -> ContractClauseResult:
        max_content_tokens = max_content_tokens or config.DOMAIN_ANALYSIS_CONTENT_TOKENS
        if not self.client.available:
            return ContractClauseResult(
                doc_id=doc_id,
                summary="LLM unavailable — configure ANTHROPIC_API_KEY.",
                warnings=[self.client.unavailable_reason() or "LLM unavailable"],
            )

        chunks = await self._gather(doc_id, _CLAUSE_QUERIES, per_query=3)
        content = self._compose(None, chunks, max_content_tokens)
        if not content.strip():
            return ContractClauseResult(
                doc_id=doc_id,
                summary="No indexed content found for this document.",
                warnings=["Document has no retrievable content."],
            )

        response = await self.client.complete(
            CLAUSE_PROMPT.format(content=content),
            system="You are a precise contracts lawyer. Return only valid JSON.",
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
        if not response.ok:
            return ContractClauseResult(
                doc_id=doc_id,
                summary="Clause analysis could not be completed.",
                chunks_analyzed=len(chunks),
                warnings=[response.error or "LLM call failed"],
            )

        parsed = extract_json(response.text)
        if not isinstance(parsed, dict):
            return ContractClauseResult(
                doc_id=doc_id,
                summary=response.text.strip()[:2000],
                chunks_analyzed=len(chunks),
                warnings=["Model returned unparseable JSON."],
            )

        clauses = [
            ContractClause(
                clause_type=str(item.get("clause_type", "Unspecified")),
                text=str(item.get("text", "")),
                page=_safe_int(item.get("page")),
                section=str(item.get("section", "")),
                risk_rating=_normalize_severity(item.get("risk_rating")),
                explanation=str(item.get("explanation", "")),
            )
            for item in parsed.get("clauses", [])
            if isinstance(item, dict)
        ]
        clauses.sort(key=lambda c: SEVERITIES.index(c.risk_rating))

        return ContractClauseResult(
            doc_id=doc_id,
            clauses=clauses,
            summary=str(parsed.get("summary", "")),
            chunks_analyzed=len(chunks),
        )

    # ------------------------------------------------------------------
    async def assess_risk(
        self,
        doc_id: str,
        client_profile: dict[str, Any] | None = None,
        fraud: FraudAnalysisResult | None = None,
        clauses: ContractClauseResult | None = None,
    ) -> RiskAssessmentResult:
        """Combine fraud + clause analysis + client profile into a 0-100 score."""
        client_profile = client_profile or {}

        if fraud is None:
            fraud = await self.extract_fraud_indicators(doc_id)
        if clauses is None:
            clauses = await self.extract_contract_clauses(doc_id)

        heuristic = _heuristic_score(fraud, clauses)

        if not self.client.available:
            return RiskAssessmentResult(
                doc_id=doc_id,
                risk_score=heuristic,
                risk_band=_band(heuristic),
                top_factors=_heuristic_factors(fraud, clauses),
                summary=(
                    "Heuristic score computed from indicator severities. "
                    "Configure ANTHROPIC_API_KEY for a full narrative assessment."
                ),
                warnings=[self.client.unavailable_reason() or "LLM unavailable"],
            )

        response = await self.client.complete(
            RISK_PROMPT.format(
                fraud=_render_fraud(fraud),
                clauses=_render_clauses(clauses),
                profile=_render_profile(client_profile),
            ),
            system="You are a risk officer. Return only valid JSON.",
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
        parsed = extract_json(response.text) if response.ok else None

        if not isinstance(parsed, dict):
            return RiskAssessmentResult(
                doc_id=doc_id,
                risk_score=heuristic,
                risk_band=_band(heuristic),
                top_factors=_heuristic_factors(fraud, clauses),
                summary="Falling back to heuristic scoring.",
                warnings=[response.error or "Model returned unparseable JSON."],
            )

        score = max(0, min(100, _safe_int(parsed.get("risk_score"), heuristic)))
        factors = [
            RiskFactor(
                factor=str(item.get("factor", "")),
                severity=_normalize_severity(item.get("severity")),
                evidence=str(item.get("evidence", "")),
                recommendation=str(item.get("recommendation", "")),
            )
            for item in parsed.get("top_factors", [])
            if isinstance(item, dict)
        ][:5]

        return RiskAssessmentResult(
            doc_id=doc_id,
            risk_score=score,
            risk_band=str(parsed.get("risk_band") or _band(score)).upper(),
            top_factors=factors or _heuristic_factors(fraud, clauses),
            summary=str(parsed.get("summary", "")),
        )

    # ------------------------------------------------------------------
    async def _gather(self, doc_id: str, queries: list[str], per_query: int):
        """Union of retrieval hits across several probe queries, best score first."""
        seen: dict[str, Any] = {}
        for query in queries:
            try:
                results = await self.retriever.retrieve(
                    query, doc_id=doc_id, top_k=per_query, retrieval_mode="hybrid"
                )
            except Exception as exc:
                logger.warning(f"Probe query failed ({query!r}): {exc}")
                continue
            for result in results:
                existing = seen.get(result.chunk_id)
                if existing is None or result.score > existing.score:
                    seen[result.chunk_id] = result
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

    @staticmethod
    def _compose(
        summary: SummaryResult | None, chunks: list, budget: int
    ) -> str:
        parts: list[str] = []
        if summary and summary.master_summary:
            parts.append(f"=== DOCUMENT SUMMARY ===\n{summary.master_summary}")
        if summary and summary.section_summaries:
            joined = "\n\n".join(
                f"[Section {i + 1}] {s}"
                for i, s in enumerate(summary.section_summaries[:20])
            )
            parts.append(f"=== SECTION SUMMARIES ===\n{joined}")

        if chunks:
            rendered = "\n\n".join(
                f"[CHUNK: {r.chunk_id} | page {r.chunk.page}"
                + (f" | {r.chunk.section_name}" if r.chunk.section_name else "")
                + f"]\n{r.chunk.text}"
                for r in chunks
            )
            parts.append(f"=== RETRIEVED SOURCE PASSAGES ===\n{rendered}")

        content, _ = token_counter.truncate_to_budget("\n\n".join(parts), budget)
        return content


# ----------------------------------------------------------------------
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in SEVERITIES else "LOW"


def _band(score: int) -> str:
    if score >= 75:
        return "SEVERE"
    if score >= 50:
        return "ELEVATED"
    if score >= 25:
        return "MODERATE"
    return "LOW"


_WEIGHTS = {"HIGH": 25, "MEDIUM": 12, "LOW": 5}


def _heuristic_score(
    fraud: FraudAnalysisResult, clauses: ContractClauseResult
) -> int:
    score = sum(_WEIGHTS[i.severity] for i in fraud.indicators)
    score += sum(_WEIGHTS[c.risk_rating] // 2 for c in clauses.clauses)
    if fraud.fraud_likelihood == "HIGH":
        score += 15
    elif fraud.fraud_likelihood == "MEDIUM":
        score += 7
    return max(0, min(100, score))


def _heuristic_factors(
    fraud: FraudAnalysisResult, clauses: ContractClauseResult
) -> list[RiskFactor]:
    factors = [
        RiskFactor(
            factor=i.type,
            severity=i.severity,
            evidence=i.evidence[:300],
            recommendation="Review the cited passage and verify against source records.",
        )
        for i in fraud.indicators[:3]
    ]
    factors.extend(
        RiskFactor(
            factor=f"{c.clause_type} clause",
            severity=c.risk_rating,
            evidence=c.text[:300],
            recommendation="Escalate to legal review before signature.",
        )
        for c in clauses.clauses
        if c.risk_rating == "HIGH"
    )
    return factors[:5]


def _render_fraud(result: FraudAnalysisResult) -> str:
    if not result.indicators:
        return f"No fraud indicators found. {result.overall_assessment}"
    lines = [
        f"- [{i.severity}] {i.type} (page {i.page}): {i.explanation} "
        f'Evidence: "{i.evidence[:240]}"'
        for i in result.indicators
    ]
    return f"Likelihood: {result.fraud_likelihood}\n" + "\n".join(lines)


def _render_clauses(result: ContractClauseResult) -> str:
    if not result.clauses:
        return "No contract clauses extracted."
    return "\n".join(
        f"- [{c.risk_rating}] {c.clause_type} (page {c.page}): {c.explanation}"
        for c in result.clauses
    )


def _render_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "(no client profile supplied)"
    return "\n".join(f"- {key}: {value}" for key, value in profile.items())
