"""Three-tier conversation memory.

    TIER 1  Short-term buffer  -> last N exchanges verbatim (fast, exact)
    TIER 2  Mid-term summary   -> rolling LLM summary of everything older
    TIER 3  Entity store       -> structured facts extracted from the conversation

``build_context_payload`` packs those tiers plus retrieved chunks into a fixed
token budget and reports exactly what didn't fit.

Note: ``add_exchange`` is async (the spec sketches it as sync). Evicting from the
short-term buffer triggers LLM summarization and entity extraction, so a sync
signature would either block the event loop or lie about what it does.
"""

from __future__ import annotations

from backend import config
from backend.core.llm import ClaudeClient, get_claude_client
from backend.core.models import (
    ContextPayload,
    ConversationExchange,
    SearchResult,
    SessionSummary,
)
from backend.core.summarizer import HierarchicalSummarizer
from backend.core.token_counter import token_counter
from backend.utils.helpers import dedupe_preserving_order, extract_json
from backend.utils.logger import logger

ENTITY_KEYS = [
    "people",
    "organizations",
    "dates",
    "amounts",
    "locations",
    "decisions",
    "risks",
    "claim_ids",
]

ENTITY_PROMPT = """Extract structured entities from the text below.

Return ONLY a JSON object with exactly these keys, each mapping to an array of
strings (use an empty array when nothing applies):
{{
  "people": [], "organizations": [], "dates": [], "amounts": [],
  "locations": [], "decisions": [], "risks": [], "claim_ids": []
}}

Copy values verbatim from the text. Do not infer or invent.

TEXT:
{text}

JSON:"""

SYSTEM_PROMPT_TEMPLATE = """You are ContextBridge, an expert document analysis AI \
specializing in Banking & Insurance.

You have access to a large document that has been intelligently chunked and indexed. \
Each response MUST:
1. Answer based ONLY on the provided document context
2. Cite specific sections using [CHUNK: chunk_id] notation when referencing information
3. If information spans multiple sections, cite all relevant sections
4. If you cannot find the answer in the provided context, say "Not found in provided \
context sections"
5. Flag any inconsistencies or anomalies you notice
6. Always specify if your answer might be incomplete due to context limitations

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_summary}

KNOWN ENTITIES FROM THIS SESSION:
{entity_store}
"""


class MemoryManager:
    """Per-session memory across three tiers."""

    def __init__(
        self,
        session_id: str,
        client: ClaudeClient | None = None,
        summarizer: HierarchicalSummarizer | None = None,
    ) -> None:
        self.session_id = session_id
        self.client = client or get_claude_client()
        self.summarizer = summarizer or HierarchicalSummarizer(self.client)

        self.short_term: list[ConversationExchange] = []
        self.mid_term_summary: str = ""
        self.entity_store: dict[str, list[str]] = {key: [] for key in ENTITY_KEYS}
        self.total_exchanges: int = 0
        self.referenced_sections: set[str] = set()

    # ------------------------------------------------------------------
    async def add_exchange(
        self,
        user_message: str,
        assistant_response: str,
        retrieved_chunks: list[SearchResult] | None = None,
    ) -> None:
        """Append an exchange, evicting the oldest into tiers 2 and 3 when full."""
        retrieved_chunks = retrieved_chunks or []
        sections = dedupe_preserving_order(
            r.chunk.section_name for r in retrieved_chunks if r.chunk.section_name
        )
        self.referenced_sections.update(sections)

        self.short_term.append(
            ConversationExchange(
                user_message=user_message,
                assistant_response=assistant_response,
                retrieved_chunk_ids=[r.chunk_id for r in retrieved_chunks],
                sections=sections,
            )
        )
        self.total_exchanges += 1

        if len(self.short_term) > config.SHORT_TERM_EXCHANGES:
            overflow_count = len(self.short_term) - config.SHORT_TERM_EXCHANGES
            evicted = self.short_term[:overflow_count]
            self.short_term = self.short_term[overflow_count:]
            await self._absorb(evicted)

    async def _absorb(self, evicted: list[ConversationExchange]) -> None:
        """Fold evicted exchanges into the mid-term summary and entity store."""
        try:
            self.mid_term_summary = await self.summarizer.summarize_conversation(
                evicted, existing_summary=self.mid_term_summary or None
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Mid-term summarization failed: {exc}")

        text = "\n".join(
            f"{e.user_message}\n{e.assistant_response}" for e in evicted
        )
        try:
            entities = await self.extract_entities(text)
            self._merge_entities(entities)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Entity extraction failed: {exc}")

    # ------------------------------------------------------------------
    async def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Ask Claude for structured entities. Returns empty lists on failure."""
        empty = {key: [] for key in ENTITY_KEYS}
        if not text.strip() or not self.client.available:
            return empty

        budgeted, _ = token_counter.truncate_to_budget(text, 6000)
        response = await self.client.complete(
            ENTITY_PROMPT.format(text=budgeted),
            system="You extract structured data and return only valid JSON.",
            max_tokens=1500,
        )
        if not response.ok:
            logger.warning(f"Entity extraction call failed: {response.error}")
            return empty

        parsed = extract_json(response.text)
        if not isinstance(parsed, dict):
            logger.warning("Entity extraction returned unparseable output")
            return empty

        return {
            key: [str(v).strip() for v in parsed.get(key, []) if str(v).strip()]
            if isinstance(parsed.get(key), list)
            else []
            for key in ENTITY_KEYS
        }

    def _merge_entities(self, entities: dict[str, list[str]]) -> None:
        for key in ENTITY_KEYS:
            merged = dedupe_preserving_order(
                self.entity_store.get(key, []) + entities.get(key, [])
            )
            self.entity_store[key] = merged[: config.ENTITY_STORE_MAX_ENTRIES]

    def merge_entities(self, entities: dict[str, list[str]] | None) -> None:
        """Public hook — lets ingestion seed the store with document-level entities."""
        if entities:
            self._merge_entities(
                {key: list(entities.get(key, []) or []) for key in ENTITY_KEYS}
            )

    # ------------------------------------------------------------------
    def build_context_payload(
        self,
        query: str,
        retrieved_chunks: list[SearchResult],
        token_budget: int = config.USABLE_CONTEXT_TOKENS,
    ) -> ContextPayload:
        """Pack context by priority:

        1. system prompt scaffold  2. entity store  3. mid-term summary
        4. retrieved chunks (best score first)  5. short-term buffer (newest first)
        """
        breakdown: dict[str, int] = {}

        entity_block = self._render_entities()
        breakdown["entity_store"] = token_counter.count(entity_block)

        summary_block = self.mid_term_summary or "(no earlier conversation)"
        summary_tokens = token_counter.count(summary_block)
        # Tier 2 is always included, but never at the cost of all retrieval.
        summary_cap = max(500, int(token_budget * 0.15))
        if summary_tokens > summary_cap:
            summary_block, summary_tokens = token_counter.truncate_to_budget(
                summary_block, summary_cap
            )
        breakdown["mid_term_summary"] = summary_tokens

        scaffold = SYSTEM_PROMPT_TEMPLATE.format(
            context="", conversation_summary="", entity_store=""
        )
        scaffold_tokens = token_counter.count(scaffold)
        breakdown["system_scaffold"] = scaffold_tokens

        query_tokens = token_counter.count(query)
        breakdown["query"] = query_tokens

        spent = (
            scaffold_tokens
            + breakdown["entity_store"]
            + summary_tokens
            + query_tokens
        )
        remaining = token_budget - spent

        # --- 4. retrieved chunks, best score first ---
        included: list[SearchResult] = []
        dropped: list[SearchResult] = []
        chunk_tokens = 0
        # Leave at least a quarter of what's left for recent conversation turns.
        chunk_allowance = max(0, int(remaining * 0.75))

        for result in sorted(retrieved_chunks, key=lambda r: r.score, reverse=True):
            cost = token_counter.count(_render_chunk(result)) + 8
            if chunk_tokens + cost <= chunk_allowance:
                included.append(result)
                chunk_tokens += cost
            else:
                dropped.append(result)

        breakdown["retrieved_chunks"] = chunk_tokens
        remaining -= chunk_tokens

        # --- 5. short-term buffer, newest first ---
        history: list[dict[str, str]] = []
        history_tokens = 0
        for exchange in reversed(self.short_term):
            pair = [
                {"role": "user", "content": exchange.user_message},
                {"role": "assistant", "content": exchange.assistant_response},
            ]
            cost = token_counter.count_messages(pair)
            if history_tokens + cost > remaining:
                break
            history = pair + history
            history_tokens += cost

        breakdown["short_term_buffer"] = history_tokens

        context_block = (
            "\n\n".join(_render_chunk(r) for r in included)
            if included
            else "(no document sections retrieved for this query)"
        )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context_block,
            conversation_summary=summary_block,
            entity_store=entity_block,
        )

        total_used = token_counter.count(system_prompt) + history_tokens + query_tokens

        return ContextPayload(
            system_prompt=system_prompt,
            conversation_history=history,
            included_chunks=included,
            dropped_chunks=dropped,
            total_tokens_used=total_used,
            token_budget=token_budget,
            breakdown=breakdown,
        )

    def _render_entities(self) -> str:
        lines = [
            f"- {key.replace('_', ' ')}: {', '.join(values)}"
            for key, values in self.entity_store.items()
            if values
        ]
        return "\n".join(lines) if lines else "(none recorded yet)"

    # ------------------------------------------------------------------
    def get_session_summary(self) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            short_term_count=len(self.short_term),
            mid_term_summary=self.mid_term_summary,
            entity_store={k: v for k, v in self.entity_store.items() if v},
            total_exchanges=self.total_exchanges,
        )

    def lookup_entity(self, needle: str) -> list[str]:
        """Search the entity store — how the demo answers follow-ups without retrieval."""
        needle = needle.lower().strip()
        hits: list[str] = []
        for key, values in self.entity_store.items():
            for value in values:
                if needle in value.lower() or needle in key:
                    hits.append(f"{key}: {value}")
        return hits

    def reset(self) -> None:
        self.short_term.clear()
        self.mid_term_summary = ""
        self.entity_store = {key: [] for key in ENTITY_KEYS}
        self.total_exchanges = 0
        self.referenced_sections.clear()


def _render_chunk(result: SearchResult) -> str:
    chunk = result.chunk
    header = f"[CHUNK: {result.chunk_id}]"
    if chunk.section_name:
        header += f" (section: {chunk.section_name}, page {chunk.page})"
    else:
        header += f" (page {chunk.page})"
    return f"{header}\n{chunk.text}"


class SessionStore:
    """In-memory registry of per-session ``MemoryManager`` instances."""

    def __init__(self) -> None:
        self._sessions: dict[str, MemoryManager] = {}

    def get(self, session_id: str) -> MemoryManager:
        manager = self._sessions.get(session_id)
        if manager is None:
            manager = MemoryManager(session_id)
            self._sessions[session_id] = manager
            logger.info(f"New session: {session_id}")
        return manager

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def all_ids(self) -> list[str]:
        return list(self._sessions)


session_store = SessionStore()
