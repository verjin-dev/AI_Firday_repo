"""Hierarchical Map-Reduce summarization.

    Level 0: raw chunks          [c1][c2][c3]...[cN]
    Level 1: chunk summaries     [s1][s2][s3]...[sN]
    Level 2: section summaries   [S1]    [S2]    [S3]
    Level 3: document summary    [MASTER]

Every intermediate level is retained, not just the final master summary — that is
what lets a fact buried on page 31 survive compression and still be citable.
"""

from __future__ import annotations

from backend import config
from backend.core.llm import ClaudeClient, get_claude_client
from backend.core.models import (
    ChunkResult,
    ConversationExchange,
    SummaryResult,
    TokenUsage,
)
from backend.core.token_counter import token_counter
from backend.utils.helpers import Timer, batched
from backend.utils.logger import logger

MAP_SYSTEM_PROMPT = (
    "You are a precise document analyst. You compress text without losing facts. "
    "Never invent information that is not in the source text."
)

MAP_PROMPT = """You are summarizing chunk {index} of {total} from a {doc_type} document.
Preserve: key facts, named entities, dates, amounts, decisions, risks, anomalies.
Be concise but complete. Do not lose numerical values or proper nouns.
{focus_line}
CHUNK TEXT:
{chunk_text}

SUMMARY:"""

REDUCE_SYSTEM_PROMPT = (
    "You are a precise document analyst combining partial summaries into a single "
    "coherent summary. Preserve every specific fact, figure, date, name and anomaly "
    "from the inputs. Never invent information."
)

REDUCE_PROMPT = """You are combining {count} partial summaries (level {level}) from a \
{doc_type} document into one consolidated summary.

Requirements:
- Preserve every named entity, date, monetary amount, policy/claim number and decision.
- Explicitly retain anything that reads as an anomaly, inconsistency or risk.
- Do not add information that is not present in the inputs.
- Write flowing prose, not a bulleted restatement.
{focus_line}
PARTIAL SUMMARIES:
{joined}

CONSOLIDATED SUMMARY:"""

CONVERSATION_PROMPT = """Summarize this conversation for long-term memory.
Focus on: decisions made, facts established, questions asked, answers given.
Preserve specific values (names, dates, amounts, identifiers) verbatim.
Write a compact paragraph.

{existing_block}NEW EXCHANGES:
{exchanges}

UPDATED SUMMARY:"""


class HierarchicalSummarizer:
    """Map-Reduce summarizer over a document's chunks."""

    def __init__(self, client: ClaudeClient | None = None) -> None:
        self.client = client or get_claude_client()

    # ------------------------------------------------------------------
    async def summarize_document(
        self,
        chunks: list[ChunkResult],
        doc_type: str = "general",
        focus: str | None = None,
    ) -> SummaryResult:
        doc_id = chunks[0].doc_id if chunks else "unknown"
        usage = TokenUsage()
        warnings: list[str] = []

        with Timer() as timer:
            if not chunks:
                return SummaryResult(
                    doc_id=doc_id,
                    master_summary="",
                    processing_time_seconds=0.0,
                    warnings=["No chunks to summarize."],
                )

            if not self.client.available:
                reason = self.client.unavailable_reason() or "LLM unavailable"
                logger.warning(f"Summarization skipped: {reason}")
                extractive = _extractive_summary(chunks)
                return SummaryResult(
                    doc_id=doc_id,
                    master_summary=extractive,
                    section_summaries=[],
                    chunk_summaries=[],
                    total_chunks_processed=0,
                    levels=0,
                    processing_time_seconds=timer.seconds,
                    completeness_score=0.0,
                    warnings=[
                        f"{reason} Showing an extractive fallback summary instead of "
                        "a hierarchical LLM summary."
                    ],
                )

            # ---------------- MAP ----------------
            chunk_summaries, map_usage, failed = await self._map_phase(
                chunks, doc_type, focus
            )
            usage.merge(map_usage)

            # Measure genuinely summarized chunks, not merely non-empty ones —
            # failed chunks carry raw-text fallback, which is also non-empty.
            completeness = (len(chunks) - failed) / len(chunks) if chunks else 0.0
            if failed:
                warnings.append(
                    f"{failed} of {len(chunks)} chunks failed to summarize; their raw "
                    "text was used instead so no content was silently dropped."
                )

            summary_map = {
                chunks[i].chunk_index: chunk_summaries[i]
                for i in range(len(chunks))
                if chunk_summaries[i]
            }

            # ---------------- REDUCE ----------------
            section_summaries: list[str] = []
            level_input = [s for s in chunk_summaries if s.strip()]
            levels = 1

            while len(level_input) > 1 and levels < config.MAX_SUMMARY_LEVELS:
                levels += 1
                level_output, reduce_usage = await self._reduce_level(
                    level_input, doc_type, focus, levels
                )
                usage.merge(reduce_usage)
                if not level_output:
                    warnings.append(f"Reduce level {levels} produced no output.")
                    break
                if levels == 2:
                    section_summaries = list(level_output)
                level_input = level_output
                if len(level_output) == 1:
                    break

            if len(level_input) > 1:
                # Hit MAX_SUMMARY_LEVELS with several branches left — force one join.
                levels += 1
                final, reduce_usage = await self._reduce_level(
                    level_input, doc_type, focus, levels, force_single=True
                )
                usage.merge(reduce_usage)
                level_input = final or level_input[:1]

            master = level_input[0] if level_input else ""
            if not section_summaries:
                section_summaries = [s for s in chunk_summaries if s.strip()]

        result = SummaryResult(
            doc_id=doc_id,
            master_summary=master,
            section_summaries=section_summaries,
            chunk_summaries=chunk_summaries,
            total_chunks_processed=len(chunks),
            levels=levels,
            token_usage=usage,
            processing_time_seconds=timer.seconds,
            completeness_score=completeness,
            warnings=warnings,
            chunk_summary_map=summary_map,
        )
        logger.info(
            f"{doc_id}: summarized {len(chunks)} chunks in {levels} levels "
            f"({timer.seconds}s, {usage.total_tokens} tokens, "
            f"completeness {completeness:.0%})"
        )
        return result

    # ------------------------------------------------------------------
    async def _map_phase(
        self, chunks: list[ChunkResult], doc_type: str, focus: str | None
    ) -> tuple[list[str], TokenUsage, int]:
        usage = TokenUsage()
        focus_line = (
            f"Pay particular attention to anything related to: {focus}\n" if focus else ""
        )

        summaries: list[str] = [""] * len(chunks)
        failed = 0
        total = len(chunks)

        # Batched so progress is visible and memory stays bounded on huge documents.
        for batch_start, batch in enumerate(
            batched(chunks, config.SUMMARY_CHUNK_BATCH_SIZE)
        ):
            offset = batch_start * config.SUMMARY_CHUNK_BATCH_SIZE
            prompts = [
                MAP_PROMPT.format(
                    index=offset + i + 1,
                    total=total,
                    doc_type=doc_type,
                    focus_line=focus_line,
                    chunk_text=chunk.text,
                )
                for i, chunk in enumerate(batch)
            ]
            responses = await self.client.complete_many(
                prompts,
                system=MAP_SYSTEM_PROMPT,
                max_tokens=config.SUMMARY_MAX_OUTPUT_TOKENS,
                model=config.SUMMARY_MODEL,
            )
            for i, response in enumerate(responses):
                position = offset + i
                usage.add(response.input_tokens, response.output_tokens)
                if response.ok and response.text.strip():
                    summaries[position] = response.text.strip()
                else:
                    failed += 1
                    # Never silently drop content: fall back to the raw chunk text.
                    truncated, _ = token_counter.truncate_to_budget(
                        batch[i].text, config.CHUNK_SIZE // 2
                    )
                    summaries[position] = truncated

        return summaries, usage, failed

    async def _reduce_level(
        self,
        inputs: list[str],
        doc_type: str,
        focus: str | None,
        level: int,
        force_single: bool = False,
    ) -> tuple[list[str], TokenUsage]:
        """Group inputs into token-budgeted batches and combine each into one summary."""
        usage = TokenUsage()
        focus_line = (
            f"Pay particular attention to anything related to: {focus}\n" if focus else ""
        )

        groups = (
            [inputs]
            if force_single
            else _group_by_token_budget(inputs, config.REDUCE_BATCH_TOKEN_BUDGET)
        )
        if len(groups) == len(inputs) and len(inputs) > 1:
            # Every input already fills the budget — halve the group count instead of
            # looping forever with no reduction.
            midpoint = max(1, len(inputs) // 2)
            groups = [inputs[:midpoint], inputs[midpoint:]]

        prompts = []
        for group in groups:
            joined = "\n\n---\n\n".join(
                f"[{i + 1}] {text}" for i, text in enumerate(group)
            )
            prompts.append(
                REDUCE_PROMPT.format(
                    count=len(group),
                    level=level,
                    doc_type=doc_type,
                    focus_line=focus_line,
                    joined=joined,
                )
            )

        # The reduce phase is where facts get merged or lost — keep it on the
        # primary (stronger) model even when the map phase uses a cheaper one.
        responses = await self.client.complete_many(
            prompts, system=REDUCE_SYSTEM_PROMPT, max_tokens=config.MAX_OUTPUT_TOKENS
        )

        outputs: list[str] = []
        for i, response in enumerate(responses):
            usage.add(response.input_tokens, response.output_tokens)
            if response.ok and response.text.strip():
                outputs.append(response.text.strip())
            else:
                # Concatenate the group so its content survives the failed call.
                outputs.append("\n\n".join(groups[i]))

        return outputs, usage

    # ------------------------------------------------------------------
    async def summarize_conversation(
        self,
        exchanges: list[ConversationExchange],
        existing_summary: str | None = None,
    ) -> str:
        """Incrementally extend a rolling conversation summary."""
        if not exchanges:
            return existing_summary or ""

        if not self.client.available:
            snippets = [
                f"User asked: {e.user_message[:160]}" for e in exchanges
            ]
            merged = " ".join(filter(None, [existing_summary or ""] + snippets))
            truncated, _ = token_counter.truncate_to_budget(merged, 800)
            return truncated

        rendered = "\n\n".join(
            f"User: {e.user_message}\nAssistant: {e.assistant_response}"
            for e in exchanges
        )
        existing_block = (
            f"EXISTING SUMMARY (extend it, do not restate it):\n{existing_summary}\n\n"
            if existing_summary
            else ""
        )
        response = await self.client.complete(
            CONVERSATION_PROMPT.format(
                existing_block=existing_block, exchanges=rendered
            ),
            system=REDUCE_SYSTEM_PROMPT,
            max_tokens=config.SUMMARY_MAX_OUTPUT_TOKENS,
        )
        if response.ok and response.text.strip():
            return response.text.strip()

        logger.warning("Conversation summarization failed; keeping previous summary")
        return existing_summary or ""


def _group_by_token_budget(inputs: list[str], budget: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for text in inputs:
        tokens = token_counter.count(text)
        if current and current_tokens + tokens > budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens

    if current:
        groups.append(current)
    return groups


def _extractive_summary(chunks: list[ChunkResult], max_tokens: int = 900) -> str:
    """No-LLM fallback: lead sentences from evenly spaced chunks across the doc."""
    if not chunks:
        return ""
    step = max(1, len(chunks) // 12)
    picked = chunks[::step][:12]
    sentences = []
    for chunk in picked:
        head = chunk.text.strip().split(". ")
        if head:
            sentences.append(head[0].strip().rstrip(".") + ".")
    text, _ = token_counter.truncate_to_budget(" ".join(sentences), max_tokens)
    return text
