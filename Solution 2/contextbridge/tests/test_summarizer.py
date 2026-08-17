"""Summarizer tests — the LLM is stubbed so these run offline and deterministically."""

from __future__ import annotations

import pytest

from backend.core.models import ClaudeResponse, ConversationExchange
from backend.core.summarizer import HierarchicalSummarizer, _group_by_token_budget
from tests.conftest import make_chunk


class StubClient:
    """Records prompts and returns canned summaries."""

    def __init__(self, available: bool = True, fail_indices: set[int] | None = None):
        self.available = available
        self.prompts: list[str] = []
        self.models: list[str | None] = []
        self.fail_indices = fail_indices or set()
        self._call_number = 0

    def unavailable_reason(self):
        return None if self.available else "stubbed as unavailable"

    async def complete(
        self, prompt, system=None, max_tokens=None, messages=None, model=None
    ):
        index = self._call_number
        self._call_number += 1
        self.prompts.append(prompt)
        self.models.append(model)
        if index in self.fail_indices:
            return ClaudeResponse(text="", error="stubbed failure")
        return ClaudeResponse(
            text=f"SUMMARY#{index}: {prompt[-60:].strip()}",
            input_tokens=100,
            output_tokens=25,
        )

    async def complete_many(
        self, prompts, system=None, max_tokens=None, concurrency=1, model=None
    ):
        return [await self.complete(p, system, max_tokens, model=model) for p in prompts]


def chunks(count: int, doc_id: str = "doc-s"):
    return [
        make_chunk(
            i,
            f"Chunk {i} content about claim CLM-2024-778341 and amount $412,500. "
            * 6,
            doc_id=doc_id,
            total=count,
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_summarize_produces_a_master_summary():
    summarizer = HierarchicalSummarizer(StubClient())
    result = await summarizer.summarize_document(chunks(6), doc_type="insurance_claim")
    assert result.master_summary
    assert result.total_chunks_processed == 6


@pytest.mark.asyncio
async def test_every_chunk_gets_a_summary():
    summarizer = HierarchicalSummarizer(StubClient())
    result = await summarizer.summarize_document(chunks(8))
    assert len(result.chunk_summaries) == 8
    assert all(s for s in result.chunk_summaries)


@pytest.mark.asyncio
async def test_completeness_is_full_when_no_call_fails():
    summarizer = HierarchicalSummarizer(StubClient())
    result = await summarizer.summarize_document(chunks(5))
    assert result.completeness_score == 1.0
    assert not result.warnings


@pytest.mark.asyncio
async def test_failed_map_call_falls_back_to_raw_text_not_silence():
    # Fail the first two map calls.
    summarizer = HierarchicalSummarizer(StubClient(fail_indices={0, 1}))
    result = await summarizer.summarize_document(chunks(5))

    assert result.completeness_score < 1.0
    assert result.warnings, "a partial failure must be surfaced as a warning"
    # The content still made it through rather than being dropped.
    assert all(s.strip() for s in result.chunk_summaries)
    assert "CLM-2024-778341" in result.chunk_summaries[0]


@pytest.mark.asyncio
async def test_hierarchy_deepens_for_many_chunks():
    summarizer = HierarchicalSummarizer(StubClient())
    small = await summarizer.summarize_document(chunks(2))
    large = await summarizer.summarize_document(chunks(40))
    assert large.levels >= small.levels
    assert large.levels >= 2


@pytest.mark.asyncio
async def test_intermediate_levels_are_retained():
    summarizer = HierarchicalSummarizer(StubClient())
    result = await summarizer.summarize_document(chunks(30))
    assert result.chunk_summaries, "level 1 must be kept"
    assert result.section_summaries, "level 2 must be kept"
    assert result.master_summary, "level 3 must be kept"


@pytest.mark.asyncio
async def test_focus_topic_reaches_the_prompt():
    client = StubClient()
    await HierarchicalSummarizer(client).summarize_document(
        chunks(3), focus="prior claims"
    )
    assert any("prior claims" in p for p in client.prompts)


@pytest.mark.asyncio
async def test_map_phase_uses_the_configured_summary_model(monkeypatch):
    """Per-chunk calls route to SUMMARY_MODEL; the reduce phase keeps the primary."""
    from backend.core import summarizer as summarizer_module

    monkeypatch.setattr(summarizer_module.config, "SUMMARY_MODEL", "auto:fast")
    client = StubClient()
    await HierarchicalSummarizer(client).summarize_document(chunks(12))

    assert "auto:fast" in client.models, "map phase must use the summary model"
    # Reduce calls pass model=None so the client's primary model applies.
    assert None in client.models, "reduce phase must stay on the primary model"


@pytest.mark.asyncio
async def test_no_llm_returns_extractive_fallback_with_warning():
    result = await HierarchicalSummarizer(StubClient(available=False)).summarize_document(
        chunks(6)
    )
    assert result.master_summary, "should still return something usable"
    assert result.warnings
    assert result.levels == 0


@pytest.mark.asyncio
async def test_empty_chunk_list_is_handled():
    result = await HierarchicalSummarizer(StubClient()).summarize_document([])
    assert result.master_summary == ""
    assert result.warnings


@pytest.mark.asyncio
async def test_token_usage_is_tracked():
    result = await HierarchicalSummarizer(StubClient()).summarize_document(chunks(4))
    assert result.token_usage.calls > 0
    assert result.token_usage.total_tokens > 0


@pytest.mark.asyncio
async def test_conversation_summary_extends_existing():
    client = StubClient()
    exchanges = [ConversationExchange("What is the claim amount?", "It is $412,500.")]
    summary = await HierarchicalSummarizer(client).summarize_conversation(
        exchanges, existing_summary="Earlier: the user asked about coverage."
    )
    assert summary
    assert any("EXISTING SUMMARY" in p for p in client.prompts)


@pytest.mark.asyncio
async def test_conversation_summary_with_no_exchanges_is_a_noop():
    summarizer = HierarchicalSummarizer(StubClient())
    assert await summarizer.summarize_conversation([], "prior") == "prior"


def test_group_by_token_budget_splits():
    inputs = ["word " * 500 for _ in range(10)]
    groups = _group_by_token_budget(inputs, budget=1000)
    assert len(groups) > 1
    assert sum(len(g) for g in groups) == 10
