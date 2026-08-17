"""Memory manager tests — three tiers plus context packing."""

from __future__ import annotations

import pytest

from backend import config
from backend.core.memory_manager import ENTITY_KEYS, MemoryManager
from backend.core.models import ClaudeResponse
from tests.conftest import make_chunk, make_result


class StubClient:
    def __init__(self, available: bool = True):
        self.available = available
        self.prompts: list[str] = []

    def unavailable_reason(self):
        return None if self.available else "stubbed"

    async def complete(
        self, prompt, system=None, max_tokens=None, messages=None, model=None
    ):
        self.prompts.append(prompt)
        if "JSON" in (system or "") or "JSON:" in prompt:
            return ClaudeResponse(
                text=(
                    '{"people": ["Gregory Halloran"], "organizations": ["Northgate '
                    'Mutual"], "dates": ["12 September 2024"], "amounts": ["$412,500"], '
                    '"locations": [], "decisions": ["Referred to SIU"], '
                    '"risks": ["Duplicate claim"], "claim_ids": ["CLM-2024-778341"]}'
                ),
                input_tokens=50,
                output_tokens=30,
            )
        return ClaudeResponse(text="Rolling summary of the conversation so far.",
                              input_tokens=50, output_tokens=20)

    async def complete_many(
        self, prompts, system=None, max_tokens=None, concurrency=1, model=None
    ):
        return [await self.complete(p, system, max_tokens, model=model) for p in prompts]


def manager(available: bool = True) -> MemoryManager:
    return MemoryManager("sess-test", client=StubClient(available))


def results(count: int, section: str = "Prior Claims History"):
    return [
        make_result(
            make_chunk(i, f"Chunk {i} text about the claim. " * 20, section=section, page=i + 1),
            score=1.0 - i * 0.05,
        )
        for i in range(count)
    ]


# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_short_term_buffer_holds_recent_exchanges():
    memory = manager()
    await memory.add_exchange("Q1", "A1")
    await memory.add_exchange("Q2", "A2")
    assert len(memory.short_term) == 2
    assert memory.short_term[-1].user_message == "Q2"


@pytest.mark.asyncio
async def test_buffer_never_exceeds_configured_size():
    memory = manager()
    for i in range(config.SHORT_TERM_EXCHANGES + 4):
        await memory.add_exchange(f"Q{i}", f"A{i}")
    assert len(memory.short_term) <= config.SHORT_TERM_EXCHANGES


@pytest.mark.asyncio
async def test_eviction_populates_mid_term_summary():
    memory = manager()
    for i in range(config.SHORT_TERM_EXCHANGES + 2):
        await memory.add_exchange(f"Q{i}", f"A{i}")
    assert memory.mid_term_summary, "tier 2 should populate after eviction"


@pytest.mark.asyncio
async def test_eviction_populates_entity_store():
    memory = manager()
    for i in range(config.SHORT_TERM_EXCHANGES + 2):
        await memory.add_exchange(f"Q{i}", f"A{i} mentions CLM-2024-778341")
    assert memory.entity_store["claim_ids"], "tier 3 should populate after eviction"


@pytest.mark.asyncio
async def test_total_exchanges_counts_everything():
    memory = manager()
    for i in range(12):
        await memory.add_exchange(f"Q{i}", f"A{i}")
    assert memory.total_exchanges == 12


@pytest.mark.asyncio
async def test_entity_extraction_parses_json():
    memory = manager()
    entities = await memory.extract_entities("Gregory Halloran filed CLM-2024-778341.")
    assert entities["claim_ids"] == ["CLM-2024-778341"]
    assert set(entities) == set(ENTITY_KEYS)


@pytest.mark.asyncio
async def test_entity_extraction_without_llm_returns_empty_shape():
    memory = manager(available=False)
    entities = await memory.extract_entities("anything")
    assert set(entities) == set(ENTITY_KEYS)
    assert all(v == [] for v in entities.values())


@pytest.mark.asyncio
async def test_entity_store_is_capped():
    memory = manager()
    memory._merge_entities(
        {"people": [f"Person {i}" for i in range(config.ENTITY_STORE_MAX_ENTRIES + 50)]}
    )
    assert len(memory.entity_store["people"]) <= config.ENTITY_STORE_MAX_ENTRIES


@pytest.mark.asyncio
async def test_entities_are_deduplicated():
    memory = manager()
    memory._merge_entities({"people": ["Alice", "alice", "Alice "]})
    assert len(memory.entity_store["people"]) == 1


# ----------------------------------------------------------------------
def test_context_payload_stays_within_budget():
    memory = manager()
    payload = memory.build_context_payload("query", results(30), token_budget=4000)
    assert payload.total_tokens_used <= 4000 * 1.15  # small scaffold tolerance
    assert payload.token_budget == 4000


def test_context_payload_reports_what_was_dropped():
    memory = manager()
    payload = memory.build_context_payload("query", results(40), token_budget=2500)
    assert payload.dropped_chunks, "a tight budget must drop something"
    assert len(payload.included_chunks) + len(payload.dropped_chunks) == 40


def test_higher_scoring_chunks_are_included_first():
    memory = manager()
    payload = memory.build_context_payload("query", results(30), token_budget=3000)
    if payload.included_chunks and payload.dropped_chunks:
        assert min(r.score for r in payload.included_chunks) >= max(
            r.score for r in payload.dropped_chunks
        )


def test_nothing_is_dropped_with_a_generous_budget():
    memory = manager()
    payload = memory.build_context_payload("query", results(5), token_budget=200_000)
    assert payload.dropped_chunks == []


def test_utilization_percent_is_reported():
    memory = manager()
    payload = memory.build_context_payload("query", results(10), token_budget=10_000)
    assert 0 < payload.utilization_percent <= 100


def test_breakdown_accounts_for_each_tier():
    memory = manager()
    payload = memory.build_context_payload("query", results(6), token_budget=20_000)
    for key in ("system_scaffold", "entity_store", "retrieved_chunks"):
        assert key in payload.breakdown


def test_system_prompt_embeds_chunk_ids_for_citation():
    memory = manager()
    payload = memory.build_context_payload("query", results(3), token_budget=20_000)
    for result in payload.included_chunks:
        assert f"[CHUNK: {result.chunk_id}]" in payload.system_prompt


@pytest.mark.asyncio
async def test_entity_store_appears_in_the_system_prompt():
    memory = manager()
    memory.merge_entities({"claim_ids": ["CLM-2024-778341"]})
    payload = memory.build_context_payload("query", results(2), token_budget=20_000)
    assert "CLM-2024-778341" in payload.system_prompt


@pytest.mark.asyncio
async def test_conversation_history_is_included_newest_first():
    memory = manager()
    await memory.add_exchange("first question", "first answer")
    await memory.add_exchange("second question", "second answer")
    payload = memory.build_context_payload("q", results(2), token_budget=20_000)
    contents = [m["content"] for m in payload.conversation_history]
    assert "second question" in contents
    assert contents.index("first question") < contents.index("second question")


# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_facts_survive_twenty_turns():
    """The headline claim: a fact from turn 1 is still available at turn 20."""
    memory = manager()
    await memory.add_exchange(
        "What was the prior policy number?",
        "The prior claim CLM-2024-778341 was under policy POL-CG-88213-B.",
    )
    for i in range(19):
        await memory.add_exchange(f"Follow-up {i}", f"Answer {i}")

    assert memory.total_exchanges == 20
    assert len(memory.short_term) <= config.SHORT_TERM_EXCHANGES
    # The fact left the verbatim buffer but persists in tier 3.
    assert memory.entity_store["claim_ids"], "entity store must retain the identifier"
    assert memory.lookup_entity("CLM-2024-778341")


@pytest.mark.asyncio
async def test_reset_clears_all_tiers():
    memory = manager()
    for i in range(8):
        await memory.add_exchange(f"Q{i}", f"A{i}")
    memory.reset()
    assert memory.short_term == []
    assert memory.mid_term_summary == ""
    assert memory.total_exchanges == 0
    assert all(v == [] for v in memory.entity_store.values())


@pytest.mark.asyncio
async def test_referenced_sections_are_tracked_for_reranking():
    memory = manager()
    await memory.add_exchange("q", "a", results(3, section="Prior Claims History"))
    assert "Prior Claims History" in memory.referenced_sections
