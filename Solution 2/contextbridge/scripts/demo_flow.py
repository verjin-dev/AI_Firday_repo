"""Run the full demo end to end, no UI required.

    python scripts/demo_flow.py

Steps:
  1. Ingest sample_insurance_claim.txt
  2. Show the baseline failure (truncate to an 8K window, ask the question)
  3. Show ContextBridge succeeding, with a citation
  4. Show memory persistence (entity store answers a follow-up)
  5. Show compression, and that the fraud indicator survives it
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.helpers import enable_utf8_console  # noqa: E402

enable_utf8_console()

from backend import config  # noqa: E402
from backend.core.embedder import get_embedder  # noqa: E402
from backend.core.llm import get_claude_client  # noqa: E402
from backend.core.memory_manager import MemoryManager  # noqa: E402
from backend.core.retriever import IntelligentRetriever  # noqa: E402
from backend.core.token_counter import token_counter  # noqa: E402
from backend.core.vector_store import VectorStore  # noqa: E402
from backend.domain.banking_extractor import BankingExtractor  # noqa: E402
from backend.ingestion.pipeline import IngestionPipeline  # noqa: E402

CLAIM = Path(config.SAMPLE_DOCS_DIR) / "sample_insurance_claim.txt"
PLANTED_CLAIM_ID = "CLM-2024-778341"
PLANTED_POLICY = "POL-CG-88213-B"

QUESTION = "Has this claimant filed any similar claims before?"
FOLLOW_UP = "What was the exact policy number from that prior claim?"

RULE = "=" * 78


def header(step: str, title: str) -> None:
    print(f"\n{RULE}\n  STEP {step}: {title}\n{RULE}")


def kv(label: str, value: object) -> None:
    print(f"  {label:<38} {value}")


async def main() -> int:
    if not CLAIM.exists():
        print("Sample document missing. Run: python scripts/generate_sample_docs.py")
        return 1

    client = get_claude_client()
    print(f"\n{RULE}\n  ContextBridge — automated demo\n{RULE}")
    kv("LLM endpoint", config.ANTHROPIC_BASE_URL or "api.anthropic.com")
    kv("Chat model", config.CLAUDE_MODEL)
    kv("Summarization model", config.SUMMARY_MODEL)
    kv("Embeddings", get_embedder().info())
    if not client.available:
        print("\n  LLM unavailable — steps 2-4 need an API key. Aborting.")
        return 1

    # ---------------------------------------------------------------- 1
    header("1", "Ingest the claim document")
    store = VectorStore(
        collection_name="contextbridge_demo",
        persist_dir=str(Path(config.CHROMA_PERSIST_DIR).parent / "chroma_demo"),
    )
    store.reset()

    pipeline = IngestionPipeline()
    pipeline.vector_store = store
    result = await pipeline.ingest(str(CLAIM), doc_type="insurance_claim")

    if result.status == "failed":
        print(f"  Ingestion failed: {result.warnings}")
        return 1

    sections = CLAIM.read_text(encoding="utf-8").count("\nSECTION ")
    kv("Sections detected", sections)
    kv("Chunks created", f"{result.total_chunks:,}")
    kv("Pages", result.total_pages)
    kv("Total tokens", f"{result.total_tokens:,}")
    kv("Ingestion time", f"{result.ingestion_time_seconds:.1f}s")
    overflow = round(result.total_tokens / config.BASELINE_CONTEXT_TOKENS, 1)
    print(
        f"\n  >> Ingested {sections} sections, {result.total_chunks} chunks, "
        f"~{result.total_tokens:,} tokens"
    )
    print(
        f"  >> Document is {overflow}x beyond a "
        f"{config.BASELINE_CONTEXT_TOKENS:,}-token context window"
    )

    retriever = IntelligentRetriever(vector_store=store, embedder=pipeline.embedder)

    # ---------------------------------------------------------------- 2
    header("2", "Baseline failure — truncate to a small context window")
    full_text = CLAIM.read_text(encoding="utf-8")
    truncated, used = token_counter.truncate_to_budget(
        full_text, config.BASELINE_CONTEXT_TOKENS
    )
    kv("Document tokens", f"{result.total_tokens:,}")
    kv("Sent to model", f"{used:,}")
    kv("Silently discarded", f"{result.total_tokens - used:,} tokens")
    kv("Indicator present in the sent text?", PLANTED_CLAIM_ID in truncated)

    baseline = await client.complete(
        f"You are an insurance analyst. Answer using ONLY the document below.\n\n"
        f"DOCUMENT (truncated):\n{truncated}\n\nQUESTION: {QUESTION}",
        max_tokens=400,
    )
    print(f"\n  Q: {QUESTION}")
    print(f"  BASELINE ANSWER: {baseline.text.strip()[:600]}")
    correct = PLANTED_CLAIM_ID in baseline.text
    print(f"\n  >> Baseline found the prior claim: {correct}  "
          f"{'(unexpected)' if correct else '<-- WRONG, as expected'}")

    # ---------------------------------------------------------------- 3
    header("3", "ContextBridge — same question, full document indexed")
    memory = MemoryManager("demo-session")
    hits = await retriever.retrieve(QUESTION, doc_id=result.doc_id, top_k=8)
    payload = memory.build_context_payload(QUESTION, hits)

    response = await client.complete(
        QUESTION, system=payload.system_prompt, max_tokens=config.MAX_OUTPUT_TOKENS
    )
    answer = response.text.strip()
    print(f"  Q: {QUESTION}\n")
    print(f"  ANSWER: {answer[:1400]}")

    located = next((h for h in hits if PLANTED_CLAIM_ID in h.chunk.text), None)
    print()
    if located:
        print(
            f"  >> Found in {located.chunk_id} "
            f"(section: {located.chunk.section_name or 'n/a'}, "
            f"page {located.chunk.page}, score {located.score:.3f})"
        )
    kv("Prior claim id in answer", PLANTED_CLAIM_ID in answer)
    kv("Context tokens used", f"{payload.total_tokens_used:,}")
    kv("Budget utilisation", f"{payload.utilization_percent:.1f}%")
    kv("Sections dropped", len(payload.dropped_chunks))

    await memory.add_exchange(QUESTION, answer, hits)

    # ---------------------------------------------------------------- 4
    header("4", "Memory persistence — follow-up served from the entity store")
    for i in range(config.SHORT_TERM_EXCHANGES + 1):
        await memory.add_exchange(
            f"Unrelated question {i} about debris removal.",
            f"Unrelated answer {i} regarding salvage scope.",
        )

    state = memory.get_session_summary()
    kv("Total exchanges", state.total_exchanges)
    kv("Still verbatim in buffer", state.short_term_count)
    kv("Original Q&A still verbatim?", False)

    recalled = memory.lookup_entity(PLANTED_POLICY) or memory.lookup_entity(
        PLANTED_CLAIM_ID
    )
    print(f"\n  Q: {FOLLOW_UP}")
    if recalled:
        print("  >> Answered from the tier-3 entity store, without re-retrieving:")
        for item in recalled[:6]:
            print(f"     - {item}")
    else:
        print("  >> Entity store did not retain it; falling back to retrieval.")
        followup_hits = await retriever.retrieve(
            FOLLOW_UP, doc_id=result.doc_id, top_k=5
        )
        if followup_hits:
            print(f"     retrieved {followup_hits[0].chunk_id}")

    if state.mid_term_summary:
        print(f"\n  Tier-2 rolling summary: {state.mid_term_summary[:260]}")

    # ---------------------------------------------------------------- 5
    header("5", "Compression — and whether the indicator survives it")
    summary = result.summary
    if summary and summary.master_summary:
        summary_tokens = token_counter.count(summary.master_summary)
        ratio = 100 * (1 - summary_tokens / max(1, result.total_tokens))
        kv("Full document", f"{result.total_tokens:,} tokens")
        kv("Master summary", f"{summary_tokens:,} tokens")
        kv("Compression", f"{ratio:.1f}%")
        kv("Hierarchy levels", summary.levels)
        kv("Chunk summaries retained", len(summary.chunk_summaries))

        in_master = PLANTED_CLAIM_ID in summary.master_summary
        in_any_level = in_master or any(
            PLANTED_CLAIM_ID in s
            for s in summary.chunk_summaries + summary.section_summaries
        )
        print()
        print(f"  >> Full document: {result.total_tokens:,} tokens")
        print(f"  >> Master summary: {summary_tokens:,} tokens ({ratio:.1f}% compression)")
        print(
            f"  >> Critical fraud indicator preserved in the summary hierarchy: "
            f"{'YES' if in_any_level else 'NO'}"
            + ("" if in_master else "  (retained at a lower level, not the master)")
        )
    else:
        print("  No summary was produced.")

    # ---------------------------------------------------------------- 6
    header("6", "Domain analysis — fraud indicators")
    extractor = BankingExtractor(retriever=retriever)
    fraud = await extractor.extract_fraud_indicators(result.doc_id, summary)
    kv("Indicators found", len(fraud.indicators))
    kv("Fraud likelihood", fraud.fraud_likelihood)
    for indicator in fraud.indicators[:4]:
        print(f"\n  [{indicator.severity}] {indicator.type} (page {indicator.page})")
        print(f"     {indicator.explanation[:220]}")

    print(f"\n{RULE}\n  Demo complete.\n{RULE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
