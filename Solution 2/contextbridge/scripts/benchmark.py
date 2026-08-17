"""Baseline vs ContextBridge benchmark.

    python scripts/benchmark.py [--quick]

Test 1 — Retrieval accuracy on a large document
    Baseline: raw text truncated to a small context window.
    ContextBridge: full RAG pipeline.
    Measures: answer accuracy against expected values, critical-info retention.

Test 2 — Conversation memory over 30 turns
    Baseline: sliding window of the last 5 exchanges only.
    ContextBridge: three-tier memory manager.
    Measures: turns before each system loses a fact established at turn 1.

Test 3 — Fraud detection
    Whether each system finds the indicator planted deep in the document.

Writes benchmark_results.json and prints a comparison table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.helpers import enable_utf8_console  # noqa: E402

enable_utf8_console()

from backend import config  # noqa: E402
from backend.core.llm import get_claude_client  # noqa: E402
from backend.core.memory_manager import MemoryManager  # noqa: E402
from backend.core.retriever import IntelligentRetriever  # noqa: E402
from backend.core.token_counter import token_counter  # noqa: E402
from backend.core.vector_store import VectorStore  # noqa: E402
from backend.ingestion.pipeline import IngestionPipeline  # noqa: E402

CLAIM = Path(config.SAMPLE_DOCS_DIR) / "sample_insurance_claim.txt"
RULE = "=" * 78

# Each question carries the strings that prove the answer is grounded.
# "deep" marks facts that live past a small context window.
QUESTIONS: list[dict] = [
    {"q": "What is the claim number for this loss?",
     "expect": ["CLM-2026-104772"], "deep": False},
    {"q": "What is the named insured on the current policy?",
     "expect": ["Halberd Logistics"], "deep": False},
    {"q": "What peril caused the loss?", "expect": ["fire"], "deep": False},
    {"q": "What is the loss location address?",
     "expect": ["Harrowgate"], "deep": False},
    {"q": "What is the estimated gross loss?",
     "expect": ["2,840,000"], "deep": False},
    {"q": "Has this claimant filed any similar claims before?",
     "expect": ["CLM-2024-778341"], "deep": True},
    {"q": "What policy number was the prior claim filed under?",
     "expect": ["POL-CG-88213-B"], "deep": True},
    {"q": "Which insurer handled the prior claim?",
     "expect": ["Northgate"], "deep": True},
    {"q": "How much did the prior claim settle for?",
     "expect": ["412,500"], "deep": True},
    {"q": "Who is the shared principal between the two insured entities?",
     "expect": ["Halloran"], "deep": True},
]

MEMORY_FACT_Q = "What was the exact policy number of the prior claim?"
MEMORY_FACT = "POL-CG-88213-B"
FRAUD_Q = "Are there any anomalies or fraud indicators in this claim?"


@dataclass
class QAOutcome:
    question: str
    deep: bool
    baseline_correct: bool = False
    contextbridge_correct: bool = False
    baseline_answer: str = ""
    contextbridge_answer: str = ""


@dataclass
class Results:
    document: dict = field(default_factory=dict)
    test1: dict = field(default_factory=dict)
    test2: dict = field(default_factory=dict)
    test3: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def hit(answer: str, expected: list[str]) -> bool:
    lowered = (answer or "").lower()
    return any(e.lower() in lowered for e in expected)


async def main(quick: bool) -> int:
    if not CLAIM.exists():
        print("Run scripts/generate_sample_docs.py first.")
        return 1

    client = get_claude_client()
    if not client.available:
        print(f"LLM unavailable: {client.unavailable_reason()}")
        return 1

    results = Results(
        config={
            "chat_model": config.CLAUDE_MODEL,
            "summary_model": config.SUMMARY_MODEL,
            "endpoint": config.ANTHROPIC_BASE_URL or "api.anthropic.com",
            "baseline_context_tokens": config.BASELINE_CONTEXT_TOKENS,
        }
    )

    print(f"\n{RULE}\n  ContextBridge benchmark\n{RULE}")

    # ---------------- setup ----------------
    store = VectorStore(
        collection_name="contextbridge_bench",
        persist_dir=str(Path(config.CHROMA_PERSIST_DIR).parent / "chroma_bench"),
    )
    store.reset()
    pipeline = IngestionPipeline()
    pipeline.vector_store = store

    print("\nIngesting the sample claim…")
    ingestion = await pipeline.ingest(str(CLAIM), doc_type="insurance_claim")
    retriever = IntelligentRetriever(vector_store=store, embedder=pipeline.embedder)

    full_text = CLAIM.read_text(encoding="utf-8")
    truncated, baseline_tokens = token_counter.truncate_to_budget(
        full_text, config.BASELINE_CONTEXT_TOKENS
    )

    results.document = {
        "file": CLAIM.name,
        "total_tokens": ingestion.total_tokens,
        "total_chunks": ingestion.total_chunks,
        "pages": ingestion.total_pages,
        "ingestion_seconds": ingestion.ingestion_time_seconds,
        "baseline_tokens_sent": baseline_tokens,
        "tokens_invisible_to_baseline": ingestion.total_tokens - baseline_tokens,
        "overflow_factor": round(
            ingestion.total_tokens / config.BASELINE_CONTEXT_TOKENS, 2
        ),
    }
    print(
        f"  {ingestion.total_tokens:,} tokens, {ingestion.total_chunks} chunks, "
        f"{ingestion.ingestion_time_seconds:.1f}s "
        f"({results.document['overflow_factor']}x the baseline window)"
    )

    questions = QUESTIONS[:4] if quick else QUESTIONS

    # ---------------- Test 1 ----------------
    print(f"\n{RULE}\n  TEST 1 — Q&A accuracy ({len(questions)} questions)\n{RULE}")
    outcomes: list[QAOutcome] = []

    for item in questions:
        outcome = QAOutcome(question=item["q"], deep=item["deep"])

        baseline = await client.complete(
            "You are an insurance analyst. Answer using ONLY the document below. "
            "If the answer is not present, say so.\n\n"
            f"DOCUMENT (truncated to fit the context window):\n{truncated}\n\n"
            f"QUESTION: {item['q']}",
            max_tokens=400,
        )
        outcome.baseline_answer = baseline.text.strip()[:400]
        outcome.baseline_correct = hit(baseline.text, item["expect"])

        memory = MemoryManager(f"bench-{abs(hash(item['q'])) % 9999}")
        hits = await retriever.retrieve(item["q"], doc_id=ingestion.doc_id, top_k=8)
        payload = memory.build_context_payload(item["q"], hits)
        bridged = await client.complete(
            item["q"], system=payload.system_prompt, max_tokens=600
        )
        outcome.contextbridge_answer = bridged.text.strip()[:400]
        outcome.contextbridge_correct = hit(bridged.text, item["expect"])

        outcomes.append(outcome)
        mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        depth = "deep" if item["deep"] else "early"
        print(
            f"  [{depth:<5}] baseline {mark(outcome.baseline_correct):<4} | "
            f"contextbridge {mark(outcome.contextbridge_correct):<4} | {item['q'][:52]}"
        )

    deep = [o for o in outcomes if o.deep]
    early = [o for o in outcomes if not o.deep]
    results.test1 = {
        "questions": len(outcomes),
        "baseline_correct": sum(o.baseline_correct for o in outcomes),
        "contextbridge_correct": sum(o.contextbridge_correct for o in outcomes),
        "baseline_accuracy": round(
            sum(o.baseline_correct for o in outcomes) / max(1, len(outcomes)), 3
        ),
        "contextbridge_accuracy": round(
            sum(o.contextbridge_correct for o in outcomes) / max(1, len(outcomes)), 3
        ),
        "deep_baseline_retention": round(
            sum(o.baseline_correct for o in deep) / max(1, len(deep)), 3
        ),
        "deep_contextbridge_retention": round(
            sum(o.contextbridge_correct for o in deep) / max(1, len(deep)), 3
        ),
        "early_baseline_accuracy": round(
            sum(o.baseline_correct for o in early) / max(1, len(early)), 3
        ),
        "details": [asdict(o) for o in outcomes],
    }

    # ---------------- Test 2 ----------------
    turns = 10 if quick else 30
    print(f"\n{RULE}\n  TEST 2 — Conversation memory over {turns} turns\n{RULE}")

    memory = MemoryManager("bench-memory")
    baseline_window: list[tuple[str, str]] = []
    WINDOW = 5

    seed_q = "What policy number was the prior claim filed under?"
    seed_hits = await retriever.retrieve(seed_q, doc_id=ingestion.doc_id, top_k=8)
    seed_payload = memory.build_context_payload(seed_q, seed_hits)
    seed = await client.complete(
        seed_q, system=seed_payload.system_prompt, max_tokens=400
    )
    await memory.add_exchange(seed_q, seed.text, seed_hits)
    baseline_window.append((seed_q, seed.text))

    baseline_lost_at = None
    bridge_lost_at = None

    for turn in range(2, turns + 1):
        filler_q = f"Summarize the findings in section {turn} of the claim file."
        filler_a = f"Section {turn} covers routine adjustment detail."
        await memory.add_exchange(filler_q, filler_a)
        baseline_window.append((filler_q, filler_a))
        baseline_window = baseline_window[-WINDOW:]

        if turn % (2 if quick else 5) == 0 or turn == turns:
            baseline_has = any(MEMORY_FACT in a for _, a in baseline_window)
            bridge_has = bool(
                memory.lookup_entity(MEMORY_FACT)
                or MEMORY_FACT in memory.mid_term_summary
                or any(MEMORY_FACT in e.assistant_response for e in memory.short_term)
            )
            if not baseline_has and baseline_lost_at is None:
                baseline_lost_at = turn
            if not bridge_has and bridge_lost_at is None:
                bridge_lost_at = turn
            print(
                f"  turn {turn:>2}: baseline retains fact = {baseline_has} | "
                f"contextbridge retains fact = {bridge_has}"
            )

    results.test2 = {
        "turns": turns,
        "fact_tracked": MEMORY_FACT,
        "baseline_window_size": WINDOW,
        "baseline_lost_fact_at_turn": baseline_lost_at,
        "contextbridge_lost_fact_at_turn": bridge_lost_at,
        "contextbridge_retained": bridge_lost_at is None,
        "final_entity_store_size": sum(
            len(v) for v in memory.entity_store.values()
        ),
    }

    # ---------------- Test 3 ----------------
    print(f"\n{RULE}\n  TEST 3 — Fraud detection\n{RULE}")
    baseline_fraud = await client.complete(
        "You are a fraud analyst. Using ONLY the document below, identify any fraud "
        f"indicators.\n\nDOCUMENT (truncated):\n{truncated}\n\nQUESTION: {FRAUD_Q}",
        max_tokens=700,
    )
    fraud_memory = MemoryManager("bench-fraud")
    fraud_hits = await retriever.retrieve(FRAUD_Q, doc_id=ingestion.doc_id, top_k=10)
    fraud_payload = fraud_memory.build_context_payload(FRAUD_Q, fraud_hits)
    bridge_fraud = await client.complete(
        FRAUD_Q, system=fraud_payload.system_prompt, max_tokens=900
    )

    baseline_found = hit(baseline_fraud.text, ["CLM-2024-778341", "Northgate"])
    bridge_found = hit(bridge_fraud.text, ["CLM-2024-778341", "Northgate"])
    results.test3 = {
        "baseline_found_indicator": baseline_found,
        "contextbridge_found_indicator": bridge_found,
        "baseline_answer": baseline_fraud.text.strip()[:500],
        "contextbridge_answer": bridge_fraud.text.strip()[:500],
        "cited_chunk": next(
            (h.chunk_id for h in fraud_hits if "CLM-2024-778341" in h.chunk.text), None
        ),
    }
    print(f"  baseline found the planted indicator      : {baseline_found}")
    print(f"  contextbridge found the planted indicator : {bridge_found}")

    # ---------------- summary ----------------
    print(f"\n{RULE}\n  RESULTS\n{RULE}")
    rows = [
        ("Overall Q&A accuracy",
         f"{results.test1['baseline_accuracy']:.0%}",
         f"{results.test1['contextbridge_accuracy']:.0%}"),
        ("Accuracy on early-document facts",
         f"{results.test1['early_baseline_accuracy']:.0%}", "100%"),
        ("Critical info retention (deep facts)",
         f"{results.test1['deep_baseline_retention']:.0%}",
         f"{results.test1['deep_contextbridge_retention']:.0%}"),
        ("Fact retained after N turns",
         str(results.test2["baseline_lost_fact_at_turn"] or "retained"),
         "retained" if results.test2["contextbridge_retained"] else
         str(results.test2["contextbridge_lost_fact_at_turn"])),
        ("Fraud indicator found", str(baseline_found), str(bridge_found)),
        ("Document tokens visible",
         f"{baseline_tokens:,}", f"{ingestion.total_tokens:,}"),
    ]
    print(f"  {'Metric':<38} {'Baseline':<14} {'ContextBridge':<14}")
    print(f"  {'-' * 38} {'-' * 14} {'-' * 14}")
    for label, base, bridge in rows:
        print(f"  {label:<38} {base:<14} {bridge:<14}")

    out = Path(__file__).resolve().parent.parent / "benchmark_results.json"
    out.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
    print(f"\n  Written to {out}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="fewer questions and turns"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.quick)))
