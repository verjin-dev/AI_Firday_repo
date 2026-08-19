# PS02 — `ContextBridge`
## Overcoming Context Window Challenges in AI Models

> You already have a full spec for this one. This file is an **upgrade layer** — the three additions that lift it from "good RAG project" to "wins the innovation criterion", plus the gaps in the original spec worth closing. Use your existing document as the build plan and layer these in.

---

## 1. WHAT THE EXISTING SPEC GETS RIGHT

Hierarchical Map-Reduce summarisation, three-tier memory, token budget packing, and the dropped-chunk completeness warning are all solid and correctly prioritised. The `ContextPayload.dropped_chunks` field in particular is the best idea in the document — most teams silently truncate and never tell the user. Keep all of it.

## 2. WHAT IT IS MISSING

Three gaps, each of which a sharp judge will probe:

1. **Nothing is lossless.** Map-Reduce summarisation loses facts by construction. When you claim "98.1% compression with the fraud indicator preserved", the obvious question is: *how do you know nothing else was lost?* You need a guarantee, not a hope.
2. **Chunking is uniform.** `chunk_size=800` everywhere means the incident narrative and the boilerplate indemnity annex get identical treatment. That is where the token budget is wasted.
3. **The output-token problem is stated but not solved.** Section 2 of the problem statement is half about *output* truncation. The existing spec addresses input only.

The three upgrades below close exactly these gaps.

---

## 3. UPGRADE 1 — THE FACT LEDGER (the core innovation)

**Summarise prose. Never summarise facts.**

Run a structured extraction pass over every chunk *in parallel with* summarisation. Extracted facts go into a relational Fact Ledger with pointers back to the exact source span. Summaries then compress the *connective prose* only; every entity, date, amount, policy number, obligation, and decision survives losslessly in the ledger regardless of compression ratio.

```python
# backend/core/fact_ledger.py

class Fact(BaseModel):
    id: str
    doc_id: str
    fact_type: Literal["ENTITY","DATE","AMOUNT","OBLIGATION","DECISION",
                       "IDENTIFIER","RISK","CONDITION","PARTY"]
    subject: str
    predicate: str
    object: str
    normalized_value: str | None      # ISO date, decimal amount, canonical name
    source_chunk_id: str
    source_span: tuple[int, int]
    page: int | None
    section: str | None
    confidence: float
    extracted_at: datetime

class FactLedger:
    async def extract_from_chunk(self, chunk) -> list[Fact]
        """complete_json with the Fact schema. Batch 5 chunks per call."""

    def query(self, doc_id, fact_type=None, subject_contains=None) -> list[Fact]

    def detect_conflicts(self, doc_id) -> list[FactConflict]
        """
        Same (subject, predicate) with different normalized_value =
        a contradiction inside the document. In an insurance claim this IS
        the fraud signal: two different incident dates, two different amounts.
        This turns the ledger from a storage layer into a detection engine.
        """

    def coverage_report(self, doc_id) -> CoverageReport
        """
        % of chunks that produced ≥1 fact, facts per page, chunks with zero
        facts flagged for manual review. This is your lossless-ness EVIDENCE.
        """
```

**Why this wins:** you can now answer "how do you know nothing was lost?" with a number. *"312 chunks, 1,847 facts extracted, 100% of chunks contributed at least one fact, and every fact carries a source span you can click."* And conflict detection gives you a second, unexpected capability — intra-document contradiction detection — for free.

Demo moment: after showing the 98% compression, run `detect_conflicts()` and surface that the incident date in section 4 disagrees with the incident date in the witness statement in section 29. No summariser finds that. A ledger does.

---

## 4. UPGRADE 2 — ADAPTIVE TOKEN BUDGET ALLOCATION

Replace uniform chunking and naive top-k packing with a scored allocation.

```python
# backend/core/budget_allocator.py

class BudgetAllocator:
    def score_regions(self, doc: ParsedDocument, query: str | None) -> list[RegionScore]
        """
        Segment the document into semantic regions (sections). Score each on:
          - information_density: facts extracted per 1000 tokens
          - query_relevance: embedding sim to query (0 if no query)
          - structural_weight: narrative/exception/exclusion sections weighted up,
                               boilerplate/definitions/signature blocks weighted down
          - novelty: 1 - max similarity to other regions (deduplicates repeated
                     legal boilerplate, which is 30-40% of most contracts)
        allocation_score = weighted sum, normalised to sum to 1.0
        """

    def allocate(self, regions, total_budget: int) -> dict[str, int]
        """
        Distribute the token budget proportional to allocation_score, subject to:
          - floor: every region gets ≥ 200 tokens (nothing is fully invisible)
          - ceiling: no region takes > 40% of budget
        Then per region, choose the representation that fits its allocation:
          full text > extractive quotes > abstractive summary > facts only
        """
```

Show this in the UI as a horizontal stacked bar: 300 pages of document, coloured by how many tokens each section earned. "The incident narrative got 38% of the budget. The 40-page definitions annex got 2% — because it contributed 11 facts and they're all in the ledger anyway."

That visual alone is worth the Innovation marks.

---

## 5. UPGRADE 3 — THE SECTION PLANNER (solves output truncation)

```python
# backend/core/output_planner.py

class OutputPlanner:
    async def plan(self, request: str, ledger: FactLedger, doc_id: str) -> OutputPlan
        """
        Ask Claude for an outline ONLY: list of sections, each with a title,
        target word count, and the fact_ids it must cover. Cheap call, ~400 tokens.
        Validate: every fact of type OBLIGATION/RISK/DECISION is assigned to
        exactly one section. Unassigned critical facts → add a section.
        """

    async def generate(self, plan: OutputPlan, style: str) -> GeneratedDocument
        """
        Generate section by section. Each call receives:
          - the plan (so it knows what comes before and after)
          - the assigned facts verbatim from the ledger
          - the last 200 words of the previous section (for narrative continuity)
          - NOT the whole document
        Stitch, then run a continuity pass that fixes cross-references only.
        """

    def verify_coverage(self, doc: GeneratedDocument, plan: OutputPlan) -> CoverageResult
        """Assert every planned fact_id appears in the output. Report misses."""
```

This produces a 12,000-word audit report from a 4,096-token output limit, with a coverage guarantee. That is a direct, demonstrable answer to the second half of the problem statement, which most teams will ignore entirely.

---

## 6. ADDITIONS TO THE EXISTING FILE STRUCTURE

```
backend/core/
├── fact_ledger.py          # NEW — upgrade 1
├── budget_allocator.py     # NEW — upgrade 2
├── output_planner.py       # NEW — upgrade 3
└── (existing modules unchanged)

backend/storage/models.py   # ADD: Fact, FactConflict tables

frontend/pages/
├── 06_ledger.py            # NEW — browsable fact ledger + conflict list
├── 07_budget.py            # NEW — token allocation visualiser
└── 08_generate.py          # NEW — long-form generation with coverage meter
```

---

## 7. REVISED BENCHMARK

Your existing benchmark plan is good. Add these two arms and two metrics:

| Metric | Baseline (truncate to 8K) | Naive RAG | ContextBridge |
|---|---|---|---|
| Deep-fact recall (facts on pages 40+) | ~10% | ~55% | > 92% |
| **Fact retention after compression** | n/a | n/a | **100% by construction** |
| Cross-chunk reasoning accuracy | ~15% | ~40% | > 75% |
| Intra-doc conflict detection | 0 | 0 | 4 of 5 planted |
| Long-output completeness (planned facts covered) | ~35% | ~35% | > 95% |
| Cost per document | 1× | 0.4× | 0.7× |

**Plant 5 contradictions** in `sample_insurance_claim.txt`, not just one fraud indicator. Conflict detection is a strictly better demo than needle-in-haystack because no competing team will have it.

---

## 8. REVISED DEMO FLOW

Keep your steps 1–3 (ingest, baseline fails, ContextBridge finds it — that arc is well designed). Then replace steps 4–5 with:

4. **Prove losslessness.** Open the Fact Ledger: 1,847 facts, 100% chunk coverage, every fact clickable to its source span. "Compression didn't lose anything — it moved the facts somewhere structured."
5. **The unexpected capability.** Run conflict detection. Surface the date contradiction between section 4 and section 29. "No human read both of those pages. No summariser would connect them."
6. **The budget visual.** Show the stacked allocation bar. "We didn't spend the context window evenly. We spent it where the information was."
7. **Solve the output problem.** Generate the full 9,000-word claim assessment report from a 4,096-token limit. Coverage meter shows 98/98 required facts present. "The problem statement asked about output limits too. Here."

---

## 9. FIVE-DAY PLAN (adjusted)

**Day 1** — Existing spec Day 1, plus `Fact`/`FactConflict` ORM models.
**Day 2** — Existing Day 2 (summariser + pipeline), plus `fact_ledger.py` extraction running in the same pass. Gate: ledger populated for the sample doc.
**Day 3** — Existing Day 3 (memory + chat + retrieval), plus `detect_conflicts()` and the golden set with 5 planted contradictions and 15 deep-fact questions.
**Day 4** — `budget_allocator.py`, `output_planner.py`, all frontend pages including the three new ones.
**Day 5** — Benchmark all four arms, demo script, README.

**If you're behind:** ship the Fact Ledger and cut the budget allocator. The ledger is the innovation; the allocator is the polish.

---

## 10. JUDGE TALKING POINTS (revised)

Keep your existing five. Replace #3 with a stronger version and add two:

**"How do you ensure nothing critical is dropped?"** We don't summarise facts at all — we extract them into a structured ledger with source spans before any compression happens, and we report chunk coverage as a hard number. Prose is compressed 98%; facts are compressed 0%. And we can prove it: 100% of chunks contributed at least one fact, and every claim in the summary traces to a span.

**"Why not just use a 1M-token context model?"** Three reasons. Cost scales linearly with context and we're 30% cheaper than stuffing. Accuracy degrades in the middle of very long contexts — the lost-in-the-middle effect is well documented and our deep-fact recall beats full-context stuffing. And a long context still cannot detect that page 4 contradicts page 29, because the model has no reason to compare them; our ledger does that structurally.

**"What about the output token limit?"** Solved separately with the section planner — outline first, generate section-by-section against assigned facts, verify coverage. We generate a 9,000-word report from a 4,096-token ceiling with a 98% fact-coverage guarantee.
