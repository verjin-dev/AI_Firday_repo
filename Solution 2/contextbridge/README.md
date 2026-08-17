# ContextBridge

**Overcoming LLM context window limitations for Banking & Insurance documents.**

AI Friday National Finals — Context Window Hackathon

---

## The problem

A 47-page commercial insurance claim is ~19,000 tokens. A 200-page one is ~90,000.
Send either to a model with an 8K context window and most of the document is
silently discarded — no error, no warning, just a confident answer built on the
first 8,000 tokens.

That failure is not theoretical. In our sample claim, the fact that the claimant
already filed an identical claim eighteen months earlier — under a different
policy, with a different insurer — sits in section 31 of 47. A truncated model
answers *"no prior claims found."* It is wrong, and it has no way to know it.

## The solution

Four layers, each addressing a different way context limits cause information loss:

| Layer | Module | What it does |
| --- | --- | --- |
| **Hierarchical summarization** | `core/summarizer.py` | True Map-Reduce over every chunk. Retains *all* intermediate levels, not just the final summary, so buried facts survive compression |
| **RAG retrieval** | `core/retriever.py` | Hybrid semantic + BM25 keyword search with section-aware reranking. Keyword matching is what reliably catches exact identifiers like `CLM-2024-778341` |
| **Multi-tier memory** | `core/memory_manager.py` | Verbatim recent turns → rolling LLM summary → structured entity store. A fact from turn 1 is still answerable at turn 30 |
| **Completeness auditing** | `domain/completeness_checker.py` | Every answer reports what *didn't* fit in context. Silent truncation is the actual enemy |

---

## Architecture

```
                            ┌──────────────────────────┐
  PDF / DOCX / TXT ───────► │   IngestionPipeline      │
                            │  parse → chunk → embed   │
                            └───────────┬──────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
        ┌────────────────┐   ┌────────────────────┐  ┌─────────────────┐
        │   ChromaDB     │   │   Summarizer       │  │ EntityExtractor │
        │ vectors +      │   │  Map-Reduce        │  │ regex + LLM     │
        │ chunk metadata │   │  L0→L1→L2→L3       │  │                 │
        └───────┬────────┘   └─────────┬──────────┘  └────────┬────────┘
                │                      │                      │
                ▼                      ▼                      ▼
        ┌───────────────────────────────────────────────────────────────┐
        │              IntelligentRetriever  (semantic + BM25)          │
        └────────────────────────────┬──────────────────────────────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  MemoryManager — packs context to a hard token budget          │
        │  1 system  2 entities  3 summary  4 chunks  5 recent turns     │
        │  → ContextPayload { included, DROPPED, tokens_used }           │
        └────────────────────────────┬──────────────────────────────────┘
                                     ▼
        ┌────────────────────┐   ┌───────────────────────────────────┐
        │  Claude / LLM      │──►│ CompletenessChecker               │
        │  cited answer      │   │ "3 sections didn't fit" + LOW/MED  │
        └────────────────────┘   └───────────────────────────────────┘
```

The **dropped chunks** path is the part that matters. Most RAG systems throw away
what doesn't fit. ContextBridge tracks it and tells the user.

---

## Quick start

```bash
cd contextbridge
python -m pip install -r requirements.txt
cp .env.example .env          # then add your API key
python scripts/generate_sample_docs.py
uvicorn backend.main:app --port 8000
```

Then, in a second terminal:

```bash
streamlit run frontend/app.py
```

Backend docs at `http://localhost:8000/docs`, UI at `http://localhost:8501`.

### Verify without the UI

```bash
python scripts/demo_flow.py
```

---

## Configuration

Everything is env-driven (`backend/config.py`). The essentials:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required for chat, summarization, extraction |
| `ANTHROPIC_BASE_URL` | unset | Any Anthropic-compatible `/v1/messages` endpoint. Accepts `host:port` or `host:port/v1` |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Chat, reduce phase, domain analysis |
| `SUMMARY_MODEL` | = `CLAUDE_MODEL` | Per-chunk map phase — the biggest cost lever |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 80 | Tokens per chunk |
| `TOP_K_CHUNKS` | 8 | Chunks retrieved per query |
| `CONTEXT_BUDGET_TOKENS` | 150000 | Hard packing budget |
| `SUMMARY_CONCURRENCY` | 6 | Parallel map-phase calls |
| `EMBEDDING_BASE_URL` | unset | Optional OpenAI-compatible `/embeddings` endpoint |

### Running against a local model router

ContextBridge talks to anything exposing an Anthropic-compatible `/v1/messages`.
Point it at a local router and use routing aliases in place of model ids:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:31415/v1
ANTHROPIC_API_KEY=<router key>
CLAUDE_MODEL=auto:smart      # chat + reduce: quality matters
SUMMARY_MODEL=auto:fast      # one call per chunk: cost matters
EMBEDDING_BASE_URL=http://127.0.0.1:31415/v1
```

> **Changing embedding backend changes vector width.** The store records the
> dimension it was built with and refuses mismatched reads and writes rather than
> returning meaningless neighbours. Delete `data/chroma_db` and re-ingest when you
> switch.

### Embedding backends

Tried in order, first success wins:

1. Remote `/embeddings` endpoint (if `EMBEDDING_BASE_URL` is set) — no download
2. `sentence-transformers` all-MiniLM-L6-v2 — best local quality, needs torch
3. ChromaDB's bundled ONNX MiniLM — same model, no torch
4. Deterministic hashing — degraded but keeps the system runnable offline

Whichever is active is reported by `GET /api/health` and shown in the sidebar.

---

## Demo walkthrough (for judges)

**1. Upload** → *Load a sample* → `sample_insurance_claim.txt`

> 47 sections · 48 chunks · ~19,000 tokens
> *"Without ContextBridge this document would exceed an 8K context window by 2.4×"*

**2. Chat** → ask:

> *Has this claimant filed any similar claims before?*

ContextBridge answers **yes**, and cites the chunk: section *Prior Claims History*,
**page 31**. It names claim `CLM-2024-778341`, policy `POL-CG-88213-B`, insurer
Northgate Mutual, settled for $412,500 — and flags the non-disclosure at
application. `scripts/demo_flow.py` shows the same question against a truncated
8K context returning *"no information regarding any prior claims."*

**3. Follow-up** → ask several unrelated questions, then:

> *What was the exact policy number from that prior claim?*

The original exchange has left the verbatim buffer, but the entity store still
holds the identifier — answered from tier-3 memory, no re-retrieval.

**4. Summarize** → the compression ratio, with all hierarchy levels browsable.

**5. Extract** → fraud indicators with severity, evidence quotes and page
references; contract clauses; a 0-100 risk gauge.

### The planted signals

| Document | Signal | Location |
| --- | --- | --- |
| `sample_insurance_claim.txt` | Duplicate prior claim, different insurer, undisclosed | §31, page 31 of 47 |
| `sample_contract.txt` | Uncapped liability carve-out | §29.3, page 29 |
| `sample_contract.txt` | Vanuatu jurisdiction + jury waiver | §34, page 34 |
| `sample_fraud_case.txt` | 14 transfers of $9,400–$9,900 (structuring) | §11 |

All are generated deterministically (seeded RNG), and none appear in the opening
sections — a truncated context window provably cannot see them.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Status, model, embedding backend, index size |
| `POST` | `/api/upload` | Ingest (multipart: `file`, `doc_type`, `run_summarization`) |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document and its chunks |
| `POST` | `/api/chat` | Cited Q&A with completeness report |
| `GET`/`DELETE` | `/api/session/{id}` | Inspect / clear session memory |
| `POST` | `/api/summarize` | Hierarchical summary + compression stats |
| `POST` | `/api/extract` | Fraud / clauses / risk / entities |

---

## Testing

```bash
python -m pytest tests/          # 109 tests
python scripts/benchmark.py      # baseline vs ContextBridge, writes JSON
```

Tests stub the LLM, so the suite runs offline and deterministically. The pipeline
tests are true integration tests against a temporary ChromaDB and assert the
planted fraud indicator survives ingestion *and* is retrievable.

---

## Technical decisions

**Section-aware chunking over fixed-size chunking.** Splitting on detected
headings first means a chunk rarely straddles two topics, and every chunk carries
a `section_name` — which is what makes citations read as *"Prior Claims History,
page 31"* rather than *"chunk 89."*

**Hybrid retrieval, not pure vector search.** Embeddings are weak at exact
identifiers; `CLM-2024-778341` is a token soup to a semantic model. BM25 catches
it. Neither alone is sufficient for financial documents, which are dense with
policy numbers and amounts.

**All hierarchy levels retained.** Compressing 19,000 tokens to 900 loses detail
by definition. Keeping level-1 and level-2 summaries means a fact absent from the
master summary is still reachable one level down, and the UI can show exactly
where it survived.

**Failures degrade, never crash.** A failed summarization call falls back to raw
chunk text rather than dropping the chunk; a dead embedder falls back to keyword
retrieval; a missing API key still allows ingestion and retrieval. Every result
object carries a `warnings` list, and `completeness_score` counts genuinely
summarized chunks — not merely non-empty ones.

**Dropped context is surfaced, not hidden.** `ContextPayload` returns
`dropped_chunks` alongside `included_chunks`, and every chat response says how
many sections didn't fit and which ones.

---

## Judge Q&A

**"What if the document is 1,000 pages?"**
Hierarchical summarization is O(N) LLM calls, not O(N²) — N map calls, then
log-ish reduce levels. `SUMMARY_CONCURRENCY` bounds parallelism, and
`REDUCE_BATCH_TOKEN_BUDGET` bounds each reduce call regardless of document size.

**"Why not just use a model with a bigger context window?"**
Cost and latency scale with context size, and retrieval accuracy degrades in the
middle of very long contexts. Sending 90,000 tokens to answer one question costs
~30× more than sending the 3,000 that matter. The token meter in the UI shows
real utilisation — the demo answers on ~4,000 tokens against a 150,000 budget.

**"How do you know nothing critical was dropped?"**
Three mechanisms: all summary levels are retained; `CompletenessChecker` audits
every response against what was dropped and downgrades confidence; the entity
store tracks identifiers independently of retrieval.

**"What's the accuracy vs baseline?"**
`scripts/benchmark.py` — accuracy on early vs deep facts, memory retention over
30 turns, and fraud detection, written to `benchmark_results.json`.

**"Does this generalize beyond Banking & Insurance?"**
The core (chunk → embed → retrieve → pack → audit) is domain-agnostic. Only
`domain/` is specific, and it plugs in via `doc_type`.

---

## Project structure

```
contextbridge/
├── backend/
│   ├── config.py           env-driven configuration
│   ├── main.py             FastAPI app
│   ├── core/               chunker, embedder, vector_store, summarizer,
│   │                       memory_manager, retriever, token_counter, llm,
│   │                       models, registry
│   ├── ingestion/          pdf/docx/text parsers + pipeline
│   ├── domain/             banking_extractor, entity_extractor,
│   │                       completeness_checker
│   ├── api/                schemas + routes
│   └── utils/              logger, helpers
├── frontend/               Streamlit app, 5 pages, 3 components
├── data/sample_docs/       generated demo documents
├── scripts/                generate_sample_docs, benchmark, demo_flow
└── tests/                  109 tests
```

### Deviations from the original spec

- **`anthropic` SDK version.** The spec pinned `0.28.0`, which predates the models
  it targets and cannot call them. Uses a current SDK.
- **Three extra modules.** `core/models.py` (shared dataclasses — the spec's types
  need one importable home), `core/llm.py` (Claude client wrapper), and
  `core/registry.py` (caches summaries between requests, which the spec's
  `/summarize` route assumes exists).
- **`MemoryManager.add_exchange` is async.** Buffer eviction triggers LLM
  summarization and entity extraction; a sync signature would block the event loop.
- **Streamlit `pages/` instead of `st.navigation()`.** `st.navigation` requires
  Streamlit ≥ 1.36; the spec pinned 1.35.
- **Requirements use ranges, not hard pins**, so the stack installs on current
  Python.

---

*Built for the AI Friday National Finals Context Window Hackathon.*
