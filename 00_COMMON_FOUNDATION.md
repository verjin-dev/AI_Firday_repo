# 00 — COMMON FOUNDATION
## Shared specification for all 15 AI Friday National Finals blueprints

> **Instructions for Claude Code:** Read this document FIRST, then read the specific problem blueprint (`PS01`–`PS15`). This file defines everything common to all projects: scaffolding, tech stack, conventions, error handling, and the Claude API integration pattern. The problem blueprint defines only what is unique. Do not duplicate this file's content into project code comments — just implement it.

---

## 1. UNIVERSAL TECH STACK

```
Language        Python 3.11+
Backend         FastAPI + uvicorn
LLM             Anthropic Claude API (claude-sonnet-4-6)
Embeddings      sentence-transformers (all-MiniLM-L6-v2, 384-dim)
Vector DB       ChromaDB (PersistentClient, local disk)
Relational      SQLite (via SQLAlchemy) for registries/traces/audit
Frontend        Streamlit (multi-page) + Plotly for charts
Tokenizer       tiktoken (cl100k_base) for budget estimation
Logging         loguru
Testing         pytest + pytest-asyncio
Config          pydantic-settings + python-dotenv
```

**Rule:** No paid third-party services beyond the Anthropic API. Everything else runs locally so the demo works offline-ish and cannot fail on a conference Wi-Fi network.

---

## 2. UNIVERSAL REQUIREMENTS.TXT BASE

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-dotenv==1.0.1
pydantic==2.7.1
pydantic-settings==2.3.0
anthropic==0.28.0
sentence-transformers==3.0.1
tiktoken==0.7.0
chromadb==0.5.3
sqlalchemy==2.0.30
streamlit==1.35.0
plotly==5.22.0
pandas==2.2.2
numpy==1.26.4
httpx==0.27.0
loguru==0.7.2
pytest==8.2.0
pytest-asyncio==0.23.7
```

Each blueprint adds its own extra dependencies in a clearly marked block.

---

## 3. UNIVERSAL PROJECT SCAFFOLD

Every project uses this skeleton. The blueprint tells you what to add under `core/`, `domain/`, and `frontend/pages/`.

```
<project_name>/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile                     # make install / make api / make ui / make demo / make test
│
├── backend/
│   ├── main.py                  # FastAPI entry, CORS, router registration, startup hooks
│   ├── config.py                # Settings class (pydantic-settings)
│   ├── core/                    # ← PROJECT-SPECIFIC MODULES GO HERE
│   ├── domain/                  # ← DOMAIN LOGIC (industry-specific)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py            # Claude wrapper (see §6)
│   │   ├── prompts.py           # All prompt templates, versioned
│   │   └── json_mode.py         # Structured output helper (see §7)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── db.py                # SQLAlchemy engine/session
│   │   ├── models.py            # ORM models
│   │   └── vector_store.py      # ChromaDB wrapper (see §8)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── health.py        # GET /api/health — always present
│   │       └── ...              # ← PROJECT-SPECIFIC ROUTES
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # loguru config, JSON sink to ./logs/app.jsonl
│       ├── errors.py            # Exception hierarchy (see §9)
│       └── timing.py            # @timed decorator, records ms into result objects
│
├── frontend/
│   ├── app.py                   # Streamlit entry + sidebar + navigation
│   ├── pages/                   # ← PROJECT-SPECIFIC PAGES
│   └── components/
│       ├── metric_card.py       # Big-number KPI card
│       ├── comparison_view.py   # Side-by-side baseline vs solution
│       └── evidence_panel.py    # Expandable source/evidence display
│
├── data/
│   ├── samples/                 # Generated demo data
│   ├── golden/                  # Labelled evaluation set (JSONL)
│   └── db/                      # SQLite + ChromaDB persistence (gitignored)
│
├── tests/
└── scripts/
    ├── generate_samples.py      # Build all demo data
    ├── benchmark.py             # Baseline vs solution, writes benchmark_results.json
    └── demo_flow.py             # Scripted end-to-end demo with printed narration
```

---

## 4. UNIVERSAL CONFIG BASE

```python
# backend/config.py
class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_FAST_MODEL: str = "claude-haiku-4-5-20251001"   # for cheap/bulk calls
    MAX_OUTPUT_TOKENS: int = 4096
    TEMPERATURE: float = 0.0                                # deterministic by default

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    CHROMA_PERSIST_DIR: str = "./data/db/chroma"
    SQLITE_URL: str = "sqlite:///./data/db/app.db"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_SECONDS: int = 120
    LLM_CACHE_ENABLED: bool = True     # disk cache keyed by hash(prompt+model+temp)
```

Project blueprints append their own constants to this class.

---

## 5. UNIVERSAL DATA CONVENTIONS

- **All IDs** are ULID-style strings: `f"{prefix}_{ulid}"` — e.g. `doc_01HX...`, `run_01HX...`.
- **All timestamps** are UTC ISO-8601 strings.
- **Every result object** carries: `status: Literal["success","partial","failed"]`, `warnings: List[str]`, `elapsed_ms: int`.
- **Every LLM call** is recorded in the `llm_traces` table: id, timestamp, prompt_hash, model, input_tokens, output_tokens, cost_usd, latency_ms, caller_module, session_id. This single table powers the metrics dashboards in almost every project — build it on Day 1.

```python
# storage/models.py — present in EVERY project
class LLMTrace(Base):
    __tablename__ = "llm_traces"
    id, created_at, session_id, caller, model, prompt_hash,
    input_tokens, output_tokens, cost_usd, latency_ms,
    temperature, cache_hit, status, error
```

**Cost table** (USD per 1M tokens) lives in `llm/client.py` as a dict so cost_usd is computed on every call.

---

## 6. UNIVERSAL CLAUDE CLIENT

```python
# backend/llm/client.py

class ClaudeClient:
    def __init__(self, settings: Settings)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        caller: str = "unknown",
        session_id: str | None = None,
    ) -> LLMResponse
        # 1. Compute prompt_hash = sha256(system + json(messages) + model + temp)
        # 2. If LLM_CACHE_ENABLED and hash in disk cache → return cached, cache_hit=True
        # 3. Call Anthropic with exponential backoff (1s, 2s, 4s) on
        #    RateLimitError / APIConnectionError / overloaded_error
        # 4. Record LLMTrace row ALWAYS (success or failure)
        # 5. Return LLMResponse(text, input_tokens, output_tokens, cost_usd,
        #                        latency_ms, cache_hit, model)

    async def complete_json(
        self, system: str, messages: list[dict],
        schema: Type[BaseModel], **kwargs
    ) -> BaseModel
        # See §7

    async def complete_batch(
        self, requests: list[BatchRequest], concurrency: int = 5
    ) -> list[LLMResponse]
        # asyncio.Semaphore-bounded concurrency; partial failures returned, never raised
```

**The disk cache is not optional.** It makes demos instant on the second run and protects you from live-demo API failures. Cache dir: `./data/db/llm_cache/`.

---

## 7. UNIVERSAL STRUCTURED OUTPUT PATTERN

Never parse free prose when you need data. Use this everywhere:

```python
# backend/llm/json_mode.py

async def complete_json(client, system, messages, schema: Type[BaseModel], retries=2):
    """
    1. Append to system prompt:
       "Respond with ONLY a JSON object matching this schema. No markdown fences,
        no preamble, no explanation.\nSCHEMA:\n{schema.model_json_schema()}"
    2. Prefill the assistant turn with "{" to force JSON start.
    3. Strip any ```json fences defensively.
    4. Validate with schema.model_validate_json().
    5. On ValidationError, retry with the error message appended as a repair turn.
    6. After `retries` failures, raise StructuredOutputError with the raw text attached.
    """
```

---

## 8. UNIVERSAL VECTOR STORE WRAPPER

```python
# backend/storage/vector_store.py

class VectorStore:
    def __init__(self, collection: str, persist_dir: str)
    def add(self, ids, texts, embeddings, metadatas) -> int
    def search(self, query_embedding, top_k=8, where: dict|None=None,
               threshold: float=0.0) -> list[SearchResult]
    def get(self, ids: list[str]) -> list[Record]
    def delete(self, where: dict) -> int
    def count(self, where: dict|None=None) -> int
```

`SearchResult` = `{id, text, score, metadata}`. Score is cosine similarity normalised to 0–1 (ChromaDB returns distance — convert with `1 - distance`).

---

## 9. UNIVERSAL ERROR HANDLING

```python
# backend/utils/errors.py
class AppError(Exception):            # base — has .code, .message, .details
class LLMError(AppError)              # API failures after retries
class StructuredOutputError(LLMError) # schema validation failed
class BudgetExceededError(AppError)   # token/cost budget hit
class RetrievalError(AppError)
class IngestionError(AppError)
class PolicyViolationError(AppError)  # governance/security refusals
```

Rules enforced in every module:
1. **Never let an exception reach the frontend raw.** FastAPI exception handler converts `AppError` → `{"status":"failed","code":...,"message":...,"details":...}` with HTTP 4xx/5xx.
2. **Never crash a batch on one item.** Partial results + `warnings[]` always.
3. **Never silently drop data.** Anything discarded goes into `warnings[]` with a count and reason.
4. **Log full traceback** with loguru at ERROR; log a structured one-line JSON event at INFO for every request.
5. **Degrade, don't die.** Embedding failure → keyword fallback. LLM judge failure → heuristic score with a `degraded=True` flag.

---

## 10. UNIVERSAL FRONTEND CONVENTIONS

Sidebar in every app:
- Project name + one-line pitch
- Active session / selected artifact
- **Live cost + token counter** for the session (read from `llm_traces`)
- "Reset demo" button
- A `st.toggle("Baseline mode")` — flipping it runs the naive approach so judges see the difference on the same screen

Every results page must include, above the fold:
- A **big-number KPI row** (3–4 `metric_card` components)
- A **baseline vs solution comparison** widget
- An **evidence panel** — nothing is asserted without a clickable source

Colour convention: green = pass/safe, amber = warning/degraded, red = fail/blocked, grey = not applicable.

---

## 11. UNIVERSAL BENCHMARK CONTRACT

`scripts/benchmark.py` in every project must:
1. Load `data/golden/*.jsonl` — the labelled evaluation set (min 40 cases).
2. Run **Arm A (baseline)**: the naive/obvious approach a competing team would build.
3. Run **Arm B (solution)**: the full system.
4. Compute the project's headline metrics for both arms.
5. Write `benchmark_results.json` with per-case rows plus aggregates.
6. Print a formatted comparison table to stdout.
7. Emit `benchmark_chart.png` (matplotlib) for the slide deck.

**The golden set is generated, labelled, and committed on Day 2 — not Day 5.** Every project's score depends on having a real number to show.

---

## 12. UNIVERSAL 5-DAY RHYTHM

| Day | Theme | Gate at end of day |
|---|---|---|
| 1 | Scaffold, config, storage, LLM client, trace table, sample data generator | `make api` runs, health check green, one LLM trace recorded |
| 2 | Core algorithm — the actual innovation | Core module passes its unit tests on sample data |
| 3 | Golden set + benchmark + evaluation loop | `make benchmark` prints a real comparison table |
| 4 | API routes + full Streamlit UI wired end to end | Judge could click through unaided |
| 5 | Demo script, error passes, README, dry runs | `make demo` runs clean 3× in a row |

**Day 3 is the most important day and teams always skip it.** Without a benchmark you lose points on Research (7%), Key Metrics (4%), and Business Value (5%) — 16% of the total.

---

## 13. UNIVERSAL README STRUCTURE

1. One-paragraph pitch
2. ASCII architecture diagram
3. Quick start (≤5 commands)
4. Demo walkthrough — exactly what to click, in order, with expected output
5. Headline metrics table (baseline vs solution)
6. Key technical decisions and why
7. Known limitations and scope boundaries ← *do not omit; Scope is 5% of the score*
8. Roadmap to production (90 days, 3 phases, named dependencies) ← *Roadmap is 10%*

---

## 14. UNIVERSAL RUBRIC MAPPING

Wire these into the deliverable deliberately:

| Criterion | Weight | Where it is earned |
|---|---|---|
| Clarity | 8% | README §1, one-slide problem statement |
| Context | 5% | README §2, industry framing in the demo |
| Research | 7% | Cite NIST AI RMF / EU AI Act / OWASP LLM Top 10 / WCAG / published benchmarks |
| Scope | 5% | README §7 — explicit non-goals |
| Logic | 8% | Architecture diagram + why-this-approach section |
| Feasibility | 7% | Working prototype on commodity hardware + local models |
| Alignment | 7% | Map each "Expected Outcome" bullet from the PS to a feature |
| Risk Assessment | 4% | Assumptions + failure modes table |
| Innovation | 4% | The named novel mechanism in each blueprint (§"Core Innovation") |
| Functionality | 6% | `make demo` runs clean |
| Usability | 5% | Streamlit UI, no CLI needed for the judge |
| Business Value | 5% | Benchmark table + cost/time saved |
| Customer Demo Readiness | 4% | Sample data pre-loaded, cached LLM responses, offline-safe |
| Value Proposition | 6% | Quantified: "X hours → Y minutes" |
| Scalability | 5% | Complexity analysis + horizontal scaling notes |
| Key Metrics | 4% | Headline metrics with baseline AND target |
| Milestones / Risk Mgmt / Dependencies | 10% | README §8 |

---

## 15. DEMO SAFETY RULES

1. **Pre-warm the cache.** Run `make demo` once before presenting; cached responses make the live run instant and immune to API issues.
2. **Ship a `--offline` flag** that serves entirely from cache with no network calls. If the venue Wi-Fi dies, you still demo.
3. **Never live-upload a file on stage.** Pre-ingest; have the upload flow available only as a "and here's how it got there" aside.
4. **Have the benchmark chart as a static PNG.** Do not compute it live.
5. **Time the demo to 4 minutes** of clicking, leaving room for questions.

---

*End of common foundation. Now read the specific problem blueprint.*
