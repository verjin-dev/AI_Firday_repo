# AI Friday National Finals — Context Window Hackathon
# Claude Code Master Prompt & Architecture Blueprint

> **Instructions for Claude Code:** Read this entire document before writing a single line of code. This is your complete specification. Build everything described here across the 5-day plan. Follow the file structure exactly, implement every module described, and ensure the demo flow works end to end.

---

## 0. PROJECT OVERVIEW

**Project Name:** `ContextBridge` — Overcoming LLM Context Window Limitations

**Problem:** LLMs have fixed context windows. Large documents (contracts, medical records, SOPs, audit trails) get truncated, causing critical information loss in enterprise workflows.

**Solution:** A multi-layered intelligent document processing system that combines:
- Hierarchical chunking + summarization (Map-Reduce)
- RAG (Retrieval-Augmented Generation) with semantic search
- Multi-level memory management for long conversations
- Domain-specific extraction for Banking & Insurance

**Demo Scenario:** Banking & Insurance — insurance claim fraud detection and contract clause analysis across 100+ page documents.

**Tech Stack:**
- Backend: Python 3.11+, FastAPI
- LLM: Anthropic Claude API (claude-sonnet-4-6)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- Vector DB: ChromaDB (local)
- Chunking: LangChain RecursiveCharacterTextSplitter
- Memory: Custom multi-tier memory manager
- Frontend: Streamlit
- File Parsing: PyMuPDF (fitz), python-docx, pandas
- Token Counting: tiktoken

---

## 1. COMPLETE FILE STRUCTURE

Build this exact directory structure:

```
contextbridge/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── backend/
│   ├── main.py                          # FastAPI app entry point
│   ├── config.py                        # All configuration, env vars, constants
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chunker.py                   # Document chunking engine
│   │   ├── embedder.py                  # Embedding generation
│   │   ├── vector_store.py              # ChromaDB wrapper
│   │   ├── summarizer.py                # Hierarchical Map-Reduce summarizer
│   │   ├── memory_manager.py            # Multi-tier conversation memory
│   │   ├── retriever.py                 # Semantic retrieval + reranking
│   │   └── token_counter.py             # Token budget management
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py                # PDF text + metadata extraction
│   │   ├── docx_parser.py               # Word document parser
│   │   ├── text_parser.py               # Plain text parser
│   │   └── pipeline.py                  # Full ingestion orchestrator
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── banking_extractor.py         # Clause, risk, fraud flag extractor
│   │   ├── entity_extractor.py          # Names, dates, amounts, decisions
│   │   └── completeness_checker.py      # Detect what was dropped/missed
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── upload.py                # POST /upload — ingest document
│   │   │   ├── chat.py                  # POST /chat — conversational Q&A
│   │   │   ├── summarize.py             # POST /summarize — full doc summary
│   │   │   ├── extract.py               # POST /extract — domain entities
│   │   │   └── health.py                # GET /health
│   │   └── schemas.py                   # Pydantic request/response models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
│
├── frontend/
│   ├── app.py                           # Streamlit main app
│   ├── pages/
│   │   ├── 01_upload.py                 # Document upload page
│   │   ├── 02_chat.py                   # Chat with document page
│   │   ├── 03_summarize.py              # Summarization results page
│   │   ├── 04_extract.py                # Entity extraction & fraud flags
│   │   └── 05_compare.py               # Side-by-side document comparison
│   └── components/
│       ├── citation_viewer.py           # Show source chunk for each answer
│       ├── token_meter.py               # Live token budget visualizer
│       └── confidence_display.py        # Answer confidence scores
│
├── data/
│   ├── sample_docs/
│   │   ├── sample_insurance_claim.txt   # Large sample doc for demo
│   │   ├── sample_contract.txt          # Multi-page contract sample
│   │   └── sample_fraud_case.txt        # Fraud case with buried indicator
│   └── chroma_db/                       # ChromaDB persisted storage (auto-created)
│
├── tests/
│   ├── test_chunker.py
│   ├── test_summarizer.py
│   ├── test_memory_manager.py
│   ├── test_retriever.py
│   └── test_pipeline.py
│
└── scripts/
    ├── generate_sample_docs.py          # Generate large demo documents
    ├── benchmark.py                      # Compare baseline vs ContextBridge
    └── demo_flow.py                      # Run full demo automatically
```

---

## 2. REQUIREMENTS.TXT

```
# Core
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-dotenv==1.0.1
pydantic==2.7.1

# LLM & AI
anthropic==0.28.0
sentence-transformers==3.0.1
tiktoken==0.7.0

# Vector DB
chromadb==0.5.3

# LangChain (chunking only)
langchain==0.2.5
langchain-text-splitters==0.2.2

# Document Parsing
PyMuPDF==1.24.5
python-docx==1.1.2
pandas==2.2.2
openpyxl==3.1.4

# Frontend
streamlit==1.35.0
plotly==5.22.0

# Utilities
httpx==0.27.0
loguru==0.7.2
numpy==1.26.4
```

---

## 3. CONFIG.PY — COMPLETE SPECIFICATION

```python
# backend/config.py
# Implement all of these with os.getenv() + sensible defaults

ANTHROPIC_API_KEY: str          # Required — from .env
CLAUDE_MODEL: str = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS: int = 4096

# Chunking
CHUNK_SIZE: int = 800           # tokens per chunk
CHUNK_OVERLAP: int = 80         # overlap between chunks (10%)
MIN_CHUNK_SIZE: int = 100       # discard chunks smaller than this

# Context window budget (leave headroom for response)
CONTEXT_BUDGET_TOKENS: int = 150000   # Claude's window
RESPONSE_RESERVE_TOKENS: int = 4096
USABLE_CONTEXT_TOKENS: int = CONTEXT_BUDGET_TOKENS - RESPONSE_RESERVE_TOKENS

# Memory tiers
SHORT_TERM_EXCHANGES: int = 5        # verbatim last N exchanges
MID_TERM_SUMMARY_EXCHANGES: int = 20 # rolling summary window
ENTITY_STORE_MAX_ENTRIES: int = 100  # key facts to track

# Retrieval
TOP_K_CHUNKS: int = 8               # chunks to retrieve per query
SIMILARITY_THRESHOLD: float = 0.35  # minimum similarity score

# Summarization
SUMMARY_CHUNK_BATCH_SIZE: int = 5   # chunks per map-reduce batch
MAX_SUMMARY_LEVELS: int = 3         # hierarchy depth

# ChromaDB
CHROMA_PERSIST_DIR: str = "./data/chroma_db"
COLLECTION_NAME: str = "contextbridge_docs"

# Embedding model
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
```

---

## 4. CORE MODULE SPECIFICATIONS

### 4.1 `core/chunker.py`

**Class: `DocumentChunker`**

Implement these methods:

```python
def chunk_text(
    self,
    text: str,
    doc_id: str,
    metadata: dict = {}
) -> List[ChunkResult]
```

- Use `RecursiveCharacterTextSplitter` with `chunk_size=CHUNK_SIZE`, `chunk_overlap=CHUNK_OVERLAP`
- Split on `["\n\n", "\n", ". ", " ", ""]` in order
- Each `ChunkResult` must contain:
  - `chunk_id: str` — `f"{doc_id}_chunk_{index:04d}"`
  - `text: str`
  - `token_count: int`
  - `char_start: int`
  - `char_end: int`
  - `chunk_index: int`
  - `total_chunks: int`
  - `metadata: dict` — inherit from doc + add chunk-level keys
- After splitting, filter out chunks below `MIN_CHUNK_SIZE` tokens
- Log: total chunks, avg token count, min/max token count

```python
def chunk_by_section(
    self,
    text: str,
    doc_id: str,
    section_markers: List[str] = None
) -> List[ChunkResult]
```

- Detect headings (ALL CAPS lines, numbered sections like "1.", "Section 2", markdown `##`)
- Split at section boundaries first, then sub-chunk if section exceeds `CHUNK_SIZE`
- Preserve section title in each chunk's metadata as `section_name`

---

### 4.2 `core/embedder.py`

**Class: `EmbeddingEngine`**

```python
def __init__(self):
    # Load sentence-transformers model once at startup
    # Cache the model — do NOT reload per request

def embed_texts(self, texts: List[str]) -> List[List[float]]
    # Batch embed, batch size 32
    # Normalize vectors (L2)
    # Return list of embedding vectors

def embed_query(self, query: str) -> List[float]
    # Single query embedding
    # Apply same normalization
```

---

### 4.3 `core/vector_store.py`

**Class: `VectorStore`**

```python
def __init__(self, collection_name: str, persist_dir: str)
    # Initialize ChromaDB PersistentClient
    # Get or create collection with cosine similarity

def add_chunks(self, chunks: List[ChunkResult], embeddings: List[List[float]]) -> int
    # Upsert chunks with their embeddings
    # Store full chunk text + all metadata in ChromaDB
    # Return count of added chunks

def search(
    self,
    query_embedding: List[float],
    doc_id: str = None,       # filter to specific document if provided
    top_k: int = TOP_K_CHUNKS,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[SearchResult]
    # Query ChromaDB
    # Filter by similarity threshold
    # Return SearchResult objects with: chunk, score, doc_id, chunk_id

def delete_document(self, doc_id: str) -> int
    # Delete all chunks for a document
    # Return count deleted

def list_documents(self) -> List[DocumentInfo]
    # Return all unique doc_ids with metadata + chunk counts

def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkResult]
    # Direct lookup by chunk_id for citation display
```

---

### 4.4 `core/summarizer.py` — CRITICAL MODULE

**Class: `HierarchicalSummarizer`**

This is the core innovation. Implement true Map-Reduce summarization:

```
Level 0: Raw chunks          [c1][c2][c3][c4][c5][c6][c7][c8]...[cN]
Level 1: Chunk summaries     [s1][s2][s3][s4][s5][s6][s7][s8]...[sN]
Level 2: Section summaries   [S1]         [S2]         [S3]
Level 3: Document summary    [MASTER]
```

```python
async def summarize_document(
    self,
    chunks: List[ChunkResult],
    doc_type: str = "general",   # "insurance_claim", "contract", "general"
    focus: str = None            # optional focus topic for targeted summarization
) -> SummaryResult
```

**Map phase** — summarize each chunk independently:
- Batch chunks in groups of `SUMMARY_CHUNK_BATCH_SIZE`
- For each chunk, call Claude with prompt:
  ```
  You are summarizing chunk {index} of {total} from a {doc_type} document.
  Preserve: key facts, named entities, dates, amounts, decisions, risks, anomalies.
  Be concise but complete. Do not lose numerical values or proper nouns.
  
  CHUNK TEXT:
  {chunk.text}
  
  SUMMARY:
  ```
- Store chunk summary alongside original chunk

**Reduce phase** — combine summaries bottom-up:
- Group chunk summaries into batches that fit within token budget
- Combine each batch into a section summary using Claude
- Repeat until single master summary remains
- Store ALL intermediate summaries (not just final)

**Return `SummaryResult`:**
```python
class SummaryResult:
    doc_id: str
    master_summary: str
    section_summaries: List[str]
    chunk_summaries: List[str]
    total_chunks_processed: int
    levels: int                      # depth of hierarchy used
    token_usage: TokenUsage
    processing_time_seconds: float
    completeness_score: float        # % of chunks successfully summarized
```

```python
async def summarize_conversation(
    self,
    exchanges: List[ConversationExchange],
    existing_summary: str = None
) -> str
    # Incrementally summarize conversation history
    # If existing_summary provided, extend it rather than reprocess
    # Focus on: decisions made, facts established, questions asked, answers given
```

---

### 4.5 `core/memory_manager.py` — CRITICAL MODULE

**Class: `MemoryManager`**

Three-tier architecture per session:

```
TIER 1: Short-term buffer    → Last N exchanges verbatim (fast, exact)
TIER 2: Mid-term summary     → Rolling LLM-generated summary (compressed)  
TIER 3: Entity store         → Structured facts extracted from conversation
```

```python
class MemoryManager:
    def __init__(self, session_id: str)

    def add_exchange(
        self,
        user_message: str,
        assistant_response: str,
        retrieved_chunks: List[SearchResult] = []
    ) -> None
        # Add to short-term buffer
        # If buffer exceeds SHORT_TERM_EXCHANGES:
        #   → Summarize oldest exchanges into mid-term summary
        #   → Extract entities from those exchanges into entity store
        #   → Remove from short-term buffer

    def build_context_payload(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
        token_budget: int = USABLE_CONTEXT_TOKENS
    ) -> ContextPayload
        # Intelligently pack context within token budget:
        # Priority order:
        #   1. System prompt (always included)
        #   2. Entity store (always included — small)
        #   3. Mid-term summary (always included — compressed)
        #   4. Retrieved chunks (fill remaining budget, best-scored first)
        #   5. Short-term buffer (fill remaining budget, newest first)
        # Return ContextPayload with what was included + what was dropped

    def extract_entities(self, text: str) -> Dict[str, List[str]]
        # Call Claude to extract:
        # {
        #   "people": [...],
        #   "organizations": [...],
        #   "dates": [...],
        #   "amounts": [...],
        #   "locations": [...],
        #   "decisions": [...],
        #   "risks": [...],
        #   "claim_ids": [...]
        # }
        # Merge into entity_store (deduplicate)

    def get_session_summary(self) -> SessionSummary
        # Return full session state for debugging/display

    def reset(self) -> None
        # Clear all tiers for this session
```

```python
class ContextPayload:
    system_prompt: str
    conversation_history: List[dict]   # OpenAI-style messages format
    included_chunks: List[SearchResult]
    dropped_chunks: List[SearchResult]  # what didn't fit — for completeness warning
    total_tokens_used: int
    token_budget: int
    utilization_percent: float
```

---

### 4.6 `core/retriever.py`

**Class: `IntelligentRetriever`**

```python
async def retrieve(
    self,
    query: str,
    doc_id: str = None,
    top_k: int = TOP_K_CHUNKS,
    retrieval_mode: str = "hybrid"   # "semantic", "keyword", "hybrid"
) -> List[SearchResult]
    # 1. Embed query
    # 2. Semantic search in ChromaDB
    # 3. For "hybrid": also do keyword matching (simple substring + BM25-like scoring)
    # 4. Merge and deduplicate results
    # 5. Rerank: boost chunks from sections already referenced in conversation
    # 6. Return top_k results with scores

async def retrieve_for_summary(
    self,
    topic: str,
    doc_id: str
) -> List[SearchResult]
    # Retrieve chunks relevant to a specific topic across the full document
    # Used for focused summarization

def get_neighboring_chunks(
    self,
    chunk_id: str,
    window: int = 1
) -> List[ChunkResult]
    # Return chunk_index ± window neighbors for context expansion
    # Used when a chunk is retrieved but surrounding context helps
```

---

### 4.7 `core/token_counter.py`

**Class: `TokenCounter`**

```python
def count(self, text: str) -> int
    # Use tiktoken cl100k_base encoding
    # Cache results for repeated strings

def count_messages(self, messages: List[dict]) -> int
    # Count tokens in OpenAI-style message list
    # Include role overhead (~4 tokens per message)

def fits_in_budget(self, text: str, budget: int) -> bool

def truncate_to_budget(self, text: str, budget: int) -> Tuple[str, int]
    # Truncate text to fit in token budget
    # Truncate at sentence boundary, not word boundary
    # Return (truncated_text, tokens_used)
```

---

## 5. INGESTION MODULE SPECIFICATIONS

### 5.1 `ingestion/pdf_parser.py`

**Class: `PDFParser`**

```python
def parse(self, file_path: str) -> ParsedDocument
    # Use PyMuPDF (fitz)
    # Extract: full text, page numbers, tables (as text), metadata
    # Preserve page breaks as "\n\n--- PAGE {n} ---\n\n"
    # Extract document metadata: title, author, creation_date, page_count
    # Detect and flag: tables, lists, section headers
    # Return ParsedDocument with text + metadata

def extract_tables(self, file_path: str) -> List[TableResult]
    # Extract tables separately with their page numbers
    # Convert to markdown table format for LLM consumption
```

### 5.2 `ingestion/pipeline.py`

**Class: `IngestionPipeline`**

This orchestrates the full flow:

```python
async def ingest(
    self,
    file_path: str,
    doc_type: str = "general",
    run_summarization: bool = True
) -> IngestionResult
```

Full pipeline:
1. Detect file type → route to correct parser
2. Parse → `ParsedDocument`
3. Chunk → `List[ChunkResult]`
4. Embed all chunks → `List[List[float]]`
5. Store in ChromaDB → confirm count
6. (If `run_summarization=True`) Run hierarchical summarization
7. Extract domain entities from master summary
8. Return `IngestionResult`:

```python
class IngestionResult:
    doc_id: str
    file_name: str
    doc_type: str
    total_pages: int
    total_chars: int
    total_tokens: int
    total_chunks: int
    chunks_stored: int
    summary: Optional[SummaryResult]
    entities: Optional[Dict]
    ingestion_time_seconds: float
    status: str   # "success", "partial", "failed"
    warnings: List[str]   # e.g. "3 chunks dropped below minimum size"
```

---

## 6. DOMAIN MODULE SPECIFICATIONS

### 6.1 `domain/banking_extractor.py`

**Class: `BankingExtractor`**

```python
async def extract_fraud_indicators(
    self,
    doc_id: str,
    summary: SummaryResult
) -> FraudAnalysisResult
    # Use Claude to analyze the full hierarchical summary
    # Look for: inconsistent dates, duplicate claims, inflated amounts,
    #           suspicious third parties, timeline anomalies, policy violations
    # For each flag: provide evidence quote, page reference, severity (HIGH/MEDIUM/LOW)
    # Return structured FraudAnalysisResult

async def extract_contract_clauses(
    self,
    doc_id: str
) -> ContractClauseResult
    # Retrieve chunks matching legal clause patterns
    # Extract: indemnification, liability caps, termination clauses,
    #          payment terms, jurisdiction, force majeure, IP rights
    # For each clause: text, page number, risk rating, plain-English explanation

async def assess_risk(
    self,
    doc_id: str,
    client_profile: dict = {}
) -> RiskAssessmentResult
    # Combine fraud indicators + clause analysis + client profile
    # Generate overall risk score (0-100)
    # Provide top 5 risk factors with evidence
```

### 6.2 `domain/completeness_checker.py`

**Class: `CompletenessChecker`**

```python
def check_response_completeness(
    self,
    query: str,
    response: str,
    dropped_chunks: List[SearchResult]
) -> CompletenessReport
    # Analyze if any dropped chunks contained information relevant to query
    # Flag: "Note: {N} document sections couldn't fit in context. They may contain relevant info."
    # List: which sections were dropped (by section name/page)
    # Confidence: how complete is the answer (HIGH/MEDIUM/LOW)

def detect_truncation_risks(
    self,
    chunks: List[ChunkResult],
    total_chunks: int
) -> List[str]
    # Identify chunks that seem to be mid-sentence (truncation artifacts)
    # Identify potential cross-chunk references ("as mentioned in section X")
```

---

## 7. API ROUTES SPECIFICATIONS

### 7.1 `api/routes/upload.py`

```
POST /api/upload
Content-Type: multipart/form-data

Request:
  file: UploadFile
  doc_type: str = "general"  # "insurance_claim", "contract", "sop", "general"
  run_summarization: bool = True

Response: IngestionResult (as JSON)
```

- Save uploaded file to temp dir
- Run `IngestionPipeline.ingest()`
- Return result immediately (no background task — demo needs synchronous result)
- Add CORS middleware for Streamlit frontend

### 7.2 `api/routes/chat.py`

```
POST /api/chat
Content-Type: application/json

Request:
{
  "session_id": "string",
  "doc_id": "string",
  "message": "string",
  "mode": "rag" | "summary" | "auto"
}

Response:
{
  "answer": "string",
  "citations": [{"chunk_id": str, "text": str, "page": int, "score": float}],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "completeness": CompletenessReport,
  "token_usage": {"context_tokens": int, "response_tokens": int, "budget_utilization": float},
  "dropped_sections": ["Section 3", "Page 47-52"],
  "session_memory_summary": "string"
}
```

**Chat logic:**
1. Get or create `MemoryManager` for `session_id`
2. Retrieve relevant chunks using `IntelligentRetriever`
3. Build `ContextPayload` via memory manager (respects token budget)
4. If any chunks dropped → run `CompletenessChecker`
5. Call Claude API with context payload
6. Parse response → extract citations from chunk references
7. Update memory manager with exchange
8. Return full structured response

### 7.3 `api/routes/summarize.py`

```
POST /api/summarize
{
  "doc_id": "string",
  "level": "chunk" | "section" | "master" | "all",
  "focus": "optional focus topic"
}

Response:
{
  "doc_id": str,
  "master_summary": str,
  "section_summaries": List[str],
  "chunk_count": int,
  "levels_used": int,
  "completeness_score": float,
  "token_savings": {"original_tokens": int, "summary_tokens": int, "compression_ratio": float}
}
```

### 7.4 `api/routes/extract.py`

```
POST /api/extract
{
  "doc_id": "string",
  "extraction_type": "fraud" | "clauses" | "risk" | "entities" | "all",
  "client_profile": {}
}

Response: Depends on extraction_type — return appropriate structured result
```

---

## 8. CLAUDE API INTEGRATION

### System Prompt Template

```python
SYSTEM_PROMPT = """You are ContextBridge, an expert document analysis AI specializing in Banking & Insurance.

You have access to a large document that has been intelligently chunked and indexed. Each response MUST:
1. Answer based ONLY on the provided document context
2. Cite specific sections using [CHUNK: chunk_id] notation when referencing information
3. If information spans multiple sections, cite all relevant sections
4. If you cannot find the answer in the provided context, say "Not found in provided context sections"
5. Flag any inconsistencies or anomalies you notice
6. Always specify if your answer might be incomplete due to context limitations

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_summary}

KNOWN ENTITIES FROM THIS SESSION:
{entity_store}

RECENT EXCHANGES:
{short_term_buffer}
"""
```

### Claude API Call Pattern

```python
async def call_claude(
    self,
    context_payload: ContextPayload,
    user_message: str
) -> ClaudeResponse:
    
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    
    # Build messages with context
    messages = context_payload.conversation_history + [
        {"role": "user", "content": user_message}
    ]
    
    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=context_payload.system_prompt,
        messages=messages
    )
    
    return ClaudeResponse(
        text=response.content[0].text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens
    )
```

---

## 9. FRONTEND SPECIFICATIONS (STREAMLIT)

### `frontend/app.py` — Main App

```python
# Multi-page Streamlit app
# Sidebar: 
#   - App title + logo
#   - Uploaded documents list (with chunk counts)
#   - Active session info
#   - Token budget meter (progress bar)
#   - Clear session button

# Pages via st.navigation():
#   📤 Upload Document
#   💬 Chat with Document  
#   📋 Summarization
#   🔍 Extract & Analyze
#   ⚖️ Compare Documents
```

### `frontend/pages/01_upload.py`

- File uploader (PDF, DOCX, TXT)
- Doc type selector dropdown
- "Run summarization on upload" toggle
- Progress bar during ingestion
- Result display: chunks created, pages, tokens, summary preview
- **Show: "Without ContextBridge this document would exceed context by Nx"**

### `frontend/pages/02_chat.py`

- Document selector (from uploaded docs)
- Chat interface (st.chat_message)
- Each assistant response shows:
  - Answer text
  - Expandable "📎 Sources" section → show cited chunks with page numbers
  - Confidence badge (🟢 HIGH / 🟡 MEDIUM / 🔴 LOW)
  - ⚠️ Warning banner if sections were dropped from context
  - Token usage meter

### `frontend/pages/03_summarize.py`

- Document selector
- Summary level tabs: Master / Sections / Chunks
- Compression ratio display: `"200 pages → 3 paragraphs (98.5% compression)"`
- Interactive chunk tree — click section to expand to chunk summaries
- Export summary as .txt button

### `frontend/pages/04_extract.py`

- Document selector + extraction type selector
- Fraud indicators table (with severity colors: RED/ORANGE/YELLOW)
- Contract clauses accordion (clause type → text → risk → explanation)
- Entity display: names, dates, amounts as colored tags
- Risk score gauge chart (0-100)
- Export as JSON button

### `frontend/pages/05_compare.py`

- Two document selectors (left + right)
- Side-by-side summary comparison
- Diff view: highlight what's in doc A but not doc B
- Shared entities table

### `frontend/components/token_meter.py`

```python
def render_token_meter(used: int, budget: int):
    # Horizontal progress bar
    # Color: green < 70%, yellow 70-90%, red > 90%
    # Show: "87,432 / 150,000 tokens (58%)"
    # Show breakdown: chunks / memory / system prompt
```

---

## 10. SAMPLE DATA GENERATION

### `scripts/generate_sample_docs.py`

Generate these three documents programmatically (write as .txt files to `data/sample_docs/`):

**1. `sample_insurance_claim.txt`** — ~15,000 words
- Insurance claim for a commercial property fire
- 47 sections: incident report, witness statements, damage assessment, prior claims history
- **Plant a fraud indicator on "page 31"**: claimant previously filed an identical claim 18 months prior under a different policy number with a different insurer for the same property
- Normal language throughout — fraud indicator should NOT be in first 10 sections
- Include: dates, amounts, policy numbers, names, locations

**2. `sample_contract.txt`** — ~12,000 words
- Multi-party commercial software licensing agreement
- 38 sections including: IP ownership, liability caps, termination, SLAs, payment terms
- Include problematic clauses: uncapped liability in section 29, unusual jurisdiction clause in section 34
- Standard legal language

**3. `sample_fraud_case.txt`** — ~8,000 words
- Customer transaction history + case notes
- Multiple transactions over 2 years
- Fraud pattern buried in middle: multiple small transactions just below reporting threshold

---

## 11. BENCHMARKING SCRIPT

### `scripts/benchmark.py`

Run this to generate demo metrics:

```python
# Test 1: Baseline vs ContextBridge on large document Q&A
# - Load sample_insurance_claim.txt
# - Ask 10 questions, some requiring info from late in document
# - Baseline: send raw text truncated to 8K tokens
# - ContextBridge: full RAG pipeline
# - Measure: answer accuracy (manual labels), critical info retention rate

# Test 2: Conversation memory
# - 30-turn conversation about the insurance claim
# - Baseline: sliding window last 5 exchanges only
# - ContextBridge: full memory manager
# - Measure: how many turns before baseline loses key facts

# Test 3: Fraud detection
# - Ask "Are there any anomalies or fraud indicators in this claim?"
# - Baseline: misses indicator (it's beyond context)
# - ContextBridge: finds and cites it precisely

# Output: benchmark_results.json with all metrics
# Print: formatted comparison table
```

---

## 12. DEMO FLOW SCRIPT

### `scripts/demo_flow.py`

This runs the winning demo automatically:

```python
# Step 1: Ingest sample_insurance_claim.txt
#   → Print: "Ingested 47 sections, 312 chunks, ~45,000 tokens"
#   → Print: "Document is 6x beyond standard context window"

# Step 2: Show baseline failure
#   → Truncate to 8K tokens
#   → Ask: "Has this claimant filed any similar claims before?"
#   → Print baseline response: "No prior claims found" (WRONG)

# Step 3: Show ContextBridge success
#   → Ask same question
#   → Print response with citation to section 31, page 31
#   → Print: "Found in chunk 089 (section: Prior Claims History, page 31)"
#   → Print: "⚠️ FRAUD INDICATOR: Duplicate claim detected"

# Step 4: Show memory persistence
#   → Ask follow-up: "What was the exact policy number from the prior claim?"
#   → Show answer retrieved from entity store (not re-retrieved from DB)

# Step 5: Show compression
#   → Print: "Full document: 45,234 tokens"
#   → Print: "Master summary: 847 tokens (98.1% compression)"
#   → Print: "Critical fraud indicator: PRESERVED in summary ✓"
```

---

## 13. ERROR HANDLING REQUIREMENTS

Every module must handle:

- `AnthropicAPIError` — log, return graceful error message, never crash
- `ChromaDBError` — log, attempt reconnect once, then fail gracefully  
- `TokenBudgetExceeded` — truncate intelligently, warn user, never silently drop
- `EmptyChunkError` — skip and warn, continue pipeline
- `FileParseError` — return partial result if possible, list what failed
- `EmbeddingError` — retry once, then fall back to keyword-only retrieval

All errors must:
1. Log full traceback with loguru
2. Return structured error response (never raw exceptions to frontend)
3. Include `warnings` list in every result object

---

## 14. DAY-BY-DAY IMPLEMENTATION ORDER

### Day 1: Foundation
1. Set up project structure (all folders + empty `__init__.py`)
2. `requirements.txt` + virtual environment
3. `config.py`
4. `core/token_counter.py`
5. `core/chunker.py` + `tests/test_chunker.py`
6. `core/embedder.py`
7. `core/vector_store.py`
8. `ingestion/pdf_parser.py` + `ingestion/text_parser.py`
9. Basic FastAPI `main.py` + `api/routes/health.py`
10. Test: ingest one document, verify chunks in ChromaDB

### Day 2: Summarization
1. `core/summarizer.py` (full Map-Reduce implementation)
2. `ingestion/pipeline.py` (full orchestration)
3. `api/routes/upload.py`
4. `api/routes/summarize.py`
5. `scripts/generate_sample_docs.py` (generate all 3 sample docs)
6. Test: full summarization of sample_insurance_claim.txt

### Day 3: Memory + Chat
1. `core/memory_manager.py` (all 3 tiers)
2. `core/retriever.py` (hybrid retrieval)
3. `api/routes/chat.py` (full chat with citations)
4. `domain/completeness_checker.py`
5. Test: 20-turn conversation retaining context

### Day 4: Domain + Frontend
1. `domain/banking_extractor.py` (fraud + clauses + risk)
2. `domain/entity_extractor.py`
3. `api/routes/extract.py`
4. ALL Streamlit pages (01-05)
5. ALL Streamlit components
6. Wire frontend to backend API

### Day 5: Polish + Demo
1. `scripts/benchmark.py` → generate metrics
2. `scripts/demo_flow.py` → automate demo
3. Edge case handling pass
4. Error handling pass
5. UI polish (colors, labels, loading spinners)
6. README.md with setup instructions + demo guide
7. Full dry run of demo flow

---

## 15. README.MD REQUIREMENTS

Include:
- One-paragraph project description
- Architecture diagram (ASCII)
- Quick start (5 commands to run)
- Demo walkthrough (what to show judges)
- Key metrics achieved
- Technical decisions + why
- Team + hackathon info

---

## 16. FINAL CHECKLIST BEFORE DEMO

- [ ] `python -m pytest tests/` — all tests pass
- [ ] Fresh ingest of sample_insurance_claim.txt takes < 60 seconds
- [ ] Chat response time < 10 seconds
- [ ] Fraud indicator found and cited correctly
- [ ] Token meter shows in UI
- [ ] Dropped sections warning appears when relevant
- [ ] Compression ratio displayed
- [ ] Benchmark results show ContextBridge > baseline on all metrics
- [ ] Demo script runs end-to-end without errors

---

## 17. JUDGE TALKING POINTS

Prepare to answer:
1. **"What happens if the document is 1000 pages?"** → Hierarchical summarization scales linearly; O(N) LLM calls, not O(N²)
2. **"Why not just use a model with a larger context window?"** → Cost, latency, and hallucination rate all scale with context size; our approach is cost-efficient and more accurate
3. **"How do you ensure nothing critical is dropped?"** → Multi-level summaries preserve critical info; completeness checker audits every response; entity store tracks all key facts
4. **"What's the accuracy vs baseline?"** → Show benchmark results from `scripts/benchmark.py`
5. **"Can this generalize beyond Banking?"** → Yes — show domain switcher, explain entity extractor is domain-agnostic, domain modules are plug-in

---

*End of architectural specification. Claude Code: begin with Day 1 implementation. Follow every module specification exactly. Do not skip the test files.*
