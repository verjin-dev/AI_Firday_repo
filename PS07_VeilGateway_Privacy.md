# PS07 — `VeilGateway`
## Safeguarding Sensitive Data: Ensuring Privacy & Regulatory Compliance

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**VeilGateway is a transparent proxy that sits in front of every LLM call and guarantees no raw PII ever reaches the model — without destroying the model's ability to reason.** Drop-in: change one base URL and your existing application becomes compliant.

---

## 2. CORE INNOVATION

Two mechanisms, both of which solve problems that naive redaction creates.

**1. Format-preserving reversible pseudonymisation.** Everyone redacts to `[REDACTED_1]`, `[NAME_2]`, `[DATE_3]`. That destroys model performance: the LLM can no longer tell that "Priya Sharma" and "Ms. Sharma" are the same person, can't reason about the gap between two dates, and produces stilted output. VeilGateway substitutes **consistent, type-preserving, realistic surrogates** — Priya Sharma becomes Anjali Verma everywhere in the session, ₹4,52,300 becomes ₹4,38,900 (order preserved, offset consistent), 12-Mar-2024 becomes 03-Apr-2024 with all intervals preserved. The model reasons normally; the vault maps back on egress. We measure and report the **utility delta** — this is the number nobody else will have.

**2. Crypto-shredding for vector stores.** GDPR Article 17 erasure against a RAG index is an unsolved operational problem: embeddings are derived data, you cannot un-embed them, and re-indexing a corpus takes hours. VeilGateway encrypts every chunk's payload under a **per-data-subject key**. Erasure = destroy one key. The embedding remains as noise, the content is unrecoverable, and it takes 40 milliseconds. Demo this live — it is the moment that wins the room.

---

## 3. ARCHITECTURE

```
   App ──▶ ┌────────────────────────────────────────────────┐
           │  VEILGATEWAY  (drop-in /v1/messages proxy)      │
           │                                                 │
   ingress │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
   ───────▶│  │ DETECT   │─▶│ CLASSIFY │─▶│ PSEUDONYMISE │──┼──▶ Claude API
           │  │ NER+regex│  │ per-type │  │ FPE + vault  │  │
           │  │ +validate│  │ policy   │  └──────────────┘  │
           │  └──────────┘  └──────────┘                    │
           │                                                 │
   egress  │  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
   ◀───────┼──│ RE-IDENTIFY  │◀─│ EGRESS   │◀─│ AUDIT    │◀─┼─── response
           │  │ authz-scoped │  │ DLP scan │  │ log      │  │
           │  └──────────────┘  └──────────┘  └──────────┘  │
           └────────────────────────────────────────────────┘
                        │                    │
                  ┌─────▼──────┐      ┌──────▼─────────┐
                  │ TOKEN VAULT│      │ SANITISED LOGS │ (never raw PII)
                  │ per-subject│      └────────────────┘
                  │ keys       │
                  └─────┬──────┘
                        ▼
              ┌────────────────────────┐
              │ ENCRYPTED RAG STORE    │  chunk payloads encrypted
              │ crypto-shred on erasure│  under subject keys
              └────────────────────────┘
```

---

## 4. EXTRA DEPENDENCIES

```
presidio-analyzer==2.2.354
presidio-anonymizer==2.2.354
spacy==3.7.5                    # + en_core_web_lg model
cryptography==42.0.7
faker==25.8.0                   # realistic surrogate generation
phonenumbers==8.13.39
python-stdnum==1.20             # Aadhaar/PAN/IBAN checksum validation
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
DETECTORS_ENABLED: list[str] = ["PERSON","EMAIL","PHONE","AADHAAR","PAN",
                                "ACCOUNT_NUMBER","CARD","IFSC","ADDRESS",
                                "DATE_OF_BIRTH","MRN","ICD_CODE","POLICY_NUMBER",
                                "IP","GEO"]
DETECTION_THRESHOLD: float = 0.55
PSEUDONYM_MODE: str = "format_preserving"    # | "tag" | "mask" | "suppress"
DATE_SHIFT_DAYS_MAX: int = 45                # consistent per subject
AMOUNT_JITTER_PCT: float = 4.0               # order-preserving
VAULT_MASTER_KEY: str                        # from .env / KMS in production
SUBJECT_KEY_ROTATION_DAYS: int = 90
RETENTION_DAYS_DEFAULT: int = 30
PURPOSE_POLICY_FILE: str = "./config/purposes.yaml"
LOG_PII_POLICY: str = "never"                # enforced, not aspirational
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/detector.py`

```python
class PIIDetector:
    def detect(self, text: str, locale="en_IN") -> list[PIIEntity]
        """
        Three layers, results merged with span-overlap resolution:
          1. Regex + CHECKSUM VALIDATION — Aadhaar (Verhoeff), PAN (format+
             checksum), card (Luhn), IFSC, IBAN. Checksums crush false
             positives: a random 12-digit number is not an Aadhaar.
          2. spaCy NER for PERSON / ORG / GPE / DATE.
          3. Context boosting: a number near 'account', 'a/c', 'policy no'
             gets its confidence raised. This is what catches the long tail.
        Returns PIIEntity(type, text, start, end, confidence, detector, subject_hint)
        """

    async def detect_llm_assisted(self, text) -> list[PIIEntity]
        """Second pass with Claude for unstructured/free-text PII that
           regex and NER miss ('the patient's brother who works at the mill
           in Coimbatore'). Runs only on flagged low-confidence regions —
           bounded cost. Report the recall lift it provides."""
```

### 6.2 `core/pseudonymizer.py`

```python
class Pseudonymizer:
    def pseudonymize(self, text, entities, session_id) -> PseudonymizedResult
        """
        For each entity, derive a deterministic surrogate:
          key = HMAC(vault_key, subject_id + entity_type + normalized_value)
          PERSON  → Faker name seeded by key; SAME input always yields SAME
                    surrogate within the session, and coreference variants
                    ('Priya', 'Ms. Sharma', 'P. Sharma') map to the SAME person
          DATE    → shift by a per-subject constant in [1, DATE_SHIFT_DAYS_MAX];
                    ALL intervals preserved exactly
          AMOUNT  → jitter by a per-subject constant %, ORDER preserved across
                    all amounts belonging to that subject
          ID      → format-preserving encryption: valid-looking PAN/Aadhaar
                    that passes checksum but maps to nothing real
          ADDRESS → same city tier and state, different street
        Store mappings in the vault, encrypted under the subject key.
        Return the text plus a TokenMap for the return path.
        """

    def reidentify(self, text, token_map, requester: Principal) -> str
        """Only surrogates the requester is authorised to see are restored.
           A support agent sees the name; an analyst sees the surrogate;
           a log sink sees neither. Same response object, three renderings."""
```

The coreference requirement is the hard part and the part that impresses. Build a unit test: a paragraph mentioning one person five ways must produce one surrogate.

### 6.3 `core/vault.py`

```python
class TokenVault:
    def get_or_create_subject_key(self, subject_id) -> bytes
        """Per-subject DEK, wrapped by the master key."""

    def store_mapping(self, subject_id, surrogate, original, entity_type) -> None
        """Original is encrypted under the subject's DEK. The plaintext
           original exists nowhere else in the system."""

    def resolve(self, surrogate, principal) -> str | None

    def crypto_shred(self, subject_id) -> ShredReceipt
        """
        1. Destroy the subject's DEK (overwrite + delete the wrapped key).
        2. Every mapping and every encrypted RAG chunk payload for that
           subject is now permanently unrecoverable.
        3. Emit a signed ShredReceipt: subject_id_hash, timestamp, counts of
           affected mappings/chunks/traces, key_id destroyed, operator.
        4. Verify: attempt to decrypt 3 random affected records, assert failure.
        Target: < 100ms regardless of corpus size. THIS IS THE DEMO.
        """
```

### 6.4 `core/encrypted_rag.py`

```python
class EncryptedRAGStore:
    def add(self, chunk, subject_ids: list[str]) -> None
        """
        Embed the PSEUDONYMISED text (so the vector space is usable and the
        embedding itself leaks nothing). Encrypt the chunk PAYLOAD under each
        associated subject's DEK (envelope encryption for multi-subject chunks).
        Store ciphertext + embedding + subject_id_hashes in Chroma metadata.
        """

    def search(self, query, principal) -> list[SearchResult]
        """Vector search, then attempt decrypt. Undecryptable chunks (shredded
           or unauthorised) are silently excluded with a count in warnings."""
```

### 6.5 `core/purpose_policy.py`

```python
# config/purposes.yaml
purposes:
  claims_triage:
    lawful_basis: contract
    permitted_types: [PERSON, POLICY_NUMBER, DATE_OF_BIRTH, MRN]
    forbidden_types: [CARD, AADHAAR]
    retention_days: 90
    egress_reidentify_roles: [claims_officer]
  marketing_analytics:
    lawful_basis: consent
    permitted_types: [GEO]
    forbidden_types: [PERSON, EMAIL, PHONE, CARD, AADHAAR, MRN]
    retention_days: 30
    egress_reidentify_roles: []

class PurposeEnforcer:
    def check(self, request, purpose) -> PolicyDecision
        """A forbidden type present for the declared purpose → BLOCK the
           request entirely with a citation to the policy clause. Purpose
           limitation is GDPR Art.5(1)(b) and almost nobody implements it."""
```

### 6.6 `core/utility_meter.py`

```python
class UtilityMeter:
    async def measure(self, tasks: list[Task]) -> UtilityReport
        """
        Run the same task set three ways: raw / tag-redacted / pseudonymised.
        Score with a task-appropriate metric (exact match, F1, judge rubric).
        Report utility retention %:  pseudonymised_score / raw_score.
        Target: > 96% for pseudonymisation, vs 70-80% for tag redaction.
        This single chart is your Innovation and Business Value evidence.
        """
```

---

## 7. API ROUTES

```
POST /v1/messages              ← drop-in Anthropic-compatible proxy endpoint
POST /api/scan                 {text} → detected entities (no LLM call)
POST /api/pseudonymize         {text, subject_id} → surrogate text + map
POST /api/erasure              {subject_id} → ShredReceipt
GET  /api/erasure/{id}/verify  → proof of unrecoverability
GET  /api/audit                DSAR-ready processing record
GET  /api/utility              utility retention report
GET  /api/policy               purpose policies
```

The drop-in proxy endpoint is the product. Show a curl command where only the base URL changed.

---

## 8. FRONTEND PAGES

**`01_gateway.py` — the three-pane view.** Paste a claim document. Left: original with PII highlighted by type. Middle: **exactly what Claude receives** (readable, natural, fully surrogate). Right: the response, re-identified for your role. A role dropdown (claims_officer / analyst / auditor) changes the right pane live.

**`02_utility.py`** — the bar chart: raw 100%, tag-redacted 74%, VeilGateway 97%. Three example outputs underneath so judges can see *why* tag redaction fails.

**`03_erasure.py` — the winning demo.** A subject with 47 chunks, 12 sessions, 340 vault mappings. Big red **"Execute Right to Erasure"** button. Timer shows 41ms. Then a verification panel that retrieves three of the subject's chunks and shows the decryption failure, plus the signed receipt.

**`04_policy.py`** — purpose editor; submit a request under `marketing_analytics` containing a name and watch it get blocked with a clause citation.

**`05_audit.py`** — DSAR export: everything processed about a subject, when, under what purpose, by whom.

---

## 9. SAMPLE DATA

Domain: **healthcare + insurance claims** (covers HIPAA and GDPR framing in one dataset).

- 200 synthetic patient/claim records with dense, realistic Indian PII: names, Aadhaar, PAN, phone, address, DOB, MRN, ICD codes, policy numbers, amounts.
- 50 free-text clinical notes and adjuster narratives with **indirect identifiers** that regex cannot catch — "the patient's employer, the textile mill on Avinashi Road" — for the LLM-assisted detection demo.
- Golden PII set: every entity span hand-labelled, so precision/recall are real numbers.
- 30 downstream tasks (summarise, extract, decide) for the utility measurement.

---

## 10. BENCHMARK

| Metric | Tag redaction | VeilGateway |
|---|---|---|
| PII detection recall | 0.87 | **0.96** |
| PII detection precision | 0.79 | **0.97** (checksums) |
| Raw PII reaching the model | 13% of entities | **0%** |
| Downstream task utility retention | 74% | **97%** |
| Coreference consistency | n/a | 100% |
| Erasure latency (47 chunks, 340 mappings) | ~40 min re-index | **41 ms** |
| PII in application logs | present | **zero** (enforced) |
| Added latency per call | +0.2s | +0.4s |

The two numbers to put on the slide: **97% utility retention** and **41 milliseconds to erase a data subject**.

---

## 11. DEMO FLOW (4 minutes)

1. **Drop-in.** Show a curl to `api.anthropic.com`, then the same curl with the base URL swapped to VeilGateway. "One line. That's the integration."
2. **The three panes.** Load a claim narrative. Left: 23 PII entities highlighted. Middle: what Claude actually sees — *read it out loud*, it reads like a normal document about Anjali Verma. Right: correct answer, re-identified.
3. **Why redaction fails.** Toggle to tag mode. Middle pane becomes `[PERSON_1] filed on [DATE_2] for [AMOUNT_3]`. Show the model's answer degrade — it can no longer tell whether `[PERSON_1]` and `[PERSON_4]` are the same person. Utility chart: 74% vs 97%.
4. **Purpose limitation.** Submit the same document under `marketing_analytics`. Blocked, with the citation: "PERSON is a forbidden type for lawful basis 'consent' without explicit marketing consent — GDPR Art. 5(1)(b)."
5. **The erasure moment.** Subject panel: 47 chunks, 12 sessions, 340 mappings. Hit Execute. **41 milliseconds.** Then hit Verify: three chunks retrieved, all fail to decrypt, receipt signed. "Your competitors' answer to this is a four-hour re-index job."
6. **The logs.** Open `logs/app.jsonl`. Grep for any of the original names. Zero hits.

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, proxy endpoint passing through unchanged, sample data generator, golden PII labels. Gate: proxy works transparently.
**Day 2** — `detector.py` all three layers with checksum validators. Gate: precision/recall on the golden set.
**Day 3** — `pseudonymizer.py` with coreference + FPE + `vault.py`, `utility_meter.py`, benchmark. Gate: utility retention number is real.
**Day 4** — `encrypted_rag.py`, `crypto_shred`, `purpose_policy.py`, all 5 pages.
**Day 5** — Audit/DSAR export, demo script, README with GDPR/HIPAA article mapping, dry runs.

**Cut list:** DSAR page, LLM-assisted detection. **Never cut** crypto-shredding or the utility chart.

---

## 13. JUDGE TALKING POINTS

**"Why not just use a self-hosted model and keep data in-house?"** Often the right answer and we support it — the gateway is model-agnostic. But it doesn't solve the problem: PII still lands in prompt logs, traces, vector indexes, and eval datasets, all of which are subject to erasure requests and access controls. Self-hosting moves the boundary; it doesn't remove the obligation. Our gateway addresses the data lifecycle, not just the network hop.

**"Doesn't pseudonymisation break the model's answers?"** That's the claim we set out to falsify, and we measured it: 97% utility retention versus 74% for tag redaction. The reason is coreference consistency and format preservation — the model can still reason about who is who, which date came first, and which amount is larger.

**"Is pseudonymised data still personal data under GDPR?"** Yes — Recital 26 is explicit, pseudonymised data remains personal data because re-identification is possible via the vault. We never claim otherwise. What we claim is Article 32 appropriate technical measures, Article 25 data protection by design, and a genuinely enforceable Article 17. The vault is the controlled re-identification boundary, and it's access-logged.

**"How does crypto-shredding satisfy erasure legally?"** Destruction of the sole decryption key renders the data permanently inaccessible, which regulators have accepted as equivalent to erasure for backed-up and derived data. We emit a signed receipt with the key ID and verification evidence. We're also explicit about the limitation: the ciphertext and embedding remain on disk as noise, and we state that in the DPIA rather than hiding it.

**"Detection is never 100%. What about the misses?"** Correct — we report 0.96 recall, not 1.0. Defence in depth: egress DLP catches PII in *outputs* even if ingress missed it in inputs, logs are structurally incapable of holding raw PII, and unrecognised high-entropy strings in sensitive purposes trigger a block rather than a pass. We'd rather fail closed.
