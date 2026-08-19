# PS01 — `ClaimLedger`
## Battling AI Hallucinations: Ensuring Reliability in Enterprise Applications

> Read `00_COMMON_FOUNDATION.md` first. This file specifies only what is unique to this project.

---

## 1. PITCH

**ClaimLedger decomposes every LLM answer into atomic claims, verifies each one against retrieved evidence, and abstains rather than guesses when the evidence isn't there.**

The insight most teams miss: hallucination is not a text problem, it is a *decision* problem. You cannot make a language model stop generating unsupported text. You can make the system refuse to ship it.

---

## 2. CORE INNOVATION (the patentable angle)

**Cost-weighted selective abstention with claim-level granularity.**

Three mechanisms in combination, none of which is standard practice:

1. **Claim-level verification, not answer-level.** An answer is not "70% correct" — it is a set of 9 claims of which 7 are supported, 1 is contradicted, 1 is unsupported. You surface exactly which sentence to distrust.
2. **Semantic entropy as a second signal.** Sample the answer N times, cluster the samples by *meaning* (bidirectional entailment), and measure entropy over meaning-clusters rather than token distributions. High semantic entropy with high token confidence is the signature of a confabulation.
3. **The abstention threshold is derived from business cost, not tuned by vibes.** Each workflow declares `cost_of_wrong_answer` and `cost_of_escalation`. The system solves for the threshold that minimises expected cost. A KYC summary abstains at 0.85; a product FAQ abstains at 0.55. Same model, same pipeline, different risk posture — configured, not re-engineered.

---

## 3. ARCHITECTURE

```
                     ┌──────────────────────────────────────┐
   User Query ──────▶│  RAG Retriever (evidence pool)        │
                     └────────────────┬─────────────────────┘
                                      ▼
                     ┌──────────────────────────────────────┐
                     │  Generator (Claude, N samples)        │
                     └────────────────┬─────────────────────┘
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  CLAIM DECOMPOSER  → atomic, self-contained claims        │
        └────────────────┬────────────────────────────────────────┘
                         ▼
   ┌──────────────┬──────────────────┬────────────────────┐
   │ NLI VERIFIER │ SEMANTIC ENTROPY │ RETRIEVAL SUPPORT  │
   │ entail/contra│ over N samples   │ max sim to evidence│
   └──────┬───────┴─────────┬────────┴──────────┬─────────┘
          └─────────────────▼───────────────────┘
              ┌──────────────────────────────┐
              │  CALIBRATOR (isotonic/Platt) │  → p(claim is true)
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  ABSTENTION POLICY ENGINE     │
              │  argmin expected business cost│
              └──────────────┬───────────────┘
                             ▼
        ANSWER (green) │ HEDGED (amber) │ ABSTAIN→HUMAN (red)
                             │
                             ▼
              ┌──────────────────────────────┐
              │  FAILURE FINGERPRINT STORE    │ → auto guardrail rules
              └──────────────────────────────┘
```

---

## 4. EXTRA DEPENDENCIES

```
# on top of the common base
transformers==4.41.2          # cross-encoder NLI model
torch==2.3.0
scikit-learn==1.5.0           # isotonic regression calibration
rank-bm25==0.2.2
matplotlib==3.9.0
```

NLI model: `cross-encoder/nli-deberta-v3-base` (local, ~180MB). Falls back to Claude-as-verifier if the model can't load.

---

## 5. PROJECT-SPECIFIC CONFIG

```python
# appended to Settings
N_SAMPLES: int = 5                        # samples for semantic entropy
SAMPLE_TEMPERATURE: float = 0.7           # samples ARE stochastic on purpose
CLAIM_MIN_WORDS: int = 4
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
NLI_ENTAIL_THRESHOLD: float = 0.60
NLI_CONTRA_THRESHOLD: float = 0.50
EVIDENCE_TOP_K: int = 10
CALIBRATION_METHOD: str = "isotonic"      # "isotonic" | "platt" | "none"

# risk profiles — the innovation lives here
RISK_PROFILES: dict = {
    "kyc_summary":      {"cost_wrong": 500.0, "cost_escalate": 12.0},
    "policy_coverage":  {"cost_wrong": 250.0, "cost_escalate": 8.0},
    "product_faq":      {"cost_wrong": 5.0,   "cost_escalate": 2.0},
}
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/claim_decomposer.py`

```python
class ClaimDecomposer:
    async def decompose(self, answer: str, query: str) -> list[Claim]
        """
        Split an answer into atomic, independently-verifiable claims.
        Rules enforced via prompt + schema:
          - Each claim must be self-contained (resolve all pronouns and
            references — "it" becomes "the 2019 Toyota Camry")
          - One assertion per claim; split conjunctions
          - Preserve the exact character span in the original answer
          - Classify claim_type: FACTUAL | NUMERIC | TEMPORAL | CAUSAL | OPINION
          - OPINION and hedged claims are marked exempt from verification
        Returns list[Claim(id, text, span_start, span_end, claim_type, exempt)]
        """

    def reassemble(self, answer: str, verdicts: list[ClaimVerdict]) -> AnnotatedAnswer
        """Map verdicts back onto character spans for UI highlighting."""
```

The self-containment rule is critical. "It was filed in March" is unverifiable; "The claim CLM-4471 was filed in March 2024" is verifiable. Verify the decomposer on Day 2 with a unit test asserting no output claim contains an unresolved pronoun.

### 6.2 `core/verifier.py`

```python
class ClaimVerifier:
    async def verify(self, claim: Claim, evidence: list[SearchResult]) -> ClaimVerdict
        """
        1. Rank evidence chunks by similarity to the claim (not to the query).
        2. For top 3 chunks, run NLI(premise=chunk, hypothesis=claim).
        3. Aggregate:
             any contradiction > NLI_CONTRA_THRESHOLD  → CONTRADICTED
             max entailment   > NLI_ENTAIL_THRESHOLD   → SUPPORTED
             otherwise                                  → UNSUPPORTED
        4. NUMERIC claims get an extra exact-match check: extract all numbers
           from claim and evidence; a numeric claim whose figure appears in no
           evidence chunk is CONTRADICTED even if NLI says entailed.
           (Number hallucination is the highest-cost failure in BFSI and
            NLI models are notoriously bad at it — this override is worth
            calling out to the judges.)
        Returns ClaimVerdict(claim_id, verdict, nli_score, best_evidence_id,
                             evidence_span, numeric_check_passed)
        """
```

### 6.3 `core/entropy.py`

```python
class SemanticEntropyScorer:
    async def score(self, query: str, evidence, n: int = N_SAMPLES) -> EntropyResult
        """
        1. Generate n answers at SAMPLE_TEMPERATURE (concurrent, via complete_batch).
        2. Cluster answers by bidirectional entailment: a and b are in the same
           cluster iff NLI(a→b) and NLI(b→a) both entail.
        3. Semantic entropy H = -Σ p_c log p_c over cluster probabilities.
        4. Also return per-claim stability: fraction of the n samples in which
           an equivalent claim appears.
        Returns EntropyResult(entropy, n_clusters, cluster_sizes,
                              claim_stability: dict[claim_text, float])
        """
```

**Cost note for the judges:** entropy scoring is N× the cost, so it runs only for claims that the NLI verifier marks UNSUPPORTED, and only in high-risk profiles. Adaptive, not blanket. Show this in the cost panel.

### 6.4 `core/calibrator.py`

```python
class ConfidenceCalibrator:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None
        """
        Features per claim: [nli_entail_score, nli_contra_score, retrieval_max_sim,
                             retrieval_mean_sim, claim_stability, n_evidence_chunks,
                             is_numeric, claim_length]
        Labels: 1 = true claim, 0 = false claim (from the golden set).
        Fit isotonic regression → maps raw score to calibrated probability.
        Persist to data/db/calibrator.pkl
        """

    def predict_proba(self, features) -> float
    def reliability_diagram(self) -> dict     # for the ECE chart in the UI
    def expected_calibration_error(self) -> float
```

Calibration is what separates this from every other hallucination project. A model that says "90% confident" and is right 90% of the time is worth vastly more than one that says 90% and is right 60% of the time. Show the reliability diagram on stage.

### 6.5 `core/abstention_policy.py`

```python
class AbstentionPolicy:
    def decide(self, verdicts, calibrated_probs, profile: str) -> Decision
        """
        Let p = min calibrated probability across non-exempt claims.
        cost_answer   = (1 - p) * profile.cost_wrong
        cost_escalate = profile.cost_escalate
        If cost_answer > cost_escalate           → ABSTAIN (route to human)
        elif any claim CONTRADICTED              → BLOCK + show contradiction
        elif any claim UNSUPPORTED               → HEDGE (strip that claim,
                                                     answer with the rest,
                                                     state what is unknown)
        else                                     → ANSWER
        Always return the arithmetic so the UI can show WHY it abstained.
        """
```

### 6.6 `domain/fingerprint_store.py`

```python
class FingerprintStore:
    def record(self, query, claim, verdict, evidence) -> None
    def cluster_failures(self, min_cluster_size=3) -> list[FailurePattern]
        """Embed failed claims, DBSCAN cluster, ask Claude to name each cluster."""
    async def propose_guardrail(self, pattern: FailurePattern) -> GuardrailRule
        """
        Generate a concrete rule: a retrieval hint, a prompt addendum, or a
        hard block pattern. Rules are proposed, human-approved, then active.
        This is the self-improving loop — demo it.
        """
```

---

## 7. DATA MODELS

```python
class Claim(BaseModel):
    id: str; text: str; span_start: int; span_end: int
    claim_type: Literal["FACTUAL","NUMERIC","TEMPORAL","CAUSAL","OPINION"]
    exempt: bool

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["SUPPORTED","CONTRADICTED","UNSUPPORTED","EXEMPT"]
    nli_entail: float; nli_contra: float
    retrieval_max_sim: float; stability: float | None
    calibrated_probability: float
    best_evidence_id: str | None; evidence_quote: str | None
    numeric_check_passed: bool | None

class Decision(BaseModel):
    action: Literal["ANSWER","HEDGE","BLOCK","ABSTAIN"]
    expected_cost_of_answering: float
    cost_of_escalation: float
    rationale: str
    min_claim_probability: float

class VerifiedAnswer(BaseModel):
    query: str; raw_answer: str; final_answer: str
    claims: list[Claim]; verdicts: list[ClaimVerdict]
    decision: Decision; entropy: EntropyResult | None
    profile: str; elapsed_ms: int; cost_usd: float
    warnings: list[str]
```

---

## 8. API ROUTES

```
POST /api/ask
  {query, doc_id?, profile="policy_coverage", verify=true}
  → VerifiedAnswer

POST /api/verify
  {answer, evidence_ids[] | query}
  → verdicts only (verify text the system did not generate — useful for
    checking a human-written summary, and a nice unexpected capability to show)

GET  /api/calibration
  → reliability diagram data + ECE

GET  /api/fingerprints
  → clustered failure patterns + proposed guardrails
POST /api/fingerprints/{id}/approve
```

---

## 9. FRONTEND PAGES

**`01_ask.py` — the money page.** Query box + risk-profile selector. Answer rendered with per-claim highlighting: green underline (supported), amber (unsupported, hedged out), red strikethrough (contradicted). Hovering a claim shows the evidence quote and NLI scores. A prominent decision banner: `✅ ANSWERED` / `⚠️ HEDGED — 2 claims removed` / `🛑 ABSTAINED — routed to human review`. Below: the expected-cost arithmetic that produced the decision.

**`02_compare.py`** — same query, baseline vs ClaimLedger, side by side. Baseline shows fluent wrong text; ClaimLedger shows the catch.

**`03_calibration.py`** — reliability diagram (Plotly), ECE number, risk-coverage curve (accuracy vs % answered as the threshold moves). The risk-coverage curve is the single most persuasive chart you can show a technical judge.

**`04_fingerprints.py`** — clustered failure patterns, proposed guardrails, approve/reject buttons.

**`05_profiles.py`** — edit `cost_wrong` / `cost_escalate` per workflow and watch the abstention rate change live. Judges love touching a slider and seeing behaviour change.

---

## 10. SAMPLE DATA

`scripts/generate_samples.py` builds:

1. **Knowledge corpus** — 60 synthetic insurance policy documents (~3 pages each) with precise, checkable facts: coverage limits, exclusions, waiting periods, claim deadlines, policy numbers.
2. **Golden set** — `data/golden/claims.jsonl`, 120 cases:
   - 40 **answerable** — the fact is in the corpus (label: all claims true)
   - 40 **unanswerable** — plausible questions whose answer is deliberately absent (correct behaviour = abstain). *This is the set baselines fail catastrophically on.*
   - 25 **partially answerable** — 3 of 4 sub-facts present (correct = hedge)
   - 15 **trap questions** — presuppose a false premise ("Why does policy P-8823 exclude flood damage?" when it doesn't). Correct = contradict the premise.
3. **Human labels** — each golden case has the true answer and the list of true atomic claims, so the calibrator has training data.

Generate with Claude in a scripted loop, then commit. 120 cases takes ~15 minutes and about $2 of API spend.

---

## 11. BENCHMARK

`scripts/benchmark.py` — Arm A = plain RAG + Claude, no verification. Arm B = ClaimLedger.

| Metric | How computed | Baseline (expect) | Target |
|---|---|---|---|
| Hallucination rate | % answers containing ≥1 false claim | 18–25% | < 4% |
| Unanswerable abstention rate | % of the 40 unanswerable cases correctly refused | 5–15% | > 90% |
| False abstention rate | % of answerable cases wrongly refused | 0% | < 8% |
| ECE (calibration error) | reliability diagram | 0.25–0.40 | < 0.08 |
| Claim-level F1 | vs human labels | — | > 0.85 |
| Cost per verified answer | from `llm_traces` | 1× | 2.5–4× |
| Expected loss (₹) | Σ cost_wrong × errors + cost_escalate × abstentions | high | 60–80% lower |

**Expected loss is your headline number.** It converts a technical metric into rupees and directly earns Business Value (5%) and Value Proposition (6%).

---

## 12. DEMO FLOW (`scripts/demo_flow.py`, 4 minutes)

1. **The setup.** "This is an insurance policy assistant. 60 policies indexed."
2. **Baseline fails, fluently.** Ask: *"What is the flood damage sub-limit under policy P-8823?"* — a policy with no flood coverage at all. Baseline answers confidently: "The flood damage sub-limit is ₹5,00,000." Let that sit for a beat.
3. **ClaimLedger catches it.** Same question. Red banner: `🛑 ABSTAINED`. The claim decomposition shows the numeric claim with `retrieval_max_sim = 0.31`, `NLI = unsupported`, `semantic entropy = 1.94 across 4 distinct meaning clusters` — the model gave four different numbers across five samples.
4. **The economics.** Show the decision arithmetic: `expected cost of answering ₹142 vs cost of escalation ₹8 → abstain`. Move the risk profile slider to "product FAQ" and watch the same question get answered with a hedge. **Same model, different risk posture, one config change.**
5. **Partial credit.** Ask a partially-answerable question. Show the hedge: 3 claims kept, 1 removed, with "I could not verify the waiting period for this policy" stated explicitly.
6. **The self-improving loop.** Open the fingerprint page: 7 failures clustered into "numeric sub-limits for perils not in the policy schedule", with a proposed guardrail rule. Approve it.
7. **The chart.** Risk-coverage curve and reliability diagram. "At 92% coverage we are 97% accurate. Baseline is 79% accurate at 100% coverage."

---

## 13. FIVE-DAY PLAN

**Day 1** — Scaffold, Claude client + trace table, corpus generator, RAG baseline working end to end. Gate: baseline answers a question with citations.

**Day 2** — `claim_decomposer.py`, `verifier.py` with NLI model + numeric override. Unit tests: no unresolved pronouns in claims; numeric override catches a planted wrong figure. Gate: decompose + verify one answer correctly.

**Day 3** — Golden set (120 cases, labelled), `entropy.py`, `calibrator.py` fitted, `benchmark.py`. Gate: real benchmark table printed, ECE computed.

**Day 4** — `abstention_policy.py`, `fingerprint_store.py`, all API routes, all 5 Streamlit pages wired. Gate: judge could click through unaided.

**Day 5** — Demo script, cache warming, offline mode, error passes, README with the metrics table, three clean dry runs.

**Cut list if behind:** drop `02_compare` (do it live with the baseline toggle), drop `fingerprint_store` auto-guardrails (show clustering only), drop the compare page. **Never cut the calibrator or the golden set** — they are the score.

---

## 14. JUDGE TALKING POINTS

**"Isn't this just RAG with citations?"** No. Citations tell you where text came from; they do not tell you whether the generated claim is entailed by it. Models routinely cite a real chunk and then state something the chunk doesn't say. We verify entailment at claim level and we act on the result by refusing to answer.

**"Why not just use a bigger/better model?"** Frontier models still hallucinate on long-tail enterprise facts that exist in no training corpus — your policy schedules, your SKUs. And model choice gives you no *control surface*: you cannot tell a model "be 3× more cautious in KYC than in FAQ." Our abstention policy is that control surface, and it works with any model.

**"Doesn't abstaining make it useless?"** Our false-abstention rate is under 8%, so we answer 92% of answerable questions. And an abstention costs ₹8 to escalate; a confident wrong KYC summary costs ₹500 and a regulator letter. We show the expected-loss curve.

**"What's the latency and cost overhead?"** 2.5–4× cost, ~1.8s added latency, and it's adaptive — full entropy scoring fires only on claims that fail the cheap NLI check, which is about 12% of claims. Low-risk profiles skip it entirely.

**"How does it scale to 1000 users?"** Verification is stateless and embarrassingly parallel. The NLI model runs on CPU at ~40 claims/sec; one GPU handles thousands. The calibrator is a single fitted curve loaded in memory.

**"Where does this fail?"** Claims requiring multi-hop reasoning across chunks that individually entail nothing — we mark those UNSUPPORTED and abstain, which is safe but costs coverage. That's the known limitation, and the fix is graph-based evidence assembly, which is our Phase 2.
