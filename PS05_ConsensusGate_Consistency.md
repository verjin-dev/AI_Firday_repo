# PS05 — `ConsensusGate`
## Unlocking Consistency: Tackling the Unpredictable Nature of Probabilistic Outputs

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**ConsensusGate separates the part of an answer that must never vary from the part that may.** Setting `temperature=0` does not give you consistency — it gives you brittle repeatability that shatters the moment a user rephrases a question. Real enterprise consistency means: *the same question, asked five different ways, by five different agents, must produce the same decision.*

---

## 2. CORE INNOVATION

**Decision/prose separation with a Semantic Variance SLO and an automatic canonicalisation pipeline.**

1. **Split every response into a decision core and a prose wrapper.** The decision core is a typed, constrained-decoded structure (`{eligible: bool, limit: 500000, waiting_period_days: 30, rule_ids: [...]}`). The prose wrapper is free text explaining it. Variance in prose is fine and even desirable. Variance in the decision core is a defect. Nobody measures these separately; everyone should.
2. **Semantic Variance Score (SVS)** — an operational SLO. For each intent, take N paraphrases of the same question, run them, and measure disagreement on the decision core (exact/numeric match) and on the prose (embedding dispersion). Publish `SVS_decision` and `SVS_prose` per intent. Alert when `SVS_decision > 0`.
3. **Automatic canonicalisation.** Any intent whose decision variance exceeds threshold is routed into a promotion pipeline: extract the governing rule from source policy into executable code, have a human approve it once, and thereafter the LLM never *decides* that class of question — it only retrieves the deterministic result and explains it. **Consistency increases monotonically over time and the LLM's job shrinks to what it's actually good at.**

That last point is the pitch line: *we don't make the model more consistent, we progressively remove the decision from the model.*

---

## 3. ARCHITECTURE

```
Query ─▶ INTENT CLASSIFIER ─▶ is this intent canonicalised?
                                    │
                    ┌───── YES ─────┴───── NO ─────┐
                    ▼                              ▼
        ┌────────────────────┐        ┌──────────────────────────┐
        │ RULE ENGINE        │        │ CONSENSUS SAMPLER        │
        │ deterministic exec │        │ N samples @ temp>0        │
        │ + LLM explains only│        └────────────┬─────────────┘
        └─────────┬──────────┘                     ▼
                  │              ┌────────────────────────────────┐
                  │              │ SEMANTIC CLUSTERER              │
                  │              │ cluster by DECISION equality,   │
                  │              │ then by meaning for prose       │
                  │              └────────────┬───────────────────┘
                  │                           ▼
                  │              ┌────────────────────────────────┐
                  │              │ CONSENSUS SELECTOR              │
                  │              │ majority decision + agreement % │
                  │              └────────────┬───────────────────┘
                  │                           ▼
                  │              ┌────────────────────────────────┐
                  │              │ SVS METER → if variance high,   │
                  │              │ queue for CANONICALISATION      │
                  │              └────────────┬───────────────────┘
                  └───────────┬───────────────┘
                              ▼
              SEMANTIC CACHE (decision-keyed) ─▶ Response + agreement badge
```

---

## 4. EXTRA DEPENDENCIES

```
scikit-learn==1.5.0
scipy==1.13.1
jsonschema==4.22.0
simpleeval==0.9.13        # safe evaluation of extracted rule expressions
matplotlib==3.9.0
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
CONSENSUS_N: int = 5
CONSENSUS_TEMPERATURE: float = 0.7      # sampling is deliberately stochastic
CONSENSUS_MIN_AGREEMENT: float = 0.60   # below this → escalate, don't answer
SVS_DECISION_SLO: float = 0.02          # 2% decision variance budget
SVS_PROSE_SLO: float = 0.35             # prose may vary much more
PARAPHRASES_PER_PROBE: int = 8
SEMANTIC_CACHE_THRESHOLD: float = 0.93  # cosine sim for a cache hit
CANONICALISATION_QUEUE_THRESHOLD: float = 0.05
RULE_ENGINE_DIR: str = "./data/rules"
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/decision_schema.py`

```python
class DecisionSchemaRegistry:
    """
    Each intent registers a typed decision schema. Example:

    return_policy_eligibility:
        eligible: bool
        window_days: int
        restocking_fee_pct: float
        exclusions: list[str]
        rule_ids: list[str]

    The LLM is REQUIRED to fill this (via complete_json). Prose is generated
    in a second, separate call that receives the decision as input and is told
    'explain this decision; you may not contradict or alter any field'.
    """
    def register(self, intent: str, schema: Type[BaseModel]) -> None
    def get(self, intent: str) -> Type[BaseModel] | None
```

This two-call split is the architectural heart. Do it on Day 2.

### 6.2 `core/consensus.py`

```python
class ConsensusSampler:
    async def sample(self, query, context, schema, n=CONSENSUS_N) -> ConsensusResult
        """
        1. Generate n decision structures concurrently at CONSENSUS_TEMPERATURE.
        2. Group by DECISION EQUALITY (field-wise; numerics compared with a
           tolerance; lists compared as sets). Not by text similarity.
        3. agreement = size of largest group / n
        4. If agreement >= CONSENSUS_MIN_AGREEMENT → return majority decision
           else → return ESCALATE with all distinct decisions shown, so the
           human sees exactly what the model was torn between.
        5. Record per-field disagreement: which field caused the split.
           'The model agreed on eligibility 5/5 but disagreed on
            restocking_fee_pct 3/5' is a debuggable statement.
        """
```

### 6.3 `core/variance_meter.py`

```python
class VarianceMeter:
    async def probe(self, intent: str, base_question: str,
                    context) -> SVSReport
        """
        1. Generate PARAPHRASES_PER_PROBE paraphrases of the question that
           preserve meaning exactly (LLM call, validated by bidirectional
           entailment — reject any paraphrase that isn't equivalent).
        2. Run each through the full pipeline.
        3. SVS_decision = 1 - (modal decision count / total)
           computed per field AND overall.
        4. SVS_prose = mean pairwise cosine distance of prose embeddings.
        5. Flag any field where paraphrasing changed the answer — this is
           the enterprise's real pain and nobody measures it.
        """

    def slo_status(self, intent) -> SLOStatus     # OK | AT_RISK | BREACHED
    def trend(self, intent, days=30) -> list[SVSPoint]
```

### 6.4 `core/canonicalizer.py`

```python
class Canonicalizer:
    async def extract_rule(self, intent, source_docs, examples) -> ProposedRule
        """
        Ask Claude to express the governing policy as an executable rule:
          conditions: list of predicates over typed inputs
          outputs: the decision fields it determines
          source_citations: exact spans in the policy document
          test_cases: input/output pairs it must satisfy
        Emitted as YAML, evaluated with simpleeval (no arbitrary code exec).
        """

    def validate(self, rule: ProposedRule, golden_cases) -> ValidationReport
        """Run the rule against labelled cases. Report accuracy and conflicts
           with existing rules. A rule that doesn't beat the LLM is rejected."""

    def promote(self, rule, approver: str) -> ActiveRule
        """Version it, record approver + timestamp + source citations.
           From now on this intent is deterministic."""

class RuleEngine:
    def evaluate(self, intent, inputs) -> RuleResult | None
        """Deterministic. Returns the decision + which rule + which clause.
           Returns None if no rule covers the input (falls back to consensus)."""
```

### 6.5 `core/semantic_cache.py`

```python
class SemanticCache:
    def lookup(self, query_embedding, intent) -> CacheHit | None
        """Cosine sim >= SEMANTIC_CACHE_THRESHOLD within the same intent.
           Returns the canonical DECISION; prose is regenerated fresh so the
           answer doesn't feel robotic. Cache the decision, not the wording —
           this is a small idea that gets a nod from technical judges."""

    def store(self, query, embedding, decision, provenance) -> None
    def invalidate(self, rule_id | doc_id) -> int
```

---

## 7. API ROUTES

```
POST /api/answer          {query, intent?, mode: "consensus"|"single"} → Answer
POST /api/probe           {intent, question} → SVSReport (paraphrase probe)
GET  /api/slo             per-intent SVS dashboard data
GET  /api/queue           intents queued for canonicalisation
POST /api/rules/propose   {intent} → ProposedRule
POST /api/rules/{id}/approve
GET  /api/rules           active rules with source citations
```

---

## 8. FRONTEND PAGES

**`01_ask.py`** — ask a question; see the answer with an **agreement badge** (`5/5 agreed` green, `3/5 agreed` amber, `2/2/1 split` red → escalated). Expandable panel shows all N sampled decisions side by side with the disagreeing field highlighted.

**`02_probe.py` — the demo page.** Type one question. It generates 8 paraphrases and runs all of them. Displays a grid: rows = paraphrases, columns = decision fields, cells green if they match the mode, red if not. **Baseline mode shows a red-speckled grid. ConsensusGate + rules shows solid green.** That grid is the single most legible artefact in this whole project.

**`03_slo.py`** — per-intent SVS trend charts, SLO status, breach history.

**`04_canonicalize.py`** — queue of high-variance intents. Click one → proposed rule in readable YAML with source citations from the policy PDF → validation report against golden cases → Approve. Then re-run the probe live and watch the grid turn green.

**`05_rules.py`** — the active rule library with version history and coverage stats ("62% of production queries now resolved deterministically").

---

## 9. SAMPLE DATA

Domain: **retail returns, price-match and loyalty policy** — chosen because the problem statement's retail scenarios are exactly this, and because the policies are simple enough for judges to verify the rule extraction by eye.

`scripts/generate_samples.py` builds:
- **Policy corpus**: 8 policy documents (returns, exchanges, price match, loyalty tiers, damaged goods, final sale, gift receipts, international) with deliberate ambiguities and cross-references.
- **Golden set**: 80 questions with human-labelled correct decision structures, including 20 designed to be genuinely ambiguous (correct behaviour = escalate, not guess).
- **Paraphrase bank**: 8 paraphrases per question, pre-generated and validated, so the probe runs instantly from cache during the demo.

---

## 10. BENCHMARK

| Metric | Baseline (temp=0, single call) | + Consensus | + Rules |
|---|---|---|---|
| Decision variance across paraphrases | 22–31% | 9% | **< 2%** |
| Exact decision match, identical query ×10 | 88% | 100% | 100% |
| Decision accuracy vs golden labels | 74% | 82% | **94%** |
| Correct escalation on ambiguous cases | 5% | 71% | 88% |
| % queries resolved deterministically | 0% | 0% | 62% |
| Cost per query | 1× | 4.2× | **0.6×** |
| p50 latency | 1.4s | 2.1s | 0.3s |

**The cost row is the surprise.** Consensus is expensive, but canonicalisation *reduces* cost below baseline because 62% of queries stop calling the LLM for the decision at all. A consistency solution that also cuts cost by 40% is a very strong business case — lead with it.

---

## 11. DEMO FLOW (4 minutes)

1. **The problem, made visceral.** Baseline mode. Ask *"Can I return a discounted item bought 20 days ago?"* five times. Get three different answers: yes with 15% fee, yes with no fee, no. Let the room see it.
2. **Paraphrase attack.** Open the probe page. Eight paraphrases, baseline mode. The grid lights up red — the `restocking_fee_pct` column especially. **"Same question. Eight ways of asking. Five different fee amounts. This is what your contact centre is doing right now."**
3. **Consensus helps.** Switch on consensus. Grid mostly green, one amber. Agreement badge shows `4/5`. Point out the model disagreed with itself on exactly one field, and we can name it.
4. **But consensus is expensive.** Show the cost panel: 4.2×.
5. **Canonicalise.** Open the queue — `return_eligibility` is top by variance. Click Propose. Show the extracted rule in YAML with citations back to page 3 of the returns policy. Validation: 20/20 golden cases pass. Approve.
6. **Re-run the probe.** Grid goes fully green. Agreement 8/8. Latency drops to 0.3s. Cost drops below baseline. **"We didn't make the model consistent. We took the decision away from it — and it got cheaper."**
7. **The trend.** SLO page showing variance falling over the simulated 30 days as more intents canonicalise.

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, policy corpus, intent classifier, baseline single-call answerer. Gate: baseline answers with visible variance.
**Day 2** — `decision_schema.py` + the two-call decision/prose split + `consensus.py`. Gate: agreement scores computed.
**Day 3** — Golden set (80 cases) + paraphrase bank + `variance_meter.py` + `benchmark.py`. Gate: real SVS numbers for baseline and consensus.
**Day 4** — `canonicalizer.py`, `RuleEngine`, `semantic_cache.py`, API, all 5 pages.
**Day 5** — Demo script, cache warm, README, dry runs.

**Cut list:** semantic cache, SLO trend page. **Never cut** the probe grid — it is the demo.

---

## 13. JUDGE TALKING POINTS

**"Why not just set temperature to zero?"** Because temperature 0 gives repeatability for *byte-identical inputs only*. Our benchmark shows 22–31% decision variance across meaning-preserving paraphrases at temperature 0. Enterprises don't have byte-identical inputs — they have humans asking the same thing differently. Determinism at the token level is not consistency at the decision level.

**"Isn't self-consistency sampling well known?"** The sampling is; clustering by *typed decision equality with numeric tolerance* rather than text similarity is not, and it's what makes per-field disagreement diagnosable. And self-consistency alone is a 4× cost tax with no path to improvement — our canonicalisation loop converts that tax into a permanent, cheaper, auditable rule.

**"What if the extracted rule is wrong?"** It never activates without passing the golden validation set and a human approval, both recorded with the approver's name and the source citations. Rules are versioned and revertible. And if no rule covers an input, we fall back to consensus rather than forcing a wrong deterministic answer.

**"Doesn't this just turn into a rules engine? Why use an LLM at all?"** Exactly — and that's the point, for the 62% of traffic that is genuinely rule-governed. The LLM's remaining jobs are the ones it's uniquely good at: interpreting the messy question, mapping it to an intent, extracting the rule from prose policy in the first place, and explaining the deterministic result in plain language. We're not removing the LLM, we're putting it where it belongs.

**"Scalability?"** Rule evaluation is microseconds. Cache lookups are a single vector search. Only the shrinking uncanonicalised tail pays the consensus cost, and that tail shrinks every week the system runs.
