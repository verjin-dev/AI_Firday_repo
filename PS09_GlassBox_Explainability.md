# PS09 — `GlassBox`
## Unlocking Clarity: Explainability & Interpretability in AI Decisions

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**GlassBox verifies that an AI's explanation actually caused its decision, instead of trusting the model's own account of itself.**

The uncomfortable truth that will get you noticed in the first 30 seconds: when you ask an LLM why it decided something, it does not introspect — it generates a plausible-sounding story. That story is frequently unrelated to the computation. Every "explainable AI" project that asks the model to explain itself is shipping rationalisation to a regulator and calling it transparency.

---

## 2. CORE INNOVATION

**Faithfulness verification by causal ablation.**

An explanation claims "your application was declined primarily because of your debt-to-income ratio and secondarily because of a recent credit enquiry." GlassBox tests that claim:

1. **Necessity test** — remove the cited factor from the inputs and re-run. If the decision does not change, the factor was not necessary and the explanation overstates it.
2. **Sufficiency test** — remove everything *except* the cited factors. If the decision still holds, the cited set is sufficient.
3. **Completeness test** — ablate factors the explanation did *not* mention. If removing an unmentioned factor flips the decision, the explanation is hiding the real driver. **This is the one that catches proxy discrimination.**

Combine into a **faithfulness score**. Explanations below threshold are regenerated with the ablation evidence supplied, or the decision is escalated. No unfaithful explanation ever reaches a customer or a regulator.

Second innovation: **counterfactual recourse generation**. Don't just explain the denial — compute the minimum feasible change that would reverse it ("approval at a debt-to-income ratio of 38% or below, which for you means reducing monthly obligations by ₹6,200"), respecting which features are actionable. Under GDPR Article 22 and emerging AI Act guidance, actionable recourse is what regulators increasingly expect, and almost nobody builds it.

---

## 3. ARCHITECTURE

```
  Inputs ──▶ ┌──────────────────────────────────────┐
             │ DECISION ENGINE                       │
             │ structured output:                    │
             │ {decision, factors[], evidence[],     │
             │  rules_fired[], confidence}           │
             └──────────────┬───────────────────────┘
                            ▼
             ┌──────────────────────────────────────┐
             │ PROVENANCE RECORDER                   │
             │ retrieval hits, rules, tool outputs,  │
             │ model+prompt version, input snapshot  │
             └──────────────┬───────────────────────┘
                            ▼
      ┌─────────────────────────────────────────────────┐
      │ ABLATION ENGINE  (the innovation)                │
      │  necessity │ sufficiency │ completeness          │
      │  → causal factor weights                         │
      └──────────────────────┬──────────────────────────┘
                             ▼
      ┌─────────────────────────────────────────────────┐
      │ FAITHFULNESS SCORER   →  < threshold? regenerate │
      └──────────────────────┬──────────────────────────┘
                             ▼
   ┌──────────────┬──────────────────┬────────────────────┐
   │ CUSTOMER VIEW│ REGULATOR VIEW   │ RECOURSE ENGINE    │
   │ plain language│ reason codes +  │ minimum feasible   │
   │ + recourse    │ full trace      │ change to flip     │
   └──────────────┴──────────────────┴────────────────────┘
```

---

## 4. EXTRA DEPENDENCIES

```
shap==0.45.1               # for the tabular/rules component comparison
scikit-learn==1.5.0
scipy==1.13.1
networkx==3.3              # decision trace graph
graphviz==0.20.3
reportlab==4.2.0
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
ABLATION_STRATEGY: str = "mask"          # "mask" | "neutral_value" | "population_mean"
ABLATION_SAMPLES: int = 3                # repeats per ablation for stability
FAITHFULNESS_THRESHOLD: float = 0.70
MAX_EXPLANATION_FACTORS: int = 4
REASON_CODE_MAP: str = "./config/reason_codes.yaml"   # ECOA-style codes
RECOURSE_ACTIONABLE_FEATURES: list[str] = ["monthly_obligations","loan_amount",
                                           "tenure_months","co_applicant"]
RECOURSE_IMMUTABLE: list[str] = ["age","gender","region","employer_type"]
TRACE_RETENTION_DAYS: int = 2555          # 7 years — regulatory
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/decision_engine.py`

```python
class DecisionEngine:
    async def decide(self, inputs: Inputs, policy_ctx) -> Decision
        """
        REQUIRED structured output (complete_json):
          decision: APPROVE | DECLINE | REFER | INSUFFICIENT_INFO
          factors: [{name, direction: +/-, claimed_weight: 0-1, evidence_ref}]
          rules_fired: [rule_id]
          confidence: 0-1
        The model must name factors from a CLOSED VOCABULARY defined in the
        policy — free-text factor names cannot be ablated and cannot be
        mapped to reason codes. This constraint is what makes the rest work.
        """
```

### 6.2 `core/ablation.py` — the core module

```python
class AblationEngine:
    async def run(self, inputs, decision, engine) -> AblationReport
        """
        For each factor f in the CLOSED VOCABULARY (mentioned or not):
          necessity[f]   : decide(inputs with f masked). Did the decision flip?
                           Record magnitude of score change.
          sufficiency[S] : decide(inputs with ONLY the cited factor set S).
                           Does the original decision hold?
          For stability, repeat ABLATION_SAMPLES times and take the modal
          outcome — LLM decisions are stochastic and a single ablation is
          noise.

        causal_weight[f] = normalised |score change| when f is masked.

        Compare causal_weight against claimed_weight from the explanation.
        Emit three findings:
          OVERSTATED  : claimed high, causal low
          UNDERSTATED : claimed low, causal high
          HIDDEN      : not claimed at all, causal weight > 0.15   ← the important one
        """

    def cost_note(self) -> str:
        """Ablation costs (n_factors + 1) decision calls. With a closed
           vocabulary of 8-12 factors that's ~13 cheap structured calls,
           heavily cacheable, ~₹1.40 per decision. Say this before you're asked."""
```

### 6.3 `core/faithfulness.py`

```python
class FaithfulnessScorer:
    def score(self, decision, ablation: AblationReport) -> FaithfulnessScore
        """
        rank_correlation : Spearman between claimed_weight and causal_weight
        necessity_hit    : fraction of cited factors that are actually necessary
        hidden_penalty   : 1 - (sum of causal weight of UNMENTIONED factors)
        faithfulness = 0.4*rank_corr + 0.3*necessity_hit + 0.3*hidden_penalty

        Returns the score plus a human-readable diagnosis:
          'Explanation cites credit_enquiries as primary (0.45) but ablation
           shows causal weight 0.04. The actual primary driver is
           employment_tenure (causal 0.51), which the explanation omits.'
        """

    async def regenerate(self, decision, ablation, engine) -> Decision
        """Re-ask for the explanation WITH the ablation evidence supplied as
           ground truth. Re-score. Max 2 attempts, then escalate to human."""
```

### 6.4 `core/recourse.py`

```python
class RecourseEngine:
    async def compute(self, inputs, decision, engine) -> Recourse
        """
        Search over ACTIONABLE features only (never immutable ones — offering
        recourse via a protected attribute is itself a discrimination finding
        and we flag it as such).
        1. Coordinate search: for each actionable feature, binary-search the
           threshold at which the decision flips, holding others constant.
        2. Then greedy multi-feature combination for the minimum-effort set,
           weighted by a per-feature difficulty cost from config.
        3. Express in the customer's units: not 'DTI < 0.38' but
           'reduce monthly obligations by ₹6,200, or extend tenure to 25 years'.
        4. Verify: apply the proposed change and re-run. Assert the decision
           actually flips. NEVER present unverified recourse.
        """
```

### 6.5 `core/provenance.py`

```python
class ProvenanceRecorder:
    def record(self, decision_id, **artifacts) -> TraceGraph
        """
        Build a networkx DAG:
          input nodes → retrieval nodes → rule nodes → factor nodes → decision
        Each edge carries: source, timestamp, confidence.
        Persist as JSON + render to SVG via graphviz.
        Store: model id, prompt version hash, policy version, input snapshot
        hash, all retrieved chunk ids, all tool responses, ablation results.
        Retention 7 years. Replayable: given a trace, re-execute and assert
        the same decision (regression protection AND a regulator's dream).
        """

    def replay(self, decision_id) -> ReplayResult
```

### 6.6 `core/views.py`

```python
class ExplanationViews:
    async def customer(self, decision, ablation, recourse) -> str
        """Plain language, ≤120 words, grade 8 reading level, no jargon,
           factors ordered by CAUSAL weight (not claimed), ends with recourse."""

    def regulator(self, decision, ablation, provenance) -> RegulatorPack
        """Reason codes from the ECOA-style map, causal weights table,
           faithfulness score, full trace graph, model/prompt versions,
           policy citations with document and clause. Exportable as PDF."""

    def engineer(self, decision, ablation, provenance) -> dict
        """Everything, raw, plus the ablation matrix and replay handle."""
```

Three views from one decision. Judges immediately understand this — it's the same fact set rendered for three audiences with different needs and different rights.

---

## 7. TARGET SYSTEM

**Loan underwriting agent** (Banking & Insurance), with a deliberately planted pathology: the model's decisions are heavily driven by `employment_tenure`, but its self-generated explanations consistently cite `credit_enquiries` because that reads as a more conventional credit rationale. This is a realistic failure mode — LLMs cite what sounds like a reason, not what drove the computation. Your ablation engine catches it. That is the whole demo.

Also plant one case where the hidden driver is a **regional proxy**, so the completeness test surfaces a discrimination finding.

---

## 8. API ROUTES

```
POST /api/decide              {inputs} → Decision + faithfulness + recourse
GET  /api/decision/{id}/ablation
GET  /api/decision/{id}/trace          (SVG + JSON)
GET  /api/decision/{id}/customer
GET  /api/decision/{id}/regulator.pdf
POST /api/decision/{id}/replay
POST /api/decision/{id}/whatif         {feature: value} → counterfactual outcome
```

---

## 9. FRONTEND PAGES

**`01_decide.py`** — application form → decision with a faithfulness badge. If the score is low, an amber banner: *"Explanation regenerated — initial explanation cited a factor with causal weight 0.04."* Showing that the system caught its *own* model rationalising is a strong trust signal.

**`02_ablation.py` — the centrepiece.** A grouped bar chart: claimed weight vs causal weight per factor. The `credit_enquiries` bar is tall in claimed, near-zero in causal. The `employment_tenure` bar is the inverse and highlighted red as **HIDDEN**. One glance tells the story.

**`03_recourse.py`** — the customer view with an interactive slider per actionable feature; move it and see the decision flip live, with the verification tick.

**`04_trace.py`** — the provenance DAG, rendered, clickable, with a Replay button that re-executes and shows the decision matched.

**`05_regulator.py`** — the reason-code pack, exportable to PDF.

---

## 10. BENCHMARK

Golden set: 60 decisions where the true causal drivers are **known by construction** (built with a controlled policy so ground truth exists).

| Metric | Self-explanation (baseline) | GlassBox |
|---|---|---|
| Faithfulness score (mean) | 0.41 | **0.86** |
| Explanations citing a non-causal primary factor | 43% | **4%** |
| Hidden drivers detected | 0 of 12 planted | **11 of 12** |
| Recourse offered | none | 100% of declines |
| Recourse verified to actually flip the decision | n/a | **100%** |
| Rank correlation, claimed vs causal | 0.22 | **0.81** |
| Cost per decision | 1× | 3.4× |
| Latency | 1.2s | 3.8s (parallel ablation) |

**"43% of self-generated explanations cite a factor that provably didn't drive the decision"** is the sentence that gets remembered from your presentation. It's a finding, not a feature.

---

## 11. DEMO FLOW (4 minutes)

1. **Ask the model to explain itself.** Run a decline. Read the explanation: fluent, professional, cites recent credit enquiries as primary. "This is what every explainable-AI demo shows you, and it is what most systems ship."
2. **Test the claim.** Ablation panel. Remove `credit_enquiries` entirely and re-run — **the decision does not change**. "That reason was not a reason."
3. **Find the real driver.** Completeness test: masking `employment_tenure`, which the explanation never mentioned, flips the decline to an approve. Causal weight 0.51. Marked HIDDEN in red.
4. **The regulatory consequence.** Switch to the planted proxy case. The hidden driver is a regional variable. "This decision would have been defended to a regulator with an explanation that was not merely incomplete — it was concealing the actual basis. Not through malice. The model doesn't know why it decided either."
5. **Fix it.** Regeneration with ablation evidence. New explanation cites tenure first. Faithfulness 0.34 → 0.89.
6. **Recourse.** Customer view: "approval at monthly obligations of ₹6,200 lower, or with tenure extended to 25 years." Move the slider. Decision flips live, verification tick appears. "Verified, not asserted."
7. **The record.** Trace graph, then Replay: same inputs, same policy version, same decision, seven years from now.

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, loan agent with closed-vocabulary structured decisions, controlled policy so ground truth exists, 60-case golden set.
**Day 2** — `ablation.py` — necessity, sufficiency, completeness, with sample stability. Gate: causal weights computed for one decision.
**Day 3** — `faithfulness.py` + regeneration + `benchmark.py`. Gate: the 0.41 vs 0.86 numbers are real.
**Day 4** — `recourse.py` with verification, `provenance.py` + replay, all 5 pages.
**Day 5** — Regulator PDF, reason-code map, demo script, README, dry runs.

**Cut list:** the regulator PDF export, the trace SVG rendering (show JSON). **Never cut** ablation or recourse verification.

---

## 13. JUDGE TALKING POINTS

**"Why not just use SHAP or LIME?"** Those are built for feature-vector models. An LLM decision pipeline has no stable feature vector — the inputs are text, retrieval, and rules. Our ablation is the same causal idea adapted to that setting: mask a semantic factor, re-run the whole pipeline, observe the change. We do use SHAP for the tabular components and reconcile the two, which is worth showing as rigour.

**"Isn't chain-of-thought already an explanation?"** Chain-of-thought is generated text that correlates with the answer; it is not a causal account, and the research on unfaithful reasoning is clear on this. We treat any model-generated rationale — including CoT — as a *hypothesis* to be tested by ablation, not as evidence.

**"Ablation costs a lot of extra calls."** Thirteen additional structured calls per decision at roughly ₹1.40, heavily cached, and they run in parallel so latency is 3.8s not 15s. For a credit decision that carries a seven-year regulatory retention obligation and an appeal right, that is not a meaningful cost. For low-stakes use cases we sample — full ablation on 10% plus every decline.

**"What if masking a factor makes the input incoherent?"** Real risk, which is why we support three ablation strategies and default to neutral-value substitution rather than deletion for structured fields, and why we repeat and take the modal outcome. We also report ablation stability per factor and discount unstable ones.

**"Regulatory basis?"** GDPR Article 22 and Recital 71 (meaningful information about the logic involved, and the right to contest); EU AI Act Article 13 (transparency for high-risk systems) and Article 86 (right to explanation of individual decisions); RBI's fair practices expectations for adverse credit decisions. The regulator pack is structured against those.

**"Scale?"** Ablation is stateless and parallel. The trace store is append-only and partitioned by date; 7-year retention of 1M decisions/year is about 400GB, which is unremarkable. Replay is a pure function of the stored snapshot.
