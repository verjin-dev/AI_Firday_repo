# PS08 — `FairLens`
## Bias & Fairness: Ensuring Equitable Outcomes in AI Outputs

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**FairLens turns fairness from a quarterly audit into a CI gate.** Every use case gets a bias budget. Exceed it and the build fails, exactly like a failing test — because that is the only mechanism that has ever made engineering teams take a non-functional requirement seriously.

---

## 2. CORE INNOVATION

**Counterfactual fairness at scale, enforced as a bias budget, with attribution to the source of the bias.**

1. **Counterfactual generation, not demographic slicing.** Group-level statistics on production data are confounded — different demographics genuinely ask different questions. Counterfactuals hold *everything* constant and change only the protected signal, so any outcome difference is causal by construction. We generate them automatically across name, gender marker, dialect/register, region, age cue, and disability disclosure.

2. **The bias budget.** Each use case declares a maximum permitted disparity per metric, chosen and signed off by a named owner. CI computes disparity and fails the build on breach. Budgets can be *spent down* deliberately — an owner can accept a 2% gap with a written justification that gets recorded — which is what makes it usable rather than aspirational.

3. **Bias attribution.** When disparity is detected, decompose *where it entered*: the retrieval layer (did different names retrieve different documents?), the training/prior of the model itself, or the prompt. Run the same counterfactual with retrieval held fixed to isolate it. **"Your bias is 71% retrieval-driven, not model-driven"** is an actionable finding with a cheap fix, and no other team will produce it.

---

## 3. ARCHITECTURE

```
   Production prompts / seed cases
              │
              ▼
   ┌────────────────────────────┐
   │ COUNTERFACTUAL GENERATOR   │  name│gender│dialect│region│age│disability
   │ validated equivalence      │  → k variants per case
   └─────────────┬──────────────┘
                 ▼
   ┌────────────────────────────┐      ┌──────────────────────────┐
   │ RUNNER (variants × system) │─────▶│ ATTRIBUTION HARNESS       │
   └─────────────┬──────────────┘      │ rerun with retrieval FIXED│
                 ▼                      └───────────┬──────────────┘
   ┌────────────────────────────┐                   │
   │ FAIRNESS METRICS            │                  │
   │ decision: DPD, EOD, flip%   │                  │
   │ language: sentiment/tone Δ,  │◀─────────────────┘
   │           readability Δ,     │
   │           refusal-rate Δ     │
   │ statistical significance     │
   └─────────────┬──────────────┘
                 ▼
   ┌────────────────────────────┐     ┌───────────────────────────┐
   │ BIAS BUDGET GATE            │────▶│ MITIGATION LAYER          │
   │ pass / warn / FAIL BUILD    │     │ blind│rebalance│guard│rerank│
   └─────────────┬──────────────┘     └───────────────────────────┘
                 ▼
   ┌────────────────────────────┐
   │ FAIRNESS REPORT CARD (PDF) │  signed, versioned, audit-ready
   └────────────────────────────┘
```

---

## 4. EXTRA DEPENDENCIES

```
scipy==1.13.1
statsmodels==0.14.2        # proportion tests, confidence intervals
scikit-learn==1.5.0
matplotlib==3.9.0
textstat==0.7.3            # readability differential
reportlab==4.2.0           # report card PDF
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
ATTRIBUTES: list[str] = ["name_origin","gender","dialect","region",
                         "age_cue","disability_disclosure"]
VARIANTS_PER_ATTRIBUTE: int = 6
BOOTSTRAP_ITERATIONS: int = 2000
SIGNIFICANCE_ALPHA: float = 0.05

# Bias budgets — per use case, signed off by a named owner
BIAS_BUDGETS: dict = {
  "credit_decision": {"demographic_parity_diff": 0.02,
                      "counterfactual_flip_rate": 0.01,
                      "tone_delta": 0.05, "owner": "risk.head@example.com"},
  "support_chat":    {"counterfactual_flip_rate": 0.03,
                      "readability_delta": 2.0, "refusal_rate_delta": 0.03,
                      "owner": "cx.head@example.com"},
}
MITIGATIONS_ENABLED: list[str] = ["attribute_blinding","retrieval_rebalance",
                                  "decision_guard","explanation_neutralize"]
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/counterfactual.py`

```python
class CounterfactualGenerator:
    def generate(self, case: Case, attributes: list[str], k=6) -> list[Variant]
        """
        name_origin  : swap names across a curated bank spanning North/South/
                       East/West Indian origins, Muslim/Hindu/Christian/Sikh/
                       Parsi names, and Anglo names. Bank is a committed data
                       file with sources cited — do not generate names ad hoc.
        gender       : swap gendered names, titles, pronouns consistently.
        dialect      : rewrite in Indian English register / non-native syntax /
                       formal standard English. MEANING MUST BE IDENTICAL —
                       validate with bidirectional entailment and drop failures.
        region       : swap city/state and PIN, holding tier constant
                       (metro↔metro, tier-2↔tier-2) so you don't confound
                       geography with genuine economic signal.
        age_cue      : 'recent graduate' / 'mid-career' / 'nearing retirement'
        disability   : add/remove a disclosure that is legally irrelevant to
                       the decision at hand.

        CRITICAL: every variant must be verified to preserve all
        decision-relevant facts. Run an LLM equivalence check that asserts
        income, tenure, amount, and all material facts are unchanged.
        Log and discard any variant that fails — report the discard rate.
        """
```

The curated name bank and the equivalence validator are what separate a serious fairness project from a demo that swaps "John" for "Jamal" and calls it science. Build both on Day 2.

### 6.2 `core/metrics.py`

```python
class FairnessMetrics:
    def decision_metrics(self, results: list[VariantResult]) -> DecisionFairness
        """
        counterfactual_flip_rate : % of cases where the DECISION changed under
                                   a protected-attribute swap. The headline —
                                   it needs no ground truth and is causal.
        demographic_parity_diff  : max - min positive rate across groups
        equal_opportunity_diff   : requires labels; computed where available
        score_delta              : mean |score difference| with bootstrap CI
        Each with a two-proportion z-test and a bootstrap 95% CI. Report
        significance — a 3% gap on n=40 is noise and saying so builds trust.
        """

    def language_metrics(self, results) -> LanguageFairness
        """
        Bias in advisory/support systems shows up in LANGUAGE before decisions:
          tone_delta        : sentiment/warmth differential (judge-scored, 0-1)
          readability_delta : Flesch-Kincaid grade difference — are some groups
                              given simpler, more patronising explanations?
          length_delta      : is one group given less detail?
          refusal_delta     : is one group refused or hedged more often?
          hedging_delta     : density of uncertainty markers
        This axis is the one competitors miss entirely and it is where
        LLM bias most commonly and most legally-dangerously appears.
        """
```

### 6.3 `core/attribution.py`

```python
class BiasAttributor:
    async def attribute(self, case, variants) -> AttributionReport
        """
        Ablation across three conditions:
          A) full pipeline                        → total disparity D_total
          B) retrieval FROZEN to the base case's  → D_model_prompt
             chunks, only generation varies
          C) attributes blinded in the prompt,    → D_retrieval
             retrieval unchanged
        retrieval_share = (D_total - D_model_prompt) / D_total
        model_share     = D_model_prompt / D_total
        Also report WHICH chunks differed between variants — usually a
        regional or demographic term in the query pulled a different policy
        document, which is a five-minute fix in the retriever.
        """
```

### 6.4 `core/mitigations.py`

```python
class MitigationLayer:
    def attribute_blinding(self, prompt) -> str
        """Strip/neutralise protected signals before the decision call.
           Keep them for the explanation call so the response stays natural."""

    def retrieval_rebalance(self, query, results) -> list[SearchResult]
        """Detect when retrieval differs across counterfactual variants and
           enforce a stable, attribute-independent retrieval set."""

    def decision_guard(self, decision, prompt) -> GuardResult
        """Post-hoc: re-run the decision on the blinded prompt. If the two
           disagree, the protected attribute materially influenced the
           outcome → BLOCK and escalate. A hard causal check at runtime,
           not just at test time."""

    async def explanation_neutralize(self, explanation, decision) -> str
        """Rewrite to a consistent register and reading level so tone and
           readability deltas collapse. Verify the decision is unchanged."""
```

`decision_guard` is a runtime control, not just a test. Point that out — it's the difference between measuring fairness and enforcing it.

### 6.5 `core/budget.py`

```python
class BiasBudget:
    def evaluate(self, use_case, metrics) -> BudgetResult
        """PASS | WARN (within 20% of budget) | BREACH. On breach, produce
           the CI annotation with the exact metric, value, budget, CI, and
           the responsible owner's email."""

    def accept_exception(self, use_case, metric, value, justification,
                         approver, expires_at) -> Exception_
        """Time-boxed, named, recorded. Appears on the report card. This is
           what makes the budget usable in a real organisation instead of
           being switched off in week two."""
```

### 6.6 `core/report_card.py`

Generates a signed PDF: use case, owner, date, model version, every metric with CI and budget, attribution breakdown, active exceptions, mitigations applied, and the counterfactual methodology appendix. This is the artefact a Chief Risk Officer files. Show it on screen.

---

## 7. TARGET SYSTEMS

Two systems under test, both deliberately imperfect:

1. **Credit decision agent** (Banking) — decides eligibility and explains it. Contains a planted retrieval bias: queries mentioning certain regions retrieve a stricter internal policy variant.
2. **Customer support assistant** (Telecom) — no formal decisions, but exhibits **language bias**: shorter, simpler, more hedged answers for non-standard dialect inputs. This is the more interesting demo because it's invisible to every decision-metric-only tool.

---

## 8. API ROUTES

```
POST /api/audit          {use_case, cases[] | sample_from_prod} → AuditRun
GET  /api/audit/{id}     metrics + CIs + significance
GET  /api/audit/{id}/attribution
POST /api/mitigate       {audit_id, mitigations[]} → post-mitigation metrics
GET  /api/budget         status per use case
POST /api/budget/exception
GET  /api/report/{id}.pdf
POST /api/gate           CI endpoint → exit code + annotations
```

---

## 9. FRONTEND PAGES

**`01_audit.py`** — pick a use case, run the audit. Progress bar over variants. Results: a big **counterfactual flip rate** number, then per-attribute breakdown bars with confidence intervals drawn.

**`02_cases.py` — the persuasion page.** A table of individual flips: base case and counterfactual side by side with only the changed token highlighted, and the two different decisions. *One row where the only difference is a surname and the outcome flipped from approve to decline is worth more than every chart in the deck.*

**`03_attribution.py`** — donut chart: retrieval 71% / model 24% / prompt 5%. Below it, the differing retrieved chunks with the offending policy document named.

**`04_mitigate.py`** — toggle each mitigation, watch metrics update live, with an accuracy-cost column so judges see the trade-off honestly.

**`05_budget.py`** — budget status board, breach history, active exceptions with approvers and expiry dates, and the CI gate output.

---

## 10. BENCHMARK

| Metric | Unmitigated | + Blinding | Full FairLens |
|---|---|---|---|
| Counterfactual flip rate (credit) | 6.8% | 3.1% | **0.7%** |
| Demographic parity difference | 0.11 | 0.05 | **0.015** |
| Tone delta (support) | 0.19 | 0.17 | **0.03** |
| Readability delta (grade levels) | 2.9 | 2.8 | **0.4** |
| Refusal rate delta | 0.07 | 0.06 | **0.01** |
| Decision accuracy (vs labels) | 0.88 | 0.85 | **0.87** |
| Runtime cost multiplier | 1× | 1× | 1.3× |

Report the **accuracy cost honestly**. A fairness project that claims zero trade-off is not believed. Ours costs 1 accuracy point and we say so — and the decision guard recovers most of it because blinding is applied only to the decision call, not the explanation.

---

## 11. DEMO FLOW (4 minutes)

1. **One case.** Show a single loan application. Change nothing but the applicant's surname. Two different outcomes. Read both explanations aloud. Silence in the room.
2. **At scale.** Run the audit: 40 seed cases × 6 attributes × 6 variants = 1,440 runs (cached, 30 seconds). Counterfactual flip rate: **6.8%**, 95% CI [5.1%, 8.6%], p < 0.001. Statistically real, not noise.
3. **The invisible bias.** Switch to the support assistant. Zero decisions, so every decision-metric tool reports zero bias. Then show the language panel: **2.9 grade levels simpler** and **19% cooler in tone** for non-standard dialect inputs. "This system has never made an unfair decision. It has been condescending to a specific group ten thousand times."
4. **Where does it come from?** Attribution: 71% retrieval. Show the two different policy documents retrieved for identical applications with different regional cues. "This isn't a model problem. It's a four-line fix in the retriever."
5. **Mitigate.** Apply retrieval rebalance + explanation neutralise. Flip rate 6.8% → 0.7%. Tone delta 0.19 → 0.03. Accuracy 0.88 → 0.87, stated plainly.
6. **Enforce.** Push a change that reintroduces the bias. CI gate fails with the annotation naming the metric, the value, the budget, and the owner. **"This is a failing test now. Not a slide in a quarterly review."**
7. **The artefact.** Open the signed PDF report card.

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, the two target systems with planted biases, curated name bank with sources.
**Day 2** — `counterfactual.py` + equivalence validator. Gate: variants generated and verified fact-preserving, discard rate reported.
**Day 3** — `metrics.py` (decision + language) with bootstrap CIs, 40 seed cases, `benchmark.py`. Gate: real flip rate with confidence intervals.
**Day 4** — `attribution.py`, `mitigations.py`, `budget.py`, all 5 pages.
**Day 5** — `report_card.py` PDF, CI gate, demo script, README, dry runs.

**Cut list:** the PDF report card, the exception workflow. **Never cut** the language-fairness axis or attribution — they are the differentiators.

---

## 13. JUDGE TALKING POINTS

**"Why counterfactuals instead of measuring production outcomes by group?"** Because production group comparisons are confounded — different groups genuinely present different applications, so a raw disparity tells you nothing about causation. Counterfactuals hold every material fact constant and vary only the protected signal, so any difference is caused by that signal. It also means we need no demographic data about real users, which is itself a privacy win.

**"Isn't swapping names a crude proxy?"** It's one of six attributes, and every variant is validated to preserve all decision-relevant facts before it's counted — we report the discard rate for variants that failed that check. Names are the most legally consequential proxy in Indian lending and hiring contexts, which is why we start there, but dialect and region routinely produce larger effects in our results.

**"You changed the model's behaviour — did accuracy drop?"** By one point, from 0.88 to 0.87, and we show it. We blind the decision call but not the explanation call, which preserves natural language while removing the causal path. Any fairness tool that claims zero cost is not measuring properly.

**"How is a bias budget different from just having a threshold?"** A threshold is a number in a document. A budget has a named owner, a CI gate that fails the build, a time-boxed exception process with a recorded approver, and a report card that goes to the risk committee. It's an organisational mechanism with a technical enforcement point.

**"Regulatory alignment?"** EU AI Act Article 10 (data governance and bias examination for high-risk systems) and Article 15 (accuracy and robustness); NIST AI RMF Measure 2.11; and in credit specifically, the disparate impact doctrine that underlies ECOA and India's fair lending expectations. The report card is structured as evidence for exactly those.

**"Scale?"** 1,440 counterfactual runs is 30 seconds warm and $2.80 cold. In CI we run a stratified 300-run subset per commit and the full 1,440 nightly.
