# PS04 — `DriftLens`
## Continuous Model Performance Monitoring: Ensuring Accuracy & Reliability

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**DriftLens is an LLM observability plane where every production incident permanently becomes a regression test.** Monitoring that only tells you something broke is a dashboard. Monitoring that makes the same break impossible next week is a system.

---

## 2. CORE INNOVATION

**The drift-to-eval closed loop.**

1. Production traces are scored online by reference-free evaluators.
2. Low-scoring and anomalous traces are **clustered by failure mode**, not listed individually — 400 alerts become 6 named problems.
3. Each cluster is labelled once by a human (one click), then **automatically promoted into a versioned regression suite** that gates every future deploy.
4. The suite grows monotonically. Your eval coverage is a function of your incident history, so the system gets harder to break over time without anyone writing tests.

Secondary innovation: **multi-signal drift detection**. Most tools watch output quality only. We watch four independent distributions — input embeddings, retrieved-context embeddings, output embeddings, and behaviour (tool-call patterns, refusal rate, length) — because they fail in different orders and the *order* tells you the root cause. Input drift first means user behaviour changed. Context drift first means your knowledge base changed. Output drift alone means the model changed.

That diagnostic ordering is the thing no competing team will have.

---

## 3. ARCHITECTURE

```
  App / Agents ──[OTel GenAI spans]──▶ COLLECTOR ──▶ trace store (SQLite+Chroma)
                                                          │
        ┌─────────────────────────────────────────────────┤
        ▼                        ▼                        ▼
  ONLINE EVALUATORS      DRIFT DETECTORS          BEHAVIOUR MONITOR
  groundedness           PSI / MMD / KS on:       refusal rate, tool-call
  relevance              input | context |        mix, latency, length,
  schema validity        output | behaviour       cost per session
  safety, latency               │
        └────────────┬──────────┘
                     ▼
          ┌────────────────────────┐
          │  ANOMALY CLUSTERER     │  embed failures → HDBSCAN → Claude names
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐      ┌───────────────────────┐
          │  ROOT CAUSE ANALYZER   │─────▶│  ALERT (with diagnosis)│
          │  slice by tenant/intent│      └───────────────────────┘
          │  /source/model version │
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐
          │  REGRESSION PROMOTER   │──▶ versioned eval suite ──▶ CI gate
          └────────────────────────┘
```

---

## 4. EXTRA DEPENDENCIES

```
opentelemetry-api==1.25.0
opentelemetry-sdk==1.25.0
scipy==1.13.1
hdbscan==0.8.36
scikit-learn==1.5.0
matplotlib==3.9.0
apscheduler==3.10.4
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
EVAL_SAMPLE_RATE: float = 0.15          # % of live traffic scored online
EVAL_SAMPLE_RATE_ERRORS: float = 1.0    # always score anything that errored
DRIFT_WINDOW_HOURS: int = 24
DRIFT_REFERENCE_DAYS: int = 7           # baseline window
PSI_ALERT_THRESHOLD: float = 0.20       # 0.1 = moderate, 0.2 = significant
MMD_ALERT_THRESHOLD: float = 0.05
QUALITY_DROP_ALERT_PCT: float = 10.0
CLUSTER_MIN_SIZE: int = 4
JUDGE_MODEL: str = "claude-sonnet-4-6"
REGRESSION_SUITE_DIR: str = "./data/eval_suites"
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/collector.py`

```python
class TraceCollector:
    async def ingest(self, span: GenAISpan) -> str
        """
        Accept OpenTelemetry GenAI-convention spans. Required attributes:
          gen_ai.system, gen_ai.request.model, gen_ai.usage.input_tokens,
          gen_ai.usage.output_tokens, plus custom: session_id, tenant_id,
          intent, retrieved_chunk_ids[], tool_calls[], user_feedback?
        Store the row in SQLite; store input/context/output embeddings in
        three separate Chroma collections (they are compared independently).
        """

    def sdk_decorator(self):
        """@driftlens.trace — one-line instrumentation for any LLM call.
           Ship this. 'Add one decorator' is a much better story than
           'integrate our platform'."""
```

### 6.2 `core/evaluators.py`

```python
class EvaluatorSuite:
    async def evaluate(self, trace: Trace) -> EvalScores
        """
        Reference-free evaluators (no ground truth needed — this is why it
        can run on live traffic):
          groundedness      : are output claims entailed by retrieved context?
                              (reuse the NLI approach; 0-1)
          relevance         : does the output address the query? (judge, 0-1)
          coherence         : self-consistency and structure (judge, 0-1)
          schema_validity   : deterministic — does structured output parse?
          safety            : toxicity/PII leak flags (deterministic + judge)
          completeness      : were retrieved chunks actually used?
        Judge calls run on a sampled subset. Deterministic checks run on 100%.
        Every judge call records its own LLMTrace (meta-monitoring — mention it,
        judges like that you monitor the monitor).
        """

    def calibrate(self, human_labels: list[HumanLabel]) -> CalibrationReport
        """
        Compute Cohen's kappa between judge scores and human labels.
        Report it in the UI. An uncalibrated LLM judge is a random number
        generator with good manners; showing kappa = 0.79 is a credibility move.
        """
```

### 6.3 `core/drift.py`

```python
class DriftDetector:
    def detect(self, dimension: Literal["input","context","output","behaviour"],
               window_hours: int, reference_days: int) -> DriftResult
        """
        For embedding dimensions:
          - PSI over the top-k PCA components (interpretable, per-bin)
          - MMD with RBF kernel (distribution-level, permutation test for p-value)
        For behaviour:
          - KS test on latency, output length, cost
          - Chi-square on tool-call mix, refusal rate, intent mix
        Returns DriftResult(dimension, statistic, p_value, severity,
                            top_contributing_examples[], first_detected_at)
        """

    def diagnose(self, results: list[DriftResult]) -> Diagnosis
        """
        THE DIAGNOSTIC TABLE — this is the innovation, implement it explicitly:

        input drifted, others stable        → user behaviour / new intent
        context drifted, input stable       → knowledge base changed or
                                              retrieval config regressed
        output + quality drifted, in/ctx ok → model version or provider change
        behaviour only                      → tool/API dependency degraded
        all four                            → deployment or config incident

        Return the diagnosis WITH the evidence that produced it and a
        confidence. Never emit a bare 'drift detected' alert.
        """
```

### 6.4 `core/clusterer.py`

```python
class FailureClusterer:
    async def cluster(self, traces: list[Trace], min_size=CLUSTER_MIN_SIZE):
        """
        1. Take traces where any eval score < threshold or user gave 👎.
        2. Build a failure embedding: concat(query_emb, output_emb, score_vector).
        3. HDBSCAN (handles noise, doesn't need k).
        4. For each cluster: ask Claude to name it and write a one-line
           description from 5 representative examples.
        5. Rank clusters by (size × severity × tenant_impact).
        Returns list[FailureCluster(id, name, description, size, severity,
                                    representatives[], affected_tenants[])]
        """
```

### 6.5 `core/root_cause.py`

```python
class RootCauseAnalyzer:
    def slice(self, cluster: FailureCluster) -> SliceReport
        """
        Automatic slicing: for each dimension (tenant, intent, model_version,
        retrieved_source, time_bucket, prompt_version), compute the failure
        rate inside the cluster vs the global base rate, with a chi-square
        significance test. Return the slices where the lift is significant.
        Output reads like: 'This cluster is 8.4× over-represented in
        retrieved_source=policy_v3.pdf (p<0.001)' — which is an actionable
        sentence, not a dashboard.
        """
```

### 6.6 `core/regression_promoter.py`

```python
class RegressionPromoter:
    def propose(self, cluster: FailureCluster) -> list[ProposedCase]
        """Pick 3 representatives, generate an assertion for each:
           expected behaviour, the evaluator that must pass, threshold."""

    def promote(self, cases: list[ProposedCase], label: HumanLabel) -> SuiteVersion
        """Append to the versioned suite (git-friendly YAML), bump version,
           record provenance: which incident, which cluster, which date."""

    async def run_suite(self, version: str, target) -> SuiteResult
        """Run the whole suite against a candidate model/prompt. This is the
           CI gate. Exit code 1 on regression."""
```

---

## 7. SIMULATOR — THE THING THAT MAKES THIS DEMOABLE

You cannot wait for real drift during a hackathon. Build `scripts/simulate_traffic.py`:

```python
"""
Generates 30 days of realistic traces for a telecom support assistant:
  Days 1-20  : healthy baseline, ~200 traces/day, quality ~0.88
  Day 21     : KNOWLEDGE INCIDENT — 30% of retrievals start hitting a stale
               tariff document. Context embeddings drift, groundedness falls.
  Day 24     : INTENT SHIFT — a new product launch introduces queries about
               a plan the KB has never heard of. Input drift, no context drift.
  Day 27     : MODEL SWAP — silently switch the generator to a weaker model.
               Output drift + quality drop, input and context stable.
Each incident has a DIFFERENT signature, so the diagnostic table proves itself.
Writes ~6,000 traces. Runs in under 2 minutes using cached LLM responses.
"""
```

This simulator is the single highest-leverage file in the project. Build it on Day 2, not Day 5.

---

## 8. API ROUTES

```
POST /api/traces                    ingest span(s)
GET  /api/metrics?window=24h        aggregate quality/latency/cost timeseries
GET  /api/drift?dimension=all       drift results + diagnosis
GET  /api/clusters                  failure clusters ranked
GET  /api/clusters/{id}/rootcause   slice report
POST /api/clusters/{id}/promote     promote to regression suite
GET  /api/suites                    list versions
POST /api/suites/{v}/run            run against a candidate
GET  /api/judge-calibration         kappa vs human labels
```

---

## 9. FRONTEND PAGES

**`01_overview.py`** — the operations dashboard. Four sparkline rows (input / context / output / behaviour drift), a quality timeseries with incident markers, cost and latency, and an active-alerts panel. Each alert shows the **diagnosis sentence**, not just a metric.

**`02_drift.py`** — per-dimension detail with PSI bin charts and the diagnostic table highlighted showing which row matched.

**`03_clusters.py`** — ranked failure clusters as cards: name, size, severity, affected tenants, three example traces. One button: **"Promote to regression suite"**.

**`04_suite.py`** — the growing suite. Version history showing "v1: 12 cases → v2: 15 cases (added from incident 2026-08-21 stale-tariff)". Run button with pass/fail matrix.

**`05_calibration.py`** — judge vs human agreement, kappa, confusion matrix.

---

## 10. BENCHMARK

Arm A = threshold alerting on average quality score only (what most teams build). Arm B = DriftLens.

| Metric | Baseline | DriftLens |
|---|---|---|
| Incidents detected (3 planted) | 1 of 3 | 3 of 3 |
| Mean time to detect | 3.2 days | 4.5 hours |
| Root cause correctly identified | 0 of 3 | 3 of 3 |
| Alert volume for the 3 incidents | 412 raw alerts | 6 clustered alerts |
| Alert precision | 11% | 83% |
| Judge–human agreement (kappa) | n/a | > 0.75 |
| Regression cases auto-generated | 0 | 9 |
| Monitoring cost as % of app LLM spend | — | < 6% |

**"412 alerts became 6"** is your headline. Alert fatigue is the thing every enterprise judge has personally suffered.

---

## 11. DEMO FLOW (4 minutes)

1. **The dashboard.** 30 days of a telecom assistant. Healthy for three weeks.
2. **Incident 1 — the KB change.** Day 21. Context drift alarm fires 4 hours in. Diagnosis: *"Context distribution shifted while input distribution is stable → knowledge source change. 8.4× over-represented: policy_v3.pdf."* Click through to the actual stale document.
3. **Incident 2 — different signature.** Day 24. Input drift, context stable. Diagnosis: *"New intent cluster detected: 'FiberMax plan' — 61 queries, 0 supporting documents in KB."* **Same tool, different diagnosis, because the signature differs.** This is the moment that lands.
4. **Incident 3 — the silent model swap.** Day 27. Output drift + quality drop, inputs and context stable. Diagnosis: model or prompt change. Show the model-version slice.
5. **Alert fatigue solved.** Toggle to baseline: 412 alerts. Toggle back: 6 clusters. 
6. **The loop closes.** Promote the stale-tariff cluster. Suite goes v3 → v4 with 3 new cases and provenance. Run it against the old broken config: fails. Against the fix: passes. **"That incident can never silently recur."**

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, trace schema, collector, `@driftlens.trace` decorator, three Chroma collections. Gate: a trace ingests and appears in the DB.
**Day 2** — `simulate_traffic.py` producing 30 days with 3 planted incidents. `evaluators.py`. Gate: 6,000 traces scored.
**Day 3** — `drift.py` with all four dimensions + the diagnostic table, `benchmark.py`. Gate: all 3 incidents detected with correct diagnoses.
**Day 4** — `clusterer.py`, `root_cause.py`, `regression_promoter.py`, API, all 5 pages.
**Day 5** — Judge calibration with 60 hand labels, demo script, README, dry runs.

**Cut list:** the calibration page and behaviour-dimension drift. **Never cut** the simulator or the diagnostic table.

---

## 13. JUDGE TALKING POINTS

**"How is this different from LangSmith / Langfuse / Arize?"** Those are excellent trace viewers and we use the same OTel conventions deliberately so we're complementary. What none of them do is close the loop: cluster failures into named problems, diagnose root cause by comparing drift across four independent distributions, and auto-promote incidents into a gating regression suite. We're not a better dashboard — we're the part after the dashboard.

**"LLM-as-judge is unreliable."** Correct, which is why we report Cohen's kappa against human labels and show the confusion matrix. Ours is 0.79. Any judge score we present comes with its calibration. And our deterministic evaluators — schema validity, PII leaks, latency — need no judge at all and run on 100% of traffic.

**"What does this cost to run?"** Under 6% of application LLM spend, because judges sample 15% of traffic and the four drift detectors are pure numpy on cached embeddings. Errored traces are always scored; healthy traffic is sampled.

**"Does it need ground truth?"** No. Every online evaluator is reference-free. Ground truth is only needed for the calibration set (60 labels, one afternoon) and for the regression suite, which acquires labels one click at a time as incidents happen.

**"Scale?"** Trace ingest is append-only and shards by tenant. Drift detection is a windowed batch job — 6,000 traces takes 1.2 seconds. At 10M traces/day you sample for the statistics; PSI and MMD are stable at n=5,000 per window.
