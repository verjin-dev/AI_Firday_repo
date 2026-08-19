# PS13 — `TokenOps`
## Cracking the Cost Curve: FinOps for Agentic AI at Enterprise Scale

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**TokenOps measures cost per business outcome, not cost per token — then learns, per task type, the cheapest route that still meets a quality bar.**

Cost per token is a vanity metric. "₹4.20 per resolved support ticket, down from ₹11.80, at the same CSAT" is a sentence a CFO can act on. Everything in this project exists to produce that sentence.

---

## 2. CORE INNOVATION

1. **Unit economics as the primary metric.** Every trace is attributed to a business outcome — ticket resolved, claim adjudicated, document processed — through an outcome-tagging middleware. All dashboards, budgets, and alerts are denominated in cost-per-outcome. This reframing is the project.

2. **A learning router (contextual bandit).** Static routing tables ("use the cheap model for classification") are guesses that rot. TokenOps runs a Thompson-sampling bandit per task type over a route space (model × prompt variant × context depth × cache policy), with reward = `quality_score − λ × cost`. It *learns* from production which route is best and adapts as models, prompts, and traffic change. Crucially it includes an **exploration budget cap** so learning never costs more than a set fraction of spend — that constraint is what makes it deployable rather than a research toy.

3. **Cost SLO burn-rate alerting.** Borrowed directly from SRE error budgets. A monthly cost budget is a *rate*; alert on burn-rate windows (a 14.4× burn over 1 hour, 6× over 6 hours) rather than on a threshold you cross on the 28th. This catches a runaway agent loop in minutes instead of at month-end invoice.

4. **Cost-aware orchestration.** Budget awareness is injected into the agent's control loop: an agent with 30% budget remaining plans fewer verification steps, uses shallower context, and prefers cached retrieval — degrading gracefully instead of either failing or blowing the budget.

---

## 3. ARCHITECTURE

```
   Agents / apps ──[outcome-tagged traces]──▶ COST LEDGER (per call, per outcome)
                                                     │
     ┌───────────────────────────────────────────────┤
     ▼                    ▼                          ▼
 UNIT ECONOMICS      BURN-RATE ALERTS         ATTRIBUTION
 cost per outcome    multi-window SLO         tenant│team│agent│step
     │                    │                          │
     └────────────────────┴──────────┬───────────────┘
                                     ▼
              ┌─────────────────────────────────────────┐
              │ OPTIMISATION STACK                       │
              │ semantic cache │ prompt compression       │
              │ context pruning│ model cascade            │
              │ batch/off-peak │ dedup                    │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │ LEARNING ROUTER (Thompson sampling)      │
              │ reward = quality − λ·cost                │
              │ exploration budget capped                │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │ GUARDRAILS: budget caps, circuit breakers│
              │ graceful degradation, loop detection     │
              └──────────────────┬──────────────────────┘
                                 ▼
                    FORECASTER → budgets, variance, showback
```

---

## 4. EXTRA DEPENDENCIES

```
scipy==1.13.1
scikit-learn==1.5.0
statsmodels==0.14.2        # forecasting
matplotlib==3.9.0
apscheduler==3.10.4
tiktoken==0.7.0
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
PRICE_TABLE: dict = {   # USD per 1M tokens — keep in one place, version it
  "claude-opus-5":   {"in": 15.00, "out": 75.00},
  "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
  "claude-haiku-4-5-20251001": {"in": 0.80, "out": 4.00},
}
USD_INR: float = 87.0

OUTCOME_TYPES: list[str] = ["ticket_resolved","claim_adjudicated",
                            "document_processed","lead_qualified"]
BANDIT_LAMBDA: float = 0.4              # cost weight in reward
BANDIT_EXPLORATION_BUDGET_PCT: float = 5.0
BANDIT_MIN_SAMPLES_PER_ARM: int = 30
QUALITY_FLOOR: float = 0.80             # never route below this

SEMANTIC_CACHE_THRESHOLD: float = 0.94
PROMPT_COMPRESSION_TARGET: float = 0.55  # keep 55% of tokens
CASCADE_ESCALATION_CONFIDENCE: float = 0.75

BURN_RATE_WINDOWS: list = [(1, 14.4), (6, 6.0), (24, 3.0)]   # (hours, multiplier)
CIRCUIT_BREAKER_MULTIPLIER: float = 25.0
LOOP_DETECTION_REPEAT_THRESHOLD: int = 4
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/ledger.py`

```python
class CostLedger:
    def record(self, call: LLMCall, tags: OutcomeTags) -> None
        """
        OutcomeTags(tenant, team, agent, workflow, step, outcome_id,
                    outcome_type, session_id).
        Cost computed from PRICE_TABLE including cache-read/write pricing
        tiers where applicable. Store in a columnar-friendly table.
        """

    def unit_economics(self, outcome_type, window, group_by) -> UnitEconomics
        """
        cost_per_outcome     = Σ cost / count(distinct successful outcome_id)
        calls_per_outcome, tokens_per_outcome, p50/p95 cost per outcome
        FAILED outcomes are counted in the numerator but not the denominator —
        wasted spend must show up in the unit cost, or you optimise the wrong
        thing. Say this explicitly; it's the kind of detail that reads as
        having actually thought about it.
        """

    def attribution(self, window) -> AttributionTree
        """Drill-down tree: tenant → team → agent → workflow → step.
           Every node carries cost, share, and week-over-week delta."""

    def waste_report(self, window) -> WasteReport
        """
        Named waste categories with rupee figures:
          duplicate_calls      : identical prompt hash within a session
          retry_waste          : failed calls that were retried
          abandoned_sessions   : spend on sessions with no outcome
          over_retrieval       : retrieved chunks never referenced in output
          verbose_output       : output tokens beyond what the schema needed
          loop_waste           : detected agent loops
        'Here is ₹1.4L of waste with a name and an owner' is the slide.
        """
```

### 6.2 `core/optimizers/`

```python
# semantic_cache.py
class SemanticCache:
    """Embed the query, match at >= threshold within (tenant, workflow).
       Store the RESULT and its quality score; never serve a cached result
       whose quality was below floor. Track hit rate and rupees saved."""

# compressor.py
class PromptCompressor:
    def compress(self, prompt, target_ratio) -> CompressedPrompt
        """
        Tiered, cheapest first:
          1. Deduplicate repeated context blocks (huge in multi-turn agents)
          2. Strip boilerplate and redundant instructions
          3. Drop retrieved chunks with attention-utility below threshold
             (learned from which chunks were actually cited historically)
          4. Only then LLM-based abstractive compression
        Always measure and report quality delta — compression that degrades
        answers is not a saving. Report the ratio AND the quality impact.
        """

# cascade.py
class ModelCascade:
    async def run(self, task) -> CascadeResult
        """
        Haiku first. Compute a confidence signal (self-reported confidence,
        schema validity, retrieval support, output entropy). Below
        CASCADE_ESCALATION_CONFIDENCE → escalate to Sonnet, then Opus.
        Track: escalation rate, cost saved, quality delta vs always-Opus.
        Report the break-even escalation rate — above ~40% the cascade costs
        more than going straight to the strong model, and knowing your own
        break-even is a credibility signal.
        """

# scheduler.py
class WorkloadScheduler:
    """Classify jobs interactive vs deferrable. Deferrable work batches and
       runs off-peak with higher concurrency and aggressive caching."""
```

### 6.3 `core/router.py` — the core module

```python
class LearningRouter:
    def __init__(self, arms: list[Route], lam=BANDIT_LAMBDA)
        """Route = (model, prompt_variant, context_depth, cache_policy).
           Typically 8-12 arms per task type."""

    def select(self, task_type, context) -> Route
        """
        Thompson sampling over a Beta posterior on normalised reward.
        Constraints applied BEFORE sampling:
          - arms whose observed quality < QUALITY_FLOOR are excluded
          - exploration spend this period is capped at
            BANDIT_EXPLORATION_BUDGET_PCT; when exhausted, exploit only
          - arms with < MIN_SAMPLES are given a warm-start prior from a
            static heuristic table, so day one isn't random
        """

    def update(self, route, quality: float, cost: float) -> None
        """reward = quality - lam * normalised_cost. Update posterior."""

    def explain(self, task_type) -> RouterExplanation
        """
        Per-arm: pulls, mean quality, mean cost, mean reward, posterior CI,
        and current selection probability. Make the bandit legible — a
        black-box router is a hard sell to an operations team, and this
        table is what makes it acceptable.
        """
```

### 6.4 `core/burn_rate.py`

```python
class BurnRateMonitor:
    def evaluate(self, budget: Budget) -> list[BurnAlert]
        """
        For each (window_hours, multiplier): if spend in that window
        extrapolates to > multiplier × the budgeted rate, alert with a
        severity derived from how fast the budget is being consumed.
        Multi-window suppresses noise: a 14.4× spike for 5 minutes is not
        an incident; a 6× burn sustained over 6 hours is.
        """
    def time_to_exhaustion(self, budget) -> timedelta
```

### 6.5 `core/guardrails.py`

```python
class CostGuardrails:
    def check(self, request, budget_state) -> GuardDecision
        """
        ALLOW | DEGRADE | QUEUE | BLOCK
        DEGRADE is the important one: at <30% budget remaining, force the
        cheap route, halve context depth, disable optional verification
        steps, raise the cache threshold. The workflow still completes.
        """
    def detect_loop(self, session) -> LoopDetection
        """Repeated identical (tool, args) or prompt hash within a session
           beyond threshold → kill the session and alert. Agent loops are
           the single largest source of surprise LLM bills."""
    def circuit_break(self, scope) -> None
```

### 6.6 `core/forecaster.py`

```python
class CostForecaster:
    def forecast(self, series, horizon_days=30) -> Forecast
        """Holt-Winters with weekly seasonality; prediction intervals.
           Also a driver decomposition: how much of the forecast growth is
           volume vs cost-per-outcome? Those need different responses —
           volume growth is good news, unit-cost growth is a regression."""
    def what_if(self, changes: dict) -> ScenarioResult
        """'If cache hit rate rises to 45% and 60% of classification moves
            to Haiku, monthly spend is ₹X ± Y.'"""
```

---

## 7. SIMULATOR

`scripts/simulate_workload.py` — 30 days of a multi-agent support platform:
- 4 agents (triage, retrieval, resolution, QA), 3 tenants, ~1,200 sessions/day
- Realistic distributions: long-tail session lengths, escalation patterns, peak hours
- **Planted incidents**: a retry storm on day 11; an agent loop on day 18 that burns 40% of a monthly budget in 6 hours; a prompt change on day 23 that triples context size
- Every session tagged with an outcome and a quality score

This is what makes the demo possible. Build it Day 2.

---

## 8. API ROUTES

```
POST /api/trace                ingest a tagged LLM call
GET  /api/unit-economics       cost per outcome, grouped
GET  /api/attribution          drill-down tree
GET  /api/waste                itemised waste report
GET  /api/router/{task_type}   bandit state + explanation
POST /api/route                get a route decision (used by agents)
GET  /api/burn                 burn-rate alerts
GET  /api/forecast             forecast + driver decomposition
POST /api/whatif               scenario modelling
GET  /api/showback             per-team/tenant chargeback report
```

---

## 9. FRONTEND PAGES

**`01_economics.py`** — the CFO view. Big number: **cost per resolved ticket, ₹4.20, down 64%**. Trend line. Volume vs unit-cost decomposition. Per-tenant showback table.

**`02_attribution.py`** — sunburst drill-down. Click through tenant → agent → step. The QA agent's verification step is 34% of spend for 6% of value — that finding should be visible in two clicks.

**`03_waste.py`** — waste categories as a ranked bar chart with rupee figures and an owner per category. A "fix" button that simulates the effect.

**`04_router.py`** — the bandit table: arms, pulls, quality, cost, posterior. A live convergence chart showing the router discovering over 30 days that Haiku handles 71% of triage at equal quality. Exploration budget consumption gauge.

**`05_burn.py`** — the incident view. Day 18's agent loop with multi-window burn-rate alerts firing, the circuit breaker engaging, and the spend curve flattening. Annotated timeline.

**`06_forecast.py`** — forecast with intervals and the what-if panel with sliders.

---

## 10. BENCHMARK

Arm A = single strong model for everything, no cache, no compression (the honest default most teams ship). Arm B = TokenOps.

| Metric | Baseline | TokenOps |
|---|---|---|
| Cost per resolved ticket | ₹11.80 | **₹4.20 (−64%)** |
| Quality score (judge, calibrated) | 0.87 | **0.86** |
| Cache hit rate | 0% | 38% |
| Cascade escalation rate | n/a | 29% |
| Context tokens per call (median) | 8,400 | 3,900 |
| Day-18 loop incident cost | ₹1,84,000 | **₹9,200** (caught in 11 min) |
| Waste identified and named | — | ₹1.4L / month |
| Router convergence | n/a | 6 days to stable policy |
| Monthly forecast accuracy (MAPE) | — | 6.2% |
| TokenOps overhead | — | 2.1% of managed spend |

**Two numbers on the slide: 64% cost reduction at 1 point of quality, and an incident contained in 11 minutes instead of 6 hours.** The second one is what actually terrifies enterprise buyers.

---

## 11. DEMO FLOW (4 minutes)

1. **Reframe the metric.** "Every AI cost dashboard shows tokens. Here is ours." Big number: ₹11.80 per resolved ticket, baseline. "That is the only number your CFO cares about."
2. **Find the money in two clicks.** Attribution sunburst: the QA verification step is 34% of spend. Drill in: it re-verifies answers that the resolution agent already scored above 0.9. "Nobody designed that. It accreted."
3. **Name the waste.** Waste page: ₹1.4L monthly, itemised — duplicate calls ₹41K, over-retrieval ₹38K, abandoned sessions ₹29K. Each with an owner.
4. **Watch the router learn.** Convergence chart over 30 simulated days: on day 1 the router splits traffic evenly; by day 6 it has discovered that 71% of triage runs on Haiku at equal quality. Open the explanation table — every arm's pulls, quality, cost, and posterior. "It's a bandit, not a black box. Here is exactly why it chose that."
5. **The incident.** Jump to day 18. An agent loop starts at 02:14. The 1-hour burn window fires at 02:25 — **11 minutes**. Circuit breaker engages, spend curve flattens. Baseline overlay shows the same incident running until the morning: ₹1.84L versus ₹9.2K. "That's the demo. Everything else is optimisation; this is insurance."
6. **Graceful degradation.** Set a tenant to 20% budget remaining. Same request: shallower context, cheap route, verification skipped. It still completes, quality 0.86 → 0.81, cost down 70%. "The workflow degrades. It doesn't stop."
7. **Forecast.** ₹18.4L next month ± 6.2%, decomposed: 82% of growth is volume, 18% is unit cost. "Volume growth is a good problem. Unit cost growth is a regression, and it has an owner."

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, price table, cost ledger with outcome tagging, tracing middleware. Gate: a tagged call lands with a rupee cost.
**Day 2** — `simulate_workload.py` 30 days with 3 planted incidents. `unit_economics` + `attribution`. Gate: cost per outcome computed.
**Day 3** — Optimisers (cache, compression, cascade), `benchmark.py`. Gate: real 64% number with a quality delta.
**Day 4** — `router.py` bandit with explanation, `burn_rate.py`, `guardrails.py`, all 6 pages.
**Day 5** — `forecaster.py` + what-if, showback, demo script, README, dry runs.

**Cut list:** forecasting, showback, workload scheduler. **Never cut** the burn-rate incident demo or the router explanation table.

---

## 13. JUDGE TALKING POINTS

**"Isn't this just cloud FinOps applied to LLMs?"** The discipline is borrowed deliberately and we say so — burn-rate alerting and showback come straight from the FinOps Foundation framework and SRE error budgets. What's different is that LLM cost is controllable at *request* time in ways cloud spend is not: you can choose a cheaper model, compress the prompt, or serve from cache, per call, based on the task. That control surface doesn't exist in cloud FinOps, and the learning router is how you exploit it.

**"Why a bandit instead of a static routing table?"** Because a static table is a snapshot of one engineer's beliefs on one Tuesday, and it decays as prompts, models, and traffic change. Our router converged in 6 days and found that 71% of triage runs fine on the cheap model — a split no one would have guessed. And it keeps adapting when the next model ships.

**"How do you stop the bandit from degrading quality while it explores?"** Three constraints. A hard quality floor excludes any arm that has ever scored below it. The exploration budget is capped at 5% of spend. And arms start with a warm-start prior from a heuristic table, so day one is sensible rather than random. The measured quality delta across the whole benchmark is one point.

**"How is quality scored in production without ground truth?"** The same reference-free evaluators used in the monitoring project — groundedness, schema validity, task-completion signals — plus real outcome signals where available: was the ticket reopened, did the customer escalate, did the human accept the draft. We report the judge's kappa against human labels alongside every quality figure.

**"What does TokenOps itself cost?"** 2.1% of managed spend. Tracing is a database write. Cache lookups are one vector search. The bandit is arithmetic. Only the quality scoring calls a model, and that samples at 15%.

**"Scale?"** The ledger is append-only and columnar-friendly — this is a standard time-series aggregation problem. The router state is a few hundred Beta distributions in memory. Nothing here becomes hard at 10M calls a day; it becomes more valuable, because the bandit converges faster with more data.
