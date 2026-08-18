# TokenOps

**FinOps for agentic AI: cost per business outcome, a router that learns, and burn-rate alerting that catches a runaway agent in two minutes instead of six hours.**

Cost per token is a vanity metric. *"₹6.24 per resolved support ticket, down from ₹17.28, at 0.874 quality against 0.892"* is a sentence a CFO can act on, and every part of this system exists to produce it — and to make sure a looping agent never gets to write a different one. On a 30-day simulated multi-agent support estate (36,862 sessions, 326,992 metered calls, three tenants, four agents), TokenOps cuts cost per resolved ticket by **63.9%** for **1.8 quality points**, and contains a planted agent-loop incident for **₹1,562** where the unmanaged arm spends **₹2.70 L**.

---

## 1. The problem, in one paragraph

An enterprise agent platform makes millions of LLM calls a month across dozens of workflows. The bill arrives as one number. Nobody can say what a resolved ticket costs, which agent step is eating the budget, how much of the spend is waste, or that a retry loop started at 02:14 and will burn 40% of the monthly budget before anyone wakes up. Existing cloud FinOps tooling does not help, because LLM cost is controllable at *request* time — you can pick a cheaper model, compress the prompt, or serve from cache, per call, based on the task. That control surface is the opportunity, and nothing exploits it.

## 2. Architecture

```
   Agents / apps ──[outcome-tagged traces]──▶ COST LEDGER (one row per call)
                                                     │
     ┌───────────────────────────────────────────────┤
     ▼                    ▼                          ▼
 UNIT ECONOMICS      BURN-RATE ALERTS         ATTRIBUTION
 cost per outcome    multi-window SLO         tenant│team│agent│workflow│step
 (failures in the    1h@14.4x 6h@6x 24h@3x    + week-over-week delta
  numerator only)    + 5-min short window
     │                    │                          │
     └────────────────────┴──────────┬───────────────┘
                                     ▼
              ┌─────────────────────────────────────────┐
              │ OPTIMISER STACK                          │
              │ semantic cache (TTL, quality-gated)      │
              │ tiered context compression               │
              │ model cascade (break-even aware)         │
              │ off-peak scheduler                       │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │ LEARNING ROUTER — Thompson sampling      │
              │ reward = quality − λ·normalised cost     │
              │ quality floor │ exploration budget cap   │
              │ warm-start priors │ full explain() table │
              └──────────────────┬──────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────┐
              │ GUARDRAILS  ALLOW│DEGRADE│QUEUE│BLOCK    │
              │ loop detection · circuit breaker         │
              └──────────────────┬──────────────────────┘
                                 ▼
                 FORECASTER → budget, drivers, what-if, showback
```

## 3. Quick start

```bash
pip install -r requirements.txt
python scripts/simulate_workload.py     # 30 days of demand, both arms (~5 min)
python scripts/benchmark.py             # writes benchmark_results.json + chart
streamlit run frontend/app.py           # the judge-facing UI
```

Optional: `uvicorn backend.main:app --port 8013` for the REST API (`/docs`), and `python scripts/demo_flow.py --offline` for the scripted narration.

**No API key is required.** TokenOps meters LLM spend; it does not generate it. The whole system runs offline on locally generated telemetry, which is also why the demo cannot fail on conference Wi-Fi.

## 4. Demo walkthrough (4 minutes)

| # | Click | What the judge sees |
|---|---|---|
| 1 | **Economics** | ₹17.28 per resolved ticket, baseline. Flip *Baseline mode* in the sidebar: ₹6.24. Same demand, different control plane. |
| 2 | **Attribution** → sunburst → `qa_agent` → `qa_verify` | 26% of spend on verification; 11,657 of those calls re-verified answers already scored ≥ 0.90. Nobody designed that. It accreted. |
| 3 | **Waste** | ₹3.12 L/month itemised, each with an owner. Toggle fixes to see what is recoverable. |
| 4 | **Router** | Traffic-share chart: the bandit finds haiku for triage (91%, stable from day 7) and keeps sonnet for generation (99%). Open the per-arm table: pulls, quality, cost, posterior interval, P(best), and which arms the quality floor excluded. |
| 5 | **Burn & incidents** | Day 18, 02:14. The two spend curves separate. Loop killed at 02:16 for ₹1,562; unmanaged, it runs to 08:00 for ₹2.70 L. |
| 6 | **Burn → degradation slider** | Set a tenant to 20% budget: context halved, verification skipped, cheap route forced. The workflow completes. |
| 7 | **Forecast** | ₹8.64 L next month ± interval, MAPE 4.5%, decomposed: 9% of growth is volume, 82% is unit cost — *that* is a regression with an owner. |

## 5. Headline metrics

Measured, not asserted: every figure is computed by `scripts/benchmark.py` from the ledger. Headline unit economics **exclude the planted agent-loop incident in both arms** — folding a one-off outage into the headline would flatter TokenOps by ~40 points and tell you nothing about a normal Tuesday. The incident is reported on its own line.

| Metric | Baseline (Arm A) | TokenOps (Arm B) | Delta |
|---|---|---|---|
| **Cost per resolved ticket** | ₹17.28 | **₹6.24** | **−63.9%** |
| Monthly spend (steady state) | ₹5.29 L | ₹1.93 L | −63.6% |
| **Mean outcome quality** | 0.892 | **0.874** | **−0.018** ← we lose here |
| Resolution rate | 94.5% | 94.5% | ±0.0 pp |
| Cache hit rate (of cacheable lookups) | 0% | 20.7% | — |
| Cascade escalation rate | n/a | 7.0% | break-even 73% |
| Median context tokens per call | 10,367 | 5,232 | −49.5% |
| p95 latency | 2,982 ms | 2,567 ms | −13.9% |
| **Day-18 loop incident cost** | **₹2.70 L** | **₹1,562** | **−99.4%** |
| Day-18 containment | 346 min | **2 min** | — |
| Day-11 retry-storm cost | ₹1,926 | ₹298 | −85% |
| Waste named and owned | ₹3.12 L/mo | ₹12,810/mo | — |
| Forecast accuracy (MAPE, post-break) | — | 4.5% | — |
| TokenOps overhead | — | 1.26% of managed spend | — |

**Two numbers for the slide: a 64% cost reduction for 1.8 quality points, and an incident contained in 2 minutes instead of 346.** The second is what actually frightens an enterprise buyer.

### Where TokenOps loses

Mean outcome quality falls 0.892 → 0.874. That is the designed trade: the router's reward is `quality − λ·cost` with λ = 0.4, and it will accept a small quality loss for a large cost one. λ is a dial, not a mystery — turn it down and you buy quality back at a known price. The hard quality floor (0.80) is what stops the trade going too far, and it is enforced by *excluding arms before sampling*, not by hoping.

## 6. Key technical decisions

**Failed outcomes count in the numerator, never the denominator.** `cost_per_outcome = Σ all spend / count(successful outcomes)`. If you divide by attempts, a system that fails cheaply looks efficient. This one decision changes what the optimiser optimises.

**A cache hit is not free.** Cache reads are billed at the discounted cache-read tier and recorded that way. Reporting cache hits as zero-cost is the most common way an LLM cost dashboard lies to its owner.

**Only some steps are cacheable.** Triage and retrieval depend on the question; resolution and QA read the customer's own words. Caching the latter is how a cache starts returning another customer's answer. That constraint is why the honest hit rate is 20.7% and not the 90%+ a naive implementation would report.

**The router is a bandit, not a table.** A static routing table is one engineer's beliefs on one Tuesday, and it rots. Thompson sampling over 10 arms per task type, with three constraints that make it deployable: a quality floor applied *before* sampling, an exploration budget capped at 5% of period spend, and weak warm-start priors so day one is sensible rather than random. `explain()` returns every arm's pulls, quality, cost, posterior interval and selection probability — an operations team will not accept a router it cannot audit.

**Multi-window burn alerting, with short windows.** Each long window (1h/6h/24h at 14.4×/6×/3×) is paired with a short window one twelfth its length; both must breach. The long window keeps it quiet; the short window makes it fast. This is the Google SRE multiwindow burn-rate rule applied to money instead of errors.

**Two independent incident mechanisms.** The loop detector matches a repeated `(step, prompt_hash)` signature inside a session — fast, because it needs no budget knowledge. The burn-rate monitor catches spend anomalies with no signature at all. The demo shows both, because the next incident will not be a loop.

**Degrade, don't block.** At <30% budget remaining the guardrail forces the cheap route, halves context, skips optional verification and relaxes the cache threshold. Blocking a workflow to save money is a support ticket; degrading it is a saving.

**Structural breaks are found, not averaged away.** The forecaster detects the day-23 prompt deploy as a level shift and reports accuracy on the current regime, because no forecaster can see a deploy coming and pretending otherwise produces a MAPE that means nothing.

## 7. Scope — what this does *not* solve

Stated deliberately; these are non-goals, not gaps we ran out of time for.

- **It does not generate LLM traffic.** TokenOps is a control plane. It meters, routes and constrains calls made by other systems.
- **It does not judge quality from scratch.** Quality scores arrive from the caller's evaluator (reference-free groundedness, schema validity, task-completion signals, human accept/reject). We sample 15% for judging and charge that cost to ourselves — it is 1.26% of managed spend.
- **No multi-cloud or multi-vendor cost normalisation.** One provider's price table, versioned in one file.
- **No prompt engineering.** The compressor removes duplication, boilerplate and never-cited retrieval chunks. It does not rewrite intent.
- **No real-time streaming ingestion.** The ledger is append-only and batch-friendly; a 30-second lag on the dashboards is acceptable and assumed.
- **The semantic cache uses a hashed bag-of-words embedding offline** so the demo needs no model download. The interface is pluggable; production would use sentence-transformers or the provider's embeddings.
- **The workload is simulated**, with parameters chosen to be defensible rather than flattering. Every number in §5 is computed from that simulation, and the simulator is in the repo for inspection.

## 8. Risk assessment

| Assumption | If it is wrong | Failure mode | Mitigation in the build |
|---|---|---|---|
| Quality scores are trustworthy enough to route on | The bandit optimises toward a bad proxy | Cost falls, quality quietly falls with it | Hard quality floor applied before sampling; 15% judge sampling; report judge–human agreement alongside every quality figure |
| Task types are stable | Route learning applies to shifting workloads | Router converges on a policy for traffic that no longer exists | Posterior updates continuously; exploration budget resets each period so the router can re-learn |
| A cache hit is semantically safe | Near-duplicate ≠ same answer | Wrong answer served confidently | 0.94 similarity threshold, per-tenant scoping, 24h TTL, quality gate on stored results |
| Price table is current | Costs are computed from stale prices | Every rupee figure is wrong | Versioned price table (`PRICE_TABLE_VERSION`), cost frozen at record time so history stays accurate |
| Loop signatures are detectable | Non-identical loops slip through | A slow-burn incident runs longer | Burn-rate alerting is the independent safety net; the demo reports both detection paths |
| Budgets are set sensibly | Alerts are noise or silence | Alert fatigue, or no alerts at all | Budget derived from prior-period baseline spend + 10% headroom, excluding incident days |

## 9. Scalability

The ledger is append-only and columnar-friendly: every analytic here is a time-series aggregation with no joins on the hot path, which is a solved problem at any volume (partition by day, roll up hourly). The router's entire state is a few hundred Beta distributions — a few kilobytes, and updates are arithmetic. The cache is one vector lookup per call. Metering is a database write. Nothing in TokenOps becomes hard at 10M calls a day; the bandit becomes *better*, because it converges faster with more data. Measured overhead is 1.26% of managed spend, and the only model call it makes is the 15% quality-judge sample.

## 10. Research grounding

- **FinOps Foundation Framework** — showback/chargeback, unit economics, and the Inform → Optimise → Operate lifecycle this project is organised around.
- **Google SRE Workbook, "Alerting on SLOs"** — multiwindow, multi-burn-rate alerting; the 14.4×/6×/3× thresholds and the short-window pairing are taken directly and applied to a cost budget instead of an error budget.
- **NIST AI RMF 1.0** — MEASURE (2.11, 2.13) and MANAGE (2.2, 4.1) functions: continuous measurement of AI system cost and performance, and documented response to degradation.
- **ISO/IEC 42001:2023** — AI management-system controls for operational monitoring and resource management (Annex A performance and resource clauses).
- **EU AI Act, Article 12 (record-keeping) and Article 72 (post-market monitoring)** — the ledger is the automatically generated event log those articles require; attribution and alerting are the monitoring plan.
- **Thompson, W.R. (1933)**, and Chapelle & Li (2011) on the empirical evaluation of Thompson sampling — the router's algorithm and its regret properties.
- **Holt (1957) / Winters (1960)** triple exponential smoothing — the forecaster, implemented directly so the project carries no heavyweight statistical dependency.

## 11. Roadmap to production (90 days)

**Phase 1, days 1–30 — Instrument.**  Ship the tracing middleware as an SDK wrapper (Python + TypeScript) around the Anthropic client, so tagging is one decorator. Replace SQLite with the customer's warehouse (ClickHouse or BigQuery) behind the existing `query_df` seam. Backfill 90 days from provider invoices for a baseline. *Dependencies:* warehouse access, one platform engineer. *Exit gate:* cost per outcome computed on live traffic for one workflow.

**Phase 2, days 31–60 — Observe and alert.**  Turn on burn-rate alerting in shadow mode, tuned against the backfilled period so thresholds are evidence-based before anyone gets paged. Ship attribution and showback to finance. Loop detection runs in detect-only mode. *Dependencies:* PagerDuty/Slack integration, an owner named per waste category. *Exit gate:* two weeks with zero false pages, and one real finding accepted by an engineering owner.

**Phase 3, days 61–90 — Act.**  Enable the router on one task type at 10% of traffic behind a flag, with the quality floor wired to the customer's own evaluator. Enable guardrail DEGRADE for non-interactive workloads first. Circuit breakers armed last, per tenant, with a manual reset. *Dependencies:* a quality signal the customer trusts, change-approval for automated routing. *Exit gate:* measured cost reduction with a quality delta inside the agreed budget, and one contained incident.

**Named risks to the plan:** a customer without any quality signal cannot run the router (Phase 3 becomes cache + compression only, roughly a third of the benefit); a provider price change mid-programme invalidates the backfilled baseline and requires a re-basing; and warehouse latency above a minute weakens burn-rate detection, which is why the alerting path reads from the write-side ledger rather than the warehouse.

## 12. Repository map

```
backend/
  config.py                  price table (versioned), thresholds, budgets
  core/
    ledger.py                the cost ledger: unit economics, attribution, waste
    pricing.py               tokens → money, once, at record time
    router.py                the learning router (Thompson sampling + constraints)
    burn_rate.py             multi-window burn alerting, minute-level detection
    guardrails.py            ALLOW/DEGRADE/QUEUE/BLOCK, loop detection, breakers
    forecaster.py            Holt-Winters, driver decomposition, what-if
    optimizers/              semantic cache · compressor · cascade · scheduler
  api/routes/                health · economics · router · burn/forecast
frontend/                    Streamlit: 6 pages, baseline toggle, evidence panels
scripts/
  simulate_workload.py       30 days, both arms, 3 planted incidents
  benchmark.py               the table above + benchmark_chart.png
  demo_flow.py               the scripted 4-minute narration
tests/                       38 tests over pricing, router, burn, guardrails, …
```

Run the tests with `python -m pytest -q`.
