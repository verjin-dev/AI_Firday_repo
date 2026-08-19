# AI Friday National Finals — Blueprint Index & Selection Guide

## How to use these

1. Pick **one** problem statement.
2. Give Claude Code `00_COMMON_FOUNDATION.md` **plus** that one `PSxx_*.md` file. Nothing else.
3. Tell it: *"Read both documents fully before writing code. Build the Day 1 gate first."*

Do not hand Claude Code all sixteen files — it will lose focus. The foundation doc holds everything shared (scaffold, LLM client, error handling, benchmark contract, demo safety rules) so each blueprint stays lean and specific.

---

## The 15 blueprints

| # | Project | Problem Statement | Core innovation | Demo strength |
|---|---|---|---|---|
| 01 | **ClaimLedger** | Hallucinations | Cost-weighted selective abstention, claim-level verification | High |
| 02 | **ContextBridge** | Context window | Fact Ledger (lossless facts, compressed prose) + section planner | High |
| 03 | **AegisBroker** | Prompt injection | Capability-based tool authority; injection can't escalate | **Very high** |
| 04 | **DriftLens** | Monitoring | Drift-to-eval loop; 4-signal diagnostic table | Medium-high |
| 05 | **ConsensusGate** | Consistency | Decision/prose split + automatic canonicalisation | High |
| 06 | **AgentSpec** | Testing framework | Metamorphic amplification (12 seeds → 487 tests) + trajectory diff | High |
| 07 | **VeilGateway** | Privacy | Format-preserving pseudonymisation + crypto-shredding | **Very high** |
| 08 | **FairLens** | Bias & fairness | Counterfactual auditing + bias budget CI gate + attribution | High |
| 09 | **GlassBox** | Explainability | Faithfulness verification by causal ablation | **Very high** |
| 10 | **PulseRAG** | Freshness | Bi-temporal facts + retroactive answer invalidation | High |
| 11 | **AccessBridge** | Accessibility | Runtime repair + source patch; context-grounded alt text | **Very high** |
| 12 | **ContextOS** | Enterprise context | Typed context contracts + 4-tier memory + SOPs as step graphs | Medium-high |
| 13 | **TokenOps** | FinOps | Cost per outcome + learning router + burn-rate alerting | High |
| 14 | **AgentPassport** | Governance | Portable signed passports + self-healing ladder + audit packs | High |
| 15 | **NexusGraph** | Knowledge unification | Per-edge lineage + hybrid SQL/graph/vector planner | High |

---

## If you want the highest expected score

**Top three by score-per-effort:**

**11 — AccessBridge.** The only demo in the set with an emotional beat: a screen reader failing on a broken page, then succeeding. Judges remember experiences over charts. Build complexity is moderate (Playwright + axe + Claude vision), the business case is regulatory and unarguable, and almost no competing team will pick accessibility.

**03 — AegisBroker.** Strongest patentability story (capability-based security applied to LLM tool calls is a genuinely novel construction), cleanest binary demo — the attack works, then it doesn't — and a security budget exists in every enterprise. Risk: needs a working agent *and* a security layer, so scope discipline matters.

**09 — GlassBox.** Has the best single line in the whole set: *"43% of self-generated explanations cite a factor that provably didn't drive the decision."* That's a research finding, not a feature, and it reframes the entire XAI category. Moderate build, very strong Innovation and Research marks.

**Honourable mention — 07 VeilGateway**, purely for the crypto-shredding moment: forty-one milliseconds to satisfy a GDPR erasure request against a RAG index, verified live. That is a "wait, do that again" moment.

---

## Which to avoid

**01 (Hallucinations), 04 (Monitoring), and 08 (Bias)** will be the three most-picked statements in the room. The baseline expectation is highest and the marginal impression is lowest. Only take one of these if you're confident you can execute its specific innovation — cost-weighted abstention, the four-signal diagnostic table, or bias attribution respectively — because without those they collapse into projects the judges will have seen four times before lunch.

**12 (ContextOS)** is the most intellectually interesting and the hardest to demo — the payoff is an absence (nothing leaked, nothing went wrong), which is difficult to make visible in four minutes.

---

## What actually decides the score

The rubric puts 20% on a working prototype and 15% on business impact, but the quiet killers are elsewhere:

- **Research (7%)** — cite real standards. Every blueprint names the specific ones: OWASP LLM Top 10, NIST AI RMF, EU AI Act articles, WCAG 2.2, ISO/IEC 42001, GDPR articles. Two or three real citations move this from Fair to Outstanding.
- **Scope (5%)** — write down what you are *not* solving. Free marks that almost every team leaves on the table.
- **Risk Assessment (4%)** — an assumptions-and-failure-modes table. Also nearly free.
- **Key Metrics (4%)** — every benchmark table in these blueprints has a baseline *and* a target. Never present a target without its baseline.
- **Day 3** — the benchmark day. It's the day teams skip when they're behind, and skipping it silently costs about 16% across Research, Key Metrics, and Business Value. Protect it.

One more thing worth doing regardless of which you pick: report **one metric where your system loses** to the baseline. A clean sweep reads as rigged; one honest loss makes the rest of the table believable.
