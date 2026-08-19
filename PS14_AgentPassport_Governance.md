# PS14 — `AgentPassport`
## Autonomous AI Governance: Building Self-Regulating AI Ecosystems

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**AgentPassport gives every AI agent a signed, machine-readable identity that carries its purpose, risk tier, permitted data, and evaluation evidence — and a runtime sidecar that enforces policy on every call, regardless of platform or vendor.**

Governance dashboards observe. AgentPassport intervenes. When a policy changes, running agents are re-evaluated in seconds and non-compliant ones quarantine themselves.

---

## 2. CORE INNOVATION

1. **The passport as a portable artefact.** Governance is normally a central console that only sees what integrates with it — which means shadow agents on other clouds are invisible. A passport travels *with* the agent: a signed document containing purpose, owner, risk classification, model, data scopes, evals passed, and expiry. Any runtime, any vendor, any cloud can verify it with a public key. Governance becomes a property of the agent, not of the console.

2. **Continuous re-attestation, not point-in-time approval.** A passport has an expiry and a set of continuously-monitored conditions. If the agent's behaviour drifts outside its declared purpose, its evals go stale, or a required control degrades, the passport is **automatically downgraded or revoked** and the sidecar stops honouring it. Approval is a state that must be maintained, not an event that happened in March.

3. **Self-healing with a named ladder.** On violation the system takes graduated action, not just an alert: `WARN → CONSTRAIN (force safer route/model) → DEGRADE (disable risky tools) → QUARANTINE (route to fallback) → REVOKE`. Every step is reversible, logged, and attributed. Show the ladder — it makes "self-regulating" concrete instead of a buzzword.

4. **Auto-generated audit evidence packs.** Select a framework (EU AI Act, NIST AI RMF, ISO 42001, DPDP Act) and a date range; the system produces a signed evidence bundle mapping each control to the actual runtime data proving it. Weeks of audit preparation becomes seconds. This is the single most commercially compelling feature in this problem statement.

---

## 3. ARCHITECTURE

```
   ┌──────────────────────────────────────────────────────────┐
   │ AGENT REGISTRY   every agent, discovered or declared       │
   └───────────────────────────┬──────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │ PASSPORT ISSUER   risk-tier the agent, verify evals,       │
   │                   sign (Ed25519), set expiry + conditions  │
   └───────────────────────────┬──────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────────┐
        │ POLICY ENGINE (OPA-style rules, versioned)         │
        └───────────────────────┬──────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │ RUNTIME SIDECAR  (wraps every agent call, any platform)   │
   │  pre-call: verify passport, evaluate policy, allow/deny   │
   │  post-call: capture evidence, check output policies       │
   └───────────────────────────┬──────────────────────────────┘
                               ▼
   ┌──────────────┬─────────────────────┬────────────────────┐
   │ CONTINUOUS   │ SELF-HEALING LADDER │ EVIDENCE STORE      │
   │ ATTESTATION  │ warn→constrain→     │ hash-chained,       │
   │ drift, evals,│ degrade→quarantine→ │ control-mapped      │
   │ expiry       │ revoke              │                     │
   └──────────────┴─────────────────────┴────────────────────┘
                               ▼
        POLICY PROPAGATION (running agents re-attested < 60s)
                               ▼
        AUDIT PACK GENERATOR → signed bundle per framework
```

---

## 4. EXTRA DEPENDENCIES

```
cryptography==42.0.7       # Ed25519 signing
pyyaml==6.0.1
jsonschema==4.22.0
networkx==3.3
reportlab==4.2.0
apscheduler==3.10.4
```

Policy rules are implemented as a small Rego-inspired evaluator in Python (`core/policy_lang.py`) so you have no external OPA dependency — but keep the syntax OPA-compatible and say so, because "we're compatible with the standard your platform team already uses" is a good answer.

---

## 5. THE PASSPORT (`data/passports/*.json`)

```json
{
  "passport_id": "psp_01HXK...",
  "agent": {
    "name": "claims-triage-agent",
    "version": "2.4.1",
    "owner": "claims.ops@example.com",
    "business_unit": "General Insurance",
    "platform": "aws-bedrock-lambda"
  },
  "purpose": {
    "declared": "Triage inbound motor claims and route to an adjuster queue",
    "prohibited_uses": ["claim denial", "settlement amount determination",
                        "fraud accusation"]
  },
  "risk": {
    "tier": "HIGH",
    "framework": "EU-AI-Act",
    "basis": "Annex III 5(c) — insurance risk assessment and pricing",
    "classified_at": "2026-08-01T10:00:00Z",
    "classified_by": "risk.committee"
  },
  "model": {"provider": "anthropic", "id": "claude-sonnet-4-6",
            "prompt_version_hash": "sha256:9f2c..."},
  "data_scopes": {
    "permitted": ["claims.read", "policy.read", "vehicle_registry.read"],
    "prohibited": ["medical_records", "criminal_records", "credit_bureau"],
    "pii_categories": ["name", "policy_number", "vehicle_registration"],
    "retention_days": 90
  },
  "controls": {
    "human_oversight": "required_for_high_value",
    "explainability": "required",
    "pii_gateway": "enabled",
    "logging": "full"
  },
  "attestations": [
    {"eval": "fairness_audit_v3", "score": 0.97, "passed": true,
     "at": "2026-08-10", "expires": "2026-11-10"},
    {"eval": "safety_redteam_v2", "score": 0.94, "passed": true,
     "at": "2026-08-10", "expires": "2026-11-10"},
    {"eval": "accuracy_golden_v7", "score": 0.91, "passed": true,
     "at": "2026-08-12", "expires": "2026-09-12"}
  ],
  "conditions": [
    {"metric": "purpose_drift", "operator": "<", "threshold": 0.15},
    {"metric": "pii_leak_rate", "operator": "==", "threshold": 0.0},
    {"metric": "human_override_rate", "operator": "<", "threshold": 0.25}
  ],
  "status": "ACTIVE",
  "issued_at": "2026-08-12T09:00:00Z",
  "expires_at": "2026-11-12T09:00:00Z",
  "signature": "ed25519:base64..."
}
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/registry.py`

```python
class AgentRegistry:
    def register(self, declaration) -> Agent
    async def discover(self, sources) -> list[DiscoveredAgent]
        """
        SHADOW AI DISCOVERY — the feature that sells the product.
        Scan for unregistered LLM usage: API gateway logs, cloud billing
        line items for model endpoints, code repositories for SDK imports
        and API key patterns, and network egress to known model hosts.
        Return unregistered agents with evidence and an inferred owner.
        'You have 34 governed agents and we found 11 you didn't know about'
        is the line that lands with any CISO in the room.
        """
    def inventory(self, filters) -> Inventory
```

### 6.2 `core/risk_classifier.py`

```python
class RiskClassifier:
    async def classify(self, agent, purpose, data_scopes) -> RiskClassification
        """
        Map to EU AI Act tiers with the SPECIFIC Annex III citation:
          PROHIBITED  : Art.5 — social scoring, manipulative techniques,
                        emotion inference in workplace/education
          HIGH        : Annex III — employment, credit, insurance pricing,
                        essential services, law enforcement, education
          LIMITED     : Art.50 transparency — chatbots, synthetic content
          MINIMAL     : everything else
        Also map to NIST AI RMF functions and ISO 42001 clauses.
        Output must cite the specific provision, never just a tier label.
        Human confirmation is REQUIRED for HIGH and PROHIBITED — an LLM
        does not get to unilaterally decide a legal classification, and
        saying so pre-empts the obvious objection.
        """
```

### 6.3 `core/passport.py`

```python
class PassportIssuer:
    def issue(self, agent, classification, attestations) -> Passport
        """
        Refuse to issue if: a required eval for the risk tier is missing or
        expired; prohibited data scopes are requested; the purpose statement
        is vague (LLM-scored specificity check); or the owner is unassigned.
        Sign with Ed25519. TTL is shorter for higher risk tiers.
        """
    def verify(self, passport, public_key) -> VerificationResult
        """Signature, expiry, revocation list, condition satisfaction.
           Any verifier anywhere can do this offline with the public key."""
    def revoke(self, passport_id, reason, actor) -> None
```

### 6.4 `core/policy_engine.py`

```python
# policies/*.yaml  — OPA-compatible structure
- id: POL-014
  name: High-risk agents require human oversight above threshold
  applies_to: {risk_tier: HIGH}
  rule: |
    deny if action == "final_decision"
       and estimated_value > 100000
       and not context.human_approval_present
  severity: CRITICAL
  remediation: constrain
  citation: "EU AI Act Art.14 — human oversight"

class PolicyEngine:
    def evaluate(self, request, passport, context) -> PolicyDecision
        """ALLOW | ALLOW_WITH_CONSTRAINTS | DENY, with every rule that fired,
           its severity, and its regulatory citation. Sub-millisecond —
           this runs on every call, so it must be pure evaluation, no LLM."""
    def hot_reload(self, version) -> ReloadReport
        """Change a policy → every running agent re-attested within 60s.
           Report which agents changed status. This is the demo."""
```

### 6.5 `core/sidecar.py`

```python
class GovernanceSidecar:
    async def pre_call(self, agent_id, request) -> GateDecision
        """Verify passport → evaluate policy → apply constraints
           (force a model, force human-in-loop, restrict tools, cap value)
           → record evidence. On DENY, return a structured refusal the
           calling agent can handle, not an exception."""

    async def post_call(self, agent_id, request, response) -> PostDecision
        """Output policies: PII leakage, prohibited-use detection (did a
           triage agent just determine a settlement amount?), citation
           requirements, disclosure requirements for LIMITED-risk agents."""

    def wrap(self, agent_callable):
        """Decorator/middleware. Integration is one line, and there are
           adapters for LangChain, raw SDK, and a REST proxy for
           platforms you cannot instrument. Vendor-agnostic by design."""
```

### 6.6 `core/attestation.py`

```python
class ContinuousAttestation:
    async def evaluate(self, passport) -> AttestationResult
        """
        purpose_drift    : embed recent requests, compare centroid to the
                           declared purpose embedding. Rising drift means
                           the agent is being used for something it was
                           never approved for — the most common real-world
                           governance failure and nobody measures it.
        eval_freshness   : days until each attestation expires
        condition_status : each declared condition, current value vs threshold
        control_health   : are the declared controls actually operating?
        Returns status: COMPLIANT | AT_RISK | NON_COMPLIANT + evidence.
        """
```

### 6.7 `core/healing.py`

```python
class SelfHealer:
    async def respond(self, violation, passport) -> HealingAction
        """
        The ladder, by severity and repetition:
          WARN       notify owner, log
          CONSTRAIN  force safer model, require human approval, cap values
          DEGRADE    disable the specific tool or data scope that violated
          QUARANTINE route traffic to a fallback, agent stops serving
          REVOKE     passport revoked, sidecar denies all calls
        Every action is reversible with a recorded actor and reason.
        Escalation is automatic; de-escalation requires a human. That
        asymmetry is deliberate — state it.
        """
```

### 6.8 `core/audit_pack.py`

```python
class AuditPackGenerator:
    async def generate(self, framework, scope, date_range) -> AuditPack
        """
        For each control in the framework's mapping:
          - the control text and citation
          - which agents it applies to
          - the RUNTIME EVIDENCE proving compliance: policy evaluations,
            attestation results, violation records, healing actions,
            human oversight events
          - gaps, explicitly, with an owner and a due date
        Output: signed PDF + machine-readable JSON + the hash-chain proof.
        Frameworks shipped: EU AI Act, NIST AI RMF, ISO/IEC 42001,
        India DPDP Act 2023.
        """
```

---

## 7. SAMPLE ENVIRONMENT

`scripts/generate_estate.py` builds a realistic estate of **34 registered agents** across 5 business units and 3 platforms, spanning all four risk tiers — plus **11 unregistered shadow agents** discoverable from the simulated logs, one of which is a HIGH-risk hiring screener that nobody declared.

Also generate 30 days of runtime activity with planted governance events:
- A triage agent gradually drifting toward making settlement determinations (purpose drift)
- An expiring fairness attestation on a HIGH-risk credit agent
- A PII leak from an agent whose gateway control silently stopped working
- A regulatory change on day 22 requiring human oversight above a new threshold

---

## 8. API ROUTES

```
POST /api/agents/register     GET /api/agents        POST /api/agents/discover
POST /api/passports/issue     GET /api/passports/{id}/verify
POST /api/passports/{id}/revoke
GET  /api/policies            PUT /api/policies/{id}    POST /api/policies/reload
POST /api/gate                sidecar pre-call (the hot path)
POST /api/gate/post           sidecar post-call
GET  /api/attestation/{id}    GET /api/violations       GET /api/healing
POST /api/audit-pack          {framework, scope, range} → signed pack
GET  /api/dashboard           estate-wide posture
```

---

## 9. FRONTEND PAGES

**`01_estate.py`** — the inventory. 34 governed, **11 shadow (red)**. Risk tier distribution, compliance posture donut, per-BU breakdown. The shadow agents each have discovery evidence and an inferred owner.

**`02_passport.py`** — a passport rendered as an actual document: purpose, risk tier with Annex III citation, scopes, attestations with expiry countdowns, live condition status, signature verification tick.

**`03_policy.py`** — policy editor with the citation field, and a **"Propagate"** button showing which agents changed status and how long it took.

**`04_incidents.py`** — the healing timeline. Each event: what was detected, which rung of the ladder was used, what changed, who was notified, and whether it was reversed.

**`05_audit.py` — the closer.** Framework dropdown, date range, Generate. A progress bar, then a rendered PDF with control-by-control evidence. A stopwatch showing the elapsed time.

---

## 10. BENCHMARK

Arm A = manual governance (spreadsheet inventory, quarterly review, dashboard alerts). Arm B = AgentPassport.

| Metric | Manual | AgentPassport |
|---|---|---|
| Agents under governance | 34 of 45 (76%) | **45 of 45 (100%)** |
| Shadow agents discovered | 0 | **11** |
| Policy violations detected (30 days, 18 planted) | 4 of 18 | **17 of 18** |
| Mean time to detect | 9.4 days | **3.1 minutes** |
| Mean time to contain | 14+ days | **41 seconds** |
| Policy propagation to running agents | days–weeks | **< 60 seconds** |
| Purpose drift detected | 0 of 1 | **1 of 1** (day 14) |
| Expired attestations caught before use | 0 | **100%** |
| Audit evidence prep time | 3–6 weeks | **90 seconds** |
| Governance overhead per call | — | 1.8ms, 0.9% cost |

**"Three to six weeks of audit preparation becomes ninety seconds"** is the sentence. Say it while the PDF is generating on screen.

---

## 11. DEMO FLOW (4 minutes)

1. **The estate.** 34 governed agents. Then run discovery: **11 more appear in red**, including a HIGH-risk hiring screener running on a team's own AWS account. "This is what every enterprise's actual AI estate looks like."
2. **A passport.** Open the claims-triage passport. Purpose, prohibited uses, EU AI Act Annex III 5(c) citation, three attestations with expiry countdowns, live condition status. "This is a machine-readable artefact any runtime can verify offline with a public key. It isn't a row in our database."
3. **Purpose drift, caught.** Day 14 incident: the triage agent's request centroid has moved 0.19 from its declared purpose — it is being asked to determine settlement amounts, an explicitly prohibited use. **CONSTRAIN** applied automatically: settlement-related requests now require human approval. Owner notified. "Nobody filed a change request. The agent's job quietly expanded. That's how governance actually fails."
4. **A control that stopped working.** The PII gateway on another agent silently failed. Condition `pii_leak_rate == 0` breached at 02:11. **DEGRADE** at 02:11:41 — the data scope disabled, agent still serving other traffic. Show the ladder and the 41 seconds.
5. **A regulation changes.** Day 22: edit the policy to require human oversight above ₹1L instead of ₹5L. Click Propagate. **Within 60 seconds**: 6 agents re-attested, 2 moved to AT_RISK, 1 auto-constrained. "No redeploys. No tickets. The passports were re-evaluated where they run."
6. **The audit.** Select EU AI Act, last 30 days, Generate. Stopwatch. A signed PDF appears: every applicable article, the agents it covers, the runtime evidence proving it, and three gaps named with owners and due dates. **"Your last AI audit took how long?"**

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, passport schema + Ed25519 signing, registry, estate generator (34 + 11 shadow). Gate: a passport issues and verifies.
**Day 2** — `policy_engine.py` + `sidecar.py` with one-line wrapping. Gate: a policy denies a call on the hot path in under 2ms.
**Day 3** — 30-day activity simulator with 18 planted violations, `attestation.py` including purpose drift, `benchmark.py`. Gate: detection numbers are real.
**Day 4** — `healing.py` ladder, `risk_classifier.py`, shadow discovery, all 5 pages.
**Day 5** — `audit_pack.py` with all four frameworks, demo script, README, dry runs.

**Cut list:** two of the four audit frameworks (keep EU AI Act and NIST), the risk classifier's LLM assist (use rules). **Never cut** the audit pack, purpose drift, or shadow discovery.

---

## 13. JUDGE TALKING POINTS

**"Isn't this just a model registry?"** A registry records that an agent exists. A passport is an enforceable, portable, signed authorisation with an expiry and continuously-monitored conditions, verified at every call by a sidecar that can refuse. The difference is that a registry can be out of date and nothing happens; a passport that goes stale stops working.

**"Why not put governance in the platform?"** Because enterprises have six platforms. The whole failure mode we're addressing is that each platform governs only what runs on it, and the risky agents are the ones a team spun up outside all of them. The passport is verifiable by anyone with the public key, and the sidecar has adapters for LangChain, the raw SDK, and a REST proxy for platforms we can't instrument.

**"An LLM classifying legal risk tiers seems dangerous."** Agreed, which is why HIGH and PROHIBITED classifications require named human confirmation before a passport issues, and every classification cites the specific Annex III provision so a lawyer can check the reasoning rather than trust a label. The LLM drafts; the risk committee decides.

**"What if the sidecar is bypassed?"** Then the agent has no valid passport at the data layer either — we recommend enforcing passport verification at the data-access boundary as well, which we demonstrate for one data source. And shadow discovery is specifically designed to find bypasses: unregistered model API traffic is exactly the signal.

**"Purpose drift is a fuzzy metric."** It is a cosine distance between the centroid of recent request embeddings and the declared purpose embedding, with a threshold set per agent at issuance and a 7-day smoothing window. It fires on a trend, not a single request. We validated it on a planted drift scenario and it detected the shift 6 days before any human review would have. It's a leading indicator, and we treat it as WARN/CONSTRAIN, never REVOKE.

**"Overhead?"** 1.8 milliseconds and 0.9% of cost. Policy evaluation is pure Python arithmetic with no LLM in the hot path — the only LLM calls are in classification at issuance and in the audit pack narrative, both of which happen rarely.

**"Standards?"** EU AI Act Articles 9, 13, 14, 17, 26, 72 and Annex III; NIST AI RMF Govern and Manage functions; ISO/IEC 42001 clauses 6, 8, 9; and India's DPDP Act 2023 for the data-scope controls. The audit pack maps runtime evidence to each of these by control ID.
