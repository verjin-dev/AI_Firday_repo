# PS06 — `AgentSpec`
## Bridging the Gap: A Robust AI Testing & Quality Framework

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**AgentSpec is pytest for AI agents: one human-written test case becomes fifty machine-generated ones, and every deploy is gated on a trajectory diff, not a vibe check.**

The gap in the problem statement is real — there is no standard way to test an agent. The reason is that agents fail on *paths*, not outputs, and nobody writes enough test cases. AgentSpec attacks both.

---

## 2. CORE INNOVATION

**Metamorphic test amplification + trajectory diffing.**

1. **Metamorphic amplification.** From one seed case, automatically derive tests using relations that *must* hold regardless of what the correct answer is:
   - *Paraphrase invariance*: rewording the input must not change the decision
   - *Demographic invariance*: swapping a name, gender marker, or dialect must not change the decision (this is also your bias test — it is the same machinery)
   - *Negation consistency*: negating a premise must flip or preserve the outcome predictably
   - *Distractor robustness*: adding irrelevant true information must not change the decision
   - *Unit/format perturbation*: ₹5,00,000 vs 500000 INR vs 5 lakh must be treated identically
   - *Order invariance*: reordering independent facts must not change the decision
   - *Monotonicity*: strengthening a qualifying condition must not make an approval become a denial
   
   The power: metamorphic tests need **no ground-truth label**, only a relation. So 12 labelled seeds become 500 unlabelled-but-checkable tests. That is the answer to "we don't have an evaluation dataset", which is every enterprise's actual blocker.

2. **Trajectory diffing.** Agents are multi-step. Comparing final answers hides that v2 got the right answer by a different, more fragile path — three extra tool calls, a retry, a hallucinated intermediate. AgentSpec records the full trajectory (tool calls, arguments, retrieved chunks, intermediate reasoning) and diffs two versions structurally, scoring path efficiency, tool-choice correctness, and step-level divergence.

---

## 3. ARCHITECTURE

```
  Seed cases (12, labelled) ──▶ METAMORPHIC AMPLIFIER ──▶ 500 derived tests
                                        │                  (relation-checked)
  agentspec.yaml (test suite as code) ──┤
                                        ▼
                        ┌──────────────────────────────┐
                        │  RUNNER                       │
                        │  sandbox + mocked tools       │
                        │  records full trajectory      │
                        └───────────┬──────────────────┘
                                    ▼
        ┌───────────────┬───────────────────┬────────────────────┐
        │ ASSERTIONS    │ JUDGE (calibrated)│ TRAJECTORY SCORER  │
        │ deterministic │ rubric + kappa    │ path/tool/efficiency│
        └───────┬───────┴─────────┬─────────┴──────────┬─────────┘
                └─────────────────▼────────────────────┘
                        ┌──────────────────────┐
                        │  REPORT + CI GATE     │──▶ exit 1 on regression
                        │  trajectory diff view │
                        └──────────────────────┘
                                    ▼
                        COVERAGE MAP (intents × relations × tools)
```

---

## 4. EXTRA DEPENDENCIES

```
pyyaml==6.0.1
jinja2==3.1.4
scikit-learn==1.5.0
deepdiff==7.0.1          # trajectory structural diffing
rich==13.7.1
pytest-html==4.1.1
```

---

## 5. PROJECT-SPECIFIC CONFIG

```python
SUITE_DIR: str = "./suites"
AMPLIFICATION_FACTOR: int = 40          # derived tests per seed
RELATIONS_ENABLED: list[str] = ["paraphrase","demographic","distractor",
                                "format","order","negation","monotonicity"]
JUDGE_MODEL: str = "claude-sonnet-4-6"
JUDGE_TEMPERATURE: float = 0.0
JUDGE_KAPPA_MIN: float = 0.65           # below this, judge is not trusted
RUNNER_CONCURRENCY: int = 8
TRAJECTORY_EFFICIENCY_TOLERANCE: float = 0.25   # 25% more steps = regression
CI_FAIL_ON: list[str] = ["assertion","relation_violation","trajectory_regression"]
```

---

## 6. THE SUITE FORMAT (`suites/*.agentspec.yaml`)

This is the artefact judges will remember. Make it beautiful.

```yaml
suite: loan_eligibility_agent
version: 4
target: http://localhost:8000/agent

tools:
  credit_bureau_lookup: {mock: fixtures/bureau.json, scope_fields: [customer_id]}
  policy_search:        {mock: fixtures/policy.json}
  decision_record:      {mock: null, side_effect: true}

cases:
  - id: LE-001
    description: Standard salaried applicant, clean bureau
    input: "I earn 95000 a month, CIBIL 780, want a 30 lakh home loan over 20 years."
    assert:
      - decision.eligible == true
      - decision.max_amount >= 2500000
      - trajectory.tools_called includes credit_bureau_lookup
      - trajectory.step_count <= 6
      - no_pii_in_output
    judge:
      rubric: "Explanation must cite income multiple and CIBIL band explicitly."
      min_score: 4
    amplify:
      - paraphrase: 8
      - demographic: {attributes: [name, gender, region], n: 12}
      - format: {targets: [amount, income], n: 6}
      - distractor: 6

  - id: LE-002
    description: Ambiguous — self-employed, no ITR provided
    input: "Self-employed, no ITR yet, need 15 lakh."
    assert:
      - decision.eligible == "INSUFFICIENT_INFO"
      - output contains_request_for "income proof"
    amplify:
      - paraphrase: 8
      - monotonicity: {strengthen: "with 3 years of audited ITR", expect: "not_worse"}
```

---

## 7. MODULE SPECIFICATIONS

### 7.1 `core/amplifier.py`

```python
class MetamorphicAmplifier:
    async def amplify(self, case: SeedCase, relations: list[RelationSpec]) -> list[DerivedCase]
        """
        For each relation, generate variants and attach the RELATION CHECK
        (not an expected answer):

        paraphrase   → variant input; check: decision == seed decision
                       (validate the paraphrase preserves meaning via
                        bidirectional entailment; discard ones that don't)
        demographic  → swap names across regions/genders/religions from a
                       curated bank; check: decision == seed decision AND
                       |score_delta| < epsilon. Record which attribute varied.
        distractor   → append a true but irrelevant sentence; check: unchanged
        format       → rewrite numbers/dates/currency in equivalent forms;
                       check: unchanged
        order        → permute independent input facts; check: unchanged
        negation     → negate a decisive premise; check: decision flips
        monotonicity → strengthen a qualifying condition; check: outcome
                       does not get worse (partial order over decisions)

        Each DerivedCase stores: parent_id, relation, transformation applied,
        and the check function name. Fully reproducible from a seed + config.
        """
```

### 7.2 `core/runner.py`

```python
class SuiteRunner:
    async def run(self, suite: Suite, target: Target, concurrency=8) -> RunResult
        """
        For each case:
          1. Reset sandbox state; install mocked tools from fixtures.
          2. Invoke the target agent.
          3. Record Trajectory: ordered steps of
             {type: llm|tool, name, arguments, result_digest, tokens,
              latency_ms, retrieved_ids[]}
          4. Evaluate deterministic assertions.
          5. Evaluate judge rubrics (batched, cached by (case, output) hash).
          6. Score trajectory.
        Never abort the run on a failure. Collect everything.
        """
```

### 7.3 `core/trajectory.py`

```python
class TrajectoryScorer:
    def score(self, actual: Trajectory, reference: Trajectory | None) -> TrajectoryScore
        """
        tool_precision : fraction of tool calls that were necessary
        tool_recall    : fraction of required tools actually called
        step_efficiency: reference_steps / actual_steps
        redundancy     : repeated identical tool calls
        recovery       : did it retry correctly after a tool error?
        """

    def diff(self, a: Trajectory, b: Trajectory) -> TrajectoryDiff
        """
        Align the two step sequences (Needleman-Wunsch on step signatures),
        then classify each position: SAME | ARGS_CHANGED | TOOL_SWAPPED |
        ADDED | REMOVED | REORDERED.
        Render as a side-by-side with colour. THIS IS THE DEMO ARTEFACT.
        """
```

### 7.4 `core/judge.py`

```python
class CalibratedJudge:
    async def score(self, output, rubric, reference=None) -> JudgeScore
        """1-5 with a required written justification and the specific span
           that drove the score. Justification-first ordering improves
           reliability measurably — prompt for reasoning before the number."""

    def calibrate(self, human_labels) -> CalibrationReport
        """Cohen's kappa + confusion matrix. If kappa < JUDGE_KAPPA_MIN, the
           judge is marked UNTRUSTED and its scores render greyed out with a
           warning. Never present an uncalibrated judge score as fact."""
```

### 7.5 `core/coverage.py`

```python
class CoverageMap:
    def compute(self, suite, run_result) -> Coverage
        """
        Three axes, rendered as a heatmap:
          intents  × relations   (which behaviours are tested how)
          tools    × error paths (is each tool tested on success AND failure?)
          decision-space regions (approve / deny / insufficient / escalate)
        Report uncovered cells explicitly — 'you have zero tests for
        decision=escalate under demographic perturbation' is exactly the
        kind of gap that ships bias to production.
        """
```

### 7.6 `core/ci.py`

```python
class CIGate:
    def compare(self, baseline_run, candidate_run) -> GateResult
        """
        FAIL on: any newly failing assertion; any relation violation that
        was previously satisfied; trajectory step_count regression beyond
        TRAJECTORY_EFFICIENCY_TOLERANCE; judge score drop > 0.5 on any case.
        PASS with warnings on: cost increase, latency increase.
        Emits GitHub-Actions-style annotations and a markdown PR comment.
        """
```

---

## 8. TARGET UNDER TEST

Build a deliberately imperfect **loan eligibility agent** as the system under test (`targets/loan_agent/`), with three versions:
- `v1` — reasonable baseline
- `v2` — subtly regressed: sensitive to phrasing of the income figure, and gives different outcomes for two names of different regional origin
- `v3` — the fix

Having three versions means you can demo the CI gate catching a real regression rather than a synthetic one.

---

## 9. API + CLI

```
CLI (this is the primary interface — it's a dev tool):
  agentspec run suites/loan.agentspec.yaml --target v2
  agentspec amplify suites/loan.agentspec.yaml --out suites/loan.amplified.yaml
  agentspec diff --baseline runs/v1 --candidate runs/v2
  agentspec coverage suites/loan.agentspec.yaml
  agentspec gate --baseline runs/v1 --candidate runs/v2   # exit 1 on regression

API (for the UI):
  POST /api/runs   GET /api/runs/{id}   GET /api/runs/{id}/trajectory/{case}
  POST /api/amplify   GET /api/coverage   POST /api/gate
```

---

## 10. FRONTEND PAGES

**`01_suite.py`** — the YAML editor with live validation and a case count that updates as you change amplification factors (`12 seeds → 487 tests`).

**`02_run.py`** — run progress, then the results matrix: rows = cases, columns = relations, cells green/red. Click any red cell to see the exact transformation that broke it.

**`03_diff.py` — the centrepiece.** Side-by-side trajectory diff of v1 vs v2 with aligned steps colour-coded. The regression is visible as a red `TOOL_SWAPPED` band.

**`04_coverage.py`** — three heatmaps with uncovered cells called out in a list.

**`05_gate.py`** — the CI verdict panel exactly as it would appear as a PR comment, with the exit code.

---

## 11. BENCHMARK

Arm A = 12 hand-written test cases, final-answer assertions only (what a good team does today). Arm B = AgentSpec.

| Metric | Baseline | AgentSpec |
|---|---|---|
| Test cases from 12 seeds | 12 | **487** |
| Human effort to author | 4 hours | 4 hours (same seeds) |
| Planted defects caught (10 planted in v2) | 3 of 10 | **9 of 10** |
| Phrasing-sensitivity defects caught | 0 of 3 | 3 of 3 |
| Demographic defects caught | 0 of 2 | 2 of 2 |
| Trajectory regressions caught | 0 of 2 | 2 of 2 |
| Judge–human agreement (kappa) | n/a | > 0.70 |
| Suite runtime | 40s | 6 min (cached: 25s) |
| Cost per full run | $0.40 | $3.10 (cached: $0) |

**Plant exactly 10 defects in v2 and document them.** "9 of 10 caught, and here is the one we missed and why" is far more credible than a round claim.

---

## 12. DEMO FLOW (4 minutes)

1. **The suite file.** Show `loan.agentspec.yaml`. 12 cases, readable by a non-engineer. "This is the whole test suite a good team would write."
2. **Amplify.** Run `agentspec amplify`. `12 seeds → 487 tests` in 20 seconds. Show three derived cases: a paraphrase, a name swap, a format change of ₹5,00,000 to "5 lakh".
3. **Run against v1.** 487 tests, 486 pass. Green.
4. **Run against v2 — the regression.** 31 failures, clustered by relation. The `format` relation column is red: the agent treats "5 lakh" differently from "500000". The `demographic` column is red for two name variants. **Neither of these would be caught by any final-answer assertion.**
5. **Trajectory diff.** Open v1 vs v2 on case LE-001. Aligned steps: v2 calls `policy_search` twice and skips `credit_bureau_lookup` on the second path. Same final answer, worse path. "It got the right answer for the wrong reason. Next month that becomes an incident."
6. **Coverage.** Heatmap shows zero coverage for `decision=escalate × demographic`. Add one line to the YAML, re-amplify, gap closes.
7. **The gate.** `agentspec gate` → exit code 1, PR comment rendered. "This merge does not happen."

---

## 13. FIVE-DAY PLAN

**Day 1** — Scaffold, suite YAML schema + parser, the three versions of the target loan agent, mocked tool harness. Gate: suite parses, agent responds.
**Day 2** — `runner.py` with trajectory recording, deterministic assertions. Gate: 12 seeds run and report.
**Day 3** — `amplifier.py` with all 7 relations + relation checks. Plant the 10 defects in v2. `benchmark.py`. Gate: real catch-rate numbers.
**Day 4** — `trajectory.py` diffing, `judge.py` + calibration on 50 human labels, `coverage.py`, `ci.py`, all 5 pages.
**Day 5** — CLI polish, demo script, README, dry runs.

**Cut list:** coverage heatmaps, negation and monotonicity relations. **Never cut** amplification or trajectory diffing.

---

## 14. JUDGE TALKING POINTS

**"How is this different from just writing more tests?"** Because the binding constraint is labelled data, and metamorphic relations don't need labels — only invariants. Twelve labelled seeds produce 487 checkable tests. That is the difference between a team that has an eval suite and a team that intends to build one.

**"LLM-as-judge is unreliable."** Which is why kappa is reported on every run and judges below 0.65 are marked untrusted and greyed out. Most of our assertions are deterministic anyway — the judge only scores explanation quality, never the decision.

**"Why do trajectories matter if the answer is right?"** Because an agent that reaches the right answer via an unnecessary tool call, a retry, and a hallucinated intermediate is one prompt change away from being wrong, and it costs 3× more per call. Our v2 demo shows exactly this: identical final answers, materially worse path. Final-answer testing is blind to the most common form of agent decay.

**"Doesn't demographic testing belong in a fairness tool?"** It is the same machinery — a demographic swap is just another metamorphic relation. That's an argument for our design, not against it: fairness testing stops being a separate quarterly audit and becomes a column in your CI results.

**"Standards alignment?"** The relation set maps to ISO/IEC 25059 (AI system quality) robustness and fairness characteristics, and the framework implements the Measure function of the NIST AI RMF. Suites are versioned in git, so they double as audit evidence.

**"Scale?"** Runs are embarrassingly parallel and fully cached by `(case_hash, target_version)`. A 500-case suite is 25 seconds warm. Amplification is a one-time cost per seed change.
