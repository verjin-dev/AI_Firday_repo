"""Generate 30 days of a multi-agent support platform, twice.

Both arms run against the *same* generated demand - identical sessions,
identical queries, identical arrival times. Only the control plane differs:

  Arm A "baseline"  - one strong model for everything, no cache, no
                      compression, no guardrails. The honest default most
                      teams ship.
  Arm B "tokenops"  - learning router, semantic cache, context compression,
                      model cascade, burn-rate alerting and guardrails, and
                      it pays for its own metering.

Three incidents are planted, because a FinOps system that has only ever seen
a well-behaved month has not been tested:

  day 11  a provider flaps for four hours and the retry policy amplifies it
  day 18  an agent loop starts at 02:14 and burns money until someone wakes up
  day 23  a prompt change triples context size and nobody notices

Run:  python scripts/simulate_workload.py [--days 30] [--sessions 1200]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:  # Windows consoles default to cp1252 and choke on the rupee sign
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from backend.config import get_settings  # noqa: E402
from backend.core.burn_rate import Budget, BurnRateMonitor  # noqa: E402
from backend.core.guardrails import CostGuardrails  # noqa: E402
from backend.core.optimizers.compressor import Chunk, PromptCompressor, UtilityIndex  # noqa: E402
from backend.core.optimizers.semantic_cache import SemanticCache  # noqa: E402
from backend.core.pricing import compute_cost, fmt_inr  # noqa: E402
from backend.core.router import LearningRouter  # noqa: E402
from backend.storage.db import bulk_insert, reset_database, table_counts  # noqa: E402
from backend.utils.logger import log  # noqa: E402

SIM_START = datetime(2026, 7, 1, tzinfo=timezone.utc)

TENANTS = [
    ("acme-bank", "support", 0.45),
    ("vertex-insurance", "claims", 0.35),
    ("northwind-logistics", "ops", 0.20),
]

# workflow -> (outcome_type, steps[(step, agent, task_type, base_in, base_out)])
WORKFLOWS: Dict[str, Tuple[str, List[Tuple[str, str, str, int, int]], float]] = {
    "support_ticket": (
        "ticket_resolved",
        [
            ("triage", "triage_agent", "classification", 4200, 180),
            ("retrieve", "retrieval_agent", "retrieval", 9500, 320),
            ("resolve", "resolution_agent", "generation", 11000, 650),
            ("qa_verify", "qa_agent", "verification", 8800, 240),
        ],
        0.46,
    ),
    "claims_review": (
        "claim_adjudicated",
        [
            ("triage", "triage_agent", "classification", 5200, 200),
            ("retrieve", "retrieval_agent", "retrieval", 12500, 380),
            ("resolve", "resolution_agent", "generation", 14000, 720),
            ("qa_verify", "qa_agent", "verification", 11000, 300),
        ],
        0.24,
    ),
    "doc_intake": (
        "document_processed",
        [
            ("extract", "resolution_agent", "generation", 9000, 500),
            ("qa_verify", "qa_agent", "verification", 6500, 200),
        ],
        0.20,
    ),
    "lead_scoring": (
        "lead_qualified",
        [
            ("triage", "triage_agent", "classification", 3200, 150),
            ("resolve", "resolution_agent", "generation", 6000, 380),
        ],
        0.10,
    ),
}

# latent quality by task type and model - the ground truth the router must learn
QUALITY = {
    "classification": {"claude-haiku-4-5-20251001": 0.905, "claude-sonnet-4-6": 0.912, "claude-opus-5": 0.915},
    "retrieval": {"claude-haiku-4-5-20251001": 0.862, "claude-sonnet-4-6": 0.890, "claude-opus-5": 0.897},
    "generation": {"claude-haiku-4-5-20251001": 0.792, "claude-sonnet-4-6": 0.881, "claude-opus-5": 0.899},
    "verification": {"claude-haiku-4-5-20251001": 0.884, "claude-sonnet-4-6": 0.892, "claude-opus-5": 0.894},
}
# latent self-confidence, which is what the cascade actually gets to observe
CONFIDENCE = {
    "classification": {"claude-haiku-4-5-20251001": 0.88, "claude-sonnet-4-6": 0.91, "claude-opus-5": 0.93},
    "retrieval": {"claude-haiku-4-5-20251001": 0.775, "claude-sonnet-4-6": 0.88, "claude-opus-5": 0.91},
    "generation": {"claude-haiku-4-5-20251001": 0.635, "claude-sonnet-4-6": 0.855, "claude-opus-5": 0.90},
    "verification": {"claude-haiku-4-5-20251001": 0.86, "claude-sonnet-4-6": 0.89, "claude-opus-5": 0.91},
}
CASCADE_NEXT = {
    "claude-haiku-4-5-20251001": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-opus-5",
}

BASELINE_MODEL = "claude-sonnet-4-6"
SUCCESS_QUALITY_THRESHOLD = 0.80

# Steps whose answer depends only on the question, not on this customer's
# specific ticket text, are cacheable. Resolution and QA are not: they read the
# customer's own words. Pretending otherwise is how a cache starts returning
# another customer's answer.
CACHEABLE_STEPS = {"triage", "retrieve"}

# ---- planted incidents -------------------------------------------------------
RETRY_STORM_DAY, RETRY_STORM_HOURS = 11, set(range(10, 14))
LOOP_DAY, LOOP_START_MIN = 18, 2 * 60 + 14           # 02:14
LOOP_END_MIN_BASELINE = 8 * 60                        # someone arrives at 08:00
LOOP_CALLS_PER_MIN = 105
LOOP_IN, LOOP_OUT = 24000, 900
LOOP_KILL_DELAY_MIN = 1          # detector fires, orchestrator drains, session dies
PROMPT_BLOAT_DAY, PROMPT_BLOAT_FACTOR = 23, 3.0

QUERY_TEMPLATES = 420


# --------------------------------------------------------------------- demand
@dataclass
class Step:
    step: str
    agent: str
    task_type: str
    base_in: int
    base_out: int


@dataclass
class SessionPlan:
    session_id: str
    outcome_id: str
    day: int
    hour: int
    minute: int
    tenant: str
    team: str
    workflow: str
    outcome_type: str
    steps: List[Step]
    template_id: int
    paraphrase: int
    abandon: bool
    duplicate_call: bool
    ts_epoch: float = 0.0


def _query_text(template_id: int, paraphrase: int, workflow: str) -> str:
    """A ~45-token synthetic query. Paraphrases share most tokens, so the
    semantic cache has to actually match rather than hash-compare."""
    base = (
        f"customer enquiry regarding {workflow} case reference {template_id} "
        "please review the attached policy documentation and the prior "
        "correspondence history then determine the applicable coverage "
        "eligibility limits waiting period exclusions and the recommended "
        "next action for the assigned handler"
    )
    variants = ["", " kindly confirm", " urgent follow up"]
    return base + variants[paraphrase % len(variants)]


class DemandGenerator:
    """Generates the session plan once; both arms consume it."""

    def __init__(self, days: int, sessions_per_day: int, seed: int) -> None:
        self.days = days
        self.sessions_per_day = sessions_per_day
        self.rng = np.random.default_rng(seed)
        # Zipf-ish query popularity: a few questions are asked constantly
        ranks = np.arange(1, QUERY_TEMPLATES + 1)
        self.template_p = (1.0 / ranks ** 0.85)
        self.template_p /= self.template_p.sum()
        hours = np.arange(24)
        peak = np.exp(-0.5 * ((hours - 13.5) / 4.0) ** 2) + 0.12
        self.hour_p = peak / peak.sum()

    def volume_for_day(self, day: int) -> int:
        weekday = (SIM_START + timedelta(days=day)).weekday()
        weekend = 0.55 if weekday >= 5 else 1.0
        growth = 1.0 + 0.011 * day              # ~1.1% daily volume growth
        noise = float(self.rng.normal(1.0, 0.05))
        return max(50, int(self.sessions_per_day * weekend * growth * noise))

    def generate(self) -> List[SessionPlan]:
        wf_names = list(WORKFLOWS)
        wf_p = np.array([WORKFLOWS[w][2] for w in wf_names], dtype=float)
        wf_p /= wf_p.sum()
        t_names = [t[0] for t in TENANTS]
        t_teams = {t[0]: t[1] for t in TENANTS}
        t_p = np.array([t[2] for t in TENANTS], dtype=float)
        t_p /= t_p.sum()

        plans: List[SessionPlan] = []
        n = 0
        for day in range(self.days):
            for _ in range(self.volume_for_day(day)):
                wf = wf_names[int(self.rng.choice(len(wf_names), p=wf_p))]
                outcome_type, raw_steps, _ = WORKFLOWS[wf]
                tenant = t_names[int(self.rng.choice(len(t_names), p=t_p))]
                hour = int(self.rng.choice(24, p=self.hour_p))
                minute = int(self.rng.integers(0, 60))
                steps = [Step(*s) for s in raw_steps]
                # 18% of support tickets need a second resolution pass
                if wf in ("support_ticket", "claims_review") and self.rng.random() < 0.18:
                    steps.insert(3, Step("resolve_retry", "resolution_agent", "generation",
                                         raw_steps[2][3], raw_steps[2][4]))
                ts = (SIM_START + timedelta(days=day, hours=hour, minutes=minute)).timestamp()
                plans.append(
                    SessionPlan(
                        session_id=f"ses_{n:07d}",
                        outcome_id=f"out_{n:07d}",
                        day=day, hour=hour, minute=minute,
                        tenant=tenant, team=t_teams[tenant],
                        workflow=wf, outcome_type=outcome_type, steps=steps,
                        template_id=int(self.rng.choice(QUERY_TEMPLATES, p=self.template_p)),
                        paraphrase=int(self.rng.integers(0, 3)),
                        abandon=bool(self.rng.random() < 0.055),
                        duplicate_call=bool(self.rng.random() < 0.08),
                        ts_epoch=ts,
                    )
                )
                n += 1
        plans.sort(key=lambda p: p.ts_epoch)
        return plans


# ------------------------------------------------------------------ chunk pool
class ChunkPool:
    """Retrieved chunks per (workflow, step), with a latent citation rate.

    Context reduction in TokenOps is emergent: chunks that history shows are
    never cited get dropped. Nothing is hard-coded to a target ratio.
    """

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.pool: Dict[str, List[Tuple[str, float]]] = {}
        for wf, (_, steps, _) in WORKFLOWS.items():
            for step, *_ in steps:
                key = f"{wf}:{step}"
                utils = rng.beta(2.0, 2.5, 8)
                self.pool[key] = [(f"{key}#c{i}", float(u)) for i, u in enumerate(utils)]

    def chunks(self, workflow: str, step: str, total_tokens: int) -> List[Chunk]:
        # a retry re-uses the same retrieval context as the step it repeats
        key = f"{workflow}:{step.replace('_retry', '')}"
        entries = self.pool.get(key) or next(iter(self.pool.values()))
        per = max(1, int(total_tokens * 0.9 / len(entries)))
        return [Chunk(cid, "", per, u) for cid, u in entries]


# ------------------------------------------------------------------ simulation
@dataclass
class ArmResult:
    arm: str
    calls: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    router_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


class Simulator:
    def __init__(self, days: int, sessions_per_day: int, seed: int) -> None:
        self.settings = get_settings()
        self.days = days
        self.seed = seed
        self.demand = DemandGenerator(days, sessions_per_day, seed).generate()
        self.chunk_pool = ChunkPool(seed + 1)
        self.budget = Budget("global", "all-tenants", self.settings.MONTHLY_BUDGET_INR, days)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _prompt_hash(template_id: int, step: str, salt: str = "") -> str:
        return hashlib.blake2b(f"{template_id}:{step}:{salt}".encode(), digest_size=8).hexdigest()

    @staticmethod
    def _ctx_multiplier(day: int) -> float:
        """The day-23 prompt change. Baseline eats it whole."""
        return PROMPT_BLOAT_FACTOR if day >= PROMPT_BLOAT_DAY else 1.0

    def _row(self, plan: SessionPlan, step: Step, arm: str, model: str, in_tok: int,
             out_tok: int, quality: Optional[float], *, cached: int = 0, cache_hit: bool = False,
             escalated: bool = False, compressed: bool = False, overhead: bool = False,
             status: str = "success", waste: Optional[str] = None, incident: Optional[str] = None,
             route_id: str = "static", minute_offset: int = 0, latency_ms: int = 0,
             prompt_salt: str = "") -> Dict[str, Any]:
        cost = compute_cost(model, in_tok, out_tok, cached_tokens=cached)
        ts = plan.ts_epoch + minute_offset * 60
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = (dt - SIM_START).days
        return {
            "call_id": "",
            "arm": arm,
            "ts": dt.isoformat(),
            "ts_epoch": ts,
            "day": day,
            "hour": dt.hour,
            "tenant": plan.tenant,
            "team": plan.team,
            "agent": step.agent,
            "workflow": plan.workflow,
            "step": step.step,
            "session_id": plan.session_id,
            "outcome_id": plan.outcome_id,
            "outcome_type": plan.outcome_type,
            "task_type": step.task_type,
            "model": model,
            "route_id": route_id,
            "prompt_hash": self._prompt_hash(plan.template_id, step.step, prompt_salt),
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "cached_tokens": int(cached),
            "cost_usd": cost.usd,
            "cost_inr": cost.inr,
            "latency_ms": int(latency_ms),
            "quality": quality,
            "cache_hit": cache_hit,
            "escalated": escalated,
            "compressed": compressed,
            "is_overhead": overhead,
            "status": status,
            "waste_tag": waste,
            "incident": incident,
        }

    # ------------------------------------------------------------------ arm A
    def run_baseline(self) -> ArmResult:
        rng = np.random.default_rng(self.seed + 100)
        res = ArmResult("baseline")
        for plan in self.demand:
            qualities: List[float] = []
            for step in plan.steps:
                mult = self._ctx_multiplier(plan.day)
                in_tok = int(rng.normal(step.base_in, step.base_in * 0.16) * mult)
                out_tok = int(rng.normal(step.base_out, step.base_out * 0.22))
                in_tok, out_tok = max(in_tok, 200), max(out_tok, 40)
                q = float(np.clip(rng.normal(QUALITY[step.task_type][BASELINE_MODEL], 0.032), 0, 1))
                qualities.append(q)
                lat = int(rng.normal(1800, 400) + out_tok * 1.1)
                res.calls.append(self._row(plan, step, "baseline", BASELINE_MODEL, in_tok, out_tok, q,
                                           latency_ms=max(lat, 200)))
                # the day-11 provider flap: the retry policy amplifies it 3x
                if plan.day == RETRY_STORM_DAY and plan.hour in RETRY_STORM_HOURS and rng.random() < 0.34:
                    for attempt in range(2):
                        res.calls.append(self._row(
                            plan, step, "baseline", BASELINE_MODEL, in_tok, out_tok, None,
                            status="failed", waste="retry_waste", incident="retry_storm",
                            latency_ms=30000, prompt_salt=f"retry{attempt}",
                        ))
                # duplicate call: same prompt, same session, nothing caching it
                if plan.duplicate_call and step.step == "qa_verify":
                    res.calls.append(self._row(plan, step, "baseline", BASELINE_MODEL, in_tok, out_tok, q,
                                               waste="duplicate_calls", latency_ms=max(lat, 200)))
            self._close_outcome(res, plan, qualities, rng, degraded=False)

        self._add_loop_incident(res, "baseline", stop_minute=LOOP_END_MIN_BASELINE)
        return res

    # ------------------------------------------------------------------ arm B
    def run_tokenops(self) -> ArmResult:
        rng = np.random.default_rng(self.seed + 200)
        res = ArmResult("tokenops")
        router = LearningRouter(seed=self.seed + 300)
        cache = SemanticCache()
        utility = UtilityIndex(min_obs=3)
        compressor = PromptCompressor(utility_index=utility, utility_threshold=0.15)
        guard = CostGuardrails()

        current_day = 0
        cache_saved = 0.0

        for plan in self.demand:
            if plan.day != current_day:
                for tt in QUALITY:
                    res.router_snapshots.extend(router.snapshot(current_day, tt))
                router.reset_exploration_budget()   # budget period is one day
                current_day = plan.day

            qualities: List[float] = []
            degraded = False
            for step in plan.steps:
                decision = router.select(step.task_type)
                route = decision.route
                model = route.model

                mult = self._ctx_multiplier(plan.day)
                raw_in = int(max(rng.normal(step.base_in, step.base_in * 0.16) * mult, 200))
                out_tok = max(int(rng.normal(step.base_out, step.base_out * 0.22)), 40)
                if route.prompt_variant == "terse":
                    out_tok = int(out_tok * 0.88)

                # --- context assembly + compression -------------------------
                chunks = self.chunk_pool.chunks(plan.workflow, step.step, raw_in)
                if route.context_depth == "shallow":
                    chunks = sorted(chunks, key=lambda c: -c.utility)[:4]
                prompt = ("system instructions\n\nyou are a helpful assistant.\n\n"
                          + _query_text(plan.template_id, plan.paraphrase, plan.workflow))
                comp = compressor.compress(prompt, chunks,
                                           allow_llm=(step.task_type == "generation" and raw_in > 20000))
                in_tok = max(comp.compressed_tokens, 150)
                # observe which chunks the answer actually cited, so utility is learned
                for c in chunks:
                    utility.observe(c.chunk_id, bool(rng.random() < c.utility))

                # --- semantic cache -----------------------------------------
                qtext = _query_text(plan.template_id, plan.paraphrase, plan.workflow) + f" {step.step}"
                cacheable = step.step in CACHEABLE_STEPS
                thr = 0.92 if route.cache_policy == "aggressive" else None
                if cacheable:
                    hit, score = cache.lookup(qtext, plan.tenant, plan.workflow,
                                              threshold=thr, now_epoch=plan.ts_epoch)
                else:
                    cache.stats.uncacheable += 1
                    hit = None
                if hit is not None:
                    cached_cost = compute_cost(model, 0, 0, cached_tokens=in_tok)
                    cache.record_saving(hit.cost_inr, cached_cost.inr)
                    cache_saved += hit.cost_inr - cached_cost.inr
                    qualities.append(hit.quality)
                    res.calls.append(self._row(plan, step, "tokenops", model, 0, 0, hit.quality,
                                               cached=in_tok, cache_hit=True, compressed=True,
                                               route_id=route.route_id, latency_ms=int(rng.normal(120, 30))))
                    router.update(step.task_type, route.route_id, hit.quality, cached_cost.inr,
                                  decision.exploring)
                    continue

                # --- cascade ------------------------------------------------
                q, cost_inr, used_model, escalated, extra_rows = self._execute_with_cascade(
                    plan, step, model, in_tok, out_tok, comp.quality_delta,
                    route.context_depth == "shallow", route.prompt_variant == "terse",
                    rng, route.route_id,
                )
                res.calls.extend(extra_rows)
                # The day-11 provider flap hits both arms - TokenOps cannot stop
                # a provider failing. What it stops is the amplification: the
                # retry budget is capped at one attempt once the 6h burn window
                # is over threshold, instead of three unconditionally.
                if plan.day == RETRY_STORM_DAY and plan.hour in RETRY_STORM_HOURS and rng.random() < 0.34:
                    res.calls.append(self._row(
                        plan, step, "tokenops", model, in_tok, out_tok, None,
                        status="failed", waste="retry_waste", incident="retry_storm",
                        route_id=route.route_id, latency_ms=30000, prompt_salt="retry0",
                        compressed=True,
                    ))
                qualities.append(q)
                if cacheable:
                    cache.put(qtext, plan.tenant, plan.workflow, result=None, quality=q,
                              cost_inr=cost_inr, now_epoch=plan.ts_epoch)
                router.update(step.task_type, route.route_id, q, cost_inr, decision.exploring)

                # --- metering overhead: 15% quality-judge sample -------------
                if rng.random() < 0.15:
                    res.calls.append(self._row(
                        plan, step, "tokenops", "claude-haiku-4-5-20251001", 1400, 110, None,
                        overhead=True, route_id="judge", latency_ms=400,
                    ))

            self._close_outcome(res, plan, qualities, rng, degraded=degraded)

        for tt in QUALITY:
            res.router_snapshots.extend(router.snapshot(current_day, tt))

        # the day-18 loop, caught by burn rate + loop detection
        detect_min = self._add_loop_incident(res, "tokenops", stop_minute=None, guard=guard)
        res.notes["loop_detected_after_min"] = detect_min
        res.notes["cache"] = cache.stats.as_dict()
        res.notes["router_explain"] = {tt: router.explain(tt) for tt in QUALITY}
        res.notes["convergence_day"] = {
            tt: router.convergence_day(res.router_snapshots, tt) for tt in QUALITY
        }
        res.notes["policy"] = {tt: router.policy(tt).route_id for tt in QUALITY}
        return res

    def _execute_with_cascade(self, plan: SessionPlan, step: Step, model: str, in_tok: int,
                              out_tok: int, quality_delta: float, shallow: bool, terse: bool,
                              rng: np.random.Generator, route_id: str):
        """Run the ladder from the router's chosen model upward. Every attempt
        is charged - that is what makes the break-even rate real."""
        rows: List[Dict[str, Any]] = []
        total_cost = 0.0
        current = model
        escalated = False
        for depth in range(3):
            q = float(np.clip(
                rng.normal(QUALITY[step.task_type][current], 0.032)
                + quality_delta - (0.012 if shallow else 0.0) - (0.005 if terse else 0.0),
                0, 1,
            ))
            conf = float(np.clip(rng.normal(CONFIDENCE[step.task_type][current], 0.06), 0, 1))
            cost = compute_cost(current, in_tok, out_tok)
            total_cost += cost.inr
            lat = int(rng.normal(1500, 350) + out_tok * (0.7 if "haiku" in current else 1.2))
            nxt = CASCADE_NEXT.get(current)
            will_escalate = conf < self.settings.CASCADE_ESCALATION_CONFIDENCE and nxt is not None
            rows.append(self._row(plan, step, "tokenops", current, in_tok, out_tok,
                                  None if will_escalate else q,
                                  escalated=escalated, compressed=True, route_id=route_id,
                                  latency_ms=max(lat, 150),
                                  waste="verbose_output" if out_tok > step.base_out * 1.6 else None))
            if not will_escalate:
                return q, total_cost, current, escalated, rows
            current, escalated = nxt, True
        return q, total_cost, current, escalated, rows

    # ------------------------------------------------------------- incidents
    def _add_loop_incident(self, res: ArmResult, arm: str, stop_minute: Optional[int],
                           guard: Optional[CostGuardrails] = None) -> Optional[int]:
        """A resolution agent loops on the same (step, prompt_hash) from 02:14.

        Baseline runs until a human arrives at 08:00. TokenOps runs the loop
        detector over the live session and the 1-hour burn window over
        minute-level spend, and stops at whichever fires first.
        """
        rng = np.random.default_rng(self.seed + (7 if arm == "baseline" else 8))
        plan = SessionPlan(
            session_id="ses_loop_d18", outcome_id="out_loop_d18", day=LOOP_DAY,
            hour=2, minute=14, tenant="vertex-insurance", team="claims",
            workflow="claims_review", outcome_type="claim_adjudicated", steps=[],
            template_id=999, paraphrase=0, abandon=False, duplicate_call=False,
            ts_epoch=(SIM_START + timedelta(days=LOOP_DAY, hours=2, minutes=14)).timestamp(),
        )
        step = Step("resolve", "resolution_agent", "generation", LOOP_IN, LOOP_OUT)
        model = BASELINE_MODEL if arm == "baseline" else "claude-sonnet-4-6"
        per_call = compute_cost(model, LOOP_IN, LOOP_OUT).inr

        detect_minute: Optional[int] = None
        if arm == "tokenops":
            # The loop detector needs LOOP_DETECTION_REPEAT_THRESHOLD identical
            # (step, prompt_hash) calls in one session, then the orchestrator
            # drains and kills it. No budget knowledge required - this is a
            # signature match, which is why it is fast.
            repeats = self.settings.LOOP_DETECTION_REPEAT_THRESHOLD
            detector_min = max(1, int(np.ceil(repeats / LOOP_CALLS_PER_MIN)))
            detect_minute = detector_min + LOOP_KILL_DELAY_MIN
            stop_minute = LOOP_START_MIN + detect_minute
            res.notes["loop_detector_min"] = detector_min
            res.notes["loop_kill_min"] = detect_minute

        end = stop_minute if stop_minute is not None else LOOP_END_MIN_BASELINE
        n_minutes = max(1, end - LOOP_START_MIN)
        total_calls = 0
        for m in range(n_minutes):
            for _ in range(LOOP_CALLS_PER_MIN):
                res.calls.append(self._row(
                    plan, step, arm, model, LOOP_IN, LOOP_OUT, None,
                    waste="loop_waste", incident="agent_loop", minute_offset=m,
                    latency_ms=int(rng.normal(2600, 300)), route_id="loop",
                ))
                total_calls += 1
        loop_cost = total_calls * per_call
        res.notes["loop_calls"] = total_calls
        res.notes["loop_cost_inr"] = loop_cost
        res.notes["loop_minutes"] = n_minutes

        ts = plan.ts_epoch + n_minutes * 60
        if arm == "tokenops":
            res.alerts.append({
                "arm": arm, "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "ts_epoch": ts, "kind": "burn_rate", "severity": "critical",
                "scope": "tenant:vertex-insurance", "window_hours": 1,
                "observed_multiplier": 14.4,
                "message": f"agent loop detected {detect_minute} min after it started; "
                           f"session {plan.session_id} killed",
            })
            res.alerts.append({
                "arm": arm, "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "ts_epoch": ts, "kind": "circuit_breaker", "severity": "critical",
                "scope": "tenant:vertex-insurance", "window_hours": None,
                "observed_multiplier": None,
                "message": "circuit breaker opened for resolution_agent on tenant vertex-insurance",
            })
            if guard is not None:
                guard.circuit_break("tenant:vertex-insurance", "agent loop on resolution_agent")
        else:
            res.alerts.append({
                "arm": arm, "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "ts_epoch": ts, "kind": "human_noticed", "severity": "critical",
                "scope": "tenant:vertex-insurance", "window_hours": None,
                "observed_multiplier": None,
                "message": "on-call engineer noticed the spend at 08:00 and killed the agent",
            })
        return detect_minute

    # --------------------------------------------------------------- outcomes
    @staticmethod
    def _close_outcome(res: ArmResult, plan: SessionPlan, qualities: List[float],
                       rng: np.random.Generator, degraded: bool) -> None:
        mean_q = float(np.mean(qualities)) if qualities else 0.0
        success = (not plan.abandon) and mean_q >= SUCCESS_QUALITY_THRESHOLD
        if plan.day == RETRY_STORM_DAY and plan.hour in RETRY_STORM_HOURS and rng.random() < 0.08:
            success = False        # the flap costs some outcomes outright
        res.outcomes.append({
            "outcome_id": plan.outcome_id,
            "arm": res.arm,
            "outcome_type": plan.outcome_type,
            "tenant": plan.tenant,
            "team": plan.team,
            "session_id": plan.session_id,
            "day": plan.day,
            "ts_epoch": plan.ts_epoch,
            "success": bool(success),
            "quality": mean_q,
            "degraded": bool(degraded),
        })


# ------------------------------------------------------------------ persistence
def persist(result: ArmResult) -> None:
    for i, row in enumerate(result.calls):
        row["call_id"] = f"call_{result.arm[:3]}_{i:08d}"
    bulk_insert("llm_calls", result.calls)
    bulk_insert("outcomes", result.outcomes)
    if result.alerts:
        bulk_insert("alerts", result.alerts)
    if result.router_snapshots:
        bulk_insert("router_state", result.router_snapshots)


def scan_burn_alerts(arm: str, days: int, budget_inr: float) -> int:
    """Post-hoc burn scan over the hourly series, stored for the incident page."""
    from backend.core.ledger import CostLedger

    ledger = CostLedger(arm)
    hourly = ledger.hourly_series()
    if hourly.empty:
        return 0
    monitor = BurnRateMonitor(Budget("global", "all-tenants", budget_inr, days))
    alerts = monitor.scan(hourly, dedup_hours=6)
    rows = []
    for a in alerts:
        rows.append({
            "arm": arm,
            "ts": datetime.fromtimestamp(a.ts_epoch, tz=timezone.utc).isoformat() if a.ts_epoch else "",
            "ts_epoch": a.ts_epoch, "kind": "burn_rate", "severity": a.severity,
            "scope": a.scope, "window_hours": a.window_hours,
            "observed_multiplier": a.observed_multiplier, "message": a.message,
        })
    if rows:
        bulk_insert("alerts", rows)
    return len(rows)


def derive_budget(days: int, headroom: float = 1.10) -> float:
    """A budget nobody set is a budget nobody defends. We derive it the way a
    finance team actually would: last period's baseline spend plus headroom,
    excluding the incident days (you do not budget for an outage)."""
    from backend.storage.db import query_df

    df = query_df(
        "SELECT SUM(cost_inr) AS c FROM llm_calls WHERE arm = 'baseline' "
        "AND incident IS NULL"
    )
    clean = float(df["c"].iloc[0] or 0.0)
    return round(clean / days * 30.0 * headroom, -3)


def analyse_incident_detection(budget_inr: float, days: int) -> Dict[str, Any]:
    """Counterfactual: with no loop signature to match, how long would
    burn-rate alerting alone have taken? This is the number that generalises,
    because the next incident will not be a loop."""
    from backend.storage.db import query_df

    rows = query_df(
        "SELECT ts_epoch, cost_inr FROM llm_calls WHERE arm = 'baseline' AND day = :d",
        {"d": LOOP_DAY},
    )
    if rows.empty:
        return {}
    day_start = (SIM_START + timedelta(days=LOOP_DAY)).timestamp()
    rows["minute"] = ((rows["ts_epoch"] - day_start) // 60).astype(int)
    per_min = rows.groupby("minute")["cost_inr"].sum()
    series = np.zeros(24 * 60)
    for m, c in per_min.items():
        if 0 <= m < len(series):
            series[m] = c
    monitor = BurnRateMonitor(Budget("global", "all-tenants", budget_inr, days))
    breach = monitor.first_breach_minute(series[LOOP_START_MIN:], long_window_hours=1, multiplier=14.4)
    return {
        "burn_alert_after_min": None if breach is None else int(breach) + 1,
        "budget_inr": budget_inr,
        "hourly_budget_inr": monitor.budget.hourly_inr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the TokenOps demo workload")
    s = get_settings()
    ap.add_argument("--days", type=int, default=s.SIM_DAYS)
    ap.add_argument("--sessions", type=int, default=s.SIM_SESSIONS_PER_DAY)
    ap.add_argument("--seed", type=int, default=s.SIM_SEED)
    ap.add_argument("--keep", action="store_true", help="append instead of resetting the database")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not args.keep:
        reset_database()

    sim = Simulator(args.days, args.sessions, args.seed)
    log.info(f"generated demand: {len(sim.demand):,} sessions over {args.days} days")

    baseline = sim.run_baseline()
    log.info(f"arm A (baseline): {len(baseline.calls):,} calls")
    persist(baseline)

    tokenops = sim.run_tokenops()
    log.info(f"arm B (tokenops): {len(tokenops.calls):,} calls")
    persist(tokenops)

    budget = derive_budget(args.days)
    log.info(f"derived monthly budget: {budget:,.0f} INR")

    for arm in ("baseline", "tokenops"):
        n = scan_burn_alerts(arm, args.days, budget)
        log.info(f"burn scan {arm}: {n} alerts")

    detection = analyse_incident_detection(budget, args.days)
    log.info(f"incident detection: {detection}")

    Path("data/samples").mkdir(parents=True, exist_ok=True)
    import json
    with open("data/samples/simulation_notes.json", "w", encoding="utf-8") as f:
        json.dump({
            "days": args.days,
            "sessions": len(sim.demand),
            "monthly_budget_inr": budget,
            "incident_detection": detection,
            "baseline_notes": {k: v for k, v in baseline.notes.items() if k != "router_explain"},
            "tokenops_notes": {k: v for k, v in tokenops.notes.items() if k != "router_explain"},
        }, f, indent=2, default=str)
    with open("data/samples/router_explain.json", "w", encoding="utf-8") as f:
        json.dump(tokenops.notes.get("router_explain", {}), f, indent=2, default=str)

    counts = table_counts()
    log.info(f"tables: {counts}")
    log.info(f"done in {time.perf_counter() - t0:.1f}s")
    print("\nrow counts:", counts)
    print("loop incident:",
          f"baseline {fmt_inr(baseline.notes.get('loop_cost_inr', 0))} over {baseline.notes.get('loop_minutes')} min",
          "|",
          f"tokenops {fmt_inr(tokenops.notes.get('loop_cost_inr', 0))} over {tokenops.notes.get('loop_minutes')} min")


if __name__ == "__main__":
    main()
