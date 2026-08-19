"""The live engine: traffic arriving now, on a wall clock.

Everything else in TokenOps is a 30-day replay. This is the part you put on a
screen in front of people: sessions arrive continuously, the router chooses a
route for each step *as it happens*, the ledger and the savings counter tick
up, and an operator can inject an incident by hand and watch the guardrail
contain it.

Nothing here is a re-enactment. The same `LearningRouter`, `SemanticCache`,
`PromptCompressor`, `CostGuardrails` and `BurnRateMonitor` objects that the
benchmark uses are driving it, so the bandit really is learning while you
watch, and the circuit breaker really does trip.

Time is compressed: `minutes_per_second` simulated minutes elapse per real
second (default 1.0, i.e. 60x). At production volume a real-time clock would
show one session every 72 seconds, which is not a demo.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from backend.config import get_settings
from backend.core.burn_rate import Budget, BurnRateMonitor
from backend.core.guardrails import CostGuardrails, GuardAction
from backend.core.optimizers.compressor import PromptCompressor, UtilityIndex
from backend.core.optimizers.semantic_cache import SemanticCache
from backend.core.pricing import compute_cost
from backend.core.router import LearningRouter
from backend.domain.workload import (
    BASELINE_MODEL, CACHEABLE_STEPS, CASCADE_NEXT, CONFIDENCE, ChunkPool,
    LOOP_CALLS_PER_MIN, LOOP_IN, LOOP_OUT, PROMPT_BLOAT_FACTOR, QUALITY,
    QUERY_TEMPLATES, SUCCESS_QUALITY_THRESHOLD, TENANTS, WORKFLOWS,
    query_text, steps_for,
)
from backend.storage.db import bulk_insert
from backend.utils.logger import log

LIVE_ARM = "live"
MINUTE_HISTORY = 24 * 60
EVENT_HISTORY = 300

INCIDENTS = {
    "agent_loop": "A resolution agent starts looping on the same prompt",
    "retry_storm": "The provider starts failing and the retry policy amplifies it",
    "prompt_bloat": "Someone deploys a prompt change that triples context size",
}


@dataclass
class LiveEvent:
    ts: str
    sim_minute: int
    kind: str            # traffic | route | alert | guardrail | incident | contain
    severity: str        # info | warning | critical | success
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveTotals:
    calls: int = 0
    sessions: int = 0
    outcomes_ok: int = 0
    outcomes_total: int = 0
    managed_inr: float = 0.0
    shadow_inr: float = 0.0          # same traffic, unmanaged
    quality_sum: float = 0.0
    shadow_quality_sum: float = 0.0
    cache_hits: int = 0
    cache_lookups: int = 0
    escalations: int = 0
    cascade_runs: int = 0
    blocked_calls: int = 0
    degraded_calls: int = 0
    incident_inr: float = 0.0
    incident_calls: int = 0


class LiveEngine:
    """One instance per process. Start it, watch it, inject at it."""

    def __init__(
        self,
        minutes_per_second: float = 1.0,
        sessions_per_day: Optional[int] = None,
        monthly_budget_inr: Optional[float] = None,
        seed: int = 41,
        persist: bool = True,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.minutes_per_second = minutes_per_second
        self.sessions_per_day = sessions_per_day or s.SIM_SESSIONS_PER_DAY
        self.persist = persist
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.router = LearningRouter(seed=seed + 1)
        self.cache = SemanticCache()
        self.utility = UtilityIndex(min_obs=3)
        self.compressor = PromptCompressor(utility_index=self.utility, utility_threshold=0.15)
        self.guard = CostGuardrails()
        self.chunks = ChunkPool(seed + 2)

        self.monthly_budget_inr = monthly_budget_inr or self._default_budget()
        self.budget = Budget("global", "all-tenants", self.monthly_budget_inr, 30)
        self.monitor = BurnRateMonitor(self.budget)

        ranks = np.arange(1, QUERY_TEMPLATES + 1)
        self.template_p = 1.0 / ranks ** 0.85
        self.template_p /= self.template_p.sum()

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _default_budget(self) -> float:
        """Reuse the budget the batch run derived from baseline spend, so the
        live page and the benchmark agree on what 'over budget' means."""
        import json
        from pathlib import Path

        p = Path("data/samples/simulation_notes.json")
        if p.exists():
            try:
                v = json.loads(p.read_text(encoding="utf-8")).get("monthly_budget_inr")
                if v:
                    return float(v)
            except Exception:
                pass
        return self.settings.MONTHLY_BUDGET_INR

    def reset(self) -> None:
        with self._lock:
            self.running = False
            self.started_at: Optional[float] = None
            self.sim_minute = 0
            self._minute_fraction = 0.0
            self.totals = LiveTotals()
            self.minute_costs: Deque[float] = deque([0.0], maxlen=MINUTE_HISTORY)
            self.minute_shadow: Deque[float] = deque([0.0], maxlen=MINUTE_HISTORY)
            self.events: Deque[LiveEvent] = deque(maxlen=EVENT_HISTORY)
            self.recent_routes: Deque[Dict[str, Any]] = deque(maxlen=40)
            self.pending_rows: List[Dict[str, Any]] = []
            self.active_incidents: Dict[str, Dict[str, Any]] = {}
            self.loop_session_calls: List[Dict[str, Any]] = []
            self.alerted_windows: Dict[int, int] = {}
            self._announced_exclusions: set = set()
            self.guard.reset()
            self.router.reset_exploration_budget()
            self._emit("info", "traffic", "engine reset; ledger cleared")

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
            self.started_at = time.time()
            self._stop.clear()
            self._emit("success", "traffic", "live traffic started")
        self._thread = threading.Thread(target=self._run, name="tokenops-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            self.running = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._flush()
        self._emit("info", "traffic", "live traffic stopped")

    def _run(self) -> None:
        last = time.time()
        while not self._stop.is_set():
            now = time.time()
            elapsed = now - last
            last = now
            try:
                self.tick(elapsed * self.minutes_per_second)
            except Exception:                      # a demo must never die on stage
                log.exception("live tick failed")
                self._emit("warning", "traffic", "tick failed; continuing")
            self._stop.wait(0.25)

    # -------------------------------------------------------------------- tick
    def tick(self, sim_minutes: float) -> None:
        """Advance the simulated clock. Safe to call directly in tests."""
        if sim_minutes <= 0:
            return
        with self._lock:
            self._minute_fraction += sim_minutes
            whole = int(self._minute_fraction)
            self._minute_fraction -= whole
            for _ in range(max(whole, 1) if whole else 0):
                self._advance_one_minute()
            if not whole:
                # sub-minute tick: still generate proportional traffic so the
                # screen keeps moving between minute boundaries
                self._generate_traffic(sim_minutes)
            self._flush()

    def _advance_one_minute(self) -> None:
        self.sim_minute += 1
        self.minute_costs.append(0.0)
        self.minute_shadow.append(0.0)
        self._generate_traffic(1.0)
        self._run_incidents()
        self._evaluate_burn()

    # ---------------------------------------------------------------- traffic
    def _generate_traffic(self, sim_minutes: float) -> None:
        rate = self.sessions_per_day / (24 * 60) * sim_minutes
        n = int(self.rng.poisson(max(rate, 0.0)))
        for _ in range(n):
            self._run_session()

    def _run_session(self) -> None:
        wf_names = list(WORKFLOWS)
        wf_p = np.array([WORKFLOWS[w][2] for w in wf_names], dtype=float)
        wf_p /= wf_p.sum()
        wf = wf_names[int(self.rng.choice(len(wf_names), p=wf_p))]
        outcome_type = WORKFLOWS[wf][0]

        t_p = np.array([t[2] for t in TENANTS], dtype=float)
        t_p /= t_p.sum()
        ti = int(self.rng.choice(len(TENANTS), p=t_p))
        tenant, team = TENANTS[ti][0], TENANTS[ti][1]

        session_id = f"live_{uuid.uuid4().hex[:10]}"
        outcome_id = f"lout_{uuid.uuid4().hex[:10]}"
        template_id = int(self.rng.choice(QUERY_TEMPLATES, p=self.template_p))
        paraphrase = int(self.rng.integers(0, 3))

        budget_state = self.budget_state()
        decision = self.guard.check(
            {"interactive": True}, {"scope": f"tenant:{tenant}",
                                    "remaining_pct": budget_state["remaining_pct"]}
        )
        if decision.action is GuardAction.BLOCK:
            self.totals.blocked_calls += 1
            return

        qualities: List[float] = []
        shadow_qualities: List[float] = []
        for step in steps_for(wf):
            q, sq = self._run_step(
                wf, step, tenant, team, session_id, outcome_id, outcome_type,
                template_id, paraphrase, decision,
            )
            qualities.append(q)
            shadow_qualities.append(sq)

        mean_q = float(np.mean(qualities)) if qualities else 0.0
        abandoned = bool(self.rng.random() < 0.055)
        success = (not abandoned) and mean_q >= SUCCESS_QUALITY_THRESHOLD
        self.totals.sessions += 1
        self.totals.outcomes_total += 1
        if success:
            self.totals.outcomes_ok += 1
            self.totals.quality_sum += mean_q
            self.totals.shadow_quality_sum += float(np.mean(shadow_qualities))

    def _run_step(self, wf, step, tenant, team, session_id, outcome_id, outcome_type,
                  template_id, paraphrase, guard_decision):
        degraded = guard_decision.action is GuardAction.DEGRADE
        rd = self.router.select(step.task_type, {"force_cheap": degraded})
        route = rd.route

        bloat = PROMPT_BLOAT_FACTOR if "prompt_bloat" in self.active_incidents else 1.0
        raw_in = int(max(self.rng.normal(step.base_in, step.base_in * 0.16) * bloat, 200))
        out_tok = max(int(self.rng.normal(step.base_out, step.base_out * 0.22)), 40)
        if route.prompt_variant == "terse":
            out_tok = int(out_tok * 0.88)

        # --- the unmanaged counterfactual, priced on the same traffic --------
        shadow_cost = compute_cost(BASELINE_MODEL, raw_in, out_tok).inr
        shadow_q = float(np.clip(
            self.rng.normal(QUALITY[step.task_type][BASELINE_MODEL], 0.032), 0, 1))
        self.totals.shadow_inr += shadow_cost
        self.minute_shadow[-1] += shadow_cost

        # --- context assembly + compression ----------------------------------
        chunks = self.chunks.chunks(wf, step.step, raw_in)
        if route.context_depth == "shallow" or degraded:
            chunks = sorted(chunks, key=lambda c: -c.utility)[:4]
        prompt = ("system instructions\n\nyou are a helpful assistant.\n\n"
                  + query_text(template_id, paraphrase, wf))
        comp = self.compressor.compress(prompt, chunks)
        in_tok = max(comp.compressed_tokens, 150)
        for c in chunks:
            self.utility.observe(c.chunk_id, bool(self.rng.random() < c.utility))

        # --- semantic cache ---------------------------------------------------
        qtext = query_text(template_id, paraphrase, wf) + f" {step.step}"
        now = self._now_epoch()
        if step.step in CACHEABLE_STEPS:
            self.totals.cache_lookups += 1
            thr = 0.92 if route.cache_policy == "aggressive" else None
            hit, _ = self.cache.lookup(qtext, tenant, wf, threshold=thr, now_epoch=now)
        else:
            hit = None
        if hit is not None:
            cached_cost = compute_cost(route.model, 0, 0, cached_tokens=in_tok)
            self.totals.cache_hits += 1
            self.cache.record_saving(hit.cost_inr, cached_cost.inr)
            self._record(wf, step, tenant, team, session_id, outcome_id, outcome_type,
                         route.model, 0, 0, hit.quality,
                         cached=in_tok, cache_hit=True, route_id=route.route_id,
                         template_id=template_id, degraded=degraded)
            self.router.update(step.task_type, route.route_id, hit.quality,
                               cached_cost.inr, rd.exploring)
            return hit.quality, shadow_q

        # --- cascade ----------------------------------------------------------
        q, cost_inr, used = self._cascade(
            wf, step, tenant, team, session_id, outcome_id, outcome_type,
            route, in_tok, out_tok, comp.quality_delta, degraded, template_id,
        )
        if step.step in CACHEABLE_STEPS:
            self.cache.put(qtext, tenant, wf, None, quality=q, cost_inr=cost_inr, now_epoch=now)
        self.router.update(step.task_type, route.route_id, q, cost_inr, rd.exploring)
        self._check_floor(step.task_type, route.route_id)

        self.recent_routes.appendleft({
            "sim_minute": self.sim_minute,
            "step": step.step,
            "task_type": step.task_type,
            "route": route.route_id,
            "model": used,
            "exploring": rd.exploring,
            "degraded": degraded,
            "quality": round(q, 3),
            "cost_inr": round(cost_inr, 4),
            "saved_inr": round(shadow_cost - cost_inr, 4),
        })
        if degraded:
            self.totals.degraded_calls += 1

        # --- the metering overhead we charge ourselves ------------------------
        if self.rng.random() < 0.15:
            self._record(wf, step, tenant, team, session_id, outcome_id, outcome_type,
                         "claude-haiku-4-5-20251001", 1400, 110, None,
                         overhead=True, route_id="judge", template_id=template_id)
        return q, shadow_q

    def _cascade(self, wf, step, tenant, team, session_id, outcome_id, outcome_type,
                 route, in_tok, out_tok, quality_delta, degraded, template_id):
        current = route.model
        total = 0.0
        escalated = False
        q = 0.0
        self.totals.cascade_runs += 1
        for _ in range(3):
            q = float(np.clip(
                self.rng.normal(QUALITY[step.task_type][current], 0.032)
                + quality_delta
                - (0.012 if route.context_depth == "shallow" or degraded else 0.0)
                - (0.005 if route.prompt_variant == "terse" else 0.0),
                0, 1,
            ))
            conf = float(np.clip(self.rng.normal(CONFIDENCE[step.task_type][current], 0.06), 0, 1))
            cost = compute_cost(current, in_tok, out_tok)
            total += cost.inr
            nxt = CASCADE_NEXT.get(current)
            will_escalate = conf < self.settings.CASCADE_ESCALATION_CONFIDENCE and nxt is not None
            self._record(wf, step, tenant, team, session_id, outcome_id, outcome_type,
                         current, in_tok, out_tok, None if will_escalate else q,
                         escalated=escalated, route_id=route.route_id,
                         template_id=template_id, degraded=degraded)
            if not will_escalate:
                return q, total, current
            current, escalated = nxt, True
            self.totals.escalations += 1
        return q, total, current

    def _check_floor(self, task_type: str, route_id: str) -> None:
        """Announce a quality-floor exclusion the moment it happens."""
        arm = self.router._task_arms(task_type).get(route_id)   # noqa: SLF001
        if arm is None or not arm.excluded:
            return
        key = f"{task_type}:{route_id}"
        if key in self._announced_exclusions:
            return
        self._announced_exclusions.add(key)
        self._emit(
            "warning", "guardrail",
            f"QUALITY FLOOR: route {route_id} removed from {task_type} - "
            f"{arm.exclusion_reason}",
            {"task_type": task_type, "route_id": route_id},
        )

    # ------------------------------------------------------------- incidents
    def inject(self, kind: str) -> Dict[str, Any]:
        if kind not in INCIDENTS:
            return {"status": "failed", "message": f"unknown incident {kind!r}",
                    "known": list(INCIDENTS)}
        with self._lock:
            if kind in self.active_incidents:
                return {"status": "partial", "message": f"{kind} already running"}
            self.active_incidents[kind] = {
                "started_minute": self.sim_minute,
                "calls": 0,
                "cost_inr": 0.0,
            }
            if kind == "agent_loop":
                self.loop_session_calls = []
            self._emit("critical", "incident", f"INJECTED: {INCIDENTS[kind]}", {"kind": kind})
        return {"status": "success", "kind": kind, "started_minute": self.sim_minute}

    def clear_incident(self, kind: str) -> None:
        with self._lock:
            self.active_incidents.pop(kind, None)

    def _run_incidents(self) -> None:
        if "agent_loop" in self.active_incidents:
            self._run_loop_minute()
        if "retry_storm" in self.active_incidents:
            self._run_retry_minute()
        # prompt_bloat needs no per-minute work: it scales context at assembly

    def _run_loop_minute(self) -> None:
        inc = self.active_incidents["agent_loop"]
        step = type("S", (), {"step": "resolve", "agent": "resolution_agent",
                              "task_type": "generation", "base_in": LOOP_IN,
                              "base_out": LOOP_OUT})()
        for _ in range(LOOP_CALLS_PER_MIN):
            cost = compute_cost(BASELINE_MODEL, LOOP_IN, LOOP_OUT)
            row = self._record(
                "claims_review", step, "vertex-insurance", "claims", "live_loop_session",
                "live_loop_outcome", "claim_adjudicated", BASELINE_MODEL, LOOP_IN, LOOP_OUT,
                None, waste="loop_waste", incident="agent_loop", route_id="loop",
                template_id=999,
            )
            inc["calls"] += 1
            inc["cost_inr"] += cost.inr
            self.totals.incident_calls += 1
            self.totals.incident_inr += cost.inr
            self.loop_session_calls.append(row)

            detection = self.guard.detect_loop(self.loop_session_calls)
            if detection.detected:
                self._contain_loop(detection, inc)
                return

    def _contain_loop(self, detection, inc: Dict[str, Any]) -> None:
        minutes = self.sim_minute - inc["started_minute"]
        self.guard.circuit_break("tenant:vertex-insurance",
                                 "agent loop on resolution_agent")
        self._emit(
            "success", "contain",
            f"LOOP KILLED after {inc['calls']} calls "
            f"({minutes} sim-min) - {detection.repeats} identical (step, prompt_hash) "
            f"in one session; circuit breaker open on tenant:vertex-insurance",
            {"wasted_inr": round(detection.wasted_inr, 2),
             "incident_inr": round(inc["cost_inr"], 2),
             "calls": inc["calls"], "sim_minutes": minutes},
        )
        self.active_incidents.pop("agent_loop", None)
        self.loop_session_calls = []

    def _run_retry_minute(self) -> None:
        """The provider is flapping. TokenOps cannot stop that; it caps the
        amplification at one retry instead of three."""
        inc = self.active_incidents["retry_storm"]
        n = int(self.rng.poisson(self.sessions_per_day / (24 * 60) * 1.2))
        step = type("S", (), {"step": "retrieve", "agent": "retrieval_agent",
                              "task_type": "retrieval", "base_in": 9500, "base_out": 320})()
        for _ in range(n):
            cost = compute_cost(BASELINE_MODEL, 9500, 320)
            self._record("support_ticket", step, "acme-bank", "support",
                         f"live_retry_{uuid.uuid4().hex[:6]}", f"lout_{uuid.uuid4().hex[:6]}",
                         "ticket_resolved", BASELINE_MODEL, 9500, 320, None,
                         status="failed", waste="retry_waste", incident="retry_storm",
                         route_id="retry", template_id=998)
            inc["calls"] += 1
            inc["cost_inr"] += cost.inr
            self.totals.incident_inr += cost.inr
            self.totals.incident_calls += 1

    # ------------------------------------------------------------- burn rate
    def _evaluate_burn(self) -> None:
        state = self.monitor.live_state(list(self.minute_costs))
        for window in state["windows"]:
            if not window["breaching"]:
                continue
            wh = window["window_hours"]
            last = self.alerted_windows.get(wh)
            if last is not None and self.sim_minute - last < 30:
                continue
            self.alerted_windows[wh] = self.sim_minute
            self._emit(
                "critical", "alert",
                f"BURN ALERT: {wh}h window at {window['observed_multiplier']:.1f}x budgeted "
                f"(threshold {window['threshold_multiplier']}x)",
                window,
            )
            if self.guard.should_break(window["observed_multiplier"]):
                self.guard.circuit_break("global:all-tenants",
                                         f"burn rate {window['observed_multiplier']:.1f}x")
                self._emit("critical", "guardrail",
                           "CIRCUIT BREAKER opened globally - new calls blocked until reset",
                           window)

    def budget_state(self) -> Dict[str, Any]:
        spent = self.totals.managed_inr + self.totals.incident_inr
        # the live window is a slice of a month; scale the budget to it so the
        # remaining-percentage means something after four simulated hours
        elapsed_frac = max(self.sim_minute, 1) / (30 * 24 * 60)
        allowance = self.monthly_budget_inr * elapsed_frac
        remaining_pct = 100.0 if allowance <= 0 else max(
            0.0, min(100.0, (allowance - spent) / allowance * 100.0))
        return {
            "scope": "global:all-tenants",
            "monthly_budget_inr": self.monthly_budget_inr,
            "allowance_so_far_inr": allowance,
            "spent_inr": spent,
            "remaining_pct": remaining_pct,
            "hourly_budget_inr": self.budget.hourly_inr,
        }

    # ------------------------------------------------------------ persistence
    def _now_epoch(self) -> float:
        return (self.started_at or time.time()) + self.sim_minute * 60.0

    def _record(self, wf, step, tenant, team, session_id, outcome_id, outcome_type,
                model, in_tok, out_tok, quality, *, cached: int = 0, cache_hit: bool = False,
                escalated: bool = False, overhead: bool = False, status: str = "success",
                waste: Optional[str] = None, incident: Optional[str] = None,
                route_id: str = "live", template_id: int = 0,
                degraded: bool = False) -> Dict[str, Any]:
        cost = compute_cost(model, in_tok, out_tok, cached_tokens=cached)
        ts = self._now_epoch()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        row = {
            "call_id": f"call_live_{uuid.uuid4().hex[:12]}",
            "arm": LIVE_ARM,
            "ts": dt.isoformat(),
            "ts_epoch": ts,
            "day": self.sim_minute // (24 * 60),
            "hour": (self.sim_minute // 60) % 24,
            "tenant": tenant,
            "team": team,
            "agent": step.agent,
            "workflow": wf,
            "step": step.step,
            "session_id": session_id,
            "outcome_id": outcome_id,
            "outcome_type": outcome_type,
            "task_type": step.task_type,
            "model": model,
            "route_id": route_id,
            "prompt_hash": hashlib.blake2b(
                f"{template_id}:{step.step}".encode(), digest_size=8).hexdigest(),
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "cached_tokens": int(cached),
            "cost_usd": cost.usd,
            "cost_inr": cost.inr,
            "latency_ms": 0,
            "quality": quality,
            "cache_hit": cache_hit,
            "escalated": escalated,
            "compressed": not cache_hit and not overhead,
            "is_overhead": overhead,
            "status": status,
            "waste_tag": waste,
            "incident": incident,
        }
        self.totals.calls += 1
        self.totals.managed_inr += cost.inr
        self.minute_costs[-1] += cost.inr
        if self.persist:
            self.pending_rows.append(row)
        return row

    def _flush(self) -> None:
        if not self.persist or not self.pending_rows:
            return
        rows, self.pending_rows = self.pending_rows, []
        try:
            bulk_insert("llm_calls", rows)
        except Exception:
            log.exception("live flush failed; dropping %d rows", len(rows))

    def _emit(self, severity: str, kind: str, message: str,
              detail: Optional[Dict[str, Any]] = None) -> None:
        self.events.appendleft(LiveEvent(
            ts=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            sim_minute=self.sim_minute, kind=kind, severity=severity,
            message=message, detail=detail or {},
        ))

    # --------------------------------------------------------------- snapshot
    def snapshot(self, minutes: int = 120) -> Dict[str, Any]:
        with self._lock:
            t = self.totals
            saved = t.shadow_inr - t.managed_inr
            ok = t.outcomes_ok or 1
            burn = self.monitor.live_state(list(self.minute_costs))
            return {
                "status": "success",
                "running": self.running,
                "sim_minute": self.sim_minute,
                "sim_clock": f"{self.sim_minute // 60 % 24:02d}:{self.sim_minute % 60:02d}",
                "minutes_per_second": self.minutes_per_second,
                "wall_seconds": (time.time() - self.started_at) if self.started_at else 0.0,
                "totals": asdict(t),
                "managed_inr": t.managed_inr + t.incident_inr,
                "shadow_inr": t.shadow_inr,
                "saved_inr": saved,
                "saved_pct": (saved / t.shadow_inr * 100.0) if t.shadow_inr else 0.0,
                "cost_per_outcome_inr": (t.managed_inr + t.incident_inr) / ok,
                "shadow_cost_per_outcome_inr": t.shadow_inr / ok,
                "mean_quality": t.quality_sum / ok,
                "shadow_mean_quality": t.shadow_quality_sum / ok,
                "cache_hit_rate": (t.cache_hits / t.cache_lookups) if t.cache_lookups else 0.0,
                "escalation_rate": (t.escalations / t.cascade_runs) if t.cascade_runs else 0.0,
                "budget": self.budget_state(),
                "burn": burn,
                "minute_costs": list(self.minute_costs)[-minutes:],
                "minute_shadow": list(self.minute_shadow)[-minutes:],
                "events": [e.as_dict() for e in list(self.events)[:60]],
                "recent_routes": list(self.recent_routes),
                "active_incidents": {k: dict(v) for k, v in self.active_incidents.items()},
                "circuit_breakers": dict(self.guard.broken_scopes),
                "policy": {tt: self.router.policy(tt).route_id for tt in QUALITY},
            }

    def set_budget(self, monthly_inr: float) -> None:
        with self._lock:
            self.monthly_budget_inr = float(monthly_inr)
            self.budget = Budget("global", "all-tenants", self.monthly_budget_inr, 30)
            self.monitor = BurnRateMonitor(self.budget)
            self._emit("info", "guardrail",
                       f"monthly budget set to {self.monthly_budget_inr:,.0f} INR")

    def set_speed(self, minutes_per_second: float) -> None:
        with self._lock:
            self.minutes_per_second = float(minutes_per_second)

    def clear_breakers(self) -> None:
        with self._lock:
            self.guard.reset()
            self._emit("info", "guardrail", "circuit breakers reset by operator")


_ENGINE: Optional[LiveEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(**kwargs: Any) -> LiveEngine:
    """Process-wide singleton, so the API and the UI drive the same engine."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = LiveEngine(**kwargs)
        return _ENGINE
