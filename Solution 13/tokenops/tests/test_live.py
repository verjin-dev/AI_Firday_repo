"""Tests for the real-time engine.

The engine is driven directly through `tick()` rather than through its
background thread, so these are deterministic. Persistence is off: these test
the control logic, not the database.
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.burn_rate import Budget, BurnRateMonitor
from backend.core.live import INCIDENTS, LiveEngine
from backend.domain.workload import LOOP_CALLS_PER_MIN, QUALITY, WORKFLOWS


def build(**kw) -> LiveEngine:
    kw.setdefault("persist", False)
    kw.setdefault("monthly_budget_inr", 582_000.0)
    kw.setdefault("seed", 11)
    return LiveEngine(**kw)


def run_minutes(engine: LiveEngine, minutes: int) -> None:
    for _ in range(minutes):
        engine.tick(1.0)


# --------------------------------------------------------------------- clock
def test_clock_advances_and_generates_traffic():
    e = build()
    run_minutes(e, 90)
    s = e.snapshot()
    assert s["sim_minute"] == 90
    assert s["totals"]["sessions"] > 0
    assert s["totals"]["calls"] > s["totals"]["sessions"]


def test_sub_minute_ticks_do_not_advance_the_minute_counter():
    e = build()
    e.tick(0.3)
    e.tick(0.3)
    assert e.sim_minute == 0


def test_reset_clears_everything():
    e = build()
    run_minutes(e, 30)
    assert e.snapshot()["totals"]["calls"] > 0
    e.reset()
    s = e.snapshot()
    assert s["sim_minute"] == 0
    assert s["totals"]["calls"] == 0
    assert s["saved_inr"] == 0.0


# ------------------------------------------------------------------ economics
def test_shadow_arm_is_priced_on_the_same_traffic():
    e = build()
    run_minutes(e, 120)
    s = e.snapshot()
    assert s["shadow_inr"] > 0
    # every managed call has a shadow counterpart, so the shadow can never be
    # zero while managed spend is non-zero
    assert s["managed_inr"] > 0


def test_savings_turn_positive_once_the_router_has_learned():
    e = build()
    run_minutes(e, 240)
    s = e.snapshot()
    assert s["saved_inr"] > 0, "managed traffic should beat the unmanaged shadow by 4 hours in"
    assert 0 < s["saved_pct"] < 100


def test_quality_stays_above_the_floor_on_average():
    e = build()
    run_minutes(e, 240)
    s = e.snapshot()
    assert s["mean_quality"] >= e.settings.QUALITY_FLOOR


# ------------------------------------------------------------------ incidents
def test_unknown_incident_is_rejected_with_the_known_list():
    e = build()
    out = e.inject("volcano")
    assert out["status"] == "failed"
    assert set(out["known"]) == set(INCIDENTS)


def test_agent_loop_is_detected_and_contained_within_one_minute():
    e = build()
    run_minutes(e, 30)
    e.inject("agent_loop")
    run_minutes(e, 2)
    s = e.snapshot()
    assert "agent_loop" not in s["active_incidents"], "loop should be killed, not still running"
    assert "tenant:vertex-insurance" in s["circuit_breakers"]
    # it must be stopped at the detector threshold, not allowed to run the minute out
    assert s["totals"]["incident_calls"] <= LOOP_CALLS_PER_MIN
    assert any(ev["kind"] == "contain" for ev in s["events"])


def test_loop_containment_costs_far_less_than_letting_it_run():
    e = build()
    run_minutes(e, 30)
    e.inject("agent_loop")
    run_minutes(e, 5)
    contained = e.snapshot()["totals"]["incident_inr"]
    # an hour of the same loop, unmanaged
    from backend.core.pricing import compute_cost
    from backend.domain.workload import BASELINE_MODEL, LOOP_IN, LOOP_OUT

    unmanaged_hour = compute_cost(BASELINE_MODEL, LOOP_IN, LOOP_OUT).inr * LOOP_CALLS_PER_MIN * 60
    assert contained < unmanaged_hour * 0.05


def test_prompt_bloat_raises_context_and_spend():
    quiet = build()
    run_minutes(quiet, 60)
    base_rate = quiet.snapshot()["managed_inr"] / 60

    bloated = build()
    run_minutes(bloated, 30)
    bloated.inject("prompt_bloat")
    before = bloated.snapshot()["managed_inr"]
    run_minutes(bloated, 30)
    after_rate = (bloated.snapshot()["managed_inr"] - before) / 30
    assert after_rate > base_rate


def test_retry_storm_produces_failed_calls_that_cost_money():
    e = build()
    run_minutes(e, 20)
    before = e.snapshot()["totals"]["incident_inr"]
    e.inject("retry_storm")
    run_minutes(e, 10)
    assert e.snapshot()["totals"]["incident_inr"] > before


# ----------------------------------------------------------------- guardrails
def test_loop_detector_beats_burn_rate_for_a_loop():
    """At a realistic budget the signature detector kills the loop before the
    rate ever moves. That ordering is the design, not an accident: burn-rate
    alerting is the net for incidents with no signature."""
    e = build()
    run_minutes(e, 20)
    e.inject("agent_loop")
    run_minutes(e, 3)
    events = e.snapshot()["events"]
    contained = next(i for i, ev in enumerate(events) if ev["kind"] == "contain")
    alerts = [i for i, ev in enumerate(events) if ev["kind"] == "alert"]
    # events are newest-first, so a later index means it happened earlier
    assert all(i > contained for i in alerts)


def test_burn_rate_catches_the_loop_when_the_signature_detector_is_blind():
    """The independent-safety-net claim, tested: blind the loop detector and
    the burn monitor must still fire."""
    from backend.core.guardrails import LoopDetection

    e = build()
    e.guard.detect_loop = lambda *a, **k: LoopDetection(detected=False)  # type: ignore[method-assign]
    run_minutes(e, 20)
    e.inject("agent_loop")
    run_minutes(e, 6)
    s = e.snapshot()
    assert any(ev["kind"] == "alert" for ev in s["events"]), "burn monitor missed an unsigned incident"
    assert s["burn"]["any_breaching"] or s["circuit_breakers"]


def test_low_budget_degrades_rather_than_stopping_work():
    e = build(monthly_budget_inr=20_000.0)
    run_minutes(e, 120)
    s = e.snapshot()
    assert s["budget"]["remaining_pct"] < 100
    assert s["totals"]["outcomes_ok"] > 0, "work must still complete under a tight budget"


def test_clear_breakers_reopens_the_scope():
    e = build()
    run_minutes(e, 20)
    e.inject("agent_loop")
    run_minutes(e, 2)
    assert e.snapshot()["circuit_breakers"]
    e.clear_breakers()
    assert not e.snapshot()["circuit_breakers"]


def test_budget_change_is_reflected_in_the_monitor():
    e = build()
    e.set_budget(1_000_000.0)
    assert e.monitor.budget.monthly_inr == 1_000_000.0
    assert e.snapshot()["budget"]["monthly_budget_inr"] == 1_000_000.0


# -------------------------------------------------------------- burn helper
def test_live_state_reports_every_window():
    m = BurnRateMonitor(Budget("global", "all", 720.0, 30))
    state = m.live_state([1 / 60.0] * 120)
    assert len(state["windows"]) == len(m.windows)
    assert not state["any_breaching"]


def test_live_state_flags_a_breach():
    m = BurnRateMonitor(Budget("global", "all", 720.0, 30))
    state = m.live_state([1 / 60.0] * 60 + [5.0] * 10)
    assert state["any_breaching"]
    assert state["worst_multiplier"] > 14.4


# ------------------------------------------------------------------ snapshot
def test_snapshot_is_json_serialisable():
    import json

    e = build()
    run_minutes(e, 20)
    json.dumps(e.snapshot(), default=str)


def test_policy_covers_every_task_type():
    e = build()
    run_minutes(e, 60)
    policy = e.snapshot()["policy"]
    assert set(policy) == set(QUALITY)


def test_every_workflow_can_be_generated():
    e = build()
    run_minutes(e, 400)
    seen = {r["step"] for r in e.snapshot()["recent_routes"]}
    assert seen, "no routing decisions recorded"
    all_steps = {s[0] for wf in WORKFLOWS.values() for s in wf[1]}
    assert seen <= all_steps
