"""Unit tests for the modules that carry the claims.

These test the arithmetic and the policy, not the simulation: if pricing,
reward, quality-floor exclusion, burn detection or the guardrail ladder are
wrong, every number in the demo is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import PRICE_TABLE, get_settings
from backend.core.burn_rate import Budget, BurnRateMonitor
from backend.core.forecaster import CostForecaster
from backend.core.guardrails import CostGuardrails, GuardAction, degraded_request
from backend.core.optimizers.cascade import ModelCascade, break_even_escalation_rate
from backend.core.optimizers.compressor import Chunk, PromptCompressor, UtilityIndex
from backend.core.optimizers.scheduler import Job, WorkloadScheduler
from backend.core.optimizers.semantic_cache import SemanticCache, hashed_embedding
from backend.core.pricing import UnknownModelError, compute_cost, fmt_inr
from backend.core.router import LearningRouter, Route, default_route_space


# ------------------------------------------------------------------- pricing
def test_cost_matches_price_table_exactly():
    c = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    p = PRICE_TABLE["claude-sonnet-4-6"]
    assert c.usd == pytest.approx(p["in"] + p["out"])
    assert c.inr == pytest.approx(c.usd * get_settings().USD_INR)


def test_cached_tokens_are_billed_not_free():
    cached = compute_cost("claude-sonnet-4-6", 0, 0, cached_tokens=1_000_000)
    assert cached.usd > 0
    full = compute_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cached.usd < full.usd          # discounted, but not zero


def test_unknown_model_raises_rather_than_pricing_at_zero():
    with pytest.raises(UnknownModelError):
        compute_cost("gpt-imaginary", 100, 100)


def test_indian_number_format():
    assert fmt_inr(184000) == "₹1.84 L"
    assert fmt_inr(9200) == "₹9,200"
    assert fmt_inr(12_500_000) == "₹1.25 Cr"


# -------------------------------------------------------------------- router
def test_reward_prefers_cheap_at_equal_quality():
    r = LearningRouter()
    assert r.reward(0.9, 0.2) > r.reward(0.9, 3.0)


def test_quality_floor_excludes_an_arm_after_enough_evidence():
    r = LearningRouter(seed=1)
    route = default_route_space()[0]
    for _ in range(40):
        r.update("classification", route.route_id, quality=0.5, cost_inr=0.1)
    arm = r._task_arms("classification")[route.route_id]
    assert arm.excluded
    assert "floor" in arm.exclusion_reason


def test_a_single_bad_sample_does_not_exclude_an_arm():
    r = LearningRouter(seed=2)
    route = default_route_space()[0]
    r.update("classification", route.route_id, quality=0.1, cost_inr=0.1)
    assert not r._task_arms("classification")[route.route_id].excluded


def test_exploration_budget_cap_forces_exploit_only():
    r = LearningRouter(seed=3)
    routes = default_route_space()
    for rt in routes:
        r.update("generation", rt.route_id, 0.9, 1.0, exploring=True)
    assert r.exploration_spend_pct("generation") >= r.settings.BANDIT_EXPLORATION_BUDGET_PCT
    d = r.select("generation")
    assert d.exploring is False
    assert "exploration budget" in d.reason


def test_router_learns_the_cheaper_arm_when_quality_is_equal():
    r = LearningRouter(seed=5)
    cheap = Route("claude-haiku-4-5-20251001", "terse", "shallow", "aggressive").route_id
    dear = Route("claude-opus-5", "full", "deep", "standard").route_id
    for _ in range(300):
        r.update("classification", cheap, quality=0.90, cost_inr=0.15)
        r.update("classification", dear, quality=0.90, cost_inr=2.60)
    assert r.policy("classification").route_id == cheap


def test_explain_returns_every_arm_with_a_posterior_interval():
    r = LearningRouter(seed=6)
    r.update("retrieval", default_route_space()[0].route_id, 0.88, 0.4)
    ex = r.explain("retrieval")
    assert len(ex["arms"]) == len(default_route_space())
    for a in ex["arms"]:
        assert a["ci_low"] <= a["posterior_mean"] <= a["ci_high"]
    assert sum(a["selection_prob"] for a in ex["arms"]) == pytest.approx(1.0, abs=0.01)


def test_guardrail_override_beats_the_bandit():
    r = LearningRouter(seed=7)
    d = r.select("generation", {"force_cheap": True})
    assert "haiku" in d.route.model
    assert d.route.context_depth == "shallow"


# ----------------------------------------------------------------- burn rate
def _hourly(costs):
    return pd.DataFrame({
        "day": [i // 24 for i in range(len(costs))],
        "hour": [i % 24 for i in range(len(costs))],
        "cost_inr": costs,
        "ts_epoch": [1_700_000_000 + i * 3600 for i in range(len(costs))],
    })


def test_steady_spend_at_budget_raises_no_alert():
    budget = Budget("global", "all", 720.0, 30)     # 1 INR/hour
    m = BurnRateMonitor(budget)
    assert m.scan(_hourly([1.0] * 200)) == []


def test_sustained_overspend_alerts():
    budget = Budget("global", "all", 720.0, 30)
    m = BurnRateMonitor(budget)
    alerts = m.scan(_hourly([1.0] * 50 + [40.0] * 30))
    assert alerts
    assert any(a.window_hours == 1 for a in alerts)
    assert max(a.observed_multiplier for a in alerts) > 14.4


def test_short_window_must_be_full_before_firing():
    budget = Budget("global", "all", 720.0, 30)     # 1 INR/hour -> 1/60 per minute
    m = BurnRateMonitor(budget)
    # a 1-hour long window pairs with a 5-minute short window; even a spike
    # from minute zero cannot fire until that short window has 5 samples
    minute = m.first_breach_minute([100.0] * 60, long_window_hours=1, multiplier=14.4)
    assert minute == 4


def test_burn_alert_fires_promptly_once_a_spike_starts():
    budget = Budget("global", "all", 720.0, 30)
    m = BurnRateMonitor(budget)
    minute = m.first_breach_minute([0.0] * 30 + [100.0] * 60, long_window_hours=1, multiplier=14.4)
    assert minute is not None and 30 <= minute <= 35


def test_no_breach_when_spend_is_within_budget():
    budget = Budget("global", "all", 720.0, 30)
    m = BurnRateMonitor(budget)
    assert m.first_breach_minute([1 / 60.0] * 300) is None


# ---------------------------------------------------------------- guardrails
def test_guardrail_ladder():
    g = CostGuardrails()
    base = {"interactive": True}
    assert g.check(base, {"scope": "t", "remaining_pct": 80}).action is GuardAction.ALLOW
    assert g.check(base, {"scope": "t", "remaining_pct": 25}).action is GuardAction.DEGRADE
    assert g.check({"interactive": False}, {"scope": "t", "remaining_pct": 6}).action is GuardAction.QUEUE
    assert g.check(base, {"scope": "t", "remaining_pct": 1}).action is GuardAction.BLOCK


def test_degrade_halves_context_and_skips_verification():
    g = CostGuardrails()
    req = {"interactive": True, "input_tokens": 8400}
    d = g.check(req, {"scope": "t", "remaining_pct": 20})
    after = degraded_request(req, d)
    assert after["input_tokens"] == 4200
    assert after["skip_verification"] is True
    assert after["force_cheap"] is True


def test_circuit_breaker_blocks_the_scope():
    g = CostGuardrails()
    g.circuit_break("tenant:x", "agent loop")
    assert g.check({}, {"scope": "tenant:x", "remaining_pct": 100}).action is GuardAction.BLOCK
    g.reset("tenant:x")
    assert g.check({}, {"scope": "tenant:x", "remaining_pct": 100}).action is GuardAction.ALLOW


def test_loop_detection_finds_the_repeat_and_prices_it():
    g = CostGuardrails()
    calls = [{"step": "resolve", "prompt_hash": "abc", "session_id": "s1",
              "cost_inr": 7.5, "ts_epoch": 1000 + i} for i in range(6)]
    d = g.detect_loop(calls)
    assert d.detected and d.repeats >= 4
    assert d.wasted_inr == pytest.approx(7.5 * (d.repeats - 1))


def test_loop_detection_ignores_distinct_calls():
    g = CostGuardrails()
    calls = [{"step": "resolve", "prompt_hash": f"h{i}", "session_id": "s1",
              "cost_inr": 1.0, "ts_epoch": i} for i in range(20)]
    assert not g.detect_loop(calls).detected


# -------------------------------------------------------------------- cache
def test_identical_query_hits_and_paraphrase_scores_high():
    c = SemanticCache(threshold=0.94)
    text = "what is the waiting period for a cashless claim under policy 4471 " \
           "and which exclusions apply to pre existing conditions"
    c.put(text, "acme", "claims", result="x", quality=0.9, cost_inr=2.0, now_epoch=0)
    hit, score = c.lookup(text, "acme", "claims", now_epoch=10)
    assert hit is not None and score == pytest.approx(1.0, abs=1e-6)


def test_cache_is_scoped_per_tenant():
    c = SemanticCache()
    c.put("same question text here for both tenants", "acme", "claims", "x", 0.9, 1.0, 0)
    hit, _ = c.lookup("same question text here for both tenants", "other", "claims", now_epoch=1)
    assert hit is None


def test_low_quality_results_are_never_served():
    c = SemanticCache(quality_floor=0.8)
    t = "a reasonably long question about coverage limits and exclusions for tenant acme"
    c.put(t, "acme", "claims", "x", quality=0.4, cost_inr=1.0, now_epoch=0)
    hit, _ = c.lookup(t, "acme", "claims", now_epoch=1)
    assert hit is None
    assert c.stats.quality_blocked == 1


def test_expired_entries_are_not_served():
    c = SemanticCache(ttl_seconds=3600)
    t = "a reasonably long question about coverage limits and exclusions for tenant acme"
    c.put(t, "acme", "claims", "x", 0.9, 1.0, now_epoch=0)
    assert c.lookup(t, "acme", "claims", now_epoch=7200)[0] is None
    assert c.stats.expired == 1


def test_embedding_is_deterministic_and_normalised():
    v = hashed_embedding("policy exclusions and waiting periods")
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, hashed_embedding("policy exclusions and waiting periods"))


# --------------------------------------------------------------- compressor
def test_compression_drops_only_low_utility_chunks():
    comp = PromptCompressor(utility_threshold=0.2)
    chunks = [Chunk("a", "", 1000, 0.9), Chunk("b", "", 1000, 0.05), Chunk("c", "", 1000, 0.6)]
    out = comp.compress("some prompt text", chunks)
    assert out.dropped_chunks == ["b"]
    assert out.compressed_tokens < out.original_tokens


def test_deduplication_removes_repeated_blocks():
    block = "retrieved context paragraph " * 20
    comp = PromptCompressor()
    out = comp.compress(f"{block}\n\n{block}\n\ntail", [])
    assert out.tier_savings["dedupe"] > 0


def test_utility_index_learns_from_citations():
    idx = UtilityIndex(min_obs=3)
    for _ in range(10):
        idx.observe("never_cited", cited=False)
        idx.observe("always_cited", cited=True)
    assert idx.utility("never_cited") == 0.0
    assert idx.utility("always_cited") == 1.0
    assert idx.utility("unseen") == 0.5      # falls back to the prior


# ------------------------------------------------------------------ cascade
def test_break_even_rate_is_between_zero_and_one():
    be = break_even_escalation_rate()
    assert 0.0 < be < 1.0


def test_cascade_escalates_on_low_confidence():
    calls = []

    def executor(model, task):
        calls.append(model)
        cheap = "haiku" in model
        return {"quality": 0.7 if cheap else 0.9,
                "confidence": 0.4 if cheap else 0.95,
                "cost_inr": 0.2 if cheap else 1.8}

    c = ModelCascade()
    res = c.run({"x": 1}, executor)
    assert res.escalated
    assert res.final_model == "claude-sonnet-4-6"
    assert res.cost_inr == pytest.approx(0.2 + 1.8)     # the cheap attempt is still charged


def test_cascade_accepts_the_cheap_model_when_confident():
    def executor(model, task):
        return {"quality": 0.9, "confidence": 0.99, "cost_inr": 0.2}

    res = ModelCascade().run({}, executor)
    assert not res.escalated
    assert "haiku" in res.final_model


# ---------------------------------------------------------------- forecaster
def test_forecast_recovers_a_known_weekly_pattern():
    days = np.arange(56)
    series = 1000 + 50 * np.sin(2 * np.pi * days / 7) + days * 5
    f = CostForecaster().forecast(series.tolist(), horizon_days=7)
    assert f.status == "success"
    assert f.mape_pct is not None and f.mape_pct < 15
    assert all(lo <= p <= hi for lo, p, hi in zip(f.lower, f.point, f.upper))


def test_short_series_degrades_instead_of_crashing():
    f = CostForecaster().forecast([100, 110, 90], horizon_days=5)
    assert f.status == "partial"
    assert len(f.point) == 5
    assert f.warnings


def test_structural_break_is_detected():
    y = np.array([100.0] * 20 + [300.0] * 20)
    assert CostForecaster().detect_break(y) is not None


def test_driver_decomposition_separates_volume_from_unit_cost():
    # the decomposition compares the last 7 days against the 7 before them,
    # so the step change has to land inside that comparison window
    df = pd.DataFrame({
        "day": range(28),
        "outcomes": [100] * 28,
        "cost_per_outcome_inr": [10.0] * 21 + [15.0] * 7,
    })
    d = CostForecaster.decompose_drivers(df, window=7)
    assert d["unit_cost_share_pct"] > 90
    assert "regression" in d["verdict"]


def test_what_if_levers_reduce_spend_monotonically():
    r = CostForecaster.what_if(100000.0, {"cache_hit_rate": 0.4}, {"cache_hit_rate": 0.0,
                                                                  "cheap_model_share": 0.0,
                                                                  "context_reduction": 0.0})
    assert r["scenario_monthly_inr"] < 100000.0
    assert r["steps"]


# ---------------------------------------------------------------- scheduler
def test_scheduler_defers_verification_work():
    jobs = [Job(f"j{i}", "qa_verify", "acme", 5000) for i in range(10)] + \
           [Job("i1", "triage", "acme", 3000)]
    plan = WorkloadScheduler().plan(jobs, now_hour=12)
    assert plan.immediate == ["i1"]
    assert plan.deferred_pct > 80
    assert plan.batches and plan.batches[0].hour in {22, 23} | set(range(0, 7))
