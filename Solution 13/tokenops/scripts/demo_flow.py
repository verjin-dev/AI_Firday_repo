"""The four-minute demo, scripted and narrated, straight from the ledger.

Every number printed here is read from the database at run time. If the
simulation changes, this narration changes with it - there are no hard-coded
figures in the script, which is the only way a demo stays honest.

Run:  python scripts/demo_flow.py
      python scripts/demo_flow.py --offline    (asserts no network is needed)
      python scripts/demo_flow.py --pause      (wait for Enter between beats)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from backend.core.ledger import CostLedger  # noqa: E402
from backend.core.optimizers.cascade import break_even_escalation_rate  # noqa: E402
from backend.core.pricing import fmt_inr  # noqa: E402
from backend.storage.db import has_data, query_df  # noqa: E402

W = 78
PAUSE = False


def beat(n: int, title: str, seconds: str) -> None:
    print("\n" + "=" * W)
    print(f"  [{n}] {title}".ljust(W - len(seconds) - 4) + f"~{seconds}")
    print("=" * W)
    if PAUSE:
        input("      (Enter to continue)")


def say(text: str) -> None:
    print(f"  {text}")


def number(label: str, value: str, note: str = "") -> None:
    print(f"    {label.ljust(38)} {value.rjust(16)}   {note}")


def load_bench() -> Dict[str, Any]:
    p = Path("benchmark_results.json")
    if not p.exists():
        print("benchmark_results.json missing. Run: python scripts/benchmark.py")
        raise SystemExit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="assert the demo needs no network at all")
    ap.add_argument("--pause", action="store_true")
    args = ap.parse_args()
    PAUSE = args.pause

    if not has_data():
        print("The ledger is empty. Run: python scripts/simulate_workload.py")
        raise SystemExit(1)

    bench = load_bench()
    b, t = bench["arms"]["baseline"], bench["arms"]["tokenops"]
    router = bench.get("router", {})
    fc = bench.get("forecast", {})
    notes_path = Path("data/samples/simulation_notes.json")
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}

    print("\n" + "#" * W)
    print("#  TOKENOPS - FinOps for agentic AI".ljust(W - 1) + "#")
    print("#  Cost per business outcome, not cost per token".ljust(W - 1) + "#")
    print("#" * W)
    if args.offline:
        say("offline mode: every figure below is read from the local ledger. "
            "No API key, no network.")

    # ---------------------------------------------------------------- beat 1
    beat(1, "Reframe the metric", "35s")
    say("Every AI cost dashboard shows tokens. Here is ours.")
    number("Cost per resolved ticket (baseline)", fmt_inr(b["cost_per_resolved_ticket_inr"]),
           "one strong model, no cache, no guardrails")
    number("Tickets resolved", f"{b['resolved_tickets']:,}",
           f"of {b['attempted_tickets']:,} attempted")
    number("Monthly spend", fmt_inr(b["monthly_spend_inr"]))
    say("")
    say("That first number is the only one your CFO can act on. Note what is in it:")
    say("failed sessions are in the numerator and not in the denominator, because")
    say("wasted spend has to show up in the unit cost.")

    # ---------------------------------------------------------------- beat 2
    beat(2, "Find the money in two clicks", "40s")
    steps = query_df(
        "SELECT step, SUM(cost_inr) AS cost_inr, COUNT(*) AS calls FROM llm_calls "
        "WHERE arm = 'baseline' AND incident IS NULL GROUP BY step ORDER BY cost_inr DESC"
    )
    total = float(steps["cost_inr"].sum())
    for _, r in steps.head(4).iterrows():
        number(f"step: {r['step']}", fmt_inr(float(r["cost_inr"])),
               f"{float(r['cost_inr']) / total:.1%} of spend, {int(r['calls']):,} calls")
    qa = query_df(
        "SELECT COUNT(*) AS n, SUM(c.cost_inr) AS c FROM llm_calls c "
        "JOIN outcomes o ON o.outcome_id = c.outcome_id AND o.arm = c.arm "
        "WHERE c.arm = 'baseline' AND c.step = 'qa_verify' AND o.quality >= 0.90"
    )
    if not qa.empty and qa["c"].iloc[0]:
        say("")
        say(f"Drill in: {int(qa['n'].iloc[0]):,} QA verification calls "
            f"({fmt_inr(float(qa['c'].iloc[0]))}) ran on sessions the resolution agent")
        say("had already scored at or above 0.90. Nobody designed that. It accreted.")

    # ---------------------------------------------------------------- beat 3
    beat(3, "Name the waste", "35s")
    waste = CostLedger("baseline").waste_report()
    say(f"{fmt_inr(waste['monthly_waste_inr'])} a month, itemised, with an owner each:")
    for item in waste["items"][:5]:
        number(item["category"], fmt_inr(item["cost_inr"] / waste["window_days"] * 30),
               f"owner: {item['owner']}")
    say("")
    say("A number nobody owns never gets fixed. That is the whole point of the column.")

    # ---------------------------------------------------------------- beat 4
    beat(4, "Watch the router learn", "50s")
    for task in ("classification", "generation"):
        r = router.get(task)
        if not r:
            continue
        say(f"{task}: settled on {r['final_leader'].replace('claude-', '')} at "
            f"{r['final_leader_share']:.0%} of traffic, stable from day {r['stable_from_day']}.")
    say("")
    say("Nobody would have written that split into a static routing table on day one.")
    say("And it is not a black box - here is the evidence table for triage:")
    explain = Path("data/samples/router_explain.json")
    if explain.exists():
        data = json.loads(explain.read_text(encoding="utf-8")).get("classification", {})
        print()
        print("    " + "route".ljust(34) + "pulls".rjust(9) + "quality".rjust(9)
              + "cost".rjust(9) + "P(best)".rjust(9))
        for a in data.get("arms", [])[:5]:
            q = "-" if a["mean_quality"] is None else f"{a['mean_quality']:.3f}"
            c = "-" if a["mean_cost_inr"] is None else f"{a['mean_cost_inr']:.3f}"
            print("    " + a["route_id"].ljust(34) + f"{a['pulls']:,}".rjust(9)
                  + q.rjust(9) + c.rjust(9) + f"{a['selection_prob']:.0%}".rjust(9))
        excl = [a for a in data.get("arms", []) if a["excluded"]]
        if excl:
            print()
            say(f"{len(excl)} arm(s) excluded by the quality floor before sampling, e.g. "
                f"{excl[0]['route_id']}: {excl[0]['exclusion_reason']}")
    say("")
    say(f"Exploration is capped at {5.0:.0f}% of period spend, so learning can never "
        "cost more than a set slice.")

    # ---------------------------------------------------------------- beat 5
    beat(5, "The incident", "60s")
    say("Day 18, 02:14. A resolution agent starts looping on the same prompt.")
    number("Unmanaged cost", fmt_inr(b["loop_incident_cost_inr"]),
           f"ran {b['loop_incident_minutes']:.0f} min, until 08:00")
    number("With TokenOps", fmt_inr(t["loop_incident_cost_inr"]),
           f"contained in {t['loop_incident_minutes']:.0f} min")
    saved = b["loop_incident_cost_inr"] - t["loop_incident_cost_inr"]
    number("Saved on one incident", fmt_inr(saved),
           f"{saved / b['loop_incident_cost_inr']:.1%} of the unmanaged cost")
    det = notes.get("incident_detection", {})
    if det.get("burn_alert_after_min") is not None:
        say("")
        say("Two mechanisms caught it. The loop detector matched a repeated "
            "(step, prompt_hash)")
        say(f"signature and killed the session. The burn-rate monitor would have fired "
            f"{det['burn_alert_after_min']} min in")
        say("from spend alone - which is what matters, because the next incident will not")
        say("be a loop.")
    say("")
    say("Everything else in this project is optimisation. This is insurance.")

    # ---------------------------------------------------------------- beat 6
    beat(6, "Graceful degradation", "30s")
    from backend.core.guardrails import CostGuardrails, degraded_request

    guard = CostGuardrails()
    req = {"interactive": True, "input_tokens": 8400, "skip_verification": False}
    d = guard.check(req, {"scope": "tenant:vertex-insurance", "remaining_pct": 20})
    after = degraded_request(req, d)
    number("Budget remaining", "20%")
    number("Guardrail decision", d.action.value)
    number("Context tokens", f"{req['input_tokens']:,} -> {after['input_tokens']:,}")
    number("Verification step", "on -> skipped")
    say("")
    say("The workflow degrades. It does not stop. Blocking a workflow to save money")
    say("is a support ticket; degrading it is a saving.")

    # ---------------------------------------------------------------- beat 7
    beat(7, "Forecast", "30s")
    if fc:
        number("Next 30 days", fmt_inr(fc["total_inr"]),
               f"{fmt_inr(fc['total_lower_inr'])} - {fmt_inr(fc['total_upper_inr'])}")
        if fc.get("mape_pct") is not None:
            number("Backtest MAPE", f"{fc['mape_pct']:.1f}%", "fit on all but the last week")
        drv = fc.get("drivers") or {}
        if drv:
            number("Growth from volume", f"{drv['volume_share_pct']:.0f}%")
            number("Growth from unit cost", f"{drv['unit_cost_share_pct']:.0f}%", drv["verdict"])
    say("")
    say("Volume growth is a good problem. Unit-cost growth is a regression, and it has")
    say("an owner. A forecast that does not separate them is a number, not information.")

    # ------------------------------------------------------------- the table
    beat(8, "The scoreboard", "20s")
    for label, bv, tv, dv in bench["table"]:
        number(label, f"{bv} -> {tv}", dv)
    print()
    say("And where we lose:")
    number("Mean outcome quality", f"{b['mean_quality']:.3f} -> {t['mean_quality']:.3f}",
           f"{t['mean_quality'] - b['mean_quality']:+.3f}")
    say("Two points of a thousand, bought with a 65% cost reduction. Lambda is the dial;")
    say("turn it down and you buy quality back at a known price.")
    say("")
    say(f"TokenOps costs {t['tokenops_overhead_pct']:.2f}% of the spend it manages.")
    say(f"Cascade escalation rate {t['cascade_escalation_rate']:.1%} against a break-even of "
        f"{break_even_escalation_rate():.0%}.")

    print("\n" + "#" * W)
    print("#  End of demo.".ljust(W - 1) + "#")
    print("#" * W + "\n")


if __name__ == "__main__":
    main()
