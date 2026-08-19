"""The live scenario, in a terminal.

Same engine as the Live ops page, driven on its real background thread, with
the incidents injected on a schedule so the whole scenario can be rehearsed —
or presented — without a browser.

    python scripts/live_demo.py                 # 90s, scripted incidents
    python scripts/live_demo.py --seconds 180
    python scripts/live_demo.py --manual        # no scripted incidents; inject by API
    python scripts/live_demo.py --no-persist    # do not write into the ledger

Nothing is pre-computed: every figure printed is read from the engine as it
runs, and the incident outcomes depend on what the guardrails actually do.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from backend.core.live import LiveEngine  # noqa: E402
from backend.core.pricing import fmt_inr  # noqa: E402

BAR = "─" * 96
SEV = {"critical": "!!", "warning": " !", "success": " +", "info": "  "}


def status_line(s: Dict) -> str:
    t = s["totals"]
    burn = s["burn"]["worst_multiplier"]
    flag = "BREACH" if s["burn"]["any_breaching"] else "ok"
    return (
        f"  {s['sim_clock']}  "
        f"calls {t['calls']:>6,}  "
        f"outcomes {t['outcomes_ok']:>4,}  "
        f"spend {fmt_inr(s['managed_inr']):>12}  "
        f"shadow {fmt_inr(s['shadow_inr']):>12}  "
        f"saved {fmt_inr(s['saved_inr']):>12} ({s['saved_pct']:>5.1f}%)  "
        f"burn {burn:>6.1f}x {flag}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=90, help="wall-clock duration")
    ap.add_argument("--speed", type=float, default=1.0, help="simulated minutes per real second")
    ap.add_argument("--manual", action="store_true", help="do not inject incidents on a schedule")
    ap.add_argument("--no-persist", action="store_true", help="do not write to the ledger")
    ap.add_argument("--budget", type=float, default=None, help="monthly budget in INR")
    args = ap.parse_args()

    engine = LiveEngine(
        minutes_per_second=args.speed,
        monthly_budget_inr=args.budget,
        persist=not args.no_persist,
    )

    # scripted beats, as (elapsed_seconds, action)
    script: List = [] if args.manual else [
        (int(args.seconds * 0.35), "agent_loop"),
        (int(args.seconds * 0.65), "prompt_bloat"),
    ]

    print("\n" + BAR)
    print("  TOKENOPS - LIVE SCENARIO")
    print(f"  {args.speed:g} simulated minute per real second  ·  "
          f"monthly budget {fmt_inr(engine.monthly_budget_inr)}  ·  "
          f"{args.seconds}s run")
    print(BAR)
    if script:
        for at, kind in script:
            print(f"  scheduled: t+{at}s  inject {kind}")
    else:
        print("  manual mode: POST /api/live/inject to trigger incidents")
    print(BAR + "\n")

    engine.start()
    seen_events = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            elapsed = time.time() - t0
            for at, kind in list(script):
                if elapsed >= at:
                    engine.inject(kind)
                    script.remove((at, kind))

            s = engine.snapshot()
            events = list(reversed(s["events"]))
            if len(events) > seen_events:
                for e in events[seen_events:]:
                    print(f"  {SEV.get(e['severity'], '  ')} [{e['ts']}] {e['message']}")
                seen_events = len(events)
            print(status_line(s), end="\r", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        engine.stop()

    s = engine.snapshot()
    t = s["totals"]
    print("\n" + BAR)
    print("  RESULT")
    print(BAR)
    rows = [
        ("Simulated time", f"{s['sim_minute']} minutes"),
        ("Sessions / calls", f"{t['sessions']:,} / {t['calls']:,}"),
        ("Managed spend", fmt_inr(s["managed_inr"])),
        ("Unmanaged shadow, same traffic", fmt_inr(s["shadow_inr"])),
        ("Saved", f"{fmt_inr(s['saved_inr'])}  ({s['saved_pct']:.1f}%)"),
        ("Cost per outcome", f"{fmt_inr(s['cost_per_outcome_inr'])} vs "
                             f"{fmt_inr(s['shadow_cost_per_outcome_inr'])} unmanaged"),
        ("Quality", f"{s['mean_quality']:.3f} vs {s['shadow_mean_quality']:.3f} unmanaged"),
        ("Cache hit rate (live window)", f"{s['cache_hit_rate']:.1%}"),
        ("Cascade escalation rate", f"{s['escalation_rate']:.1%}"),
        ("Incident spend", f"{fmt_inr(t['incident_inr'])} over {t['incident_calls']:,} calls"),
        ("Degraded / blocked calls", f"{t['degraded_calls']:,} / {t['blocked_calls']:,}"),
        ("Budget remaining", f"{s['budget']['remaining_pct']:.0f}%"),
    ]
    for label, value in rows:
        print(f"    {label.ljust(34)} {value}")

    print("\n    Policy the router settled on:")
    for task, route in s["policy"].items():
        print(f"      {task.ljust(16)} {route}")

    if s["circuit_breakers"]:
        print("\n    Circuit breakers open:")
        for scope, reason in s["circuit_breakers"].items():
            print(f"      {scope}: {reason}")
    print(BAR + "\n")


if __name__ == "__main__":
    main()
