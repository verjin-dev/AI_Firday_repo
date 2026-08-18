"""Cost SLO burn-rate alerting, borrowed straight from SRE error budgets.

A monthly cost budget is a *rate*. Alerting on "80% of budget consumed" tells
you on the 28th. Alerting on burn rate over multiple windows tells you in
minutes, and the multi-window scheme is what keeps it quiet: a 14.4x spike
for five minutes is not an incident, a 6x burn sustained over six hours is.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import get_settings

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Budget:
    scope: str                 # "global" | "tenant" | "team"
    scope_value: str
    monthly_inr: float
    period_days: int = 30

    @property
    def hourly_inr(self) -> float:
        return self.monthly_inr / (self.period_days * 24.0)


@dataclass
class BurnAlert:
    ts_epoch: float
    day: int
    hour: int
    scope: str
    window_hours: int
    threshold_multiplier: float
    observed_multiplier: float
    window_spend_inr: float
    severity: str
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BurnState:
    alerts: List[BurnAlert] = field(default_factory=list)
    fired_windows: Dict[int, float] = field(default_factory=dict)


class BurnRateMonitor:
    """Stateless over a series: `evaluate_at` scores one point in time,
    `scan` walks an hourly series and returns every alert that would have
    fired, with the minute it fired."""

    def __init__(self, budget: Budget, windows: Optional[List] = None) -> None:
        self.settings = get_settings()
        self.budget = budget
        self.windows = windows or self.settings.BURN_RATE_WINDOWS

    @staticmethod
    def _severity(window_hours: int, observed: float, threshold: float) -> str:
        if observed >= threshold * 1.5 or window_hours <= 1:
            return "critical"
        if observed >= threshold:
            return "critical" if window_hours <= 6 else "warning"
        return "info"

    def evaluate_at(self, hourly: pd.DataFrame, at_index: int) -> List[BurnAlert]:
        """`hourly` needs columns: day, hour, cost_inr (one row per clock hour,
        gaps allowed). `at_index` is the row being evaluated."""
        alerts: List[BurnAlert] = []
        row = hourly.iloc[at_index]
        for window_hours, multiplier in self.windows:
            lo = max(0, at_index - window_hours + 1)
            window = hourly.iloc[lo : at_index + 1]
            hours_covered = max(len(window), 1)
            spend = float(window["cost_inr"].sum())
            observed_rate = spend / hours_covered
            observed_multiplier = observed_rate / self.budget.hourly_inr if self.budget.hourly_inr else 0.0
            if observed_multiplier >= multiplier and hours_covered >= min(window_hours, 1):
                sev = self._severity(window_hours, observed_multiplier, multiplier)
                alerts.append(
                    BurnAlert(
                        ts_epoch=float(row.get("ts_epoch", 0.0) or 0.0),
                        day=int(row["day"]),
                        hour=int(row["hour"]),
                        scope=f"{self.budget.scope}:{self.budget.scope_value}",
                        window_hours=window_hours,
                        threshold_multiplier=multiplier,
                        observed_multiplier=round(observed_multiplier, 2),
                        window_spend_inr=spend,
                        severity=sev,
                        message=(
                            f"{window_hours}h burn rate {observed_multiplier:.1f}x budgeted "
                            f"(threshold {multiplier}x) - {spend:,.0f} INR in {hours_covered}h"
                        ),
                    )
                )
        return alerts

    def scan(self, hourly: pd.DataFrame, dedup_hours: int = 6) -> List[BurnAlert]:
        """Walk the whole series. Within `dedup_hours` the same window is not
        re-alerted, so one incident produces one page, not forty."""
        hourly = hourly.sort_values(["day", "hour"]).reset_index(drop=True)
        out: List[BurnAlert] = []
        last_fired: Dict[int, int] = {}
        for i in range(len(hourly)):
            abs_hour = int(hourly.iloc[i]["day"]) * 24 + int(hourly.iloc[i]["hour"])
            for alert in self.evaluate_at(hourly, i):
                prev = last_fired.get(alert.window_hours)
                if prev is not None and abs_hour - prev < dedup_hours:
                    continue
                last_fired[alert.window_hours] = abs_hour
                out.append(alert)
        return out

    def first_alert_for_incident(self, hourly: pd.DataFrame, start_day: int,
                                 start_hour: int) -> Optional[BurnAlert]:
        """Detection latency for a known incident start - the number that
        actually matters on stage."""
        alerts = self.scan(hourly, dedup_hours=1)
        start_abs = start_day * 24 + start_hour
        for a in sorted(alerts, key=lambda x: (x.day, x.hour, x.window_hours)):
            if a.day * 24 + a.hour >= start_abs:
                return a
        return None

    def first_breach_minute(
        self,
        minute_costs: Sequence[float],
        long_window_hours: int = 1,
        multiplier: float = 14.4,
        short_window_fraction: float = 1 / 12.0,
    ) -> Optional[int]:
        """Detection latency in minutes, using the standard multiwindow rule.

        A long window alone is slow to fire: an hour-long window needs an hour
        of bad spend to reach its own threshold. The multiwindow burn alert
        pairs each long window with a short window of one twelfth its length,
        and requires BOTH to be over threshold. The short window is what makes
        detection fast; the long window is what stops it being noisy.
        """
        long_min = int(long_window_hours * 60)
        short_min = max(1, int(long_min * short_window_fraction))
        budget_per_min = self.budget.hourly_inr / 60.0
        if budget_per_min <= 0:
            return None
        threshold = multiplier * budget_per_min
        costs = np.asarray(list(minute_costs), dtype=float)
        for t in range(len(costs)):
            if t < short_min - 1:
                continue          # the short window must be full before it may fire
            short_rate = costs[t - short_min + 1 : t + 1].mean()
            l_lo = max(0, t - long_min + 1)
            elapsed = len(costs[l_lo : t + 1])
            long_rate = costs[l_lo : t + 1].mean()
            if short_rate >= threshold and long_rate >= threshold * min(1.0, elapsed / long_min):
                return t
        return None

    def time_to_exhaustion_hours(self, hourly: pd.DataFrame, lookback_hours: int = 24) -> Optional[float]:
        """At the current burn rate, how long until the monthly budget is gone."""
        if hourly.empty:
            return None
        recent = hourly.tail(lookback_hours)
        rate = float(recent["cost_inr"].sum()) / max(len(recent), 1)
        if rate <= 0:
            return None
        spent = float(hourly["cost_inr"].sum())
        remaining = self.budget.monthly_inr - spent
        return max(remaining, 0.0) / rate

    def budget_state(self, hourly: pd.DataFrame) -> Dict[str, Any]:
        spent = float(hourly["cost_inr"].sum()) if not hourly.empty else 0.0
        remaining_pct = max(0.0, (self.budget.monthly_inr - spent) / self.budget.monthly_inr * 100.0)
        tte = self.time_to_exhaustion_hours(hourly)
        return {
            "scope": f"{self.budget.scope}:{self.budget.scope_value}",
            "monthly_budget_inr": self.budget.monthly_inr,
            "spent_inr": spent,
            "remaining_inr": self.budget.monthly_inr - spent,
            "remaining_pct": remaining_pct,
            "hourly_budget_inr": self.budget.hourly_inr,
            "time_to_exhaustion_hours": tte,
            "status": "success",
        }


def summarise_alerts(alerts: List[BurnAlert]) -> Dict[str, Any]:
    by_sev: Dict[str, int] = {}
    for a in alerts:
        by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
    worst = max(alerts, key=lambda a: (SEVERITY_ORDER[a.severity], a.observed_multiplier), default=None)
    return {
        "count": len(alerts),
        "by_severity": by_sev,
        "worst": worst.as_dict() if worst else None,
    }
