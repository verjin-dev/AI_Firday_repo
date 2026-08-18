"""Cost forecasting: Holt-Winters with weekly seasonality, plus the piece
that makes a forecast actionable - driver decomposition.

Growth from volume and growth from unit cost need opposite responses. Volume
growth is a good problem. Unit-cost growth is a regression, and it has an
owner. A forecast that does not separate them is a number, not information.

Implemented directly in numpy (additive triple exponential smoothing with a
small parameter grid search) so the project has no statsmodels/scipy
dependency - it runs on a laptop with nothing but the base stack.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    horizon_days: int
    point: List[float]
    lower: List[float]
    upper: List[float]
    total_inr: float
    total_lower_inr: float
    total_upper_inr: float
    mape_pct: Optional[float]
    method: str
    params: Dict[str, float] = field(default_factory=dict)
    drivers: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    status: str = "success"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _holt_winters_fit(y: np.ndarray, season: int, alpha: float, beta: float,
                      gamma: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Additive Holt-Winters. Returns (level, trend, seasonals, fitted)."""
    n = len(y)
    n_seasons = max(n // season, 1)
    season_means = [np.mean(y[i * season : (i + 1) * season]) for i in range(n_seasons)]
    level = float(season_means[0])
    trend = float((np.mean(y[season : 2 * season]) - np.mean(y[:season])) / season) if n >= 2 * season else 0.0
    seasonals = np.array(
        [float(np.mean([y[j * season + i] - season_means[j] for j in range(n_seasons)
                        if j * season + i < n])) for i in range(season)]
    )
    fitted = np.zeros(n)
    for t in range(n):
        s_idx = t % season
        fitted[t] = level + trend + seasonals[s_idx]
        last_level = level
        level = alpha * (y[t] - seasonals[s_idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        seasonals[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasonals[s_idx]
    return level, trend, seasonals, fitted


def _grid_search(y: np.ndarray, season: int) -> Tuple[float, float, float]:
    best, best_sse = (0.3, 0.1, 0.2), float("inf")
    grid = [0.05, 0.15, 0.3, 0.5, 0.7]
    for a in grid:
        for b in [0.0, 0.05, 0.15, 0.3]:
            for g in [0.05, 0.2, 0.4]:
                _, _, _, fitted = _holt_winters_fit(y, season, a, b, g)
                sse = float(np.sum((y[season:] - fitted[season:]) ** 2))
                if np.isfinite(sse) and sse < best_sse:
                    best_sse, best = sse, (a, b, g)
    return best


class CostForecaster:
    def __init__(self, season: int = 7, z: float = 1.96) -> None:
        self.season = season
        self.z = z

    # --------------------------------------------------------------- forecast
    def forecast(self, series: Sequence[float], horizon_days: int = 30,
                 drivers: Optional[pd.DataFrame] = None,
                 exclude_incident_days: Optional[Sequence[int]] = None) -> Forecast:
        y = np.asarray([float(v) for v in series], dtype=float)
        warnings: List[str] = []

        if exclude_incident_days:
            keep = [i for i in range(len(y)) if i not in set(exclude_incident_days)]
            if len(keep) >= self.season * 2:
                median = float(np.median(y[keep]))
                for i in set(exclude_incident_days):
                    if i < len(y):
                        y[i] = median
                warnings.append(
                    f"incident days {sorted(set(exclude_incident_days))} replaced with the median: "
                    "a forecast trained on an outage forecasts outages"
                )

        if len(y) < self.season * 2:
            mean = float(np.mean(y)) if len(y) else 0.0
            point = [mean] * horizon_days
            sd = float(np.std(y)) if len(y) > 1 else mean * 0.2
            warnings.append(f"only {len(y)} days of history; falling back to a flat mean forecast")
            return Forecast(
                horizon_days, point, [max(p - self.z * sd, 0.0) for p in point],
                [p + self.z * sd for p in point], float(np.sum(point)),
                float(np.sum(point)) - self.z * sd * horizon_days ** 0.5,
                float(np.sum(point)) + self.z * sd * horizon_days ** 0.5,
                None, "mean_fallback", {}, {}, warnings, "partial",
            )

        a, b, g = _grid_search(y, self.season)
        level, trend, seasonals, fitted = _holt_winters_fit(y, self.season, a, b, g)
        resid = y[self.season :] - fitted[self.season :]
        sd = float(np.std(resid)) if len(resid) > 1 else float(np.std(y))

        point = [float(max(level + (h + 1) * trend + seasonals[(len(y) + h) % self.season], 0.0))
                 for h in range(horizon_days)]
        # interval widens with the square root of horizon, as it should
        lower = [max(p - self.z * sd * np.sqrt(h + 1) / 2.0, 0.0) for h, p in enumerate(point)]
        upper = [p + self.z * sd * np.sqrt(h + 1) / 2.0 for h, p in enumerate(point)]

        total = float(np.sum(point))
        # accuracy is measured on the current regime; a structural break is
        # reported rather than averaged into the error
        brk = self.detect_break(y)
        mape = self.backtest_mape(y)
        if brk is not None:
            post = y[brk:]
            post_mape = self.backtest_mape(post) if len(post) >= self.season * 2 + 7 else None
            pre_mape = self.backtest_mape(y[:brk]) if brk >= self.season * 2 + 7 else None
            warnings.append(
                f"structural break detected at day {brk} (level shift): a forecaster cannot "
                "see a deploy coming. Accuracy on the stable segment is reported separately."
            )
            mape = post_mape if post_mape is not None else (pre_mape if pre_mape is not None else mape)

        return Forecast(
            horizon_days=horizon_days,
            point=point,
            lower=lower,
            upper=upper,
            total_inr=total,
            total_lower_inr=float(np.sum(lower)),
            total_upper_inr=float(np.sum(upper)),
            mape_pct=mape,
            method="holt_winters_additive",
            params={"alpha": a, "beta": b, "gamma": g, "season": self.season,
                    "resid_sd": sd, "level": float(level), "trend": float(trend),
                    "structural_break_day": brk},
            drivers=self.decompose_drivers(drivers) if drivers is not None else {},
            warnings=warnings,
        )

    # ------------------------------------------------------- structural break
    @staticmethod
    def detect_break(y: np.ndarray, min_segment: int = 7, ratio: float = 1.4) -> Optional[int]:
        """Find a step change in level - a prompt deploy, a model swap, a new
        tenant. No forecaster can predict across one, so the honest thing is to
        find it, say so, and measure accuracy on each side separately."""
        best_day, best_ratio = None, ratio
        for t in range(min_segment, len(y) - min_segment):
            before, after = float(np.mean(y[:t])), float(np.mean(y[t:]))
            if before <= 0:
                continue
            r = after / before
            if r > best_ratio:
                best_ratio, best_day = r, t
        return best_day

    # --------------------------------------------------------------- backtest
    def backtest_mape(self, y: np.ndarray, holdout: int = 7) -> Optional[float]:
        """Honest accuracy: fit on everything but the last week, predict it."""
        if len(y) < self.season * 2 + holdout:
            return None
        train, test = y[:-holdout], y[-holdout:]
        a, b, g = _grid_search(train, self.season)
        level, trend, seasonals, _ = _holt_winters_fit(train, self.season, a, b, g)
        pred = np.array([max(level + (h + 1) * trend + seasonals[(len(train) + h) % self.season], 0.0)
                         for h in range(holdout)])
        mask = test > 0
        if not mask.any():
            return None
        return float(np.mean(np.abs((test[mask] - pred[mask]) / test[mask])) * 100.0)

    # -------------------------------------------------------------- drivers
    @staticmethod
    def decompose_drivers(df: pd.DataFrame, window: int = 7) -> Dict[str, Any]:
        """Split period-over-period cost growth into volume and unit-cost
        contributions.  dC = u0*dV (volume) + V0*du (unit cost) + dV*du."""
        need = {"outcomes", "cost_per_outcome_inr"}
        if df is None or df.empty or not need.issubset(df.columns):
            return {}
        d = df.dropna(subset=["cost_per_outcome_inr"])
        if len(d) < window * 2:
            return {}
        prev, curr = d.iloc[-2 * window : -window], d.iloc[-window:]
        v0, v1 = float(prev["outcomes"].mean()), float(curr["outcomes"].mean())
        u0, u1 = float(prev["cost_per_outcome_inr"].mean()), float(curr["cost_per_outcome_inr"].mean())
        dv, du = v1 - v0, u1 - u0
        vol_contrib, unit_contrib, inter = u0 * dv, v0 * du, dv * du
        total = vol_contrib + unit_contrib + inter
        share = lambda x: (abs(x) / (abs(vol_contrib) + abs(unit_contrib) + abs(inter)) * 100.0) \
            if (abs(vol_contrib) + abs(unit_contrib) + abs(inter)) > 0 else 0.0
        return {
            "window_days": window,
            "volume_prev": v0, "volume_curr": v1, "volume_delta_pct": (dv / v0 * 100.0) if v0 else None,
            "unit_cost_prev": u0, "unit_cost_curr": u1, "unit_cost_delta_pct": (du / u0 * 100.0) if u0 else None,
            "growth_inr": total,
            "volume_contribution_inr": vol_contrib,
            "unit_cost_contribution_inr": unit_contrib,
            "interaction_inr": inter,
            "volume_share_pct": share(vol_contrib),
            "unit_cost_share_pct": share(unit_contrib),
            "verdict": (
                "unit cost is rising - this is a regression with an owner"
                if du > 0 and share(unit_contrib) > 25
                else "growth is volume-driven - a good problem"
            ),
        }

    # ---------------------------------------------------------------- what-if
    @staticmethod
    def what_if(baseline_monthly_inr: float, changes: Dict[str, float],
                current: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Scenario model over the levers TokenOps actually controls.

        changes accepts: cache_hit_rate (0-1), cheap_model_share (0-1),
        context_reduction (0-1), volume_growth (0-1 fraction).
        Effects are multiplicative on the remaining billable spend, which is
        the right first-order model: each lever removes a slice of the tokens
        the previous lever left behind.
        """
        current = current or {"cache_hit_rate": 0.0, "cheap_model_share": 0.0, "context_reduction": 0.0}
        cost = baseline_monthly_inr
        steps: List[Dict[str, Any]] = []

        def apply(name: str, factor: float, note: str) -> None:
            nonlocal cost
            before = cost
            cost *= factor
            steps.append({"lever": name, "before_inr": before, "after_inr": cost,
                          "delta_inr": cost - before, "note": note})

        d_cache = max(0.0, changes.get("cache_hit_rate", current["cache_hit_rate"]) - current["cache_hit_rate"])
        if d_cache > 0:
            # a cache hit still costs the discounted cache-read tier, ~10%
            apply("semantic cache", 1.0 - d_cache * 0.9, f"+{d_cache:.0%} hit rate at ~90% saving per hit")

        d_cheap = max(0.0, changes.get("cheap_model_share", current["cheap_model_share"]) - current["cheap_model_share"])
        if d_cheap > 0:
            # haiku vs sonnet is roughly a 3.75x price ratio on input tokens
            apply("model cascade", 1.0 - d_cheap * 0.73, f"+{d_cheap:.0%} of traffic moved to the cheap model")

        d_ctx = max(0.0, changes.get("context_reduction", current["context_reduction"]) - current["context_reduction"])
        if d_ctx > 0:
            # input tokens are ~70% of spend in a retrieval-heavy workload
            apply("context pruning", 1.0 - d_ctx * 0.7, f"context down {d_ctx:.0%}")

        growth = changes.get("volume_growth", 0.0)
        if growth:
            apply("volume growth", 1.0 + growth, f"volume {growth:+.0%}")

        return {
            "baseline_monthly_inr": baseline_monthly_inr,
            "scenario_monthly_inr": cost,
            "delta_inr": cost - baseline_monthly_inr,
            "delta_pct": (cost - baseline_monthly_inr) / baseline_monthly_inr * 100.0 if baseline_monthly_inr else 0.0,
            "steps": steps,
            "status": "success",
            "warnings": ["first-order model: levers are applied multiplicatively and "
                         "assume the traffic mix holds"],
        }
