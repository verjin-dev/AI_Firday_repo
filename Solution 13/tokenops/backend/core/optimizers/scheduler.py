"""Workload scheduler: interactive work runs now, deferrable work batches.

Deferrable jobs (nightly summarisation, backfills, evaluation sweeps) do not
need a p95 latency guarantee, so they can run off-peak with high concurrency,
an aggressive cache threshold, and the cheap route. This is the least
innovative module in the project and one of the largest single savings, which
is usually how it goes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

INTERACTIVE_STEPS = {"triage", "resolution"}
DEFERRABLE_STEPS = {"qa_verify", "summarise", "backfill", "eval"}
OFF_PEAK_HOURS = set(range(0, 7)) | {22, 23}


@dataclass
class Job:
    job_id: str
    step: str
    tenant: str
    est_input_tokens: int
    deadline_hours: Optional[float] = None
    interactive: Optional[bool] = None


@dataclass
class Batch:
    hour: int
    jobs: List[str]
    concurrency: int
    cache_threshold: float
    est_saving_pct: float


@dataclass
class SchedulePlan:
    immediate: List[str] = field(default_factory=list)
    batches: List[Batch] = field(default_factory=list)
    deferred_pct: float = 0.0
    est_saving_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "immediate": self.immediate,
            "batches": [asdict(b) for b in self.batches],
            "deferred_pct": self.deferred_pct,
            "est_saving_pct": self.est_saving_pct,
            "warnings": self.warnings,
        }


class WorkloadScheduler:
    def __init__(self, off_peak_hours: Optional[set] = None, batch_size: int = 64,
                 off_peak_concurrency: int = 16, off_peak_cache_threshold: float = 0.90) -> None:
        self.off_peak_hours = off_peak_hours or OFF_PEAK_HOURS
        self.batch_size = batch_size
        self.off_peak_concurrency = off_peak_concurrency
        self.off_peak_cache_threshold = off_peak_cache_threshold

    @staticmethod
    def classify(job: Job) -> bool:
        """True if the job is interactive (must run now)."""
        if job.interactive is not None:
            return job.interactive
        if job.step in DEFERRABLE_STEPS:
            return False
        if job.deadline_hours is not None and job.deadline_hours > 4:
            return False
        return job.step in INTERACTIVE_STEPS or True

    def plan(self, jobs: Iterable[Job], now_hour: int = 12) -> SchedulePlan:
        jobs = list(jobs)
        plan = SchedulePlan()
        deferrable: List[Job] = []
        for j in jobs:
            if self.classify(j):
                plan.immediate.append(j.job_id)
            else:
                deferrable.append(j)

        target_hour = next((h for h in sorted(self.off_peak_hours) if h > now_hour), min(self.off_peak_hours))
        for i in range(0, len(deferrable), self.batch_size):
            chunk = deferrable[i : i + self.batch_size]
            plan.batches.append(
                Batch(
                    hour=target_hour,
                    jobs=[j.job_id for j in chunk],
                    concurrency=self.off_peak_concurrency,
                    cache_threshold=self.off_peak_cache_threshold,
                    est_saving_pct=42.0,   # cheap route + relaxed cache threshold
                )
            )
        total = len(jobs) or 1
        plan.deferred_pct = len(deferrable) / total * 100.0
        plan.est_saving_pct = plan.deferred_pct * 0.42
        if not deferrable:
            plan.warnings.append("no deferrable work found: every job claimed to be interactive")
        return plan
