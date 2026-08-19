"""The simulated estate: a multi-agent support platform for three tenants.

This module is the single definition of the workload. Both consumers import
it, so they cannot drift apart:

  * `scripts/simulate_workload.py` - the 30-day batch run behind the benchmark
  * `backend/core/live.py`         - the real-time engine behind the Live Ops
                                     page, where traffic arrives now and
                                     incidents are injected by hand

Latent quality and confidence tables are the *ground truth* the router has to
discover. They are never read by the router - only by the executor that
simulates a model responding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

TENANTS: List[Tuple[str, str, float]] = [
    ("acme-bank", "support", 0.45),
    ("vertex-insurance", "claims", 0.35),
    ("northwind-logistics", "ops", 0.20),
]

# workflow -> (outcome_type, steps[(step, agent, task_type, base_in, base_out)], weight)
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

# ---- agent-loop shape, shared by the batch incident and the live injector ----
LOOP_CALLS_PER_MIN = 105
LOOP_IN, LOOP_OUT = 24000, 900
LOOP_KILL_DELAY_MIN = 1        # detector fires, orchestrator drains, session dies
PROMPT_BLOAT_FACTOR = 3.0

QUERY_TEMPLATES = 420


@dataclass
class Step:
    step: str
    agent: str
    task_type: str
    base_in: int
    base_out: int


def steps_for(workflow: str) -> List[Step]:
    return [Step(*s) for s in WORKFLOWS[workflow][1]]


def query_text(template_id: int, paraphrase: int, workflow: str) -> str:
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


class ChunkPool:
    """Retrieved chunks per (workflow, step), with a latent citation rate.

    Context reduction is emergent: chunks that history shows are never cited
    get dropped. Nothing is hard-coded to a target ratio.
    """

    def __init__(self, seed: int) -> None:
        import numpy as np

        from backend.core.optimizers.compressor import Chunk  # local import: keeps domain light

        self._Chunk = Chunk
        rng = np.random.default_rng(seed)
        self.pool: Dict[str, List[Tuple[str, float]]] = {}
        for wf, (_, steps, _) in WORKFLOWS.items():
            for step, *_ in steps:
                key = f"{wf}:{step}"
                utils = rng.beta(2.0, 2.5, 8)
                self.pool[key] = [(f"{key}#c{i}", float(u)) for i, u in enumerate(utils)]

    def chunks(self, workflow: str, step: str, total_tokens: int):
        # a retry re-uses the same retrieval context as the step it repeats
        key = f"{workflow}:{step.replace('_retry', '')}"
        entries = self.pool.get(key) or next(iter(self.pool.values()))
        per = max(1, int(total_tokens * 0.9 / len(entries)))
        return [self._Chunk(cid, "", per, u) for cid, u in entries]
