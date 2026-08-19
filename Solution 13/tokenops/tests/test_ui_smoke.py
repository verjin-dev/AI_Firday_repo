"""Smoke tests: every Streamlit page must render without raising.

A judge clicking through the app is the functionality score. These run the
real pages against the real ledger via Streamlit's AppTest harness, so a
broken column name or a missing table fails here rather than on stage.

Skipped automatically when the ledger is empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.storage.db import has_data  # noqa: E402

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "frontend/app.py",
    "frontend/pages/01_economics.py",
    "frontend/pages/02_attribution.py",
    "frontend/pages/03_waste.py",
    "frontend/pages/04_router.py",
    "frontend/pages/05_burn.py",
    "frontend/pages/06_forecast.py",
    "frontend/pages/07_live.py",
]

pytestmark = pytest.mark.skipif(
    not has_data(), reason="ledger is empty; run scripts/simulate_workload.py"
)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


def test_baseline_toggle_switches_the_arm() -> None:
    """The baseline toggle is the demo's whole comparison mechanism."""
    at = AppTest.from_file(str(ROOT / "frontend/pages/01_economics.py"), default_timeout=120)
    at.run()
    assert not at.exception
    before = [m.value for m in at.metric]
    at.toggle[0].set_value(True).run()
    assert not at.exception
    after = [m.value for m in at.metric]
    assert before != after, "flipping baseline mode changed nothing on the page"
