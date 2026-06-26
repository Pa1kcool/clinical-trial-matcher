"""Milestone 4 CI gate: fail the build if reasoning quality regresses past frozen thresholds.

This runs LIVE model calls, so it's gated behind an env flag + API key. The fast local
and CI suites skip it automatically; the scheduled eval job sets the flag + secret to run it.
"""

import os

import pytest

from ctmatch.config import settings

_RUN = os.getenv("CTMATCH_RUN_EVAL") == "1" and bool(settings.anthropic_api_key)

pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="set CTMATCH_RUN_EVAL=1 and CTMATCH_ANTHROPIC_API_KEY to run the live eval gate",
)

# Frozen floors: loose enough not to flap on a 2-verdict swing, tight enough to catch
# a real regression. Tighten as the golden set grows and the numbers stabilise.
MIN_CONFIDENT_ACCURACY = 0.90
MAX_OVERCLAIM_RATE = 0.15
GATE_CASES = 6  # fixed cheap subset


def test_reasoning_quality_gate():
    from ctmatch.eval.dataset import GOLDEN
    from ctmatch.eval.run import run_eval

    r = run_eval(settings.model, delay=1.0, cases=GOLDEN[:GATE_CASES])
    assert r["confident_accuracy"] >= MIN_CONFIDENT_ACCURACY, r
    assert r["unsafe_overclaim_rate"] <= MAX_OVERCLAIM_RATE, r
