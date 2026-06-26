"""Milestone 4 (dataset): load the hand-labelled golden set from golden.json.

Keeping it as JSON (data, not code) means the answer key can be reviewed, edited,
and grown by anyone — including a clinician — without touching Python. Each label:
met = summary supports it · not_met = summary contradicts it · unknown = summary is silent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class GoldCriterion(TypedDict):
    text: str
    gold: str  # met | not_met | unknown


class GoldCase(TypedDict):
    patient: str
    criteria: list[GoldCriterion]


_GOLDEN_PATH = Path(__file__).parent / "golden.json"


def load_golden() -> list[GoldCase]:
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        return json.load(f)


GOLDEN: list[GoldCase] = load_golden()
