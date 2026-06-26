"""Milestone 5 (guardrails): screen patient text BEFORE it reaches the agent.

Defense-in-depth at the input boundary. This is a heuristic layer — deliberately NOT
a second LLM — that cheaply blocks the three things a public, LLM-backed clinical
endpoint must not accept blindly:

  1. prompt injection / jailbreak attempts (security)
  2. real personal identifiers (privacy — this is a demo on public data)
  3. input with no clinical signal at all (quality + cost: don't pay for garbage)

Heuristics are a first line, not a guarantee; they are paired with the structured
output and groundedness gate downstream, which constrain what the model can emit even
if something slips through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    allowed: bool
    category: str | None = None  # injection | pii | not_clinical
    reason: str | None = None


# 1. Prompt-injection / jailbreak signatures (specific enough to avoid clinical hits).
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?(previous|prior|above|earlier)\s+(instruction|prompt|message|rule)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|earlier)",
    r"forget\s+(everything|all|your|the)\s+(instruction|prompt|rule|context)",
    r"(reveal|print|show|repeat|output)\s+(your|the)\s+(system\s+)?(prompt|instruction)",
    r"system\s+prompt",
    r"you\s+are\s+now\s+(a|an)\b",
    r"new\s+(instruction|rule|task|persona)s?\s*:",
    r"pretend\s+(to\s+be|you\s+are|that)",
    r"act\s+as\s+(a|an|if|though)\b",
    r"</?(system|assistant|user|instruction)s?>",  # delimiter / role injection
    r"\boverride\s+(your|the|all|safety)",
    r"\b(jailbreak|do\s+anything\s+now)\b",
]

# 2. Personal identifiers — tight patterns so clinical numbers (HbA1c 8.2, LVEF 30%) pass.
_PII_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",  # email
    r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b",  # phone
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like
]

# 3. Clinical signal — a plausible summary mentions at least one of these. Broad on
#    purpose: this only rejects input with ZERO clinical relevance.
_CLINICAL_HINTS = [
    r"\b\d{1,3}\s*[-]?\s*(year|yr|yo|y/o|years?\s*old)\b",
    r"\b(male|female|man|woman|patient|pt|boy|girl)\b",
    r"\b(cancer|carcinoma|tumou?r|diabet|leuk|lymphoma|melanoma|arthritis|asthma|"
    r"hypertension|failure|disease|syndrome|infection|mutation|metasta|stage|"
    r"chemo|therapy|treatment|diagnos|positive|negative|her2|egfr|braf|ecog|"
    r"mg\b|dose|biopsy|relapse|refractory)\b",
]


def _matches(text: str, patterns: list[str]) -> str | None:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return p
    return None


def screen(text: str) -> GuardrailResult:
    """Run all input checks in order; first failure short-circuits."""
    if _matches(text, _INJECTION_PATTERNS):
        return GuardrailResult(False, "injection", "Input looks like a prompt-injection attempt.")
    if _matches(text, _PII_PATTERNS):
        return GuardrailResult(
            False,
            "pii",
            "Input appears to contain personal identifiers; this is a "
            "demo on public data — do not enter real patient identifiers.",
        )
    if not _matches(text, _CLINICAL_HINTS):
        return GuardrailResult(
            False, "not_clinical", "Input does not read like a clinical patient summary."
        )
    return GuardrailResult(True)
