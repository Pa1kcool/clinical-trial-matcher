"""Milestone 4 (run): execute the golden set through the model's reasoning and score it.

We judge the *given* criteria directly (no retrieval) so the measurement is stable and
isolates reasoning quality. A small inter-case delay keeps us under the rate limit;
disagreements (predicted != gold) are collected so you can inspect failures, not just
the score.
"""

from __future__ import annotations

import time
from typing import Any

from ctmatch.eval.dataset import GOLDEN, GoldCase
from ctmatch.eval.metrics import summarize
from ctmatch.llm import call_tool_usage
from ctmatch.nodes import _VERDICTS_SCHEMA

# Illustrative USD per *million* tokens — verify on the pricing page and update.
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}

_SYSTEM = (
    "You are a clinical-trial eligibility assistant. For the patient and the numbered "
    "list of criteria, judge EACH criterion as 'met', 'not_met', or 'unknown', in the "
    "same order. Use 'unknown' whenever the summary lacks the information to decide — "
    "never guess. Put the exact patient fact you relied on in 'patient_evidence' (null "
    "if none)."
)


def judge_case(case: GoldCase, model: str) -> tuple[list[str], int, int]:
    criteria = case["criteria"]
    numbered = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(criteria))
    user = f"Patient: {case['patient']}\n\nCriteria:\n{numbered}"
    data, in_tok, out_tok = call_tool_usage(
        _SYSTEM, user, "verdicts", _VERDICTS_SCHEMA, model=model, max_tokens=2048
    )
    preds = [v.get("verdict", "unknown") for v in data.get("verdicts", [])]
    preds = (preds + ["unknown"] * len(criteria))[: len(criteria)]  # align to gold length
    return preds, in_tok, out_tok


def run_eval(model: str, delay: float = 0.0, cases: list[GoldCase] | None = None) -> dict[str, Any]:
    cases = cases if cases is not None else GOLDEN
    pairs: list[tuple[str, str]] = []
    disagreements: list[dict[str, str]] = []
    in_tok = out_tok = 0
    start = time.time()
    for case in cases:
        preds, itok, otok = judge_case(case, model)
        for crit, pred in zip(case["criteria"], preds, strict=True):
            pairs.append((pred, crit["gold"]))
            if pred != crit["gold"]:
                disagreements.append(
                    {
                        "patient": case["patient"][:60],
                        "criterion": crit["text"],
                        "predicted": pred,
                        "gold": crit["gold"],
                    }
                )
        in_tok += itok
        out_tok += otok
        if delay:
            time.sleep(delay)
    elapsed = time.time() - start

    price = PRICES.get(model, {"input": 0.0, "output": 0.0})
    cost = in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"]
    result = summarize(pairs)
    result.update(
        {
            "model": model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_cost_usd": round(cost, 4),
            "seconds": round(elapsed, 1),
            "disagreements": disagreements,
        }
    )
    return result
