"""Milestone 3 (schemas): the typed data shapes the agent passes between nodes.

Defining these *before* the node logic is the real-world habit — every step then
agrees on the same structure, and the three-way eligibility label (the abstention)
is baked into the type system, not left to free-text the model might fudge.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class Eligibility(StrEnum):
    """A criterion is met, not met, or — crucially — unverifiable from what we know."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"  # the abstention: missing info, NOT a guess either way


class CriterionVerdict(BaseModel):
    criterion: str = Field(description="The exact eligibility criterion text being judged.")
    verdict: Eligibility
    rationale: str = Field(description="One-sentence justification.")
    patient_evidence: str | None = Field(
        default=None,
        description="The specific patient fact relied on; None when nothing supports a call.",
    )


class TrialMatch(BaseModel):
    nct_id: str
    title: str
    verdicts: list[CriterionVerdict]
    overall: str  # likely_eligible | likely_ineligible | needs_review


class AgentState(TypedDict, total=False):
    """What flows through the LangGraph. total=False so nodes fill it in progressively."""

    patient: str  # the input patient summary
    keywords: list[str]  # analyzer's search terms
    retrieved: list[dict]  # candidate trial chunks from HybridRetriever
    grade_ok: bool  # relevance grader's verdict on the retrieval
    retries: int  # corrective-loop counter (caps re-retrieval)
    matches: list[dict]  # final per-trial results
