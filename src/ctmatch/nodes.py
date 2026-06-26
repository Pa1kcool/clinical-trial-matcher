"""Milestone 3 (nodes): the agent's reasoning steps.

Each node forces structured output via a JSON schema (call_tool) and is traced
(@observe). Cheap model for the light steps, strong model for the judgment. The
generate step fans its independent per-trial judgments out across threads.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from ctmatch.config import settings
from ctmatch.llm import call_tool
from ctmatch.retrieval import HybridRetriever
from ctmatch.schema import AgentState
from ctmatch.tracing import observe

MAX_RETRIES = 2
MAX_GENERATE_WORKERS = 5

_KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
    "required": ["keywords"],
}
_RELEVANT_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
}
_VERDICTS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["met", "not_met", "unknown"]},
                    "rationale": {"type": "string"},
                    "patient_evidence": {"type": ["string", "null"]},
                },
                "required": ["criterion", "verdict", "rationale", "patient_evidence"],
            },
        }
    },
    "required": ["verdicts"],
}


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever()


@observe()
def analyze(state: AgentState) -> dict:
    """Turn the free-text patient summary into search keywords."""
    system = (
        "Extract concise clinical-trial search keywords from a patient summary. "
        "Focus on condition, biomarkers, stage, and prior treatments."
    )
    data = call_tool(
        system, state["patient"], "keywords", _KEYWORDS_SCHEMA, model=settings.cheap_model
    )
    return {"keywords": data["keywords"], "retries": state.get("retries", 0)}


@observe()
def retrieve(state: AgentState) -> dict:
    """Pull candidate trial criteria with the hybrid retriever from M2."""
    query = " ".join(state.get("keywords") or []) or state["patient"]
    chunks = _retriever().search(query, top_k=5)
    retrieved = [
        {"nct_id": c.payload["nct_id"], "title": c.payload["title"], "criteria": c.text}
        for c in chunks
    ]
    return {"retrieved": retrieved}


@observe()
def grade(state: AgentState) -> dict:
    """Relevance grader: are the retrieved trials worth reasoning over?"""
    system = "Decide whether the retrieved trials are relevant to the patient's primary condition."
    trials = "\n".join(f"- {r['title']}" for r in state["retrieved"])
    data = call_tool(
        system,
        f"Patient: {state['patient']}\n\nTrials:\n{trials}",
        "relevance",
        _RELEVANT_SCHEMA,
        model=settings.cheap_model,
    )
    return {"grade_ok": bool(data.get("relevant"))}


@observe()
def broaden(state: AgentState) -> dict:
    """Corrective loop: widen the search terms when grading failed."""
    system = (
        "The previous trial search was too narrow. Produce broader or alternative "
        "keywords for the same patient."
    )
    data = call_tool(
        system, state["patient"], "keywords", _KEYWORDS_SCHEMA, model=settings.cheap_model
    )
    return {"keywords": data["keywords"], "retries": state.get("retries", 0) + 1}


def _judge_trial(trial: dict, patient: str) -> dict:
    """Judge one trial's criteria. Independent of every other trial, so it runs in a thread."""
    user = f"Patient: {patient}\n\nTrial {trial['nct_id']} criteria:\n{trial['criteria']}"
    data = call_tool(
        _JUDGE_SYSTEM, user, "verdicts", _VERDICTS_SCHEMA, model=settings.model, max_tokens=2048
    )
    return {"nct_id": trial["nct_id"], "title": trial["title"], "verdicts": data["verdicts"]}


_JUDGE_SYSTEM = (
    "You are a clinical-trial eligibility assistant. For the patient and one trial's "
    "criteria, judge EACH criterion as 'met', 'not_met', or 'unknown'. Use 'unknown' "
    "whenever the summary lacks the information to decide — never guess. Put the exact "
    "patient fact you relied on in 'patient_evidence' (null if none)."
)


@observe()
def generate(state: AgentState) -> dict:
    """Per-criterion reasoning for each trial, fanned out across threads.

    The trials are independent, so we judge them concurrently. We copy the current
    context into each worker so the trace spans still nest under this node.
    """
    trials = state["retrieved"]
    patient = state["patient"]
    workers = min(len(trials), MAX_GENERATE_WORKERS) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for trial in trials:
            ctx = contextvars.copy_context()
            futures.append(pool.submit(ctx.run, _judge_trial, trial, patient))
        matches = [f.result() for f in futures]  # ordered to match `trials`
    return {"matches": matches}


@observe()
def verify(state: AgentState) -> dict:
    """Groundedness gate (deterministic): a met/not_met verdict with no patient
    evidence is downgraded to 'unknown'. The model can't talk its way past this."""
    for trial in state["matches"]:
        for v in trial["verdicts"]:
            if v.get("verdict") in ("met", "not_met") and not v.get("patient_evidence"):
                v["verdict"] = "unknown"
                v["rationale"] = "Downgraded: no patient evidence supported this judgment."
        trial["overall"] = _overall(trial["verdicts"])
    return {"matches": state["matches"]}


def _overall(verdicts: list[dict]) -> str:
    if any(v.get("verdict") == "not_met" for v in verdicts):
        return "likely_ineligible"
    if any(v.get("verdict") == "unknown" for v in verdicts):
        return "needs_review"
    return "likely_eligible"
