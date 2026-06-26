"""Milestone 5/6 (api): FastAPI service wrapping the agent + the web UI.

GET /            serves the single-page demo.
POST /match        runs the agent and returns the full ranked result (for API consumers).
POST /match/stream streams step-by-step progress as Server-Sent Events (for the UI).
GET /health        deploy/uptime probe.

Before the agent runs, input is screened (length + guardrails), rate-limited per client,
and served from a small response cache so repeated/sample queries are free and instant.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ctmatch.guardrails import screen

app = FastAPI(title="Clinical Trial Matcher", version="0.1.0")

_STATIC = Path(__file__).parent / "static"

# --- cost guards -----------------------------------------------------------------
# In-memory and single-instance: fine for a demo. For multi-instance you'd move these
# to Redis. Rate limit is per client IP; the cache is keyed by the patient text so
# repeated clicks and the sample patients never re-spend on the model.
_RATE_MAX = 12  # requests
_RATE_WINDOW = 60.0  # seconds
_CACHE_MAX = 256
_hits: dict[str, list[float]] = defaultdict(list)
_cache: OrderedDict[str, list[dict]] = OrderedDict()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _hits[ip] if now - t < _RATE_WINDOW]
    _hits[ip] = recent
    if len(recent) >= _RATE_MAX:
        return True
    recent.append(now)
    return False


def _cache_get(key: str) -> list[dict] | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: str, value: list[dict]) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


class MatchRequest(BaseModel):
    patient: str = Field(min_length=10, max_length=2000, description="Patient summary.")


class CriterionOut(BaseModel):
    criterion: str
    verdict: str
    rationale: str
    patient_evidence: str | None = None


class TrialOut(BaseModel):
    nct_id: str
    title: str
    overall: str
    verdicts: list[CriterionOut]


class MatchResponse(BaseModel):
    matches: list[TrialOut]


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest, request: Request) -> MatchResponse:
    if _rate_limited(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "reason": "Too many requests. Wait a minute and try again.",
            },
        )
    verdict = screen(req.patient)
    if not verdict.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "input_rejected",
                "category": verdict.category,
                "reason": verdict.reason,
            },
        )

    key = req.patient.strip()
    cached = _cache_get(key)
    if cached is not None:
        return MatchResponse(matches=cached)

    from ctmatch.graph import run
    from ctmatch.tracing import flush

    try:
        matches = run(req.patient)
    finally:
        flush()
    _cache_put(key, matches)
    return MatchResponse(matches=matches)


@app.post("/match/stream")
def match_stream(req: MatchRequest, request: Request) -> StreamingResponse:
    """Stream the agent's per-node progress, then the ranked result, as SSE."""
    limited = _rate_limited(_client_ip(request))
    verdict = screen(req.patient)
    key = req.patient.strip()
    cached = _cache_get(key)

    def gen():
        from ctmatch.graph import build_graph, rank_matches
        from ctmatch.tracing import flush

        if limited:
            yield _sse(
                {"type": "error", "reason": "Too many requests. Wait a minute and try again."}
            )
            return
        if not verdict.allowed:
            yield _sse({"type": "error", "category": verdict.category, "reason": verdict.reason})
            return
        if cached is not None:
            for node in ("analyze", "retrieve", "grade", "generate", "verify"):
                yield _sse({"type": "step", "node": node})
            yield _sse({"type": "done", "matches": cached})
            return
        try:
            matches: list[dict] = []
            for update in build_graph().stream({"patient": req.patient, "retries": 0}):
                for node, partial in update.items():
                    if isinstance(partial, dict) and "matches" in partial:
                        matches = partial["matches"]
                    yield _sse({"type": "step", "node": node})
            ranked = rank_matches(matches)
            _cache_put(key, ranked)
            yield _sse({"type": "done", "matches": ranked})
        except Exception:
            yield _sse({"type": "error", "reason": "The agent failed to complete the request."})
        finally:
            flush()

    return StreamingResponse(gen(), media_type="text/event-stream")
