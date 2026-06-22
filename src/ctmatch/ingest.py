"""Milestone 2 (ingestion): pull trials from the public ClinicalTrials.gov API,
extract eligibility criteria + metadata, embed, and upsert into Qdrant.

This populates the *dense* half of the hybrid retriever. The BM25 (sparse) half
is built at query time in retrieval.py from the same stored documents.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ctmatch.config import settings

logger = logging.getLogger(__name__)

API_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_CONDITIONS = ["breast cancer", "type 2 diabetes", "atrial fibrillation"]


def fetch_trials(condition: str, max_pages: int = 2, page_size: int = 100) -> list[dict[str, Any]]:
    """Fetch studies for a condition, following pagination tokens."""
    studies: list[dict[str, Any]] = []
    token: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {"query.cond": condition, "pageSize": page_size}
        if token:
            params["pageToken"] = token
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        studies.extend(data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return studies


def parse_trial(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ClinicalTrials.gov study into the fields we index and cite."""
    section = study.get("protocolSection", {})
    ident = section.get("identificationModule", {})
    elig = section.get("eligibilityModule", {})
    conds = section.get("conditionsModule", {}).get("conditions", [])
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle", ""),
        "conditions": ", ".join(conds),
        "criteria": elig.get("eligibilityCriteria", "") or "",
        "sex": elig.get("sex"),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
    }


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Character chunker with overlap. Deterministic so eval results are reproducible."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def build_records(conditions: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition in conditions:
        trials = fetch_trials(condition)
        logger.info("fetched %d trials for %r", len(trials), condition)
        for study in trials:
            trial = parse_trial(study)
            if not trial["nct_id"] or not trial["criteria"]:
                continue
            for piece in chunk_text(trial["criteria"]):
                records.append({"text": piece, "source": "ClinicalTrials.gov", **trial})
    return records


def ingest(conditions: list[str] | None = None) -> int:
    """End-to-end ingestion. Returns number of chunks upserted."""
    conditions = conditions or DEFAULT_CONDITIONS
    records = build_records(conditions)
    n_trials = len({r["nct_id"] for r in records})
    logger.info("prepared %d criteria chunks across %d trials", len(records), n_trials)

    model = SentenceTransformer(settings.embed_model)
    vectors = model.encode(
        [r["text"] for r in records],
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = QdrantClient(url=settings.qdrant_url)
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=settings.collection,
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=rec)
            for vec, rec in zip(vectors, records, strict=True)
        ],
    )
    logger.info("upserted %d chunks into %r", len(records), settings.collection)
    return len(records)
