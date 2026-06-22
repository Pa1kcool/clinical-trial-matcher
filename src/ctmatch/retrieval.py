"""Milestone 2 (retrieval): the production retrieval cascade.

    dense (Qdrant) ─┐
                    ├─ RRF fusion ─ top-N candidates ─ cross-encoder rerank ─ top-K
    BM25 (sparse) ──┘

Why hybrid: BM25 catches exact clinical terms (biomarkers, codes, thresholds)
that dense embeddings smooth over; dense catches paraphrase/semantics BM25 misses.
RRF fuses the two ranked lists without needing their scores to share a scale.

Heavy ML imports are deferred into methods so the pure fusion function stays
importable in milliseconds — keeping unit tests and CLI startup fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from ctmatch.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Fuse several ranked id-lists into one. Pure function — unit tested.

    Score for a doc = sum over lists of 1 / (k + rank), rank starting at 1.
    k=60 is the canonical RRF constant (Cormack et al.).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for index, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + index + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    payload: dict[str, Any]
    rerank_score: float


class HybridRetriever:
    """Loads the corpus once, then serves hybrid + reranked retrieval."""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = SentenceTransformer(settings.embed_model)
        self._reranker: CrossEncoder | None = None
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Pull all stored chunks once to back the in-memory BM25 index."""
        from rank_bm25 import BM25Okapi

        points, _ = self.client.scroll(
            collection_name=settings.collection,
            limit=100_000,
            with_payload=True,
            with_vectors=False,
        )
        self.ids: list[str] = [str(p.id) for p in points]
        self.texts: list[str] = [p.payload["text"] for p in points]
        self.payloads: dict[str, dict[str, Any]] = {str(p.id): p.payload for p in points}
        self._bm25 = BM25Okapi([t.lower().split() for t in self.texts])

    @cached_property
    def reranker(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.reranker_model)

    def dense_search(self, query: str, k: int) -> list[str]:
        vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.client.search(
            collection_name=settings.collection, query_vector=vector, limit=k
        )
        return [str(h.id) for h in hits]

    def bm25_search(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in top]

    def search(self, query: str, top_k: int = 5, candidate_k: int = 50) -> list[RetrievedChunk]:
        """Full cascade: hybrid retrieve -> RRF fuse -> rerank -> top_k."""
        dense = self.dense_search(query, candidate_k)
        sparse = self.bm25_search(query, candidate_k)
        fused = reciprocal_rank_fusion([dense, sparse])[:candidate_k]

        pairs = [(query, self.payloads[doc_id]["text"]) for doc_id in fused]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(fused, scores, strict=True), key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                id=doc_id,
                text=self.payloads[doc_id]["text"],
                payload=self.payloads[doc_id],
                rerank_score=float(score),
            )
            for doc_id, score in ranked[:top_k]
        ]
