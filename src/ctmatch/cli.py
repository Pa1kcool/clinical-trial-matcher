"""Command-line entrypoint. Run via `ctmatch ...` or `python -m ctmatch.cli ...`."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

app = typer.Typer(help="Clinical-trial eligibility matching agent.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    condition: Annotated[
        list[str] | None,
        typer.Option("--condition", "-c", help="Repeatable; defaults to a starter set."),
    ] = None,
) -> None:
    """Fetch trials from ClinicalTrials.gov and index them into Qdrant."""
    from ctmatch.ingest import ingest as run_ingest

    n = run_ingest(conditions=condition or None)
    typer.echo(f"Ingested {n} criteria chunks.")


@app.command()
def search(
    query: str,
    top_k: Annotated[int, typer.Option(help="How many trials to return.")] = 5,
) -> None:
    """Hybrid + reranked retrieval against the indexed trials."""
    from ctmatch.retrieval import HybridRetriever

    retriever = HybridRetriever()
    for i, chunk in enumerate(retriever.search(query, top_k=top_k), start=1):
        p = chunk.payload
        typer.echo(f"\n[{i}] {p['nct_id']} · {p['title']}  (score {chunk.rerank_score:.3f})")
        typer.echo(f"    {chunk.text[:220]}…")


if __name__ == "__main__":
    app()
