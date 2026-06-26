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
    """Hybrid + reranked retrieval against the indexed trials (no reasoning)."""
    from ctmatch.retrieval import HybridRetriever

    retriever = HybridRetriever()
    for i, chunk in enumerate(retriever.search(query, top_k=top_k), start=1):
        p = chunk.payload
        typer.echo(f"\n[{i}] {p['nct_id']} · {p['title']}  (score {chunk.rerank_score:.3f})")
        typer.echo(f"    {chunk.text[:220]}…")


@app.command()
def match(patient: str) -> None:
    """Run the full agent: judge each trial's criteria for a patient summary."""
    from ctmatch.graph import run
    from ctmatch.tracing import flush

    marks = {"met": "✓", "not_met": "✗", "unknown": "?"}
    try:
        for m in run(patient):
            typer.echo(f"\n{m['nct_id']} · {m['title']}  →  {m['overall'].upper()}")
            for v in m["verdicts"]:
                typer.echo(f"  [{marks.get(v['verdict'], '?')}] {v['criterion'][:90]}")
                typer.echo(f"      {v['verdict']} — {v['rationale']}")
                if v.get("patient_evidence"):
                    typer.echo(f"      evidence: {v['patient_evidence']}")
    finally:
        flush()


@app.command(name="eval")
def evaluate(
    compare: Annotated[bool, typer.Option(help="Run both the strong and cheap model.")] = False,
    verbose: Annotated[bool, typer.Option(help="Print every model-vs-gold disagreement.")] = False,
    delay: Annotated[float, typer.Option(help="Seconds between cases; eases rate limits.")] = 1.0,
) -> None:
    """Score the agent's reasoning on the golden set: accuracy, over-claim rate, cost."""
    from ctmatch.config import settings
    from ctmatch.eval.run import run_eval

    models = [settings.model, settings.cheap_model] if compare else [settings.model]
    for mdl in models:
        typer.echo(f"\nEvaluating {mdl} …")
        r = run_eval(mdl, delay=delay)
        typer.echo(
            f"  criteria evaluated : {r['n']}\n"
            f"  accuracy           : {r['accuracy']}\n"
            f"  confident accuracy : {r['confident_accuracy']}   (met/not_met only)\n"
            f"  unsafe over-claim  : {r['unsafe_overclaim_rate']}   (lower is safer)\n"
            f"  tokens in/out      : {r['input_tokens']}/{r['output_tokens']}\n"
            f"  est. cost          : ${r['est_cost_usd']}\n"
            f"  wall time          : {r['seconds']}s"
        )
        if verbose and r["disagreements"]:
            typer.echo(f"  disagreements ({len(r['disagreements'])}):")
            for d in r["disagreements"]:
                typer.echo(f"    [{d['predicted']} vs gold {d['gold']}] {d['criterion']}")
                typer.echo(f"        patient: {d['patient']}…")


if __name__ == "__main__":
    app()
