# Clinical Trial Matcher

An AI agent that reads a patient summary, finds candidate clinical trials, and judges each eligibility criterion as **met**, **not met**, or **unknown**, with the evidence it used and an honest "I don't know" when the summary doesn't say.

**Live demo:** _add your URL here after deploying_

> Decision-support prototype built on public ClinicalTrials.gov data. It is not a clinical tool and should not be used for real medical decisions.

---

## What it does

You paste something like:

> 62-year-old postmenopausal woman with HER2-positive metastatic breast cancer, prior chemotherapy with docetaxel

and the agent retrieves real trials, then goes criterion by criterion and tells you, for each one, whether the patient meets it, fails it, or whether the summary simply doesn't contain enough to say. Every "met" or "not met" comes with the exact patient fact it relied on. Trials are ranked so the ones worth reviewing sit on top and the dead ends sink to the bottom (still visible, with the reasoning, so you can see *why* they were ruled out).

The thing I care about most: when the summary is silent on a criterion, the agent says **unknown** instead of guessing. For a tool that touches eligibility, a confident wrong answer is worse than an honest "needs a human to check."

## Demo


https://github.com/user-attachments/assets/c4a41299-e044-43f4-a74d-901b74e95047



## Screenshots

![The live step-by-step progress](docs/screenshot-progress.png)

![The demo page with a result](docs/screenshot-results.png)

![Langfuse Dashboards](docs/screenshot-langfuse.png)

![Langfuse Latency](docs/screenshot-langfusel.png)

## Who it is for

The intended user is a clinician or clinical research coordinator screening patients for trials. The whole design assumes a human in the loop: the agent does the tedious first pass and flags everything it can't verify, and a person makes the actual call.

## How it works

Each request runs a small self-correcting agent built with LangGraph:

```
analyze -> retrieve -> grade -(relevant)-> generate -> verify -> rank
                          ^
                          |-(too narrow)- broaden -|
```

1. **analyze** turns the free text into search keywords (cheap model).
2. **retrieve** pulls candidate trials using hybrid search: BM25 keyword matching plus PubMedBERT dense vectors, fused with Reciprocal Rank Fusion, then re-ranked with a cross-encoder.
3. **grade** checks whether the retrieved trials are actually relevant. If they look too narrow, the agent loops back and broadens the search (up to twice) before giving up.
4. **generate** judges every criterion of every trial (strong model), and is forced to return structured output through a tool schema so the answer is always valid JSON in the exact shape I want.
5. **verify** is plain Python, not a model. It downgrades any "met" or "not met" that has no supporting patient evidence back to "unknown." The model cannot talk its way past this gate.
6. **rank** sorts trials so the actionable ones come first.

Two model sizes are used on purpose: a cheap fast model (Haiku) for the light steps, a stronger model (Sonnet) for the actual judgment. That keeps cost down without hurting quality, and I have the eval numbers to back that up.

## What makes it more than a RAG demo

- **Abstention.** It refuses to answer criteria the summary can't support, and I measure how often it fails to do that (see the over-claim rate below).
- **An eval harness with a safety metric.** A hand-labelled golden set of 147 criteria across 24 disease areas, scored for accuracy and for an "unsafe over-claim rate" that the reference system (NIH's TrialGPT) does not report.
- **Observability.** Every step and model call is a traced span in Langfuse with token cost attached, so I know exactly what each match costs and where the time goes.
- **Input guardrails.** Patient text is screened for prompt injection, personal identifiers, and obvious non-clinical garbage before it ever reaches a model.
- **A CI gate.** A scheduled job runs the eval and fails if quality drops below frozen thresholds, so a future change can't silently make it worse.

## Results

Measured on the 147-criterion golden set (24 disease areas):

| Model | Accuracy | Confident accuracy | Unsafe over-claim | Cost / run |
|-------|----------|--------------------|--------------------|------------|
| Sonnet | 0.98 | 0.99 | 0.043 | ~$0.24 |
| Haiku  | 1.00 | 1.00 | 0.00  | ~$0.08 |

"Confident accuracy" is the score on criteria that actually have a yes/no answer (the number most comparable to TrialGPT's published ~87%). "Unsafe over-claim" is the fraction of unknowable criteria where the model committed to an answer it shouldn't have. Both models sit near the ceiling on this set, so they are within noise of each other, which means the cost routing decision stands on cost, not quality.

One real match costs about 7 cents and now takes roughly 20 seconds (the five per-trial judgments run in parallel).

## Tech stack

- Python 3.13, packaged with **uv**
- **LangGraph** for the agent state machine
- **Anthropic API** (Claude Sonnet and Haiku) for reasoning
- **Qdrant** for vector search, **PubMedBERT** embeddings, a cross-encoder re-ranker (sentence-transformers)
- **FastAPI** for the API and to serve the web UI
- **Langfuse** for tracing and cost
- **pytest**, **ruff**, **pre-commit**, **GitHub Actions** for quality and CI
- A single static HTML/CSS/JS page for the demo (no frontend framework)

## Run it yourself

You need: Python 3.13, [uv](https://docs.astral.sh/uv/), Docker (for Qdrant), and an Anthropic API key.

```bash
# 1. clone and install
git clone https://github.com/YOUR_USERNAME/clinical-trial-matcher.git
cd clinical-trial-matcher
uv sync --extra dev

# 2. set your key
cp .env.example .env
# open .env and put your key in CTMATCH_ANTHROPIC_API_KEY

# 3. start Qdrant
docker compose up -d

# 4. build the trial index (fetches public trials and embeds them)
uv run ctmatch ingest

# 5a. try it from the command line
uv run ctmatch match "62-year-old postmenopausal woman with HER2-positive metastatic breast cancer, prior chemotherapy with docetaxel"

# 5b. or run the web app and open http://localhost:8000
uv run uvicorn ctmatch.api:app --reload --port 8000
```

Run the eval:

```bash
uv run ctmatch eval --compare          # score both models on the golden set
uv run ctmatch eval --verbose          # show every disagreement vs the gold labels
```

Run the checks:

```bash
uv run ruff check .
uv run pytest -q
```

### Optional: tracing with Langfuse

Add three keys to `.env` and every run shows up as a costed trace in your Langfuse dashboard. Leave them out and tracing quietly turns off, nothing breaks.

```
CTMATCH_LANGFUSE_PUBLIC_KEY=pk-lf-...
CTMATCH_LANGFUSE_SECRET_KEY=sk-lf-...
CTMATCH_LANGFUSE_HOST=https://cloud.langfuse.com
```

## Project layout

```
src/ctmatch/
  config.py        typed settings from env / .env
  ingest.py        fetch trials, chunk criteria, embed into Qdrant
  retrieval.py     hybrid BM25 + dense search, RRF fusion, cross-encoder rerank
  schema.py        shared types (verdicts, agent state)
  llm.py           Anthropic wrapper, forced structured output, token usage
  nodes.py         the agent's steps (analyze, retrieve, grade, broaden, generate, verify)
  graph.py         wires the nodes into the LangGraph state machine + ranking
  tracing.py       Langfuse on/off
  guardrails.py    input screening (injection / PII / non-clinical)
  cli.py           ingest / search / match / eval commands
  api.py           FastAPI: /, /match, /match/stream, /health
  static/index.html  the web demo
  eval/
    metrics.py     accuracy, confident accuracy, unsafe over-claim
    dataset.py     loads the golden set
    golden.json    147 hand-labelled criteria
    run.py         runs the golden set through a model and scores it
tests/             unit tests + the env-gated eval CI gate
```

## Limitations and roadmap

Honest about what this is not yet.

The system has two halves that can each be wrong: retrieval (which trials even reach the agent) and reasoning (how it judges them). Today I have a real number for the reasoning half (the 147-criterion eval and the over-claim rate), but retrieval quality is only evaluated indirectly. That matters: if the retriever never surfaces the trial a patient actually qualifies for, perfect reasoning still misses the best match. So this is a first-class gap, not a footnote. The planned TREC Clinical Trials benchmark (M7) measures retrieval recall directly on an expert-labelled dataset, alongside reasoning accuracy.

Other known gaps:

- The trial-level summary doesn't yet distinguish inclusion from exclusion criteria, so a triggered exclusion can show as "needs review" instead of "likely ineligible." The per-criterion reasoning is correct; only the rollup logic is naive. Small fix, planned.
- The golden labels are mine, not a clinician's. For real use, a clinician signs off on the answer key.

Planned next:

- Add rate limiting and cached sample cases, then deploy to a public URL.
- Benchmark on the public TREC Clinical Trials dataset, which measures both retrieval recall and reasoning accuracy on the same footing as TrialGPT.
- Distinguish inclusion vs exclusion criteria in the trial-level rollup.

## Disclaimer

This is a demonstration project built on public data. It is not a medical device, not validated, and must not be used for real clinical or patient decisions.
