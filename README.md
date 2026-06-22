# Clinical-Trial Eligibility Matching Agent

A self-correcting agent that matches a patient summary to clinical trials, reasons
over each eligibility criterion (met / not met / unknown), cites the exact criterion
and patient fact, and abstains when a criterion can't be verified. Benchmarked against
NIH TrialGPT. Public data only; decision-support prototype, not a clinical tool.

## Quickstart (macOS)
    brew install uv
    brew install --cask docker
    uv sync --extra dev && cp .env.example .env
    docker compose up -d
    uv run ctmatch ingest
    uv run ctmatch search "HER2-positive breast cancer, prior chemotherapy"

## Milestones
- M1 scaffold done  · M2 ingestion + hybrid retrieval done
- M3 agent graph · M4 eval harness · M5 API · M6 deploy + dashboard
