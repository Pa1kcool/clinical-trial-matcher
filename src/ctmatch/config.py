"""Typed application settings, loaded once from environment / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CTMATCH_",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "clinical_trials"

    embed_model: str = "NeuML/pubmedbert-base-embeddings"
    embed_dim: int = 768
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Reasoning models: strong for judgment, cheap for the light steps.
    model: str = "claude-sonnet-4-6"
    cheap_model: str = "claude-haiku-4-5-20251001"

    anthropic_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
