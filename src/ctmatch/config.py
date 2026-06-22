"""Typed application settings, loaded once from environment / .env.

Using pydantic-settings instead of scattered os.getenv calls is the real-world
pattern: one validated, typed, importable source of truth for configuration.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CTMATCH_",
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    collection: str = "clinical_trials"

    embed_model: str = "NeuML/pubmedbert-base-embeddings"
    embed_dim: int = 768
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    anthropic_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
