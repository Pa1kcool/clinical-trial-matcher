"""Milestone 3 (tracing): Langfuse instrumentation.

Active only when CTMATCH_LANGFUSE_* keys are set. When they aren't, `observe` is a
transparent no-op and nothing touches Langfuse — so the CLI stays silent until you
opt in. Decorate nodes with @observe() and LLM calls with @observe(as_type="generation").
"""

from __future__ import annotations

import os
from typing import Any

from ctmatch.config import settings

_ENABLED = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if _ENABLED:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    from langfuse import get_client, observe

    def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
        """Attach model + token counts to the current generation so Langfuse shows cost."""
        get_client().update_current_generation(
            model=model,
            usage_details={"input": input_tokens, "output": output_tokens},
        )

    def flush() -> None:
        """Send queued traces — important for a short-lived CLI process."""
        get_client().flush()

else:

    def observe(arg: Any = None, **kwargs: Any):
        if callable(arg):
            return arg

        def deco(fn):
            return fn

        return deco

    def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
        return None

    def flush() -> None:
        return None
