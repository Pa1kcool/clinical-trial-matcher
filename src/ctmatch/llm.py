"""Milestone 3/4 (llm): thin Anthropic wrapper with forced structured output + tracing.

`call_tool` is the traced version the agent uses. `call_tool_usage` returns token
counts too, so the eval harness can report cost per model. Both share `_tool_call`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from anthropic import Anthropic

from ctmatch.config import settings
from ctmatch.tracing import observe, record_usage


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic(api_key=settings.anthropic_api_key)


def _tool_call(
    system: str, user: str, tool_name: str, schema: dict[str, Any], model: str, max_tokens: int
) -> tuple[dict[str, Any], str, int, int]:
    """Shared core: returns (parsed_dict, model, input_tokens, output_tokens)."""
    resp = _client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": "Return the structured result.",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    for block in resp.content:
        if block.type == "tool_use":
            return dict(block.input), resp.model, resp.usage.input_tokens, resp.usage.output_tokens
    raise ValueError("model returned no tool_use block")


@observe(as_type="generation")
def call_tool(
    system: str,
    user: str,
    tool_name: str,
    schema: dict[str, Any],
    model: str | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Force structured output; traced. Returns the parsed dict."""
    data, used_model, in_tok, out_tok = _tool_call(
        system, user, tool_name, schema, model or settings.model, max_tokens
    )
    record_usage(used_model, in_tok, out_tok)
    return data


def call_tool_usage(
    system: str,
    user: str,
    tool_name: str,
    schema: dict[str, Any],
    model: str | None = None,
    max_tokens: int = 1024,
) -> tuple[dict[str, Any], int, int]:
    """Like call_tool but returns (data, input_tokens, output_tokens) for cost accounting."""
    data, _model, in_tok, out_tok = _tool_call(
        system, user, tool_name, schema, model or settings.model, max_tokens
    )
    return data, in_tok, out_tok
