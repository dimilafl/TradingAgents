"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call fails, raise immediately so errors
   are visible rather than silently falling back to free text.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        bound = llm.with_structured_output(schema)
        bound._structured_schema = schema
        return bound
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _build_schema_hint(structured_llm: Any) -> str:
    """Extract JSON schema from the bound Pydantic model and format as a prompt hint."""
    schema_cls = getattr(structured_llm, "_structured_schema", None)
    if schema_cls is None or not issubclass(schema_cls, BaseModel):
        return "\n\nRespond in JSON format."
    schema_dict = schema_cls.model_json_schema()
    return (
        "\n\nRespond in JSON format matching this schema:\n"
        f"```json\n{json.dumps(schema_dict, indent=2)}\n```"
    )


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; raise on failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            json_prompt = prompt
            if isinstance(prompt, str) and "json" not in prompt.lower():
                json_prompt = prompt + _build_schema_hint(structured_llm)
            elif isinstance(prompt, list):
                hint = _build_schema_hint(structured_llm)
                last = prompt[-1]
                if isinstance(last, dict) and "content" in last and "json" not in last["content"].lower():
                    json_prompt = [*prompt[:-1], {**last, "content": last["content"] + hint}]
            result = structured_llm.invoke(json_prompt)
            return render(result)
        except Exception as exc:
            raise RuntimeError(
                f"{agent_name}: structured-output failed. "
                f"Error: {exc}"
            ) from exc

    response = plain_llm.invoke(prompt)
    return response.content
