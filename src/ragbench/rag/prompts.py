"""Loading and rendering of immutable, versioned grounded-answer prompts."""

from __future__ import annotations

import json
from enum import StrEnum
from importlib.resources import files

from ragbench.rag.context import ContextBundle


class PromptVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


def render_prompt(
    version: PromptVersion,
    *,
    question: str,
    context: ContextBundle,
) -> str:
    """Render one known prompt without evaluating braces or controls found in source data."""
    if not isinstance(version, PromptVersion):
        raise ValueError("unknown prompt version")
    if not question.strip():
        raise ValueError("question cannot be blank")
    template = (
        files("ragbench.rag.prompt_templates")
        .joinpath(f"{version.value}.txt")
        .read_text(encoding="utf-8")
    )
    if template.count("{{QUESTION_JSON}}") != 1 or template.count("{{CONTEXT}}") != 1:
        raise RuntimeError(f"prompt version {version.value} has an invalid template contract")
    question_json = _safe_json(json.dumps(question, ensure_ascii=False))
    return template.replace("{{QUESTION_JSON}}", question_json).replace(
        "{{CONTEXT}}", context.text
    )


def _safe_json(value: str) -> str:
    return value.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
