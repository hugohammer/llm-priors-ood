"""Querying LLMs for regression coefficients.

One elicitation call asks a model for a coefficient per feature, in raw feature
units, as a JSON object keyed exactly by the feature names. Repeated calls at
non-zero temperature give the sample cloud that becomes the prior.

Two elicitation modes:

  direct     the response is the coefficient object alone
  reasoning  the response is {"reasoning": <text>, "coefficients": {...}}

In reasoning mode `reasoning` is generated *before* `coefficients`, so the
coefficients are conditioned on the model's own written justification. This is
prompted chain-of-thought, and it is also the only way to obtain an inspectable
rationale from providers that do not expose their internal reasoning trace.

Each provider exposes this differently: Anthropic enables extended thinking
(which the API runs at temperature 1.0 and cannot combine with forced tool use);
OpenAI and Google do not expose an internal trace, so the elicited `reasoning`
field is the reasoning record.

Schema enforcement is by API contract wherever possible: OpenAI strict
`json_schema`, Anthropic forced tool use, Google `response_schema`. Only the
Anthropic reasoning path falls back to prompt-level enforcement.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Sequence

TOOL_NAME = "prior_coefficients"
ANTHROPIC_THINKING_BUDGET = 3000
ANTHROPIC_THINKING_TEMPERATURE = 1.0  # required by the API when thinking is on


# --------------------------------------------------------------------------- #
# Schemas and the prompt-level output contract
# --------------------------------------------------------------------------- #

def coefficient_schema(feature_names: Sequence[str]) -> dict:
    """Flat object: one number per feature, no extras."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {f: {"type": "number"} for f in feature_names},
        "required": list(feature_names),
    }


def reasoning_schema(feature_names: Sequence[str], *, strict: bool = True) -> dict:
    """Wrapped object. `reasoning` precedes `coefficients` so it is generated first."""
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "coefficients": coefficient_schema(feature_names),
        },
        "required": ["reasoning", "coefficients"],
    }
    if strict:
        schema["additionalProperties"] = False
    return schema


def output_contract(user_prompt: str, feature_names: Sequence[str], *, reasoning: bool) -> str:
    """Append the output-format contract to a dataset-specific user prompt."""
    keys = json.dumps(list(feature_names), ensure_ascii=False)
    if not reasoning:
        return (
            user_prompt.rstrip()
            + "\n\nOutput format (strict):\n"
            + "- Return only a JSON object. No markdown, code fences, or commentary.\n"
            + f"- Include exactly these keys (no more, no fewer): {keys}\n"
            + "- Every value must be a finite numeric coefficient in raw-feature units.\n"
        )
    return (
        user_prompt.rstrip()
        + "\n\nOutput format (strict):\n"
        + "- Return only a JSON object. No markdown or code fences around the JSON.\n"
        + "- The JSON object must have exactly two top-level keys: 'reasoning' and 'coefficients'.\n"
        + "- 'reasoning' is a string: walk through the predictors and justify sign AND magnitude "
        + "given the feature-scale summaries and the domain. Multi-paragraph is fine.\n"
        + f"- 'coefficients' is an object whose keys are exactly (no more, no fewer): {keys}\n"
        + "- Every coefficient value must be a finite numeric coefficient in raw-feature units.\n"
    )


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #

class InvalidResponse(ValueError):
    """Response could not be parsed into a complete, finite coefficient vector."""


def parse(text: str, feature_names: Sequence[str]) -> tuple[dict[str, float], str | None]:
    """Extract (coefficients, reasoning) from a raw response.

    A response is rejected unless it carries exactly the requested keys and all
    values are finite. Rejected draws are resampled rather than repaired, so a
    partial or hallucinated key set never reaches the prior.
    """
    payload = _loads(text)
    reasoning = None
    if isinstance(payload, dict) and "coefficients" in payload:
        reasoning = payload.get("reasoning")
        payload = payload["coefficients"]
    if not isinstance(payload, dict):
        raise InvalidResponse("response is not a JSON object")

    expected = set(feature_names)
    missing, extra = expected - payload.keys(), payload.keys() - expected
    if missing or extra:
        raise InvalidResponse(f"key mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    coefficients = {}
    for name in feature_names:
        value = float(payload[name])
        if not _finite(value):
            raise InvalidResponse(f"non-finite coefficient for {name!r}")
        coefficients[name] = value
    return coefficients, reasoning


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Prompt-enforced paths (Anthropic reasoning) may wrap the JSON in prose.
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise InvalidResponse("no JSON object found in response")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidResponse(f"malformed JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Model:
    provider: str   # "openai" | "anthropic" | "google"
    name: str
    reasoning: bool = False

    @property
    def slug(self) -> str:
        """Directory name for this cell, matching the prior bundle layout."""
        base = f"{self.provider}__{self.name.replace('.', '-')}"
        return f"{base}__reasoning" if self.reasoning else base


def call(
    model: Model,
    *,
    system_prompt: str,
    user_prompt: str,
    feature_names: Sequence[str],
    temperature: float,
) -> str:
    """Dispatch one elicitation call and return the raw response text."""
    prompt = output_contract(user_prompt, feature_names, reasoning=model.reasoning)
    if model.provider == "openai":
        return _call_openai(model, system_prompt, prompt, feature_names, temperature)
    if model.provider == "anthropic":
        return _call_anthropic(model, system_prompt, prompt, feature_names, temperature)
    if model.provider == "google":
        return _call_google(model, system_prompt, prompt, feature_names, temperature)
    raise ValueError(f"unknown provider {model.provider!r}")


def _key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing {name} in environment")
    return value


def _call_openai(model, system_prompt, prompt, feature_names, temperature) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_key("OPENAI_API_KEY"), timeout=120.0, max_retries=0)
    schema = (reasoning_schema(feature_names) if model.reasoning
              else coefficient_schema(feature_names))
    params = {
        "model": model.name,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {"type": "json_schema", "name": TOOL_NAME, "schema": schema, "strict": True},
            "verbosity": "medium" if model.reasoning else "low",
        },
        # Identical in both modes: the R/NR contrast here is elicited reasoning,
        # not a change in how hard the model thinks internally.
        "reasoning": {"effort": "low"},
        "temperature": float(temperature),
        "max_output_tokens": 6000 if model.reasoning else 1600,
    }
    try:
        response = client.responses.create(**params)
    except Exception as exc:
        if "temperature" not in str(exc).lower():
            raise
        # Some reasoning models reject an explicit temperature. Dropping it
        # means the call runs at the provider default, so the effective
        # temperature for that cell is NOT the requested value.
        params.pop("temperature")
        response = client.responses.create(**params)
    return response.output_text or ""


def _call_anthropic(model, system_prompt, prompt, feature_names, temperature) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))

    if model.reasoning:
        # Extended thinking forbids forced tool use, so the schema is enforced
        # by the prompt and recovered from text. It also pins temperature to
        # 1.0 regardless of what the caller asked for.
        response = client.messages.create(
            model=model.name,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=ANTHROPIC_THINKING_TEMPERATURE,
            thinking={"type": "enabled", "budget_tokens": ANTHROPIC_THINKING_BUDGET},
        )
        return "\n".join(
            block.text for block in response.content
            if getattr(block, "type", None) != "thinking" and getattr(block, "text", None)
        )

    response = client.messages.create(
        model=model.name,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=float(temperature),
        tools=[{
            "name": TOOL_NAME,
            "description": "Return prior coefficients as a strict JSON object.",
            "input_schema": coefficient_schema(feature_names),
        }],
        tool_choice={"type": "tool", "name": TOOL_NAME},
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return json.dumps(block.input)
    raise InvalidResponse("no tool_use block in Anthropic response")


def _call_google(model, system_prompt, prompt, feature_names, temperature) -> str:
    import google.generativeai as genai

    genai.configure(api_key=_key("GEMINI_API_KEY"))
    client = genai.GenerativeModel(model.name, system_instruction=system_prompt)
    # No thinking configuration either way: Gemini 2.5 thinks by default, so
    # both modes reason internally and differ only in the response schema.
    schema = (reasoning_schema(feature_names, strict=False) if model.reasoning
              else coefficient_schema(feature_names))
    response = client.generate_content(
        prompt,
        generation_config={
            "temperature": float(temperature),
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    return (response.text or "").strip()


def elicit(
    model: Model,
    *,
    system_prompt: str,
    user_prompt: str,
    feature_names: Sequence[str],
    n_samples: int,
    temperature: float,
    max_attempts: int | None = None,
) -> tuple[list[dict[str, float]], list[str | None]]:
    """Draw `n_samples` valid coefficient vectors, resampling invalid responses.

    `max_attempts` bounds total calls so a misbehaving model cannot spend
    without limit; it defaults to four times the requested sample count.
    """
    max_attempts = max_attempts or 4 * n_samples
    samples: list[dict[str, float]] = []
    reasonings: list[str | None] = []

    for _ in range(max_attempts):
        if len(samples) >= n_samples:
            break
        try:
            text = call(model, system_prompt=system_prompt, user_prompt=user_prompt,
                        feature_names=feature_names, temperature=temperature)
            coefficients, reasoning = parse(text, feature_names)
        except InvalidResponse:
            continue
        samples.append(coefficients)
        reasonings.append(reasoning)

    if len(samples) < n_samples:
        raise RuntimeError(f"only {len(samples)}/{n_samples} valid samples within {max_attempts} attempts")
    return samples, reasonings
