"""Building the elicitation prompts.

A prompt is two parts:

  system  stable elicitation rules -- the role to adopt, that coefficients are
          in raw feature units, that there is no intercept, and what the output
          must look like. Varies only with the elicitation mode.
  user    dataset-specific content -- the modelling task, domain context,
          feature meanings, and feature-scale summaries computed from the
          source-domain training pool.

Two rules shape what goes in:

  * Scale summaries come from the source training pool only. The ID test set
    and the OOD set never inform a prompt.
  * The prompt describes the source domain but never the OOD target. Boston's
    prompt says the source is newer-housing tracts; it does not say the
    evaluation is on older ones. Any OOD gain therefore reflects domain
    knowledge rather than knowing the shift in advance.

The intercept is excluded because it is fixed by the outcome's base rate and
the feature centring, neither of which needs domain expertise.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .prompt_content import CONTENT


def _summary_stats(series: pd.Series) -> dict:
    """Distribution summary for one feature, in raw units."""
    values = pd.to_numeric(series, errors="coerce")
    n_total = int(values.shape[0])
    present = values.dropna()

    base = {
        "n_total": n_total,
        "n_non_null": int(present.shape[0]),
        "missing_count": n_total - int(present.shape[0]),
        "missing_rate": float((n_total - present.shape[0]) / n_total) if n_total else 1.0,
    }
    if present.empty:
        return {**base, "summary_kind": "unavailable"}

    array = present.to_numpy(dtype=float)
    if set(np.unique(array)).issubset({0, 1}):
        prevalence = float(array.mean())
        return {**base, "summary_kind": "binary_indicator",
                "prevalence_1": prevalence, "prevalence_0": float(1.0 - prevalence)}

    q05, q25, q50, q75, q95 = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        **base, "summary_kind": "continuous_numeric",
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "std": float(array.std()),
        "q05": float(q05), "q25": float(q25), "q50": float(q50),
        "q75": float(q75), "q95": float(q95),
    }


def _summary_text(stats: Mapping[str, object]) -> str:
    kind = str(stats.get("summary_kind"))
    missing_rate = float(stats.get("missing_rate", 0.0))
    suffix = f", missing={missing_rate:.3f}" if missing_rate > 0 else ""

    if kind == "binary_indicator":
        return f"binary 0/1 indicator; prevalence={float(stats['prevalence_1']):.3f}{suffix}"
    if kind == "continuous_numeric":
        return (
            "continuous/raw numeric; "
            f"median={float(stats['q50']):.3g}, "
            f"IQR=[{float(stats['q25']):.3g}, {float(stats['q75']):.3g}], "
            f"central95%≈[{float(stats['q05']):.3g}, {float(stats['q95']):.3g}]"
            f"{suffix}"
        )
    return "summary unavailable"


def feature_summaries(X: pd.DataFrame, feature_names: Sequence[str]) -> str:
    """One `- name: summary` line per feature, in design-matrix order."""
    return "\n".join(f"- {name}: {_summary_text(_summary_stats(X[name]))}" for name in feature_names)


def summaries_payload(X: pd.DataFrame, feature_names: Sequence[str]) -> dict:
    """Machine-readable form of the same summaries, saved alongside a bundle."""
    return {
        "row_count": int(X.shape[0]),
        "feature_count": len(feature_names),
        "features": {name: _summary_stats(X[name]) for name in feature_names},
    }


def system_prompt(dataset: str, *, sample_mode: bool = True, reasoning: bool = False) -> str:
    """Stable elicitation rules.

    `sample_mode` asks for one draw from the model's belief distribution rather
    than a central estimate -- that is what makes repeated calls informative
    about uncertainty instead of returning the same vector each time.
    """
    content = CONTENT[dataset]
    rules = [
        "You are helping construct informative priors for a regression model.",
        f"Adopt the perspective of {content['role']}.",
        "Use the feature definitions and feature-scale summaries to reason about BOTH sign and magnitude.",
        "All coefficients must be given in the original raw-feature units used by the model matrix.",
        "Do not standardize, rescale, or reinterpret the coefficients as standardized effects.",
        "Do not include an intercept.",
    ]

    if reasoning:
        rules += [
            "Before committing to numbers, reason step-by-step: for each predictor, weigh its likely sign, "
            "its magnitude given the feature-scale summary, and the mechanism that links it to the outcome. "
            "Also consider how predictors overlap, so that correlated variables share signal rather than each getting a large coefficient.",
            "Record that reasoning in the 'reasoning' field of the response JSON, then write the numeric coefficients in the 'coefficients' field.",
            "Return a single JSON object with exactly the keys 'reasoning' and 'coefficients'. Do not wrap it in markdown or code fences.",
        ]
    else:
        rules += [
            "Output must be only a JSON object matching the provided schema.",
            "Do not include explanations, markdown, code fences, or extra keys.",
        ]

    if sample_mode:
        rules += [
            "Return one plausible coefficient vector draw from your belief distribution.",
            "Do not collapse uncertainty into a single average if multiple plausible settings exist.",
            "Keep the draw realistic and domain-grounded rather than intentionally noisy.",
        ]
    else:
        rules += [
            "Return your best central estimate of the coefficient vector.",
            "Prefer stable, representative values over edge-case values.",
        ]

    rules += [
        "Prefer interpretable, conservative, domain-plausible coefficients.",
        "Avoid extreme magnitudes unless they are clearly justified by the variable scale and domain mechanism.",
    ]
    return "\n".join(rules).strip() + "\n"


def user_prompt(dataset: str, X: pd.DataFrame, feature_names: Sequence[str]) -> str:
    """Dataset-specific content, including scale summaries computed from `X`.

    `X` must be the source-domain training pool.
    """
    content = CONTENT[dataset]
    descriptions = content["feature_descriptions"]

    def bullets(lines: Sequence[str]) -> str:
        return "\n".join(f"- {line}" for line in lines) if lines else "- None"

    sections = [
        "Task context",
        content["outcome_desc"],
        "",
        "Domain context:",
        bullets(content["domain_context"]),
        "",
        "Dataset notes:",
        bullets(content["extra_notes"]),
        "",
        "Guidance:",
        bullets(content["guidance"]),
        "",
        "Feature meanings:",
        "\n".join(
            f"- {name}: {descriptions.get(name, 'raw predictor from the dataset')}"
            for name in feature_names
        ),
        "",
        "Feature scale summaries from the actual design matrix:",
        feature_summaries(X, feature_names),
    ]

    if content["signal_hints"]:
        sections += ["", "Relevant domain patterns to weigh alongside the summaries:",
                     bullets(content["signal_hints"])]

    sections += [
        "",
        "Important modeling reminders:",
        "- Binary indicators use 0/1 coding, so their coefficients represent the change associated with turning the indicator on.",
        "- Continuous-feature coefficients should be scaled to the feature summaries shown above; do not assume a one-unit change is small unless the summary suggests it is.",
        "- When multiple variables capture overlapping structure, distribute signal conservatively rather than giving each one a large effect.",
        "- The output schema already fixes the keys. Choose only the numeric values.",
    ]
    return "\n".join(sections).strip() + "\n"
