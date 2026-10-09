"""Turning repeated LLM coefficient samples into prior quantities.

The LLM returns coefficients in *raw* feature units, one JSON object per draw.
Models are fitted on standardised features, so each sample is rescaled by the
feature standard deviations of whatever split is being fitted:

    beta_standardised = beta_raw * scale

That rescaling is redone per cross-validation fold, because each fold has its
own scaler fitted on its own training portion.

From the M rescaled samples come the three prior quantities used by the model
families: the mean, the per-feature variance, and the full covariance (whose
inverse is the precision).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import COVARIANCE_JITTER, VARIANCE_EPS


def load_samples(path: str | Path, max_samples: int | None = None) -> list[dict[str, float]]:
    """Load a prior bundle, optionally truncating to the first `max_samples`.

    Truncation is by position, not random: bundles holding more draws than the
    configured M are cut to the leading M so a bundle can grow without changing
    what earlier runs consumed.
    """
    with open(path) as f:
        samples = json.load(f)
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(f"prior bundle must be a list of >= 2 samples: {path}")
    if max_samples is not None:
        samples = samples[:max_samples]
    return [{str(k): float(v) for k, v in s.items()} for s in samples]


def as_matrix(samples: Sequence[dict[str, float]], feature_names: Sequence[str]) -> np.ndarray:
    """Stack samples into an (M, p) array with columns ordered by `feature_names`.

    Raises if the bundle's keys and the design matrix's features disagree: a
    silent mismatch here would misalign every coefficient.
    """
    expected = set(feature_names)
    missing = [f for f in feature_names if f not in samples[0]]
    extra = [k for k in samples[0] if k not in expected]
    if missing or extra:
        raise ValueError(f"prior keys do not match features. missing={missing[:5]} extra={extra[:5]}")
    return np.array([[s[f] for f in feature_names] for s in samples], dtype=float)


@dataclass(frozen=True)
class PriorQuantities:
    """Prior mean, variances and precision in standardised feature space."""

    mean: np.ndarray           # (p,)   mu_hat
    variances: np.ndarray      # (p,)   per-feature sample variance
    precision: np.ndarray      # (p, p) inverse sample covariance

    @property
    def identity(self) -> np.ndarray:
        """P = I, for the Prior-Mean family."""
        return np.eye(self.mean.size)

    @property
    def diagonal_precision(self):
        """Independent-Prior precision, as a function of alpha.

        Returns a callable rather than a matrix so the diagonal is formed as
        `alpha / (var_j + eps)` in a single division, matching how this family
        is defined. See `estimators.Precision`.
        """
        variances = self.variances

        def precision(alpha: float) -> np.ndarray:
            return np.diag(alpha / (variances + VARIANCE_EPS))

        return precision


def prior_quantities(
    samples: Sequence[dict[str, float]],
    feature_names: Sequence[str],
    scale: np.ndarray,
) -> PriorQuantities:
    """Rescale raw-unit samples to standardised space and summarise them.

    `scale` is the StandardScaler `scale_` of the split being fitted.

    The covariance is jittered before inversion (see COVARIANCE_JITTER) so it
    is invertible.
    """
    raw = as_matrix(samples, feature_names)
    standardised = raw * np.asarray(scale, dtype=float)

    mean = standardised.mean(axis=0)
    ddof = 1 if standardised.shape[0] > 1 else 0
    variances = standardised.var(axis=0, ddof=ddof)

    covariance = np.cov(standardised, rowvar=False) + COVARIANCE_JITTER * np.eye(mean.size)
    precision = np.linalg.pinv(covariance)

    return PriorQuantities(mean=mean, variances=variances, precision=precision)


def raw_mean(samples: Sequence[dict[str, float]], feature_names: Sequence[str]) -> np.ndarray:
    """Prior mean in raw feature units, before any standardisation."""
    return as_matrix(samples, feature_names).mean(axis=0)
