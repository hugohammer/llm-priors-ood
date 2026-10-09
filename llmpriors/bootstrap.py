"""Percentile bootstrap over the evaluation set.

Models are fixed; only the evaluation sample is resampled. This measures how
sensitive a reported metric is to the composition of the test set. It does not
capture variability in training, hyperparameter selection, or elicitation.

Every model sees the *same* resample indices within a call, so the per-resample
draws are paired. That yields two tables:

  marginal   per-model CI on the model's own metric
  paired     CI on the per-resample difference (model - baseline)

Improvement-over-baseline claims should use the paired table: it accounts for
the correlation between a prior-informed model and its baseline on the same
evaluation sample, which the marginal intervals ignore.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .metrics import classification_metrics, mse


def _quantiles(ci_level: float) -> tuple[float, float]:
    alpha = 1.0 - float(ci_level)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    return alpha / 2.0, 1.0 - alpha / 2.0


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample within each class so class balance is preserved."""
    classes = np.unique(y)
    if classes.size < 2:
        return rng.integers(0, y.size, size=y.size)
    pieces = [rng.choice(np.flatnonzero(y == c), size=int(np.sum(y == c)), replace=True) for c in classes]
    indices = np.concatenate(pieces)
    rng.shuffle(indices)
    return indices


def _summarise(
    point: Mapping[str, Mapping[str, float]],
    draws: Mapping[str, Mapping[str, np.ndarray]],
    ci_level: float,
    n_resamples: int,
) -> pd.DataFrame:
    lower_q, upper_q = _quantiles(ci_level)
    rows = []
    for model, metrics in point.items():
        for metric, estimate in metrics.items():
            d = np.asarray(draws[model][metric], dtype=float)
            rows.append(
                {
                    "Model": model,
                    "Metric": metric,
                    "Estimate": float(estimate),
                    "BootstrapMean": float(np.nanmean(d)),
                    "BootstrapSE": float(np.nanstd(d, ddof=1)) if d.size > 1 else 0.0,
                    "CI_Lower": float(np.nanquantile(d, lower_q)),
                    "CI_Upper": float(np.nanquantile(d, upper_q)),
                    "CI_Level": float(ci_level),
                    "N_Resamples": int(n_resamples),
                }
            )
    return pd.DataFrame(rows)


def _summarise_paired(
    point: Mapping[str, Mapping[str, float]],
    draws: Mapping[str, Mapping[str, np.ndarray]],
    baseline: str,
    ci_level: float,
    n_resamples: int,
) -> pd.DataFrame:
    lower_q, upper_q = _quantiles(ci_level)
    base_point, base_draws = point[baseline], draws[baseline]
    rows = []
    for model, metrics in point.items():
        for metric, estimate in metrics.items():
            if metric not in base_point:
                continue
            d = np.asarray(draws[model][metric], dtype=float) - np.asarray(base_draws[metric], dtype=float)
            rows.append(
                {
                    "Model": model,
                    "Baseline": baseline,
                    "Metric": metric,
                    "Delta": float(estimate) - float(base_point[metric]),
                    "BootstrapMean": float(np.nanmean(d)),
                    "BootstrapSE": float(np.nanstd(d, ddof=1)) if d.size > 1 else 0.0,
                    "CI_Lower": float(np.nanquantile(d, lower_q)),
                    "CI_Upper": float(np.nanquantile(d, upper_q)),
                    "CI_Level": float(ci_level),
                    "N_Resamples": int(n_resamples),
                }
            )
    return pd.DataFrame(rows)


def _run(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    evaluate: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    index_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    baseline: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = np.asarray(y_true).ravel()
    point = {name: evaluate(y_true, np.asarray(p)) for name, p in predictions.items()}
    metric_names = tuple(next(iter(point.values())).keys())
    draws = {
        name: {m: np.empty(n_resamples, dtype=float) for m in metric_names}
        for name in predictions
    }

    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = index_fn(y_true, rng)
        y_boot = y_true[idx]
        for name, p in predictions.items():
            for metric, value in evaluate(y_boot, np.asarray(p)[idx]).items():
                draws[name][metric][i] = float(value)

    return (
        _summarise(point, draws, ci_level, n_resamples),
        _summarise_paired(point, draws, baseline, ci_level, n_resamples),
    )


def bootstrap_regression(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    baseline: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run(
        y_true,
        predictions,
        lambda y, p: {"MSE": mse(y, p)},
        lambda y, rng: rng.integers(0, y.size, size=y.size),
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        baseline=baseline,
    )


def bootstrap_classification(
    y_true: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    baseline: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _run(
        y_true,
        probabilities,
        classification_metrics,
        _stratified_indices,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        baseline=baseline,
    )
