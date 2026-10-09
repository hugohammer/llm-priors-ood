"""Evaluation metrics.

Regression reports MSE. Classification reports the threshold-free metrics (AUC,
LogLoss, Brier) alongside threshold-0.5 metrics (Acc, Precision, Sensitivity,
Specificity, F1).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CLASSIFICATION_THRESHOLD

_EPS = 1e-12


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def classification_metrics(
    y_true: np.ndarray,
    p1: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Metric dict for predicted P(y=1).

    Specificity is computed directly rather than via `recall_score` on the
    flipped labels so that an evaluation sample containing no negatives yields
    NaN instead of a silent zero. AUC is NaN when only one class is present,
    which happens in some bootstrap resamples of small evaluation sets.
    """
    p1 = np.clip(np.asarray(p1, dtype=float), _EPS, 1 - _EPS)
    y_true = np.asarray(y_true).astype(int)
    y_pred = (p1 >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, p1))
    except ValueError:
        auc = float("nan")

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    n_negative = true_negative + int(np.sum((y_true == 0) & (y_pred == 1)))
    specificity = (true_negative / n_negative) if n_negative > 0 else float("nan")

    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred),
        "AUC": auc,
        "LogLoss": log_loss(y_true, p1),
        "Brier": brier_score_loss(y_true, p1),
    }


def tuning_score(y_true: np.ndarray, p1: np.ndarray, metric: str) -> float:
    """Validation score for classification tuning, always higher-is-better."""
    p1 = np.clip(np.asarray(p1, dtype=float), _EPS, 1 - _EPS)
    if metric == "auc":
        return float(roc_auc_score(y_true, p1))
    if metric == "logloss":
        return -float(log_loss(y_true, p1))
    raise ValueError(f"unknown tuning metric: {metric}")
