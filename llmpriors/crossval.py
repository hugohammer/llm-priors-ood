"""Fold construction, per-fold preprocessing, and hyperparameter selection.

Selection rule: **strict-best mean CV**. For each grid value, average the
per-fold validation score and take the best average. The standard error across
folds is computed and reported for diagnostics but does not influence the
choice.

Folds are built once per dataset from (seed, n_folds, stratify) and reused by
every LLM variant of that dataset, so all variants are scored on identical
splits. The SHA of the fold indices is written into each run's config so that
invariant is checkable after the fact.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .preprocessing import Preprocessor
from .priors import PriorQuantities, prior_quantities

Fold = tuple[np.ndarray, np.ndarray]


def make_folds(X: np.ndarray, y: np.ndarray, *, n_folds: int, seed: int, stratify: bool) -> list[Fold]:
    """Return `n_folds` (train_idx, val_idx) pairs, deterministic in `seed`."""
    splitter_cls = StratifiedKFold if stratify else KFold
    splitter = splitter_cls(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in splitter.split(X, y)]


def fold_hash(folds: list[Fold]) -> str:
    """SHA-256 over the fold index byte stream, for cross-variant comparison."""
    h = hashlib.sha256()
    for train_idx, val_idx in folds:
        h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
        h.update(b"|")
        h.update(np.asarray(val_idx, dtype=np.int64).tobytes())
        h.update(b";")
    return h.hexdigest()


@dataclass(frozen=True)
class FoldData:
    """One fold, preprocessed and with prior quantities on that fold's scale."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    prior: PriorQuantities
    scale: np.ndarray


def prepare_folds(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    samples: list[dict[str, float]],
    feature_names: list[str],
) -> list[FoldData]:
    """Fit the imputer/scaler inside each fold and rescale the prior to match.

    Preprocessing is fitted on the fold's training portion only, so the
    validation portion is never seen by the imputer or scaler. The prior is
    rescaled with that same fold's standard deviations.
    """
    prepared = []
    for train_idx, val_idx in folds:
        pre = Preprocessor().fit(X[train_idx])
        prepared.append(
            FoldData(
                X_train=pre.transform(X[train_idx]),
                y_train=y[train_idx],
                X_val=pre.transform(X[val_idx]),
                y_val=y[val_idx],
                prior=prior_quantities(samples, feature_names, pre.scale),
                scale=pre.scale,
            )
        )
    return prepared


@dataclass(frozen=True)
class Selection:
    """Outcome of a grid search over one hyperparameter."""

    grid: np.ndarray
    fold_scores: np.ndarray   # (n_folds, n_grid)
    mean: np.ndarray
    std: np.ndarray
    se: np.ndarray
    index: int

    @property
    def value(self) -> float:
        return float(self.grid[self.index])

    @property
    def se_at_best(self) -> float:
        return float(self.se[self.index])

    def to_frame(self, *, grid_column: str, score_column: str, higher_is_better: bool) -> pd.DataFrame:
        """Tuning table, one row per grid value, best first."""
        chosen = np.zeros(self.grid.size, dtype=bool)
        chosen[self.index] = True
        frame = pd.DataFrame(
            {
                grid_column: self.grid,
                score_column: self.mean,
                f"{score_column}_std": self.std,
                f"{score_column}_se": self.se,
                "is_chosen": chosen,
            }
        )
        for i in range(self.fold_scores.shape[0]):
            frame[f"{score_column}_fold{i}"] = self.fold_scores[i]
        return frame.sort_values(score_column, ascending=not higher_is_better).reset_index(drop=True)


def select_best_mean(grid: np.ndarray, fold_scores: np.ndarray, *, higher_is_better: bool) -> Selection:
    """Pick the grid value with the best mean score across folds."""
    grid = np.asarray(grid, dtype=float)
    fold_scores = np.asarray(fold_scores, dtype=float)
    if fold_scores.shape[1] != grid.size:
        raise ValueError(f"fold_scores {fold_scores.shape} does not match grid size {grid.size}")

    mean = fold_scores.mean(axis=0)
    n_folds = fold_scores.shape[0]
    std = fold_scores.std(axis=0, ddof=1) if n_folds > 1 else np.zeros_like(mean)

    return Selection(
        grid=grid,
        fold_scores=fold_scores,
        mean=mean,
        std=std,
        se=std / np.sqrt(n_folds),
        index=int(np.argmax(mean) if higher_is_better else np.argmin(mean)),
    )
