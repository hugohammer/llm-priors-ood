"""Running one (dataset, LLM variant) cell end to end.

The protocol is the same for both task types:

  1. Split the source domain into an 80% training pool and a 20% ID test set.
     The OOD set is whatever the dataset's domain splitter put on the other side.
  2. Build CV folds on the training pool. Fit preprocessing inside each fold and
     rescale the prior to that fold's feature scales.
  3. Tune alpha per family by strict-best mean CV, then gamma for the Mixed
     family using the already-chosen alphas.
  4. Refit every family on the full training pool and evaluate once on the ID
     test set and once on the OOD set.
  5. Bootstrap the evaluation sets for marginal and paired-vs-baseline CIs.

Only the training pool is ever used for tuning. The ID test set and the OOD set
are untouched until step 4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.model_selection import train_test_split

from . import config as cfg
from .bootstrap import bootstrap_classification, bootstrap_regression
from .crossval import Selection, fold_hash, make_folds, prepare_folds, select_best_mean
from .estimators import GaussianPriorLogistic, GaussianPriorRidge
from .metrics import classification_metrics, mse, tuning_score
from .preprocessing import Preprocessor
from .priors import prior_quantities, raw_mean


def _ridge(alpha: float) -> SklearnRidge:
    """sklearn Ridge on this study's penalty scale.

    sklearn minimises ||y - Xb||^2 + alpha*||b||^2, while every prior family
    here uses 0.5*alpha*penalty. Halving alpha on the way in puts the baseline
    on the same scale, so tuned alpha values are comparable across families.
    """
    return SklearnRidge(alpha=0.5 * float(alpha), fit_intercept=True)


@dataclass
class CellResult:
    id_metrics: pd.DataFrame
    ood_metrics: pd.DataFrame
    chosen: dict[str, float]


def _split_source(X: np.ndarray, y: np.ndarray, *, seed: int, stratify: bool):
    return train_test_split(
        X, y,
        test_size=cfg.ID_TEST_FRACTION,
        random_state=seed,
        stratify=y if stratify else None,
    )


def _write(out_dir: Path, name: str, frame: pd.DataFrame) -> None:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(out_dir: Path, name: str, payload) -> None:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _common_outputs(
    out_dir: Path,
    *,
    run_name: str,
    seed: int,
    stratified: bool,
    folds,
    chosen: dict[str, float],
    selections: dict[str, Selection],
    n_train: int,
    n_id: int,
    n_ood: int,
    feature_names: list[str],
    samples: list[dict[str, float]],
    prior_raw: np.ndarray,
    final_scale: np.ndarray,
    fold_scales: list[np.ndarray],
) -> None:
    _write_json(out_dir, "config.json", {
        "run_name": run_name,
        "seed": seed,
        "cv": {
            "n_folds": len(folds),
            "selection_rule": "strict_best_mean",
            "stratified": stratified,
            "fold_indices_sha256": fold_hash(folds),
            "fold_val_sizes": [int(v.size) for _, v in folds],
        },
        "best": chosen,
        "se_at_best": {k: s.se_at_best for k, s in selections.items()},
        "bootstrap": {
            "n_resamples": cfg.BOOTSTRAP_RESAMPLES,
            "ci_level": cfg.BOOTSTRAP_CI_LEVEL,
            "scope": "fixed-model evaluation-set bootstrap",
        },
    })
    _write_json(out_dir, "split_sizes.json", {"train_full": n_train, "id_test": n_id, "ood": n_ood})
    _write_json(out_dir, "priors_raw_samples.json", samples)
    _write_json(out_dir, "priors_raw_single.json", dict(zip(feature_names, prior_raw.tolist())))

    scaling = {"feature": feature_names, "std_final": final_scale, "prior_raw_mean": prior_raw}
    for i, s in enumerate(fold_scales):
        scaling[f"std_fold{i}"] = s
    _write(out_dir, "prior_scaling.csv", pd.DataFrame(scaling))


def run_regression(
    *,
    run_name: str,
    out_dir: str | Path,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_ood: np.ndarray,
    y_ood: np.ndarray,
    feature_names: list[str],
    samples: list[dict[str, float]],
    seed: int = cfg.SEED,
) -> CellResult:
    out_dir = Path(out_dir)
    labels = cfg.REGRESSION_LABELS
    alpha_grid = cfg.ALPHA_GRID

    X_train, X_id, y_train, y_id = _split_source(X_source, y_source, seed=seed, stratify=False)

    pre = Preprocessor().fit(X_train)
    X_train_s, X_id_s, X_ood_s = pre.transform(X_train), pre.transform(X_id), pre.transform(X_ood)
    prior = prior_quantities(samples, feature_names, pre.scale)
    prior_raw = raw_mean(samples, feature_names)

    folds = make_folds(X_train, y_train, n_folds=cfg.N_FOLDS, seed=seed, stratify=False)
    fold_data = prepare_folds(X_train, y_train, folds, samples, feature_names)

    # (mu, P) per family, using each fold's own rescaled prior during tuning.
    families = {
        "prior_mean": lambda p: (p.mean, p.identity),
        "correlated": lambda p: (p.mean, p.precision),
        "independent": lambda p: (p.mean, p.diagonal_precision),
    }

    selections: dict[str, Selection] = {}

    scores = np.zeros((len(folds), alpha_grid.size))
    for i, fd in enumerate(fold_data):
        for j, alpha in enumerate(alpha_grid):
            model = _ridge(alpha).fit(fd.X_train, fd.y_train)
            scores[i, j] = mse(fd.y_val, model.predict(fd.X_val))
    selections["baseline"] = select_best_mean(alpha_grid, scores, higher_is_better=False)

    for family, spec in families.items():
        scores = np.zeros((len(folds), alpha_grid.size))
        for i, fd in enumerate(fold_data):
            mu, P = spec(fd.prior)
            for j, alpha in enumerate(alpha_grid):
                model = GaussianPriorRidge(mu=mu, P=P, alpha=float(alpha)).fit(fd.X_train, fd.y_train)
                scores[i, j] = mse(fd.y_val, model.predict(fd.X_val))
        selections[family] = select_best_mean(alpha_grid, scores, higher_is_better=False)

    # Mixed: blend predictions of the two already-tuned models across gamma.
    gamma_scores = np.zeros((len(folds), cfg.GAMMA_GRID.size))
    for i, fd in enumerate(fold_data):
        p_base = _ridge(selections["baseline"].value).fit(fd.X_train, fd.y_train).predict(fd.X_val)
        mu, P = families["prior_mean"](fd.prior)
        p_prior = GaussianPriorRidge(mu=mu, P=P, alpha=selections["prior_mean"].value) \
            .fit(fd.X_train, fd.y_train).predict(fd.X_val)
        for j, gamma in enumerate(cfg.GAMMA_GRID):
            gamma_scores[i, j] = mse(fd.y_val, (1 - gamma) * p_base + gamma * p_prior)
    gamma_selection = select_best_mean(cfg.GAMMA_GRID, gamma_scores, higher_is_better=False)
    gamma = gamma_selection.value

    # Deployment fits on the full training pool.
    fitted = {
        labels["unregularised"]: LinearRegression().fit(X_train_s, y_train),
        labels["baseline"]: _ridge(selections["baseline"].value).fit(X_train_s, y_train),
    }
    for family, spec in families.items():
        mu, P = spec(prior)
        fitted[labels[family]] = GaussianPriorRidge(mu=mu, P=P, alpha=selections[family].value) \
            .fit(X_train_s, y_train)

    def predictions(X: np.ndarray) -> dict[str, np.ndarray]:
        preds = {name: model.predict(X) for name, model in fitted.items()}
        preds[cfg.mixed_label(gamma)] = (
            (1 - gamma) * preds[labels["baseline"]] + gamma * preds[labels["prior_mean"]]
        )
        return preds

    pred_id, pred_ood = predictions(X_id_s), predictions(X_ood_s)
    id_metrics = pd.DataFrame([{"Model": n, "MSE": mse(y_id, p)} for n, p in pred_id.items()])
    ood_metrics = pd.DataFrame([{"Model": n, "MSE": mse(y_ood, p)} for n, p in pred_ood.items()])

    id_marginal, id_paired = bootstrap_regression(
        y_id, pred_id, n_resamples=cfg.BOOTSTRAP_RESAMPLES, ci_level=cfg.BOOTSTRAP_CI_LEVEL,
        seed=seed, baseline=labels["baseline"])
    ood_marginal, ood_paired = bootstrap_regression(
        y_ood, pred_ood, n_resamples=cfg.BOOTSTRAP_RESAMPLES, ci_level=cfg.BOOTSTRAP_CI_LEVEL,
        seed=seed + 1, baseline=labels["baseline"])

    chosen = {f"{k}_alpha": v.value for k, v in selections.items()} | {"mixed_gamma": gamma}
    _common_outputs(
        out_dir, run_name=run_name, seed=seed, stratified=False, folds=folds, chosen=chosen,
        selections=selections, n_train=len(y_train), n_id=len(y_id), n_ood=len(y_ood),
        feature_names=feature_names, samples=samples, prior_raw=prior_raw,
        final_scale=pre.scale, fold_scales=[fd.scale for fd in fold_data],
    )
    _write(out_dir, "id_metrics.csv", id_metrics)
    _write(out_dir, "ood_metrics.csv", ood_metrics)
    _write(out_dir, "bootstrap/id_metrics_bootstrap.csv", id_marginal)
    _write(out_dir, "bootstrap/ood_metrics_bootstrap.csv", ood_marginal)
    _write(out_dir, "bootstrap/id_metrics_bootstrap_diff.csv", id_paired)
    _write(out_dir, "bootstrap/ood_metrics_bootstrap_diff.csv", ood_paired)
    for family, selection in selections.items():
        _write(out_dir, f"tuning/{family}_alpha_tuning.csv",
               selection.to_frame(grid_column="alpha", score_column="val_mse", higher_is_better=False))
    _write(out_dir, "tuning/mixed_gamma_tuning.csv",
           gamma_selection.to_frame(grid_column="gamma", score_column="val_mse", higher_is_better=False))

    return CellResult(id_metrics=id_metrics, ood_metrics=ood_metrics, chosen=chosen)


def run_classification(
    *,
    run_name: str,
    out_dir: str | Path,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_ood: np.ndarray,
    y_ood: np.ndarray,
    feature_names: list[str],
    samples: list[dict[str, float]],
    seed: int = cfg.SEED,
    tune_metric: str = "auc",
) -> CellResult:
    out_dir = Path(out_dir)
    labels = cfg.CLASSIFICATION_LABELS
    alpha_grid = cfg.ALPHA_GRID
    # Baseline tunes on alpha_equiv = 1/C so its axis matches the prior families'.
    alpha_equiv_grid = np.sort(1.0 / cfg.C_GRID)

    X_train, X_id, y_train, y_id = _split_source(X_source, y_source, seed=seed, stratify=True)

    pre = Preprocessor().fit(X_train)
    X_train_s, X_id_s, X_ood_s = pre.transform(X_train), pre.transform(X_id), pre.transform(X_ood)
    prior = prior_quantities(samples, feature_names, pre.scale)
    prior_raw = raw_mean(samples, feature_names)

    folds = make_folds(X_train, y_train, n_folds=cfg.N_FOLDS, seed=seed, stratify=True)
    fold_data = prepare_folds(X_train, y_train, folds, samples, feature_names)

    # Correlated-Prior is fitted from a zero start; the other two from a seeded
    # random start. This mirrors the published configuration and changes the
    # L-BFGS path, so it is part of the specification rather than an accident.
    families = {
        "prior_mean": (lambda p: (p.mean, p.identity), seed),
        "correlated": (lambda p: (p.mean, p.precision), None),
        "independent": (lambda p: (p.mean, p.diagonal_precision), seed),
    }

    selections: dict[str, Selection] = {}

    scores = np.zeros((len(folds), alpha_equiv_grid.size))
    for i, fd in enumerate(fold_data):
        for j, alpha_equiv in enumerate(alpha_equiv_grid):
            model = LogisticRegression(C=1.0 / float(alpha_equiv), penalty="l2",
                                       solver="lbfgs", max_iter=5000).fit(fd.X_train, fd.y_train)
            scores[i, j] = tuning_score(fd.y_val, model.predict_proba(fd.X_val)[:, 1], tune_metric)
    selections["baseline"] = select_best_mean(alpha_equiv_grid, scores, higher_is_better=True)
    best_C = 1.0 / selections["baseline"].value

    for family, (spec, random_state) in families.items():
        scores = np.zeros((len(folds), alpha_grid.size))
        for i, fd in enumerate(fold_data):
            mu, P = spec(fd.prior)
            # One model per fold, walking alpha downwards so each fit
            # warm-starts from the previous (more regularised) solution.
            model = GaussianPriorLogistic(mu=mu, P=P, alpha=float(alpha_grid[-1]),
                                          random_state=random_state)
            for j in range(alpha_grid.size - 1, -1, -1):
                model.alpha = float(alpha_grid[j])
                model.fit(fd.X_train, fd.y_train)
                scores[i, j] = tuning_score(fd.y_val, model.predict_proba(fd.X_val), tune_metric)
        selections[family] = select_best_mean(alpha_grid, scores, higher_is_better=True)

    gamma_scores = np.zeros((len(folds), cfg.GAMMA_GRID.size))
    for i, fd in enumerate(fold_data):
        base = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=5000) \
            .fit(fd.X_train, fd.y_train)
        p_base = base.predict_proba(fd.X_val)[:, 1]
        spec, random_state = families["prior_mean"]
        mu, P = spec(fd.prior)
        p_prior = GaussianPriorLogistic(mu=mu, P=P, alpha=selections["prior_mean"].value,
                                        random_state=random_state) \
            .fit(fd.X_train, fd.y_train).predict_proba(fd.X_val)
        for j, gamma in enumerate(cfg.GAMMA_GRID):
            gamma_scores[i, j] = tuning_score(fd.y_val, (1 - gamma) * p_base + gamma * p_prior, tune_metric)
    gamma_selection = select_best_mean(cfg.GAMMA_GRID, gamma_scores, higher_is_better=True)
    gamma = gamma_selection.value

    unregularised = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs",
                                       max_iter=10000).fit(X_train_s, y_train)
    baseline = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs",
                                  max_iter=5000).fit(X_train_s, y_train)
    fitted = {}
    for family, (spec, random_state) in families.items():
        mu, P = spec(prior)
        fitted[labels[family]] = GaussianPriorLogistic(mu=mu, P=P, alpha=selections[family].value,
                                                       random_state=random_state).fit(X_train_s, y_train)

    def probabilities(X: np.ndarray) -> dict[str, np.ndarray]:
        probs = {
            labels["unregularised"]: unregularised.predict_proba(X)[:, 1],
            labels["baseline"]: baseline.predict_proba(X)[:, 1],
        }
        for name, model in fitted.items():
            probs[name] = model.predict_proba(X)
        probs[cfg.mixed_label(gamma)] = (
            (1 - gamma) * probs[labels["baseline"]] + gamma * probs[labels["prior_mean"]]
        )
        return probs

    prob_id, prob_ood = probabilities(X_id_s), probabilities(X_ood_s)
    id_metrics = pd.DataFrame([{"Model": n, **classification_metrics(y_id, p)} for n, p in prob_id.items()])
    ood_metrics = pd.DataFrame([{"Model": n, **classification_metrics(y_ood, p)} for n, p in prob_ood.items()])

    id_marginal, id_paired = bootstrap_classification(
        y_id, prob_id, n_resamples=cfg.BOOTSTRAP_RESAMPLES, ci_level=cfg.BOOTSTRAP_CI_LEVEL,
        seed=seed, baseline=labels["baseline"])
    ood_marginal, ood_paired = bootstrap_classification(
        y_ood, prob_ood, n_resamples=cfg.BOOTSTRAP_RESAMPLES, ci_level=cfg.BOOTSTRAP_CI_LEVEL,
        seed=seed + 1, baseline=labels["baseline"])

    chosen = {f"{k}_alpha": v.value for k, v in selections.items()} | {
        "baseline_C": best_C, "mixed_gamma": gamma}
    _common_outputs(
        out_dir, run_name=run_name, seed=seed, stratified=True, folds=folds, chosen=chosen,
        selections=selections, n_train=len(y_train), n_id=len(y_id), n_ood=len(y_ood),
        feature_names=feature_names, samples=samples, prior_raw=prior_raw,
        final_scale=pre.scale, fold_scales=[fd.scale for fd in fold_data],
    )
    _write(out_dir, "id_metrics.csv", id_metrics)
    _write(out_dir, "ood_metrics.csv", ood_metrics)
    _write(out_dir, "bootstrap/id_metrics_bootstrap.csv", id_marginal)
    _write(out_dir, "bootstrap/ood_metrics_bootstrap.csv", ood_marginal)
    _write(out_dir, "bootstrap/id_metrics_bootstrap_diff.csv", id_paired)
    _write(out_dir, "bootstrap/ood_metrics_bootstrap_diff.csv", ood_paired)
    grid_col = {"baseline": "alpha_equiv"}
    for family, selection in selections.items():
        _write(out_dir, f"tuning/{family}_alpha_tuning.csv",
               selection.to_frame(grid_column=grid_col.get(family, "alpha"),
                                  score_column="val_metric", higher_is_better=True))
    _write(out_dir, "tuning/mixed_gamma_tuning.csv",
           gamma_selection.to_frame(grid_column="gamma", score_column="val_metric", higher_is_better=True))

    return CellResult(id_metrics=id_metrics, ood_metrics=ood_metrics, chosen=chosen)
