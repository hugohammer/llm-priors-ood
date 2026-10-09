"""Shared scaffolding for the four follow-up experiments.

Everything here follows the thesis pipeline's own conventions so the numbers
are comparable to the published tables: the same alpha grid, the same 5-fold
strict-best-mean selection rule, preprocessing and prior rescaling fitted
inside each fold, and the same seed.

The deployed (variant, family) cells are transcribed from the thesis result
tables so these experiments analyse the models the thesis actually deployed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # repository root: llmpriors/, data/ and priors/ live here
sys.path.insert(0, str(REPO))

from llmpriors import config as cfg  # noqa: E402
from llmpriors import datasets, priors  # noqa: E402
from llmpriors.estimators import GaussianPriorLogistic, GaussianPriorRidge  # noqa: E402
from llmpriors.preprocessing import Preprocessor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge as SklearnRidge  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402

DATA_ROOT = REPO / "data"
PRIOR_ROOT = REPO / "priors"
OUT = HERE / "out"

# Thesis label -> prior bundle directory.
VARIANT_DIR = {
    "Opus 4.5": "anthropic__opus-4-5",
    "Opus 4.5 (R)": "anthropic__opus-4-5__reasoning",
    "Haiku 4.5": "anthropic__haiku-4-5",
    "Haiku 4.5 (R)": "anthropic__haiku-4-5__reasoning",
    "GPT-5.1": "openai__gpt-5-1",
    "GPT-5.1 (R)": "openai__gpt-5-1__reasoning",
    "GPT-5-mini": "openai__gpt-5-mini",
    "GPT-5-mini (R)": "openai__gpt-5-mini__reasoning",
    "Gemini Pro": "google__gemini-2-5-pro",
    "Gemini Pro (R)": "google__gemini-2-5-pro__reasoning",
    "Gemini Flash": "google__gemini-2-5-flash",
    "Gemini Flash (R)": "google__gemini-2-5-flash__reasoning",
}

TASK = {
    "boston_housing": "regression",
    "cancer": "regression",
    "heart_disease": "classification",
    "diabetes": "classification",
    "adult": "classification",
}

PRETTY = {
    "boston_housing": "Boston Housing",
    "cancer": "Cancer",
    "heart_disease": "Heart Disease",
    "diabetes": "Diabetes",
    "adult": "Adult",
}

FAMILIES = ["prior_mean", "independent", "correlated", "mixed"]

FAMILY_LABEL = {
    "prior_mean": "Prior-Mean",
    "independent": "Independent-Prior",
    "correlated": "Correlated-Prior",
    "mixed": "Mixed",
}

#: Deployed cells, transcribed from the thesis tables 4-1 .. 4-8.
#: value = (variant label, tuned lambda) ; for "mixed", lambda is the tuned gamma.
DEPLOYED: dict[str, dict[str, tuple[str, float]]] = {
    # Re-selected after dropping the B feature (see DROP_FEATURES).
    "boston_housing": {
        "baseline": ("-", 25.1189),
        "prior_mean": ("Gemini Pro", 251.189),
        "independent": ("Opus 4.5", 10.0),
        "correlated": ("Opus 4.5", 15.8489),
        "mixed": ("Gemini Pro", 1.00),
    },
    "cancer": {
        "baseline": ("-", 100.0),
        "prior_mean": ("Opus 4.5", 158.489),
        "independent": ("GPT-5.1", 158.489),
        "correlated": ("Gemini Pro (R)", 158.489),
        "mixed": ("Opus 4.5", 0.68),
    },
    "heart_disease": {
        "baseline": ("-", 25.1189),
        "prior_mean": ("Opus 4.5", 1000.0),
        "independent": ("Gemini Pro (R)", 0.251189),
        "correlated": ("Opus 4.5", 0.01),
        "mixed": ("Opus 4.5", 1.00),
    },
    "diabetes": {
        "baseline": ("-", 25.1189),
        "prior_mean": ("Opus 4.5", 398.107),
        "independent": ("Opus 4.5", 0.158489),
        "correlated": ("Opus 4.5", 0.630957),
        "mixed": ("Opus 4.5", 0.97),
    },
    "adult": {
        "baseline": ("-", 1.0),
        "prior_mean": ("GPT-5-mini", 1.0),
        "independent": ("Gemini Flash (R)", 0.000630957),
        "correlated": ("Opus 4.5 (R)", 0.001),
        "mixed": ("GPT-5-mini (R)", 0.95),
    },
}


#: Features removed from a dataset before fitting. Boston Housing's "B" is
#: 1000*(Bk - 0.63)^2 with Bk the proportion of Black residents: a
#: non-invertible transform that builds a racial-segregation hypothesis into
#: the feature itself. scikit-learn removed the dataset over it. We drop the
#: column from the design matrix and the matching coordinate from every
#: elicited prior, which leaves the rest of the study unchanged because the
#: feature exists only in this dataset.
DROP_FEATURES = {"boston_housing": ["B"]}


def load(name: str):
    """Dataset with DROP_FEATURES removed from X and from feature_names."""
    d = datasets.load(name, root=DATA_ROOT)
    drop = DROP_FEATURES.get(name, [])
    if not drop:
        return d
    keep = [i for i, f in enumerate(d.feature_names) if f not in drop]
    names = [d.feature_names[i] for i in keep]
    assert len(names) == len(d.feature_names) - len(drop), f"drop failed for {name}"
    return datasets.Dataset(
        name=d.name, task=d.task,
        X_source=d.X_source[:, keep], y_source=d.y_source,
        X_ood=d.X_ood[:, keep], y_ood=d.y_ood,
        feature_names=names,
    )


def load_samples(dataset: str, variant_label: str, m: int = 50):
    """Elicited samples, with DROP_FEATURES coordinates removed.

    Deleting a key drops that coordinate from mu_hat and takes the principal
    submatrix of Sigma_hat, which is the marginal prior over the features we
    actually fit.
    """
    path = PRIOR_ROOT / dataset / VARIANT_DIR[variant_label] / "prior_samples.json"
    samples = priors.load_samples(path, max_samples=m)
    drop = DROP_FEATURES.get(dataset, [])
    if drop:
        samples = [{k: v for k, v in smp.items() if k not in drop} for smp in samples]
    return samples


def split_source(X, y, *, seed: int, stratify: bool):
    """The thesis 80/20 ID split."""
    return train_test_split(
        X, y, test_size=cfg.ID_TEST_FRACTION, random_state=seed,
        stratify=y if stratify else None,
    )


def make_folds(X, y, *, seed: int, stratify: bool):
    cls = StratifiedKFold if stratify else KFold
    return list(cls(n_splits=cfg.N_FOLDS, shuffle=True, random_state=seed).split(X, y))


def precision_for(family: str, pq):
    """(mu, P) for a family, in the study's parameterisation."""
    if family in ("prior_mean", "mixed"):
        return pq.identity
    if family == "independent":
        return pq.diagonal_precision
    if family == "correlated":
        return pq.precision
    raise ValueError(family)


def start_seed_for(family: str) -> int | None:
    """L-BFGS cold-start convention of the source pipeline.

    Correlated-Prior starts at zero; Prior-Mean and Independent-Prior start at
    a seeded 0.01*N(0,1) draw. This affects the optimiser path and therefore
    the third decimal of the reported metrics, so it is matched exactly.
    """
    return None if family == "correlated" else cfg.SEED


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit_prior_ridge(Xtr, ytr, mu, P, alpha):
    return GaussianPriorRidge(mu=mu, P=P, alpha=alpha).fit(Xtr, ytr)


def fit_baseline_ridge(Xtr, ytr, alpha):
    """sklearn Ridge on the study's 0.5*alpha penalty scale."""
    return SklearnRidge(alpha=0.5 * float(alpha), fit_intercept=True).fit(Xtr, ytr)


def fit_prior_logistic(Xtr, ytr, mu, P, alpha, seed=cfg.SEED):
    return GaussianPriorLogistic(mu=mu, P=P, alpha=alpha, random_state=seed).fit(Xtr, ytr)


def fit_baseline_logistic(Xtr, ytr, alpha):
    """Baseline is tuned on alpha_equiv = 1/C, penalty 0.5*alpha  =>  C = 1/alpha."""
    return LogisticRegression(C=1.0 / float(alpha), max_iter=5000,
                              fit_intercept=True).fit(Xtr, ytr)


def mse(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


# --------------------------------------------------------------------------
# lambda selection, by the study's strict-best-mean CV rule
# --------------------------------------------------------------------------

@dataclass
class FoldPack:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    pq: object


def prepare(X, y, folds, samples, feature_names):
    packs = []
    for tr, va in folds:
        pre = Preprocessor().fit(X[tr])
        packs.append(FoldPack(
            X_train=pre.transform(X[tr]), y_train=y[tr],
            X_val=pre.transform(X[va]), y_val=y[va],
            pq=priors.prior_quantities(samples, feature_names, pre.scale),
        ))
    return packs


def tune_alpha_regression(packs, family, grid=None, mu_transform=None):
    """Lowest mean-CV MSE. `mu_transform` lets a placebo reuse the same path."""
    grid = cfg.ALPHA_GRID if grid is None else grid
    scores = np.zeros((len(packs), grid.size))
    for k, f in enumerate(packs):
        mu = f.pq.mean if mu_transform is None else mu_transform(f.pq.mean)
        P = precision_for(family, f.pq)
        for j, a in enumerate(grid):
            m = fit_prior_ridge(f.X_train, f.y_train, mu, P, a)
            scores[k, j] = mse(f.y_val, m.predict(f.X_val))
    return float(grid[int(np.argmin(scores.mean(axis=0)))]), scores


def tune_alpha_baseline_regression(packs, grid=None):
    grid = cfg.ALPHA_GRID if grid is None else grid
    scores = np.zeros((len(packs), grid.size))
    for k, f in enumerate(packs):
        for j, a in enumerate(grid):
            m = fit_baseline_ridge(f.X_train, f.y_train, a)
            scores[k, j] = mse(f.y_val, m.predict(f.X_val))
    return float(grid[int(np.argmin(scores.mean(axis=0)))]), scores


def tune_alpha_classification(packs, family, grid=None, mu_transform=None,
                              metric="auc"):
    """Highest mean-CV score. Walks the grid downwards to warm-start, as the
    study's logistic tuning does."""
    from llmpriors.metrics import tuning_score
    grid = cfg.ALPHA_GRID if grid is None else grid
    order = np.argsort(grid)[::-1]
    scores = np.zeros((len(packs), grid.size))
    for k, f in enumerate(packs):
        mu = f.pq.mean if mu_transform is None else mu_transform(f.pq.mean)
        P = precision_for(family, f.pq)
        model = GaussianPriorLogistic(mu=mu, P=P, alpha=float(grid[order[0]]),
                                      random_state=start_seed_for(family))
        for j in order:
            model.alpha = float(grid[j])
            model.fit(f.X_train, f.y_train)
            scores[k, j] = tuning_score(f.y_val, model.predict_proba(f.X_val), metric)
    return float(grid[int(np.argmax(scores.mean(axis=0)))]), scores


def tune_alpha_baseline_classification(packs, grid=None, metric="auc"):
    from llmpriors.metrics import tuning_score
    grid = np.sort(1.0 / cfg.C_GRID) if grid is None else grid
    scores = np.zeros((len(packs), grid.size))
    for k, f in enumerate(packs):
        for j, a in enumerate(grid):
            m = fit_baseline_logistic(f.X_train, f.y_train, a)
            scores[k, j] = tuning_score(f.y_val, m.predict_proba(f.X_val)[:, 1], metric)
    return float(grid[int(np.argmax(scores.mean(axis=0)))]), scores


# --------------------------------------------------------------------------
# paired bootstrap over the evaluation set (the study's inferential test)
# --------------------------------------------------------------------------

def paired_bootstrap(y, pred_a, pred_b, stat, *, stratify, n=cfg.BOOTSTRAP_RESAMPLES,
                     seed=cfg.SEED, level=cfg.BOOTSTRAP_CI_LEVEL):
    """Percentile CI on stat(a) - stat(b), models fixed, evaluation set resampled.

    Same resample indices for both models, so the difference is paired.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    classes = np.unique(y)
    for i in range(n):
        if stratify and classes.size >= 2:
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == c), size=int(np.sum(y == c)), replace=True)
                for c in classes])
            rng.shuffle(idx)
        else:
            idx = rng.integers(0, y.size, size=y.size)
        try:
            draws[i] = stat(y[idx], pred_a[idx]) - stat(y[idx], pred_b[idx])
        except ValueError:
            draws[i] = np.nan
    lo = (1.0 - level) / 2.0
    d = draws[~np.isnan(draws)]
    return float(np.quantile(d, lo)), float(np.quantile(d, 1.0 - lo))


def write(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False)
    print(f"  -> {path.relative_to(HERE)}")
    return path
