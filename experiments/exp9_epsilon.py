"""Experiment 9: is the variance floor a safeguard or a hard constraint?

The Methods call eps = 1e-8 (added to the elicited variances before inversion)
and the 1e-6 covariance jitter "numerical safeguards with no modelling content"
that "cap how much confidence a unanimous elicitation is allowed to express".

Reviewer 1 observed that they do the opposite. One variant -- Opus 4.5,
non-reasoning -- returns bit-identical coefficients across all 50 draws for
some features, so the elicited variance is exactly zero and the Independent
Prior penalty weight becomes alpha/(0 + 1e-8) = alpha * 1e8. Those coefficients
are pinned to the elicited value, which is an equality constraint, not a prior.
Opus 4.5 is the deployed variant for both Boston headline cells.

This script re-tunes lambda and recomputes the deployed OOD lift across a range
of eps, and -- for Correlated-Prior, whose covariance is singular in exactly
the same directions -- across a range of jitters plus a Ledoit-Wolf shrinkage
estimator, which is the principled alternative the manuscript defers to future
work. If the headline lifts are an artefact of the floor they will move; if
they survive, the floor can be reported as the safeguard the paper claims.

Only the two families whose precision depends on the elicited spread are swept.
Prior-Mean (P = I) and Mixed cannot be affected and are omitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import log_loss, roc_auc_score

import common as C
from llmpriors import priors as P
from llmpriors.priors import PriorQuantities

EPS_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
STRIDE = {"adult": 4}
FAMILIES = ("independent", "correlated")


def zero_spread_count(dataset, variant):
    """How many features the variant returns bit-identically across all draws."""
    s = C.load_samples(dataset, variant)
    keys = list(s[0].keys())
    return sum(1 for k in keys if len({round(r[k], 12) for r in s}) == 1), len(keys)


def shrunk_pq(pq_mean, standardised):
    """PriorQuantities with a Ledoit-Wolf precision in place of jitter+pinv."""
    lw = LedoitWolf().fit(standardised)
    return PriorQuantities(mean=pq_mean,
                           variances=standardised.var(axis=0, ddof=1),
                           precision=lw.precision_)


def run(dataset):
    task = C.TASK[dataset]
    stratify = task == "classification"
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED,
                                    stratify=stratify)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=stratify)
    pre = C.Preprocessor().fit(Xtr)
    A, O = pre.transform(Xtr), pre.transform(d.X_ood)
    grid = C.cfg.ALPHA_GRID[::STRIDE.get(dataset, 1)]

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    if task == "regression":
        a_b, _ = C.tune_alpha_baseline_regression(packs0, grid=grid)
        base = C.mse(d.y_ood, C.fit_baseline_ridge(A, ytr, a_b).predict(O))
        base_ll = None
    else:
        a_b, _ = C.tune_alpha_baseline_classification(
            packs0, grid=np.sort(1.0 / C.cfg.C_GRID)[::STRIDE.get(dataset, 1)])
        pb = C.fit_baseline_logistic(A, ytr, a_b).predict_proba(O)[:, 1]
        base = float(roc_auc_score(d.y_ood, pb))
        base_ll = float(log_loss(d.y_ood, np.clip(pb, 1e-12, 1 - 1e-12)))

    rows = []
    for family in FAMILIES:
        variant = C.DEPLOYED[dataset][family][0]
        samples = C.load_samples(dataset, variant)
        nz, ntot = zero_spread_count(dataset, variant)

        for eps in EPS_GRID:
            # patch the real constants, then rebuild every prior through the
            # production code path so nothing is approximated here
            P.VARIANCE_EPS = eps
            P.COVARIANCE_JITTER = eps
            packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
            pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
            lift, a = fit_and_score(task, packs, family, pq, A, ytr, O, d.y_ood,
                                    base, base_ll, grid)
            rows.append(dict(dataset=C.PRETTY[dataset], family=C.FAMILY_LABEL[family],
                             variant=variant, estimator=f"eps={eps:g}", eps=eps,
                             zero_spread=nz, n_features=ntot, tuned=a, **lift))

        # Ledoit-Wolf shrinkage, Correlated-Prior only (it is a covariance fix)
        if family == "correlated":
            P.VARIANCE_EPS, P.COVARIANCE_JITTER = 1e-8, 1e-6
            raw = P.as_matrix(samples, d.feature_names)
            std_full = raw * np.asarray(pre.scale, dtype=float)
            pq_lw = shrunk_pq(std_full.mean(axis=0), std_full)
            packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
            for fp in packs:
                raw_f = P.as_matrix(samples, d.feature_names)
                std_f = raw_f * np.asarray(fp.scale, dtype=float) \
                    if hasattr(fp, "scale") else std_full
                fp.pq = shrunk_pq(std_f.mean(axis=0), std_f)
            lift, a = fit_and_score(task, packs, family, pq_lw, A, ytr, O, d.y_ood,
                                    base, base_ll, grid)
            rows.append(dict(dataset=C.PRETTY[dataset], family=C.FAMILY_LABEL[family],
                             variant=variant, estimator="Ledoit-Wolf", eps=np.nan,
                             zero_spread=nz, n_features=ntot, tuned=a, **lift))

    P.VARIANCE_EPS, P.COVARIANCE_JITTER = 1e-8, 1e-6
    return rows


def fit_and_score(task, packs, family, pq, A, ytr, O, y_ood, base, base_ll, grid):
    if task == "regression":
        a, _ = C.tune_alpha_regression(packs, family, grid=grid)
        m = C.fit_prior_ridge(A, ytr, pq.mean, C.precision_for(family, pq), a)
        v = C.mse(y_ood, m.predict(O))
        return dict(lift_pct=100.0 * (base - v) / base, logloss_lift_pct=np.nan), a
    a, _ = C.tune_alpha_classification(packs, family, grid=grid)
    m = C.fit_prior_logistic(A, ytr, pq.mean, C.precision_for(family, pq), a,
                             seed=C.start_seed_for(family))
    p = m.predict_proba(O)
    auc = float(roc_auc_score(y_ood, p))
    ll = float(log_loss(y_ood, np.clip(p, 1e-12, 1 - 1e-12)))
    return dict(lift_pct=100.0 * (auc - base) / base,
                logloss_lift_pct=100.0 * (base_ll - ll) / base_ll), a


def main():
    out = []
    for dataset in C.TASK:
        print(f"\n=== {C.PRETTY[dataset]} ===", flush=True)
        rows = run(dataset)
        out.extend(rows)
        for r in rows:
            extra = ("" if np.isnan(r["logloss_lift_pct"])
                     else f"  logloss {r['logloss_lift_pct']:+7.2f}%")
            print(f"  {r['family']:18s} {r['estimator']:14s} "
                  f"zero-spread {r['zero_spread']:2d}/{r['n_features']:2d}  "
                  f"lambda {r['tuned']:10.4g}  lift {r['lift_pct']:+7.2f}%{extra}",
                  flush=True)
    C.write(pd.DataFrame(out), "exp9_epsilon.csv")


if __name__ == "__main__":
    main()
