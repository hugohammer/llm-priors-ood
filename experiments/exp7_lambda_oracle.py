"""Experiment 7: what does source-domain lambda selection leave on the table?

Two explanations compete for Adult's null result.

  (a) The prior carries nothing useful for this dataset, so no lambda would
      have helped.
  (b) The prior does carry something useful OOD, but the selection rule cannot
      see it: lambda is chosen to minimise source-domain CV error, and a prior
      that trades in-domain fit for out-of-domain transfer looks strictly bad
      by that criterion. With n_train ~ 27k the source likelihood also
      dominates any fixed prior, so the weight the posterior gives the prior
      goes to zero as n grows.

These have different signatures. Under (a) the OOD metric is monotone in
lambda with its best value at the grid floor. Under (b) the OOD metric has an
interior optimum that source CV declines to pick.

This script sweeps the whole lambda grid for every dataset and family, records
the OOD metric at each point, and reports the CV-selected lambda against the
OOD-optimal one. The gap between them is the cost of selecting on the source
domain -- an oracle quantity, not an achievable one, which is exactly why it
belongs in a discussion of what a better merging rule might buy.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import common as C

STRIDE = {"adult": 2}  # 26 of 51 grid points; each logistic fit there costs ~2 s


def sweep(dataset, family):
    task = C.TASK[dataset]
    stratify = task == "classification"
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED,
                                    stratify=stratify)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=stratify)
    variant = C.DEPLOYED[dataset][family][0]
    samples = C.load_samples(dataset, variant)
    packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)

    pre = C.Preprocessor().fit(Xtr)
    A, O = pre.transform(Xtr), pre.transform(d.X_ood)
    pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
    P = C.precision_for(family, pq)

    grid = C.cfg.ALPHA_GRID[::STRIDE.get(dataset, 1)]

    # baseline, tuned by the study's own rule
    if task == "regression":
        a_b, _ = C.tune_alpha_baseline_regression(packs, grid=grid)
        base = C.mse(d.y_ood, C.fit_baseline_ridge(A, ytr, a_b).predict(O))
        a_cv, _ = C.tune_alpha_regression(packs, family, grid=grid)
    else:
        a_b, _ = C.tune_alpha_baseline_classification(
            packs, grid=np.sort(1.0 / C.cfg.C_GRID)[::STRIDE.get(dataset, 1)])
        pb = C.fit_baseline_logistic(A, ytr, a_b).predict_proba(O)[:, 1]
        base_auc = float(roc_auc_score(d.y_ood, pb))
        base_ll = float(log_loss(d.y_ood, np.clip(pb, 1e-12, 1 - 1e-12)))
        a_cv, _ = C.tune_alpha_classification(packs, family, grid=grid)

    rows = []
    order = np.argsort(grid)[::-1]
    model = None
    for j in order:
        a = float(grid[j])
        if task == "regression":
            m = C.fit_prior_ridge(A, ytr, pq.mean, P, a)
            val = C.mse(d.y_ood, m.predict(O))
            rows.append(dict(alpha=a, ood_mse=val,
                             lift=100.0 * (base - val) / base))
        else:
            if model is None:
                model = C.GaussianPriorLogistic(mu=pq.mean, P=P, alpha=a,
                                                random_state=C.start_seed_for(family))
            model.alpha = a
            model.fit(A, ytr)
            p = model.predict_proba(O)
            auc = float(roc_auc_score(d.y_ood, p))
            ll = float(log_loss(d.y_ood, np.clip(p, 1e-12, 1 - 1e-12)))
            rows.append(dict(alpha=a, ood_auc=auc, ood_logloss=ll,
                             lift=100.0 * (auc - base_auc) / base_auc,
                             lift_ll=100.0 * (base_ll - ll) / base_ll))

    df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    df["dataset"] = C.PRETTY[dataset]
    df["family"] = C.FAMILY_LABEL[family]
    df["task"] = task
    df["alpha_cv"] = a_cv
    return df


frames = []
ONLY = os.environ.get("ONLY_DATASET")
for dataset in C.TASK:
    if ONLY and dataset != ONLY:
        continue
    for family in ("prior_mean", "independent", "correlated"):
        frames.append(sweep(dataset, family))
        f = frames[-1]
        best = f.loc[f["lift"].idxmax()]
        at_cv = f.iloc[(f["alpha"] - f["alpha_cv"].iloc[0]).abs().argmin()]
        extra = ""
        if "lift_ll" in f:
            bl = f.loc[f["lift_ll"].idxmax()]
            extra = (f"   logloss: cv {at_cv['lift_ll']:+6.2f}%  "
                     f"oracle {bl['lift_ll']:+6.2f}% at lambda {bl['alpha']:g}")
        print(f"  {C.PRETTY[dataset]:15s} {C.FAMILY_LABEL[family]:18s} "
              f"cv lambda {at_cv['alpha']:9.4g} -> {at_cv['lift']:+6.2f}%   "
              f"oracle lambda {best['alpha']:9.4g} -> {best['lift']:+6.2f}%{extra}",
              flush=True)

out = pd.concat(frames, ignore_index=True)
C.write(out, "exp7_lambda_oracle.csv")

print("\n" + "=" * 104)
print("Cost of selecting lambda on the source domain (oracle gap)")
print("=" * 104)
summ = []
for (ds, fam), g in out.groupby(["dataset", "family"], sort=False):
    at_cv = g.iloc[(g["alpha"] - g["alpha_cv"].iloc[0]).abs().argmin()]
    best = g.loc[g["lift"].idxmax()]
    interior = 0 < (g["alpha"] < best["alpha"]).sum() < len(g)
    summ.append(dict(dataset=ds, family=fam, alpha_cv=at_cv["alpha"],
                     lift_at_cv=at_cv["lift"], alpha_oracle=best["alpha"],
                     lift_oracle=best["lift"],
                     gap=best["lift"] - at_cv["lift"],
                     oracle_interior=bool(interior)))
sm = pd.DataFrame(summ)
C.write(sm, "exp7_lambda_oracle_summary.csv")
print(f"{'dataset':16s} {'family':19s} {'cv lambda':>10s} {'lift':>7s} "
      f"{'oracle lambda':>14s} {'lift':>7s} {'gap':>7s} interior")
for _, r in sm.iterrows():
    print(f"{r.dataset:16s} {r.family:19s} {r.alpha_cv:10.4g} {r.lift_at_cv:+6.2f}% "
          f"{r.alpha_oracle:14.4g} {r.lift_oracle:+6.2f}% {r.gap:+6.2f}% "
          f"{'yes' if r.oracle_interior else 'no'}")
