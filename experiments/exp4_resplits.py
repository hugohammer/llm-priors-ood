"""Experiment 4 (strongly recommended): stability across ID resplits.

The thesis reports one 80/20 ID split (seed 42) and bootstraps only the
evaluation set, which by construction cannot see variability in training or in
lambda selection. This repeats the whole deployment step over many ID resplits.

The OOD partition is fixed by the domain split, so resampling the ID side
varies only which source rows train the model - exactly the component the
bootstrap leaves out. Lambda is re-tuned from scratch on every resplit.

The deployed LLM variant per family is held fixed at the thesis's choice, so
this measures stability of the lift given that variant, not stability of the
variant-selection step itself. Re-running the 12-variant selection per resplit
would be the fuller experiment; it is far more expensive and is noted as such.

Adult uses a strided alpha grid and fewer resplits: each logistic fit there
costs ~2 s, and its lambda collapses to the grid floor regardless.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import common as C

# dataset -> (n_resplits, alpha-grid stride)
BUDGET = {
    "boston_housing": (50, 1),
    "cancer": (50, 1),
    "heart_disease": (30, 1),
    "diabetes": (30, 1),
    "adult": (10, 4),
}

if os.environ.get("SMOKE"):
    # 2 resplits and a coarse grid, to check the wiring end to end
    BUDGET = {k: (2, 10) for k in BUDGET}


def deploy_once(dataset, family, seed, grid):
    """One full deployment on one ID resplit; returns lift over baseline."""
    task = C.TASK[dataset]
    d = C.load(dataset)
    stratify = task == "classification"
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=seed, stratify=stratify)

    variant = C.DEPLOYED[dataset][family][0]
    samples = C.load_samples(dataset, variant)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=stratify)
    packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)

    pre = C.Preprocessor().fit(Xtr)
    A, O = pre.transform(Xtr), pre.transform(d.X_ood)
    pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)

    if task == "regression":
        a_b, _ = C.tune_alpha_baseline_regression(packs, grid=grid)
        base = C.fit_baseline_ridge(A, ytr, a_b)
        m_base = C.mse(d.y_ood, base.predict(O))

        if family == "mixed":
            a_pm, _ = C.tune_alpha_regression(packs, "prior_mean", grid=grid)
            gamma = C.DEPLOYED[dataset]["mixed"][1]
            pm = C.fit_prior_ridge(A, ytr, pq.mean, pq.identity, a_pm)
            pred = (1 - gamma) * base.predict(O) + gamma * pm.predict(O)
            a_p = a_pm
        else:
            a_p, _ = C.tune_alpha_regression(packs, family, grid=grid)
            pred = C.fit_prior_ridge(A, ytr, pq.mean,
                                     C.precision_for(family, pq), a_p).predict(O)
        m_p = C.mse(d.y_ood, pred)
        return {"lift_pct": 100.0 * (m_base - m_p) / m_base,
                "baseline_metric": m_base, "prior_metric": m_p,
                "alpha_baseline": a_b, "alpha_prior": a_p}

    a_b, _ = C.tune_alpha_baseline_classification(
        packs, grid=np.sort(1.0 / C.cfg.C_GRID)[::BUDGET[dataset][1]])
    base = C.fit_baseline_logistic(A, ytr, a_b)
    pb = base.predict_proba(O)[:, 1]

    if family == "mixed":
        a_pm, _ = C.tune_alpha_classification(packs, "prior_mean", grid=grid)
        gamma = C.DEPLOYED[dataset]["mixed"][1]
        pm = C.fit_prior_logistic(A, ytr, pq.mean, pq.identity, a_pm)
        pp = (1 - gamma) * pb + gamma * pm.predict_proba(O)
        a_p = a_pm
    else:
        a_p, _ = C.tune_alpha_classification(packs, family, grid=grid)
        pp = C.fit_prior_logistic(A, ytr, pq.mean,
                                  C.precision_for(family, pq), a_p,
                                  seed=C.start_seed_for(family)).predict_proba(O)

    auc_b, auc_p = roc_auc_score(d.y_ood, pb), roc_auc_score(d.y_ood, pp)
    ll_b = log_loss(d.y_ood, np.clip(pb, 1e-12, 1 - 1e-12))
    ll_p = log_loss(d.y_ood, np.clip(pp, 1e-12, 1 - 1e-12))
    return {"lift_pct": 100.0 * (auc_p - auc_b) / auc_b,
            "logloss_lift_pct": 100.0 * (ll_b - ll_p) / ll_b,
            "baseline_metric": float(auc_b), "prior_metric": float(auc_p),
            "alpha_baseline": a_b, "alpha_prior": a_p}


rows = []
ONLY = os.environ.get('ONLY_DATASET')
for dataset, (n_seeds, stride) in BUDGET.items():
    if ONLY and dataset != ONLY:
        continue
    grid = C.cfg.ALPHA_GRID[::stride]
    for family in C.FAMILIES:
        t0 = time.time()
        for seed in range(1000, 1000 + n_seeds):
            r = deploy_once(dataset, family, seed, grid)
            rows.append({
                "dataset": C.PRETTY[dataset],
                "task": C.TASK[dataset],
                "family": C.FAMILY_LABEL[family],
                "variant": C.DEPLOYED[dataset][family][0],
                "seed": seed,
                **r,
            })
        el = time.time() - t0
        sub = pd.DataFrame([x for x in rows if x["dataset"] == C.PRETTY[dataset]
                            and x["family"] == C.FAMILY_LABEL[family]])
        print(f"  {C.PRETTY[dataset]:15s} {C.FAMILY_LABEL[family]:18s}"
              f" n={n_seeds:3d}  mean {sub['lift_pct'].mean():+7.2f}%"
              f"  sd {sub['lift_pct'].std(ddof=1):5.2f}"
              f"  [{sub['lift_pct'].min():+7.2f}, {sub['lift_pct'].max():+7.2f}]"
              f"  ({el:.0f}s)", flush=True)

frame = pd.DataFrame(rows)
C.write(frame, "exp4_resplits_raw.csv")

summary = (frame.groupby(["dataset", "task", "family", "variant"], sort=False)
           .agg(n_resplits=("lift_pct", "size"),
                mean_lift_pct=("lift_pct", "mean"),
                sd_lift_pct=("lift_pct", lambda s: s.std(ddof=1)),
                min_lift_pct=("lift_pct", "min"),
                median_lift_pct=("lift_pct", "median"),
                max_lift_pct=("lift_pct", "max"),
                frac_resplits_below_baseline=("lift_pct", lambda s: float(np.mean(s < 0))),
                q025=("lift_pct", lambda s: float(np.quantile(s, 0.025))),
                q975=("lift_pct", lambda s: float(np.quantile(s, 0.975))))
           .reset_index())
C.write(summary, "exp4_resplits_summary.csv")

print("\n" + "=" * 108)
print("OOD lift over baseline across ID resplits  (MSE lift for regression, AUC lift for classification)")
print("=" * 108)
print(f"{'dataset':16s} {'family':19s} {'n':>4s} {'mean':>8s} {'sd':>7s} {'median':>8s}"
      f" {'2.5-97.5 pct':>20s} {'<baseline':>10s}")
for _, r in summary.iterrows():
    ci = f"[{r['q025']:+.2f}, {r['q975']:+.2f}]"
    print(f"{r['dataset']:16s} {r['family']:19s} {r['n_resplits']:4.0f}"
          f" {r['mean_lift_pct']:+7.2f}% {r['sd_lift_pct']:7.2f} {r['median_lift_pct']:+7.2f}%"
          f" {ci:>20s} {r['frac_resplits_below_baseline']:9.0%}")
