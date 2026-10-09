"""Experiment 1 (near-free): the prior-only row, lambda -> infinity.

The coefficient vector is frozen at the LLM prior mean; only the intercept is
fitted on the source training pool. This is the limit the prior families
approach as lambda grows, and it separates "the prior carries usable signal"
from "the data plus a nudge from the prior carries usable signal".

Reported for every LLM variant, so the deployed variant can be read against
the rest of the field rather than in isolation.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import common as C


def fit_intercept_regression(X, y, mu):
    return float(np.mean(y - X @ mu))


def fit_intercept_logistic(X, y, mu):
    """1-D MLE for the intercept with coefficients held at mu."""
    z0 = X @ mu

    def nll(b):
        z = np.clip(z0 + b, -500, 500)
        return -float(np.sum(y * z - np.logaddexp(0, z)))

    return float(minimize_scalar(nll, bounds=(-50, 50), method="bounded").x)


rows = []
ONLY = os.environ.get('ONLY_DATASET')
for dataset, task in C.TASK.items():
    if ONLY and dataset != ONLY:
        continue
    d = C.load(dataset)
    stratify = task == "classification"
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source,
                                    seed=C.cfg.SEED, stratify=stratify)
    pre = C.Preprocessor().fit(Xtr)
    Xtr_s, Xood_s = pre.transform(Xtr), pre.transform(d.X_ood)

    # baseline at its deployed lambda, for the reference column
    a_base = C.DEPLOYED[dataset]["baseline"][1]
    if task == "regression":
        base = C.fit_baseline_ridge(Xtr_s, ytr, a_base)
        p_base = base.predict(Xood_s)
        base_metric = C.mse(d.y_ood, p_base)
    else:
        base = C.fit_baseline_logistic(Xtr_s, ytr, a_base)
        p_base = base.predict_proba(Xood_s)[:, 1]
        base_metric = roc_auc_score(d.y_ood, p_base)

    deployed_variant = C.DEPLOYED[dataset]["prior_mean"][0]

    for label in C.VARIANT_DIR:
        samples = C.load_samples(dataset, label)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
        mu = pq.mean

        row = {
            "dataset": C.PRETTY[dataset],
            "task": task,
            "variant": label,
            "is_deployed_prior_mean": label == deployed_variant,
        }

        if task == "regression":
            b = fit_intercept_regression(Xtr_s, ytr, mu)
            pred = Xood_s @ mu + b
            m = C.mse(d.y_ood, pred)
            lo, hi = C.paired_bootstrap(
                d.y_ood, pred, p_base,
                lambda yy, pp: float(np.mean((yy - pp) ** 2)), stratify=False)
            row.update({
                "baseline_OOD_MSE": base_metric,
                "prior_only_OOD_MSE": m,
                "pct_vs_baseline": 100.0 * (base_metric - m) / base_metric,
                "paired_delta_lo": lo,
                "paired_delta_hi": hi,
            })
        else:
            b = fit_intercept_logistic(Xtr_s, ytr, mu)
            z = np.clip(Xood_s @ mu + b, -500, 500)
            pred = 1.0 / (1.0 + np.exp(-z))
            auc = roc_auc_score(d.y_ood, pred)
            lo, hi = C.paired_bootstrap(
                d.y_ood, pred, p_base,
                lambda yy, pp: float(roc_auc_score(yy, pp)), stratify=True)
            row.update({
                "baseline_OOD_AUC": base_metric,
                "prior_only_OOD_AUC": auc,
                "prior_only_OOD_LogLoss": log_loss(d.y_ood, np.clip(pred, 1e-12, 1 - 1e-12)),
                "prior_only_OOD_Brier": brier_score_loss(d.y_ood, pred),
                "pct_vs_baseline": 100.0 * (auc - base_metric) / base_metric,
                "paired_delta_lo": lo,
                "paired_delta_hi": hi,
            })

        rows.append(row)

    print(f"{C.PRETTY[dataset]:16s} done")

frame = pd.DataFrame(rows)
C.write(frame, "exp1_prior_only.csv")

print("\n--- prior-only, deployed Prior-Mean variant per dataset ---")
sel = frame[frame["is_deployed_prior_mean"]]
for _, r in sel.iterrows():
    print(f"  {r['dataset']:16s} {r['variant']:16s} {r['pct_vs_baseline']:+7.2f}% vs baseline"
          f"   paired delta [{r['paired_delta_lo']:+.4g}, {r['paired_delta_hi']:+.4g}]")

print("\n--- best and worst prior-only variant per dataset ---")
for ds, g in frame.groupby("dataset", sort=False):
    b, w = g.loc[g["pct_vs_baseline"].idxmax()], g.loc[g["pct_vs_baseline"].idxmin()]
    print(f"  {ds:16s} best {b['variant']:16s} {b['pct_vs_baseline']:+7.2f}%"
          f"   worst {w['variant']:16s} {w['pct_vs_baseline']:+8.2f}%")
