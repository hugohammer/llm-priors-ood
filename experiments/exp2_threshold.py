"""Experiment 2 (must do): is the OOD F1 gain an operating-point artefact?

The thesis compares every classification model at a fixed threshold of 0.5.
Under label shift that is not a neutral choice: the source-fitted baseline is
simply at the wrong operating point on the OOD set, and any model that inflates
its predicted probabilities will score better on F1 without being a better
model.

This experiment gives every model - baseline included - the same operating-point
freedom, with the threshold chosen by 5-fold CV on the source training pool
(never on OOD data), and re-tests the F1 claim. The threshold-free metrics
(AUC, LogLoss, Brier) are reported alongside, since those need no such defence.

Also reports the OOD-oracle threshold as a diagnostic upper bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

import common as C

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
DATASETS = ["heart_disease", "diabetes", "adult"]


def f1_at(y, p, t):
    return float(f1_score(np.asarray(y).astype(int), (np.asarray(p) >= t).astype(int),
                          zero_division=0))


def predict_family(dataset, family, Xtr_s, ytr, X_eval_s, pq, samples, feature_names):
    """Deployed model for a family, at its deployed lambda / gamma."""
    label, tuned = C.DEPLOYED[dataset][family]
    if family == "mixed":
        gamma = tuned
        a_pm = C.DEPLOYED[dataset]["prior_mean"][1]
        a_b = C.DEPLOYED[dataset]["baseline"][1]
        pm = C.fit_prior_logistic(Xtr_s, ytr, pq.mean, pq.identity, a_pm)
        base = C.fit_baseline_logistic(Xtr_s, ytr, a_b)
        return (1 - gamma) * base.predict_proba(X_eval_s)[:, 1] + gamma * pm.predict_proba(X_eval_s)
    P = C.precision_for(family, pq)
    m = C.fit_prior_logistic(Xtr_s, ytr, pq.mean, P, tuned,
                             seed=C.start_seed_for(family))
    return m.predict_proba(X_eval_s)


def cv_threshold(dataset, family, X, y, samples, feature_names):
    """Threshold maximising mean 5-fold CV F1 on the source training pool."""
    folds = C.make_folds(X, y, seed=C.cfg.SEED, stratify=True)
    scores = np.zeros((len(folds), THRESHOLD_GRID.size))
    for k, (tr, va) in enumerate(folds):
        pre = C.Preprocessor().fit(X[tr])
        A, B = pre.transform(X[tr]), pre.transform(X[va])
        pq = C.priors.prior_quantities(samples, feature_names, pre.scale)
        if family == "baseline":
            p = C.fit_baseline_logistic(A, y[tr], C.DEPLOYED[dataset]["baseline"][1]
                                        ).predict_proba(B)[:, 1]
        else:
            p = predict_family(dataset, family, A, y[tr], B, pq, samples, feature_names)
        for j, t in enumerate(THRESHOLD_GRID):
            scores[k, j] = f1_at(y[va], p, t)
    return float(THRESHOLD_GRID[int(np.argmax(scores.mean(axis=0)))])


rows = []
for dataset in DATASETS:
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED, stratify=True)
    pre = C.Preprocessor().fit(Xtr)
    Xtr_s, Xood_s = pre.transform(Xtr), pre.transform(d.X_ood)

    y_ood = np.asarray(d.y_ood).astype(int)
    prev_tr, prev_ood = float(np.mean(ytr)), float(np.mean(y_ood))

    preds, thresholds = {}, {}
    for family in ["baseline"] + C.FAMILIES:
        label = C.DEPLOYED[dataset][family][0]
        variant = label if family != "baseline" else C.DEPLOYED[dataset]["prior_mean"][0]
        samples = C.load_samples(dataset, variant)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
        if family == "baseline":
            preds[family] = C.fit_baseline_logistic(
                Xtr_s, ytr, C.DEPLOYED[dataset]["baseline"][1]).predict_proba(Xood_s)[:, 1]
        else:
            preds[family] = predict_family(dataset, family, Xtr_s, ytr, Xood_s,
                                           pq, samples, d.feature_names)
        thresholds[family] = cv_threshold(dataset, family, Xtr, ytr, samples, d.feature_names)
        print(f"  {C.PRETTY[dataset]:15s} {family:18s} CV threshold {thresholds[family]:.2f}")

    p_base, t_base = preds["baseline"], thresholds["baseline"]

    for family in ["baseline"] + C.FAMILIES:
        p, t = preds[family], thresholds[family]
        sweep = np.array([f1_at(y_ood, p, tt) for tt in np.linspace(0.01, 0.99, 99)])

        row = {
            "dataset": C.PRETTY[dataset],
            "family": "Baseline" if family == "baseline" else C.FAMILY_LABEL[family],
            "variant": "-" if family == "baseline" else C.DEPLOYED[dataset][family][0],
            "prevalence_id_train": prev_tr,
            "prevalence_ood": prev_ood,
            "cv_threshold": t,
            "F1_at_0.5": f1_at(y_ood, p, 0.5),
            "F1_at_cv_threshold": f1_at(y_ood, p, t),
            "F1_at_oracle_threshold": float(sweep.max()),
            "oracle_threshold": float(np.linspace(0.01, 0.99, 99)[int(sweep.argmax())]),
            "AUC": float(roc_auc_score(y_ood, p)),
            "LogLoss": float(log_loss(y_ood, np.clip(p, 1e-12, 1 - 1e-12))),
            "Brier": float(brier_score_loss(y_ood, p)),
        }

        if family != "baseline":
            # paired delta at the fixed 0.5 threshold (what the thesis reports)
            lo, hi = C.paired_bootstrap(
                y_ood, p, p_base, lambda yy, pp: f1_at(yy, pp, 0.5), stratify=True)
            row["dF1_at_0.5_lo"], row["dF1_at_0.5_hi"] = lo, hi
            # Paired delta with BOTH models at their own CV-chosen threshold.
            # Done inline rather than through paired_bootstrap, because the two
            # models are scored at *different* thresholds and that helper
            # applies one statistic to both.
            t_self, t_b = t, t_base
            rng = np.random.default_rng(C.cfg.SEED)
            draws = np.empty(C.cfg.BOOTSTRAP_RESAMPLES)
            classes = np.unique(y_ood)
            for i in range(C.cfg.BOOTSTRAP_RESAMPLES):
                idx = np.concatenate([
                    rng.choice(np.flatnonzero(y_ood == c), size=int(np.sum(y_ood == c)),
                               replace=True) for c in classes])
                rng.shuffle(idx)
                draws[i] = f1_at(y_ood[idx], p[idx], t_self) - f1_at(y_ood[idx], p_base[idx], t_b)
            q = (1.0 - C.cfg.BOOTSTRAP_CI_LEVEL) / 2.0
            row["dF1_at_cv_lo"] = float(np.quantile(draws, q))
            row["dF1_at_cv_hi"] = float(np.quantile(draws, 1 - q))
            # threshold-free deltas
            lo, hi = C.paired_bootstrap(y_ood, p, p_base,
                                        lambda yy, pp: float(roc_auc_score(yy, pp)), stratify=True)
            row["dAUC_lo"], row["dAUC_hi"] = lo, hi
            lo, hi = C.paired_bootstrap(
                y_ood, p, p_base,
                lambda yy, pp: float(log_loss(yy, np.clip(pp, 1e-12, 1 - 1e-12))), stratify=True)
            row["dLogLoss_lo"], row["dLogLoss_hi"] = lo, hi
            lo, hi = C.paired_bootstrap(y_ood, p, p_base,
                                        lambda yy, pp: float(brier_score_loss(yy, pp)),
                                        stratify=True)
            row["dBrier_lo"], row["dBrier_hi"] = lo, hi

        rows.append(row)

frame = pd.DataFrame(rows)
C.write(frame, "exp2_threshold.csv")


def sig(lo, hi):
    if np.isnan(lo):
        return "   "
    return " * " if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "   "


print("\n" + "=" * 100)
print("F1: fixed 0.5 threshold (as in the thesis) vs both models recalibrated on source data")
print("=" * 100)
for ds, g in frame.groupby("dataset", sort=False):
    b = g[g["family"] == "Baseline"].iloc[0]
    print(f"\n{ds}   prevalence ID-train {b['prevalence_id_train']:.3f} -> OOD {b['prevalence_ood']:.3f}"
          f"   (baseline CV threshold {b['cv_threshold']:.2f})")
    print(f"  {'family':20s} {'F1@0.5':>8s} {'dF1@0.5 95% CI':>24s}   "
          f"{'F1@CVthr':>9s} {'dF1@CVthr 95% CI':>24s}")
    print(f"  {'Baseline':20s} {b['F1_at_0.5']:8.4f} {'-':>24s}   {b['F1_at_cv_threshold']:9.4f} {'-':>24s}")
    for _, r in g[g["family"] != "Baseline"].iterrows():
        c1 = f"[{r['dF1_at_0.5_lo']:+.4f}, {r['dF1_at_0.5_hi']:+.4f}]"
        c2 = f"[{r['dF1_at_cv_lo']:+.4f}, {r['dF1_at_cv_hi']:+.4f}]"
        print(f"  {r['family']:20s} {r['F1_at_0.5']:8.4f} {c1:>22s}{sig(r['dF1_at_0.5_lo'], r['dF1_at_0.5_hi'])}"
              f" {r['F1_at_cv_threshold']:9.4f} {c2:>22s}{sig(r['dF1_at_cv_lo'], r['dF1_at_cv_hi'])}")

print("\n" + "=" * 100)
print("Threshold-free metrics (no operating-point defence needed)")
print("=" * 100)
for ds, g in frame.groupby("dataset", sort=False):
    b = g[g["family"] == "Baseline"].iloc[0]
    print(f"\n{ds}")
    print(f"  {'family':20s} {'AUC':>8s} {'dAUC 95% CI':>22s}  {'LogLoss':>8s} {'dLogLoss 95% CI':>22s}  {'Brier':>7s}")
    print(f"  {'Baseline':20s} {b['AUC']:8.4f} {'-':>22s}  {b['LogLoss']:8.4f} {'-':>22s}  {b['Brier']:7.4f}")
    for _, r in g[g["family"] != "Baseline"].iterrows():
        c1 = f"[{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]"
        c2 = f"[{r['dLogLoss_lo']:+.4f}, {r['dLogLoss_hi']:+.4f}]"
        print(f"  {r['family']:20s} {r['AUC']:8.4f} {c1:>20s}{sig(r['dAUC_lo'], r['dAUC_hi'])}"
              f" {r['LogLoss']:8.4f} {c2:>20s}{sig(r['dLogLoss_lo'], r['dLogLoss_hi'])} {r['Brier']:7.4f}")
print("\n(* = paired 95% CI excludes zero)")
