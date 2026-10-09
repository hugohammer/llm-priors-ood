"""Experiment 11 (exp5 re-run, split index recorded): one interval covering source-split, lambda-selection AND
target-sample uncertainty at once.

Motivation. The two uncertainty analyses in the paper cover *disjoint* sources,
not nested ones:

  paired bootstrap (exp2/tables)  resamples the OOD set, models held fixed
                                  -> target-sample noise only
  resplits (exp4)                 redraws the ID split and re-tunes lambda,
                                  OOD partition held fixed
                                  -> training + tuning noise only

So neither dominates the other, and reporting only one of them hides a real
component. This script nests them: for every ID resplit the model is refitted
and lambda re-tuned, and the lift is then evaluated on B bootstrap resamples of
the OOD set. Pooling S x B draws gives a percentile interval that propagates
all three sources.

Elicitation remains a fourth source that is not covered here: the deployed LLM
variant and its M = 50 draws are held fixed, as in exp4.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import common as C

# dataset -> (n_resplits, alpha-grid stride, inner bootstrap draws)
BUDGET = {
    "boston_housing": (50, 1, 200),
    "cancer": (50, 1, 200),
    "heart_disease": (30, 1, 200),
    "diabetes": (30, 1, 200),
    "adult": (10, 4, 200),
}

if os.environ.get("SMOKE"):
    BUDGET = {k: (2, 10, 20) for k in BUDGET}


def boot_indices(y, n_draws, rng, stratify):
    """Index sets for the inner bootstrap of the OOD set."""
    y = np.asarray(y)
    out = []
    classes = np.unique(y)
    for _ in range(n_draws):
        if stratify and classes.size >= 2:
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == c), size=int(np.sum(y == c)),
                           replace=True) for c in classes])
            rng.shuffle(idx)
        else:
            idx = rng.integers(0, y.size, size=y.size)
        out.append(idx)
    return out


def deploy_once(dataset, family, seed, grid):
    """One resplit: retune lambda, refit, return OOD predictions of both models."""
    task = C.TASK[dataset]
    d = C.load(dataset)
    stratify = task == "classification"
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=seed,
                                    stratify=stratify)
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
        pb = base.predict(O)
        if family == "mixed":
            a_p, _ = C.tune_alpha_regression(packs, "prior_mean", grid=grid)
            gamma = C.DEPLOYED[dataset]["mixed"][1]
            pm = C.fit_prior_ridge(A, ytr, pq.mean, pq.identity, a_p)
            pp = (1 - gamma) * pb + gamma * pm.predict(O)
        else:
            a_p, _ = C.tune_alpha_regression(packs, family, grid=grid)
            pp = C.fit_prior_ridge(A, ytr, pq.mean,
                                   C.precision_for(family, pq), a_p).predict(O)
        return pb, pp, d.y_ood

    a_b, _ = C.tune_alpha_baseline_classification(
        packs, grid=np.sort(1.0 / C.cfg.C_GRID)[::BUDGET[dataset][1]])
    base = C.fit_baseline_logistic(A, ytr, a_b)
    pb = base.predict_proba(O)[:, 1]
    if family == "mixed":
        a_p, _ = C.tune_alpha_classification(packs, "prior_mean", grid=grid)
        gamma = C.DEPLOYED[dataset]["mixed"][1]
        pm = C.fit_prior_logistic(A, ytr, pq.mean, pq.identity, a_p)
        pp = (1 - gamma) * pb + gamma * pm.predict_proba(O)
    else:
        a_p, _ = C.tune_alpha_classification(packs, family, grid=grid)
        pp = C.fit_prior_logistic(A, ytr, pq.mean, C.precision_for(family, pq),
                                  a_p, seed=C.start_seed_for(family)).predict_proba(O)
    return pb, pp, d.y_ood


def lifts_on(y, pb, pp, idx, task):
    """Percentage lift of the prior model over baseline on one resample."""
    yy, b, p = np.asarray(y)[idx], pb[idx], pp[idx]
    if task == "regression":
        mb = float(np.mean((yy - b) ** 2))
        mp = float(np.mean((yy - p) ** 2))
        return {"primary": 100.0 * (mb - mp) / mb}
    if np.unique(yy).size < 2:
        return None
    ab, ap = roc_auc_score(yy, b), roc_auc_score(yy, p)
    lb = log_loss(yy, np.clip(b, 1e-12, 1 - 1e-12))
    lp = log_loss(yy, np.clip(p, 1e-12, 1 - 1e-12))
    return {"primary": 100.0 * (ap - ab) / ab,
            "logloss": 100.0 * (lb - lp) / lb}


rows, draws = [], []
ONLY = os.environ.get("ONLY_DATASET")
for dataset, (n_seeds, stride, n_boot) in BUDGET.items():
    if ONLY and dataset != ONLY:
        continue
    task = C.TASK[dataset]
    grid = C.cfg.ALPHA_GRID[::stride]
    stratify = task == "classification"

    for family in C.FAMILIES:
        t0 = time.time()
        rng = np.random.default_rng(C.cfg.SEED)
        joint, resplit_only = {"primary": [], "logloss": []}, {"primary": [], "logloss": []}
        joint_split = {"primary": [], "logloss": []}

        for split_idx, seed in enumerate(range(1000, 1000 + n_seeds)):
            pb, pp, y_ood = deploy_once(dataset, family, seed, grid)

            full = lifts_on(y_ood, pb, pp, np.arange(len(y_ood)), task)
            for k, v in full.items():
                resplit_only[k].append(v)

            for idx in boot_indices(y_ood, n_boot, rng, stratify):
                got = lifts_on(y_ood, pb, pp, idx, task)
                if got is None:
                    continue
                for k, v in got.items():
                    joint[k].append(v)
                    joint_split[k].append(split_idx)

        def summarise(vals):
            v = np.asarray(vals, dtype=float)
            return dict(mean=float(v.mean()),
                        q025=float(np.quantile(v, 0.025)),
                        q975=float(np.quantile(v, 0.975)),
                        frac_below=float(np.mean(v < 0)), n=int(v.size))

        for metric in ("primary", "logloss"):
            if not joint[metric]:
                continue
            j, r = summarise(joint[metric]), summarise(resplit_only[metric])
            rows.append(dict(
                dataset=C.PRETTY[dataset], task=task,
                family=C.FAMILY_LABEL[family],
                variant=C.DEPLOYED[dataset][family][0],
                metric=("MSE" if task == "regression" else
                        ("AUC" if metric == "primary" else "LogLoss")),
                n_resplits=n_seeds, n_boot_per_resplit=n_boot,
                joint_mean=j["mean"], joint_q025=j["q025"], joint_q975=j["q975"],
                joint_frac_below=j["frac_below"], joint_n_draws=j["n"],
                resplit_mean=r["mean"], resplit_q025=r["q025"],
                resplit_q975=r["q975"], resplit_frac_below=r["frac_below"]))
            for v, sp in zip(joint[metric], joint_split[metric]):
                draws.append(dict(dataset=C.PRETTY[dataset],
                                  family=C.FAMILY_LABEL[family],
                                  metric=rows[-1]["metric"], split=sp, lift_pct=v))

        el = time.time() - t0
        last = [x for x in rows if x["dataset"] == C.PRETTY[dataset]
                and x["family"] == C.FAMILY_LABEL[family]]
        pr = last[0]
        print(f"  {C.PRETTY[dataset]:15s} {C.FAMILY_LABEL[family]:18s} "
              f"joint {pr['joint_mean']:+6.2f}% [{pr['joint_q025']:+6.2f}, "
              f"{pr['joint_q975']:+6.2f}]  below {pr['joint_frac_below']:5.0%}   "
              f"(resplit-only [{pr['resplit_q025']:+6.2f}, {pr['resplit_q975']:+6.2f}])"
              f"  ({el:.0f}s)", flush=True)

frame = pd.DataFrame(rows)
C.write(frame, "exp11_joint.csv")
C.write(pd.DataFrame(draws), "exp11_joint_draws.csv")

print("\n" + "=" * 112)
print("Joint interval over source split + lambda selection + target sample")
print("=" * 112)
print(f"{'dataset':16s} {'family':19s} {'metric':8s} {'mean':>8s} "
      f"{'joint 2.5-97.5':>20s} {'below':>7s} {'resplit-only 2.5-97.5':>24s}")
for _, r in frame.iterrows():
    jj = f"[{r['joint_q025']:+.2f}, {r['joint_q975']:+.2f}]"
    rr = f"[{r['resplit_q025']:+.2f}, {r['resplit_q975']:+.2f}]"
    print(f"{r['dataset']:16s} {r['family']:19s} {r['metric']:8s} "
          f"{r['joint_mean']:+7.2f}% {jj:>20s} {r['joint_frac_below']:6.0%} {rr:>24s}")
