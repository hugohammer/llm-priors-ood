"""Experiment 3 (must do): placebo control on the elicited prior.

Does the OOD gain come from the *content* of the LLM prior, or would any
shrinkage target of roughly the same shape and magnitude do as well? Two
placebos, each a structure-preserving corruption of the real prior:

  permuted   the whole prior is re-indexed onto a random permutation of the
             features - same values, same covariance structure, wrong features
  signflip   each coefficient's sign is flipped at random - same magnitudes,
             same correlation magnitudes, scrambled directions

Both corruptions preserve the prior's scale, so a model that only benefits from
"shrink toward something non-zero of about this size" scores the same under them
as under the real prior. Lambda is re-tuned by the study's own 5-fold CV rule
for every draw, so each placebo gets the same tuning advantage the real prior gets.

Adult is excluded: CV drives lambda to the grid floor there and the prior
families collapse onto the baseline, so a placebo is vacuous by construction.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import common as C
from llmpriors.priors import PriorQuantities

N_DRAWS_REGRESSION = 200
N_DRAWS_CLASSIFICATION = 100

# All three families on all four datasets. The classification cells were
# originally restricted to prior_mean on cost grounds; a timing check put a
# single logistic placebo draw at 0.75-1.7 s, so the full set is ~14 min and
# the restriction was not worth the gap it left in the figure.
CELLS = [
    ("boston_housing", ["prior_mean", "independent", "correlated"], N_DRAWS_REGRESSION),
    ("cancer", ["prior_mean", "independent", "correlated"], N_DRAWS_REGRESSION),
    ("heart_disease", ["prior_mean", "independent", "correlated"], N_DRAWS_CLASSIFICATION),
    ("diabetes", ["prior_mean", "independent", "correlated"], N_DRAWS_CLASSIFICATION),
]


def permute_prior(pq: PriorQuantities, perm: np.ndarray) -> PriorQuantities:
    """Re-index the entire prior onto `perm`: mean, variances and precision."""
    return PriorQuantities(
        mean=pq.mean[perm],
        variances=pq.variances[perm],
        precision=pq.precision[np.ix_(perm, perm)],
    )


def signflip_prior(pq: PriorQuantities, signs: np.ndarray) -> PriorQuantities:
    """Flip coefficient signs. With D = diag(s) and D^-1 = D, Sigma -> D Sigma D,
    so the precision transforms the same way."""
    D = np.diag(signs)
    return PriorQuantities(
        mean=pq.mean * signs,
        variances=pq.variances,
        precision=D @ pq.precision @ D,
    )


def prior_of(pack_pq, kind, perm, signs):
    if kind == "real":
        return pack_pq
    if kind == "permuted":
        return permute_prior(pack_pq, perm)
    return signflip_prior(pack_pq, signs)


def eval_regression(dataset, family, kind, perm, signs, Xtr, ytr, X_ood, y_ood,
                    samples, names, mse_base):
    """Re-tune lambda by CV on ID data under this prior, refit, score OOD."""
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=False)
    packs = C.prepare(Xtr, ytr, folds, samples, names)

    scores = np.zeros((len(packs), C.cfg.ALPHA_GRID.size))
    for k, f in enumerate(packs):
        pq = prior_of(f.pq, kind, perm, signs)
        P = C.precision_for(family, pq)
        for j, a in enumerate(C.cfg.ALPHA_GRID):
            m = C.fit_prior_ridge(f.X_train, f.y_train, pq.mean, P, a)
            scores[k, j] = C.mse(f.y_val, m.predict(f.X_val))
    alpha = float(C.cfg.ALPHA_GRID[int(np.argmin(scores.mean(axis=0)))])

    pre = C.Preprocessor().fit(Xtr)
    pq = prior_of(C.priors.prior_quantities(samples, names, pre.scale), kind, perm, signs)
    m = C.fit_prior_ridge(pre.transform(Xtr), ytr, pq.mean,
                          C.precision_for(family, pq), alpha)
    m_ood = C.mse(y_ood, m.predict(pre.transform(X_ood)))
    return 100.0 * (mse_base - m_ood) / mse_base, m_ood, alpha


def eval_classification(dataset, family, kind, perm, signs, Xtr, ytr, X_ood, y_ood,
                        samples, names, auc_base, ll_base):
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=True)
    packs = C.prepare(Xtr, ytr, folds, samples, names)
    from llmpriors.metrics import tuning_score

    grid = C.cfg.ALPHA_GRID
    order = np.argsort(grid)[::-1]
    scores = np.zeros((len(packs), grid.size))
    for k, f in enumerate(packs):
        pq = prior_of(f.pq, kind, perm, signs)
        P = C.precision_for(family, pq)
        model = C.GaussianPriorLogistic(mu=pq.mean, P=P, alpha=float(grid[order[0]]),
                                        random_state=C.cfg.SEED)
        for j in order:
            model.alpha = float(grid[j])
            model.fit(f.X_train, f.y_train)
            scores[k, j] = tuning_score(f.y_val, model.predict_proba(f.X_val), "auc")
    alpha = float(grid[int(np.argmax(scores.mean(axis=0)))])

    pre = C.Preprocessor().fit(Xtr)
    pq = prior_of(C.priors.prior_quantities(samples, names, pre.scale), kind, perm, signs)
    m = C.fit_prior_logistic(pre.transform(Xtr), ytr, pq.mean,
                             C.precision_for(family, pq), alpha)
    p = m.predict_proba(pre.transform(X_ood))
    auc = float(roc_auc_score(y_ood, p))
    ll = float(log_loss(y_ood, np.clip(p, 1e-12, 1 - 1e-12)))
    return (100.0 * (auc - auc_base) / auc_base, 100.0 * (ll_base - ll) / ll_base,
            auc, ll, alpha)


rows = []
draw_rows = []
ONLY = os.environ.get('ONLY_DATASET')
for dataset, families, n_draws in CELLS:
    if ONLY and dataset != ONLY:
        continue
    task = C.TASK[dataset]
    d = C.load(dataset)
    stratify = task == "classification"
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source,
                                    seed=C.cfg.SEED, stratify=stratify)
    p_feat = len(d.feature_names)
    rng = np.random.default_rng(0)

    # baseline at deployed lambda
    pre = C.Preprocessor().fit(Xtr)
    a_base = C.DEPLOYED[dataset]["baseline"][1]
    if task == "regression":
        base = C.fit_baseline_ridge(pre.transform(Xtr), ytr, a_base)
        mse_base = C.mse(d.y_ood, base.predict(pre.transform(d.X_ood)))
        auc_base = ll_base = None
    else:
        base = C.fit_baseline_logistic(pre.transform(Xtr), ytr, a_base)
        pb = base.predict_proba(pre.transform(d.X_ood))[:, 1]
        auc_base = float(roc_auc_score(d.y_ood, pb))
        ll_base = float(log_loss(d.y_ood, np.clip(pb, 1e-12, 1 - 1e-12)))
        mse_base = None

    for family in families:
        variant = C.DEPLOYED[dataset][family][0]
        samples = C.load_samples(dataset, variant)

        ident = np.arange(p_feat)
        ones = np.ones(p_feat)

        if task == "regression":
            real_lift, real_val, real_a = eval_regression(
                dataset, family, "real", ident, ones, Xtr, ytr, d.X_ood, d.y_ood,
                samples, d.feature_names, mse_base)
        else:
            real_lift, real_ll_lift, real_auc, real_ll, real_a = eval_classification(
                dataset, family, "real", ident, ones, Xtr, ytr, d.X_ood, d.y_ood,
                samples, d.feature_names, auc_base, ll_base)

        for kind in ("permuted", "signflip"):
            lifts, ll_lifts = [], []
            for _ in range(n_draws):
                perm = rng.permutation(p_feat)
                signs = rng.choice([-1.0, 1.0], size=p_feat)
                if task == "regression":
                    lift, _, _ = eval_regression(
                        dataset, family, kind, perm, signs, Xtr, ytr, d.X_ood, d.y_ood,
                        samples, d.feature_names, mse_base)
                    lifts.append(lift)
                else:
                    lift, ll_lift, _, _, _ = eval_classification(
                        dataset, family, kind, perm, signs, Xtr, ytr, d.X_ood, d.y_ood,
                        samples, d.feature_names, auc_base, ll_base)
                    lifts.append(lift)
                    ll_lifts.append(ll_lift)
            lifts = np.array(lifts)
            for i, v in enumerate(lifts):
                row_d = {
                    "dataset": C.PRETTY[dataset], "task": task,
                    "family": C.FAMILY_LABEL[family], "placebo": kind,
                    "draw": i, "lift_pct": float(v),
                    "real_lift_pct": real_lift,
                }
                # Classification also carries the log-loss lift per draw. The
                # study tunes lambda on AUC, but AUC is the metric the priors
                # leave flat; log loss is where the effect lives, so the figure
                # needs its draws too.
                if task == "classification":
                    row_d["logloss_lift_pct"] = float(ll_lifts[i])
                    row_d["real_logloss_lift_pct"] = real_ll_lift
                draw_rows.append(row_d)

            row = {
                "dataset": C.PRETTY[dataset],
                "task": task,
                "family": C.FAMILY_LABEL[family],
                "variant": variant,
                "placebo": kind,
                "n_draws": n_draws,
                "real_lift_pct": real_lift,
                "placebo_mean_pct": float(lifts.mean()),
                "placebo_sd_pct": float(lifts.std(ddof=1)),
                "placebo_min_pct": float(lifts.min()),
                "placebo_max_pct": float(lifts.max()),
                "frac_placebo_beats_baseline": float(np.mean(lifts > 0)),
                "frac_placebo_beats_real": float(np.mean(lifts >= real_lift)),
                "empirical_p": float((np.sum(lifts >= real_lift) + 1) / (n_draws + 1)),
            }
            if task == "classification":
                ll_lifts = np.array(ll_lifts)
                row.update({
                    "real_logloss_lift_pct": real_ll_lift,
                    "placebo_logloss_mean_pct": float(ll_lifts.mean()),
                    "placebo_logloss_max_pct": float(ll_lifts.max()),
                    "frac_placebo_logloss_beats_real": float(np.mean(ll_lifts >= real_ll_lift)),
                    "empirical_p_logloss": float((np.sum(ll_lifts >= real_ll_lift) + 1) / (n_draws + 1)),
                })
            rows.append(row)
            print(f"  {C.PRETTY[dataset]:15s} {C.FAMILY_LABEL[family]:18s} {kind:9s}"
                  f" real {real_lift:+7.2f}%  placebo mean {lifts.mean():+7.2f}%"
                  f"  p={row['empirical_p']:.4f}", flush=True)

frame = pd.DataFrame(rows)
C.write(frame, "exp3_placebo.csv")
C.write(pd.DataFrame(draw_rows), "exp3_placebo_draws.csv")

print("\n" + "=" * 104)
print("Placebo control: does the prior's CONTENT drive the gain?")
print("  metric = OOD MSE lift for regression, OOD AUC lift for classification")
print("=" * 104)
print(f"{'dataset':16s} {'family':19s} {'placebo':9s} {'real':>8s} {'placebo mean':>13s}"
      f" {'placebo range':>20s} {'p':>8s}")
for _, r in frame.iterrows():
    rng_s = f"[{r['placebo_min_pct']:+.2f}, {r['placebo_max_pct']:+.2f}]"
    print(f"{r['dataset']:16s} {r['family']:19s} {r['placebo']:9s}"
          f" {r['real_lift_pct']:+7.2f}% {r['placebo_mean_pct']:+12.2f}%"
          f" {rng_s:>20s} {r['empirical_p']:8.4f}")

if "empirical_p_logloss" in frame:
    print("\n--- classification, same test on OOD log loss ---")
    sub = frame[frame["task"] == "classification"]
    for _, r in sub.iterrows():
        print(f"{r['dataset']:16s} {r['family']:19s} {r['placebo']:9s}"
              f" real {r['real_logloss_lift_pct']:+7.2f}%"
              f"  placebo mean {r['placebo_logloss_mean_pct']:+7.2f}%"
              f"  p={r['empirical_p_logloss']:.4f}")
