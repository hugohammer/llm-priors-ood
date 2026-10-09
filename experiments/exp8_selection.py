"""Experiment 8: an honest, source-only selection rule over the full cell grid.

Both referees found the same defect: the "deployed pipeline" marker in the
selection figure was computed as the maximum OOD lift over the four families
(`make_thesis_figures.py`: `deployed=float(dep.lift_pct.max())`, with three
datasets hard-coded at their own per-family maxima). That is selection on the
test set, which contradicts the paper's stated governing principle.

The Methods also define a selection rule only *within* family, while the text
and abstract claim selection over the 48-cell (variant, family) grid. This
script defines the missing rule and applies it:

  within-family   per family, the variant with the best mean 5-fold CV score
                  on the source training pool   (the study's documented rule)
  cross-family    the single best cell over all 48, by the same CV score
                  (the rule the abstract implies but never specified)

For every cell it records the CV score, its fold-level standard error, the
in-domain held-out metric and the OOD metric, so that three separate referee
points can be answered from one sweep:

  * what the honest deployed lift actually is, per dataset;
  * whether the CV differences being selected on exceed their own noise
    (Reviewer 1: "selection among 48 candidates separated by far less than one
    fold-level SE is close to selection at random");
  * whether the lift is OOD-specific or also present in-domain (Reviewer 1
    found Boston's in-domain lift exceeds its OOD lift, which is not reported
    anywhere in the manuscript).

Regression scores are CV MSE (lower better); classification uses the study's
own tuning metric, CV AUC (higher better). Both are comparable across families
within a dataset because the folds and the target are identical, which is what
makes a cross-family rule well defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import common as C

GAMMA_GRID = C.cfg.GAMMA_GRID
FAMILIES = ("prior_mean", "independent", "correlated")
STRIDE = {"adult": 4}  # logistic fits on n ~ 27k; lambda collapses to the floor anyway


def _grid(dataset):
    return C.cfg.ALPHA_GRID[::STRIDE.get(dataset, 1)]


def sweep_regression(dataset):
    d = C.load(dataset)
    Xtr, Xid, ytr, yid = C.split_source(d.X_source, d.y_source,
                                        seed=C.cfg.SEED, stratify=False)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=False)
    pre = C.Preprocessor().fit(Xtr)
    A, ID, O = pre.transform(Xtr), pre.transform(Xid), pre.transform(d.X_ood)
    grid = _grid(dataset)

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    a_base, _ = C.tune_alpha_baseline_regression(packs0, grid=grid)
    base = C.fit_baseline_ridge(A, ytr, a_base)
    pred_base = base.predict(O)
    base_ood = C.mse(d.y_ood, pred_base)
    base_id = C.mse(yid, base.predict(ID))

    rows = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(dataset, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)

        for family in FAMILIES:
            alpha, scores = C.tune_alpha_regression(packs, family, grid=grid)
            mean = scores.mean(axis=0)
            j = int(np.argmin(mean))
            m = C.fit_prior_ridge(A, ytr, pq.mean, C.precision_for(family, pq), alpha)
            rows.append(dict(variant=variant, family=family, tuned=alpha,
                             cv=float(mean[j]),
                             cv_se=float(scores[:, j].std(ddof=1) / np.sqrt(len(packs))),
                             id_metric=C.mse(yid, m.predict(ID)),
                             ood_metric=C.mse(d.y_ood, m.predict(O))))

        # Mixed: blend baseline and Prior-Mean predictions; gamma by CV
        a_pm, _ = C.tune_alpha_regression(packs, "prior_mean", grid=grid)
        fold_pred = []
        for f in packs:
            pm = C.fit_prior_ridge(f.X_train, f.y_train, f.pq.mean, f.pq.identity, a_pm)
            bl = C.fit_baseline_ridge(f.X_train, f.y_train, a_base)
            fold_pred.append((f.y_val, bl.predict(f.X_val), pm.predict(f.X_val)))
        gs = np.array([[C.mse(yv, (1 - g) * pb + g * pp) for g in GAMMA_GRID]
                       for yv, pb, pp in fold_pred])
        mean = gs.mean(axis=0)
        j = int(np.argmin(mean))
        gamma = float(GAMMA_GRID[j])
        pm_full = C.fit_prior_ridge(A, ytr, pq.mean, pq.identity, a_pm)
        rows.append(dict(variant=variant, family="mixed", tuned=gamma,
                         cv=float(mean[j]),
                         cv_se=float(gs[:, j].std(ddof=1) / np.sqrt(len(packs))),
                         id_metric=C.mse(yid, (1 - gamma) * base.predict(ID)
                                         + gamma * pm_full.predict(ID)),
                         ood_metric=C.mse(d.y_ood, (1 - gamma) * pred_base
                                          + gamma * pm_full.predict(O))))

    cells = pd.DataFrame(rows)
    cells["lift_pct"] = 100.0 * (base_ood - cells["ood_metric"]) / base_ood
    cells["id_lift_pct"] = 100.0 * (base_id - cells["id_metric"]) / base_id
    cells["higher_is_better"] = False
    return cells, base_ood, base_id


def sweep_classification(dataset):
    d = C.load(dataset)
    Xtr, Xid, ytr, yid = C.split_source(d.X_source, d.y_source,
                                        seed=C.cfg.SEED, stratify=True)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=True)
    pre = C.Preprocessor().fit(Xtr)
    A, ID, O = pre.transform(Xtr), pre.transform(Xid), pre.transform(d.X_ood)
    grid = _grid(dataset)

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    a_base, _ = C.tune_alpha_baseline_classification(
        packs0, grid=np.sort(1.0 / C.cfg.C_GRID)[::STRIDE.get(dataset, 1)])
    base = C.fit_baseline_logistic(A, ytr, a_base)
    pb_ood = base.predict_proba(O)[:, 1]
    pb_id = base.predict_proba(ID)[:, 1]
    base_ood = float(roc_auc_score(d.y_ood, pb_ood))
    base_id = float(roc_auc_score(yid, pb_id))
    base_ll = float(log_loss(d.y_ood, np.clip(pb_ood, 1e-12, 1 - 1e-12)))

    rows = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(dataset, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)

        for family in FAMILIES:
            alpha, scores = C.tune_alpha_classification(packs, family, grid=grid)
            mean = scores.mean(axis=0)
            j = int(np.argmax(mean))
            m = C.fit_prior_logistic(A, ytr, pq.mean, C.precision_for(family, pq),
                                     alpha, seed=C.start_seed_for(family))
            p_ood, p_id = m.predict_proba(O), m.predict_proba(ID)
            rows.append(dict(variant=variant, family=family, tuned=alpha,
                             cv=float(mean[j]),
                             cv_se=float(scores[:, j].std(ddof=1) / np.sqrt(len(packs))),
                             id_metric=float(roc_auc_score(yid, p_id)),
                             ood_metric=float(roc_auc_score(d.y_ood, p_ood)),
                             ood_logloss=float(log_loss(
                                 d.y_ood, np.clip(p_ood, 1e-12, 1 - 1e-12)))))

        from llmpriors.metrics import tuning_score
        a_pm, _ = C.tune_alpha_classification(packs, "prior_mean", grid=grid)
        fold_pred = []
        for f in packs:
            pm = C.fit_prior_logistic(f.X_train, f.y_train, f.pq.mean,
                                      f.pq.identity, a_pm, seed=C.cfg.SEED)
            bl = C.fit_baseline_logistic(f.X_train, f.y_train, a_base)
            fold_pred.append((f.y_val, bl.predict_proba(f.X_val)[:, 1],
                              pm.predict_proba(f.X_val)))
        gs = np.array([[tuning_score(yv, (1 - g) * pb + g * pp, "auc")
                        for g in GAMMA_GRID] for yv, pb, pp in fold_pred])
        mean = gs.mean(axis=0)
        j = int(np.argmax(mean))
        gamma = float(GAMMA_GRID[j])
        pm_full = C.fit_prior_logistic(A, ytr, pq.mean, pq.identity, a_pm,
                                       seed=C.cfg.SEED)
        mix_ood = (1 - gamma) * pb_ood + gamma * pm_full.predict_proba(O)
        mix_id = (1 - gamma) * pb_id + gamma * pm_full.predict_proba(ID)
        rows.append(dict(variant=variant, family="mixed", tuned=gamma,
                         cv=float(mean[j]),
                         cv_se=float(gs[:, j].std(ddof=1) / np.sqrt(len(packs))),
                         id_metric=float(roc_auc_score(yid, mix_id)),
                         ood_metric=float(roc_auc_score(d.y_ood, mix_ood)),
                         ood_logloss=float(log_loss(
                             d.y_ood, np.clip(mix_ood, 1e-12, 1 - 1e-12)))))

    cells = pd.DataFrame(rows)
    cells["lift_pct"] = 100.0 * (cells["ood_metric"] - base_ood) / base_ood
    cells["id_lift_pct"] = 100.0 * (cells["id_metric"] - base_id) / base_id
    cells["logloss_lift_pct"] = 100.0 * (base_ll - cells["ood_logloss"]) / base_ll
    cells["higher_is_better"] = True
    return cells, base_ood, base_id


def main():
    frames, summary = [], []
    for dataset in C.TASK:
        task = C.TASK[dataset]
        cells, base_ood, base_id = (sweep_regression(dataset) if task == "regression"
                                    else sweep_classification(dataset))
        cells.insert(0, "dataset", C.PRETTY[dataset])
        cells.insert(1, "task", task)
        frames.append(cells)

        hib = bool(cells["higher_is_better"].iloc[0])
        pick = (lambda g: g.loc[g["cv"].idxmax()]) if hib else (lambda g: g.loc[g["cv"].idxmin()])

        # the study's documented rule: best CV variant within each family
        within = {fam: pick(cells[cells.family == fam]) for fam in
                  ("prior_mean", "independent", "correlated", "mixed")}
        # the rule the abstract implies: best CV cell over the whole grid
        cross = pick(cells)

        print(f"\n=== {C.PRETTY[dataset]} ({task}) ===")
        print(f"  cells {len(cells)}   OOD lift floor {cells.lift_pct.min():+.2f}%"
              f"   ceiling {cells.lift_pct.max():+.2f}%")
        print(f"  {'rule':28s} {'selected cell':34s} {'CV':>10s} {'OOD lift':>9s}")
        for fam, r in within.items():
            print(f"  within-family: {C.FAMILY_LABEL[fam]:13s} "
                  f"{r.variant + ' / ' + fam:34s} {r.cv:10.4f} {r.lift_pct:+8.2f}%")
        print(f"  {'CROSS-FAMILY (source CV)':28s} "
              f"{cross.variant + ' / ' + cross.family:34s} {cross.cv:10.4f} "
              f"{cross.lift_pct:+8.2f}%")
        print(f"  {'OOD-BEST (what Fig 4 plotted)':28s} "
              f"{cells.loc[cells.lift_pct.idxmax()].variant:34s} "
              f"{'':10s} {cells.lift_pct.max():+8.2f}%")

        # is the CV margin bigger than its own noise?
        srt = cells.sort_values("cv", ascending=not hib)
        margin = abs(float(srt.iloc[0].cv) - float(srt.iloc[1].cv))
        print(f"  CV margin over runner-up {margin:.5f}   "
              f"fold SE at winner {float(srt.iloc[0].cv_se):.5f}   "
              f"margin/SE = {margin / max(float(srt.iloc[0].cv_se), 1e-12):.2f}")

        summary.append(dict(
            dataset=C.PRETTY[dataset], task=task,
            floor=float(cells.lift_pct.min()), ceiling=float(cells.lift_pct.max()),
            ood_best_lift=float(cells.lift_pct.max()),
            cross_family_variant=cross.variant, cross_family_family=cross.family,
            cross_family_cv=float(cross.cv), cross_family_lift=float(cross.lift_pct),
            cross_family_id_lift=float(cross.id_lift_pct),
            cv_margin=margin, cv_se_at_winner=float(srt.iloc[0].cv_se),
            margin_over_se=margin / max(float(srt.iloc[0].cv_se), 1e-12),
            **{f"within_{f}_lift": float(r.lift_pct) for f, r in within.items()},
            **{f"within_{f}_variant": r.variant for f, r in within.items()}))

    C.write(pd.concat(frames, ignore_index=True), "exp8_selection_cells.csv")
    C.write(pd.DataFrame(summary), "exp8_selection_summary.csv")

    print("\n" + "=" * 100)
    print("HONEST SELECTION: source-CV cross-family vs what the figure plotted")
    print("=" * 100)
    print(f"{'dataset':16s} {'floor':>8s} {'ceiling':>9s} {'cross-family (CV)':>19s} "
          f"{'OOD-best (old)':>15s} {'gap':>8s}")
    for s in summary:
        print(f"{s['dataset']:16s} {s['floor']:+7.2f}% {s['ceiling']:+8.2f}% "
              f"{s['cross_family_lift']:+18.2f}% {s['ood_best_lift']:+14.2f}% "
              f"{s['ood_best_lift'] - s['cross_family_lift']:+7.2f}%")


if __name__ == "__main__":
    main()
