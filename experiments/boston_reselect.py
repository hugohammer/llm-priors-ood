"""Re-run provider/family selection for Boston Housing without the B feature.

Dropping B changes only this dataset, so only this dataset is re-selected. The
protocol is the study's own: lambda and the winning LLM variant are both chosen
by best mean 5-fold CV on the source training pool, and the OOD partition is
touched once, at evaluation.

Outputs
  boston_nob_cells.csv       all 12 x 4 (variant, family) cells at their CV lambda
  boston_nob_deployment.csv  the deployed row per family, with paired bootstrap CIs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

DATASET = "boston_housing"
GAMMA_GRID = C.cfg.GAMMA_GRID


def main():
    d = C.load(DATASET)
    Xtr, Xid, ytr, yid = C.split_source(d.X_source, d.y_source,
                                        seed=C.cfg.SEED, stratify=False)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=False)

    pre = C.Preprocessor().fit(Xtr)
    A, ID, O = pre.transform(Xtr), pre.transform(Xid), pre.transform(d.X_ood)

    # ---- baseline, lambda by the same CV rule ---------------------------
    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(DATASET, "Opus 4.5"),
                       d.feature_names)
    a_base, _ = C.tune_alpha_baseline_regression(packs0)
    base = C.fit_baseline_ridge(A, ytr, a_base)
    pred_base = base.predict(O)
    mse_base_ood = C.mse(d.y_ood, pred_base)
    mse_base_id = C.mse(yid, base.predict(ID))

    unreg = C.fit_baseline_ridge(A, ytr, 0.0)
    print(f"baseline ridge  lambda={a_base:g}  ID MSE={mse_base_id:.2f}  "
          f"OOD MSE={mse_base_ood:.2f}")
    print(f"unregularised   OOD MSE={C.mse(d.y_ood, unreg.predict(O)):.2f}\n")

    rows = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(DATASET, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)

        for family in ("prior_mean", "independent", "correlated"):
            alpha, scores = C.tune_alpha_regression(packs, family)
            m = C.fit_prior_ridge(A, ytr, pq.mean, C.precision_for(family, pq), alpha)
            rows.append(dict(variant=variant, family=family, tuned=alpha,
                             cv=float(scores.mean(axis=0).min()),
                             id_mse=C.mse(yid, m.predict(ID)),
                             ood_mse=C.mse(d.y_ood, m.predict(O))))

        # Mixed: blend baseline and Prior-Mean predictions, gamma by CV
        a_pm, _ = C.tune_alpha_regression(packs, "prior_mean")
        fold_pred = []
        for k, f in enumerate(packs):
            pm = C.fit_prior_ridge(f.X_train, f.y_train, f.pq.mean, f.pq.identity, a_pm)
            bl = C.fit_baseline_ridge(f.X_train, f.y_train, a_base)
            fold_pred.append((f.y_val, bl.predict(f.X_val), pm.predict(f.X_val)))
        gscore = np.array([[C.mse(yv, (1 - g) * pb + g * pp) for g in GAMMA_GRID]
                           for yv, pb, pp in fold_pred])
        gamma = float(GAMMA_GRID[int(np.argmin(gscore.mean(axis=0)))])
        pm_full = C.fit_prior_ridge(A, ytr, pq.mean, pq.identity, a_pm)
        mix_ood = (1 - gamma) * pred_base + gamma * pm_full.predict(O)
        mix_id = (1 - gamma) * base.predict(ID) + gamma * pm_full.predict(ID)
        rows.append(dict(variant=variant, family="mixed", tuned=gamma,
                         cv=float(gscore.mean(axis=0).min()),
                         id_mse=C.mse(yid, mix_id),
                         ood_mse=C.mse(d.y_ood, mix_ood)))

    cells = pd.DataFrame(rows)
    cells["lift_pct"] = 100.0 * (mse_base_ood - cells["ood_mse"]) / mse_base_ood
    C.write(cells, "boston_nob_cells.csv")

    # ---- deployed model per family: best mean-CV score ------------------
    print(f"{'family':20s} {'selected variant':17s} {'tuned':>9s} {'ID MSE':>8s} "
          f"{'OOD MSE':>8s} {'lift':>8s}  paired delta 95% CI")
    dep = []
    for family in ("prior_mean", "independent", "correlated", "mixed"):
        g = cells[cells.family == family]
        win = g.loc[g["cv"].idxmin()]
        samples = C.load_samples(DATASET, win.variant)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
        if family == "mixed":
            packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
            a_pm, _ = C.tune_alpha_regression(packs, "prior_mean")
            pm_full = C.fit_prior_ridge(A, ytr, pq.mean, pq.identity, a_pm)
            pred = (1 - win.tuned) * pred_base + win.tuned * pm_full.predict(O)
        else:
            m = C.fit_prior_ridge(A, ytr, pq.mean, C.precision_for(family, pq), win.tuned)
            pred = m.predict(O)
        lo, hi = C.paired_bootstrap(d.y_ood, pred, pred_base,
                                    lambda yy, pp: float(np.mean((yy - pp) ** 2)),
                                    stratify=False)
        dep.append(dict(family=C.FAMILY_LABEL[family], variant=win.variant,
                        tuned=win.tuned, id_mse=win.id_mse, ood_mse=win.ood_mse,
                        lift_pct=win.lift_pct, delta_lo=lo, delta_hi=hi,
                        baseline_ood_mse=mse_base_ood, baseline_id_mse=mse_base_id,
                        baseline_lambda=a_base))
        print(f"{C.FAMILY_LABEL[family]:20s} {win.variant:17s} {win.tuned:9.4g} "
              f"{win.id_mse:8.2f} {win.ood_mse:8.2f} {win.lift_pct:+7.2f}%  "
              f"[{lo:+.2f}, {hi:+.2f}]")

    C.write(pd.DataFrame(dep), "boston_nob_deployment.csv")

    print(f"\nselection floor / ceiling across all {len(cells)} cells: "
          f"{cells.lift_pct.min():+.2f}% / {cells.lift_pct.max():+.2f}%")
    print(f"deployed (Independent-Prior is the study's headline family): "
          f"{dep[1]['lift_pct']:+.2f}%")


if __name__ == "__main__":
    main()
