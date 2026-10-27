"""Experiment 13: how far apart are the selection candidates, measured correctly?

exp8 compared the CV margin between the winning cell and the runner-up against
the fold-level standard error of the winner's own CV score. That is the wrong
yardstick. Every candidate is scored on the *same* five folds, so fold-to-fold
variation is common to all of them and cancels in a comparison. The margin
should be compared against the standard error of the paired difference,

    SE_paired = sd_k( s_winner,k - s_runnerup,k ) / sqrt(K),

which is typically far smaller than either candidate's own SE and is the
quantity a paired comparison actually turns on.

Two further corrections to the same analysis:

  * Mixed collapses onto Prior-Mean whenever its tuned gamma is 1, so the two
    cells are the same fitted model and an exact tie between them says nothing
    about selection. Duplicate cells are identified by comparing the per-fold
    score vectors and are excluded from the runner-up comparison.
  * The runner-up is taken over *distinct* models, so the margin reported here
    is between the winner and the best genuinely different candidate.

This records the per-fold CV score at each cell's selected lambda, which exp8
discarded, and recomputes the margin on both scales so the two can be compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

FAMILIES = ("prior_mean", "independent", "correlated")
STRIDE = {"adult": 4}
GAMMA_GRID = C.cfg.GAMMA_GRID


def per_fold_scores(dataset):
    """For every (variant, family) cell: the per-fold CV scores at its own tuned lambda."""
    task = C.TASK[dataset]
    stratify = task == "classification"
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED,
                                    stratify=stratify)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=stratify)
    pre = C.Preprocessor().fit(Xtr)
    A = pre.transform(Xtr)
    grid = C.cfg.ALPHA_GRID[::STRIDE.get(dataset, 1)]
    higher = task == "classification"

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    if task == "regression":
        a_base, _ = C.tune_alpha_baseline_regression(packs0, grid=grid)
    else:
        a_base, _ = C.tune_alpha_baseline_classification(
            packs0, grid=np.sort(1.0 / C.cfg.C_GRID)[::STRIDE.get(dataset, 1)])

    rows = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(dataset, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)

        for family in FAMILIES:
            if task == "regression":
                alpha, scores = C.tune_alpha_regression(packs, family, grid=grid)
            else:
                alpha, scores = C.tune_alpha_classification(packs, family, grid=grid)
            mean = scores.mean(axis=0)
            j = int(np.argmax(mean) if higher else np.argmin(mean))
            rows.append(dict(variant=variant, family=family, tuned=alpha,
                             cv=float(mean[j]), folds=scores[:, j].copy()))

        # Mixed: blended predictions, gamma by CV on the same folds
        if task == "regression":
            a_pm, _ = C.tune_alpha_regression(packs, "prior_mean", grid=grid)
            fp = []
            for f in packs:
                pm = C.fit_prior_ridge(f.X_train, f.y_train, f.pq.mean, f.pq.identity, a_pm)
                bl = C.fit_baseline_ridge(f.X_train, f.y_train, a_base)
                fp.append((f.y_val, bl.predict(f.X_val), pm.predict(f.X_val)))
            gs = np.array([[C.mse(yv, (1 - g) * pb + g * pp) for g in GAMMA_GRID]
                           for yv, pb, pp in fp])
        else:
            from llmpriors.metrics import tuning_score
            a_pm, _ = C.tune_alpha_classification(packs, "prior_mean", grid=grid)
            fp = []
            for f in packs:
                pm = C.fit_prior_logistic(f.X_train, f.y_train, f.pq.mean,
                                          f.pq.identity, a_pm, seed=C.cfg.SEED)
                bl = C.fit_baseline_logistic(f.X_train, f.y_train, a_base)
                fp.append((f.y_val, bl.predict_proba(f.X_val)[:, 1],
                           pm.predict_proba(f.X_val)))
            gs = np.array([[tuning_score(yv, (1 - g) * pb + g * pp, "auc")
                            for g in GAMMA_GRID] for yv, pb, pp in fp])
        mean = gs.mean(axis=0)
        j = int(np.argmax(mean) if higher else np.argmin(mean))
        rows.append(dict(variant=variant, family="mixed", tuned=float(GAMMA_GRID[j]),
                         cv=float(mean[j]), folds=gs[:, j].copy()))
    return rows, higher


def main():
    out = []
    for dataset in C.TASK:
        rows, higher = per_fold_scores(dataset)
        rows.sort(key=lambda r: -r["cv"] if higher else r["cv"])
        win = rows[0]

        # a candidate is a duplicate of the winner if its per-fold scores match
        def same(a, b):
            return np.allclose(a["folds"], b["folds"], rtol=0, atol=1e-10)

        dupes = [r for r in rows[1:] if same(win, r)]
        distinct = [r for r in rows[1:] if not same(win, r)]
        runner = distinct[0]

        diff = win["folds"] - runner["folds"]
        K = len(diff)
        se_paired = float(diff.std(ddof=1) / np.sqrt(K))
        se_single = float(win["folds"].std(ddof=1) / np.sqrt(K))
        margin = abs(float(win["cv"] - runner["cv"]))

        # where in the pool does the winner's OOD performance sit?
        print(f"\n=== {C.PRETTY[dataset]} ===", flush=True)
        print(f"  winner    {win['variant']} / {win['family']}  CV {win['cv']:.5f}")
        if dupes:
            print(f"  duplicates of the winner (same fitted model): "
                  f"{', '.join(d['variant']+'/'+d['family'] for d in dupes)}")
        print(f"  runner-up {runner['variant']} / {runner['family']}  CV {runner['cv']:.5f}")
        print(f"  margin {margin:.5f}")
        print(f"    vs single-candidate SE {se_single:.5f}  ->  {margin/max(se_single,1e-12):5.2f} SE")
        print(f"    vs PAIRED SE           {se_paired:.5f}  ->  {margin/max(se_paired,1e-12):5.2f} SE")

        out.append(dict(dataset=C.PRETTY[dataset],
                        winner=f"{win['variant']}/{win['family']}",
                        runner_up=f"{runner['variant']}/{runner['family']}",
                        n_duplicates=len(dupes), margin=margin,
                        se_single=se_single, se_paired=se_paired,
                        margin_over_se_single=margin / max(se_single, 1e-12),
                        margin_over_se_paired=margin / max(se_paired, 1e-12)))

    df = pd.DataFrame(out)
    C.write(df, "exp13_paired_margin.csv")
    print("\n" + "=" * 92)
    print("Selection margin on the correct (paired) scale")
    print("=" * 92)
    print(f"{'dataset':16s} {'margin':>10s} {'/ single SE':>12s} {'/ paired SE':>12s} {'dupes':>6s}")
    for r in out:
        print(f"{r['dataset']:16s} {r['margin']:10.5f} "
              f"{r['margin_over_se_single']:12.2f} {r['margin_over_se_paired']:12.2f} "
              f"{r['n_duplicates']:6d}")
    lo, hi = df.margin_over_se_paired.min(), df.margin_over_se_paired.max()
    print(f"\npaired margins span {lo:.2f} to {hi:.2f} standard errors")


if __name__ == "__main__":
    main()
