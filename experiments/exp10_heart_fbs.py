"""Experiment 10: does the Heart Disease result depend on the audited feature?

The Heart Disease loader's docstring records a decision the manuscript does not:

    `ca`, `thal` and `slope` are dropped: each is >= 50% missing in the OOD
    cohorts. `fbs` is kept at roughly 20% OOD missingness - an audit found
    that dropping it slightly hurt prior performance with no offsetting
    benefit.

Two problems. The Methods state the drop rule as >= 50% *source-domain*
missingness; on Cleveland the rates are ca 1.3%, thal 0.7%, slope 0.0%, so the
stated rule would drop nothing. And `fbs` was retained after checking its
effect on the prior-informed model -- a feature decision taken with reference
to the outcome being tested, on the dataset carrying the largest log-loss
claim in the paper.

Restoring ca/thal/slope is not possible from stored data: the elicited Heart
priors cover 18 one-hot features and contain no coordinates for them, so that
would need re-elicitation. The `fbs` decision *is* testable, by the same
principal-submatrix device used to drop Boston's B feature, and that is the
decision that was made by looking at the answer.

This script re-runs the full 48-cell selection for Heart Disease with and
without fbs, re-selecting the deployed variant by source CV in each case, so
the comparison is between two complete pipelines rather than two refits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from exp8_selection import sweep_classification

FBS_KEYS = ["cat__fbs=0", "cat__fbs=1"]
ARMS = [("with fbs (as published)", []), ("without fbs", FBS_KEYS)]


def main():
    frames, summary = [], []
    for label, drop in ARMS:
        C.DROP_FEATURES["heart_disease"] = drop
        cells, base_auc, base_id = sweep_classification("heart_disease")
        cells.insert(0, "arm", label)
        frames.append(cells)

        d = C.load("heart_disease")
        print(f"\n=== {label} === p = {len(d.feature_names)} features, "
              f"baseline OOD AUC {base_auc:.4f}", flush=True)

        pick = lambda g: g.loc[g["cv"].idxmax()]
        within = {f: pick(cells[cells.family == f])
                  for f in ("prior_mean", "independent", "correlated")}
        cross = pick(cells)
        for fam, r in within.items():
            print(f"  {C.FAMILY_LABEL[fam]:18s} {r.variant:16s} "
                  f"AUC lift {r.lift_pct:+6.2f}%   "
                  f"log-loss lift {r.logloss_lift_pct:+7.2f}%", flush=True)
        print(f"  CROSS-FAMILY (CV)  {cross.variant:16s} "
              f"AUC lift {cross.lift_pct:+6.2f}%   "
              f"log-loss lift {cross.logloss_lift_pct:+7.2f}%", flush=True)

        row = dict(arm=label, n_features=len(d.feature_names),
                   baseline_ood_auc=base_auc,
                   cross_variant=cross.variant, cross_family=cross.family,
                   cross_auc_lift=float(cross.lift_pct),
                   cross_logloss_lift=float(cross.logloss_lift_pct))
        for fam, r in within.items():
            row[f"{fam}_variant"] = r.variant
            row[f"{fam}_auc_lift"] = float(r.lift_pct)
            row[f"{fam}_logloss_lift"] = float(r.logloss_lift_pct)
        summary.append(row)

    C.DROP_FEATURES.pop("heart_disease", None)
    C.write(pd.concat(frames, ignore_index=True), "exp10_heart_fbs_cells.csv")
    sm = pd.DataFrame(summary)
    C.write(sm, "exp10_heart_fbs_summary.csv")

    print("\n" + "=" * 92)
    print("Does the Heart Disease claim survive dropping the audited feature?")
    print("=" * 92)
    a, b = summary[0], summary[1]
    print(f"{'family':20s} {'with fbs':>22s} {'without fbs':>22s} {'change':>12s}")
    for fam in ("prior_mean", "independent", "correlated"):
        for metric, suffix in (("AUC", "auc_lift"), ("log loss", "logloss_lift")):
            k = f"{fam}_{suffix}"
            print(f"{C.FAMILY_LABEL[fam] + ' ' + metric:20s} "
                  f"{a[k]:+21.2f}% {b[k]:+21.2f}% {b[k] - a[k]:+11.2f}%")
    print(f"{'cross-family AUC':20s} {a['cross_auc_lift']:+21.2f}% "
          f"{b['cross_auc_lift']:+21.2f}% {b['cross_auc_lift'] - a['cross_auc_lift']:+11.2f}%")
    print(f"{'cross-family logloss':20s} {a['cross_logloss_lift']:+21.2f}% "
          f"{b['cross_logloss_lift']:+21.2f}% "
          f"{b['cross_logloss_lift'] - a['cross_logloss_lift']:+11.2f}%")


if __name__ == "__main__":
    main()
