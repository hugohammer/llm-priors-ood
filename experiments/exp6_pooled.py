"""Experiment 6: pool the joint-uncertainty results across datasets.

The per-dataset joint intervals of exp5 all contain zero while the point
estimates are consistently positive, so the obvious question is whether
aggregating across datasets sharpens the evidence. It does, but only under
assumptions that have to be checked rather than asserted, so this script
computes every reasonable variant and reports them side by side.

For each family and metric we build the sampling distribution of a pooled lift
by drawing one exp5 joint draw per dataset independently and combining them,
repeated B times. Two combination rules:

  equal      unweighted mean of the per-dataset lifts. Treats every dataset as
             an equally interesting instance. Dominated by whichever dataset has
             the widest interval.
  invvar     inverse-variance (fixed-effect) weights, w_i = 1/var_i. Optimal
             *if* all datasets estimate a common effect. Dominated by whichever
             dataset is measured most precisely.

Both are reported because they answer different questions and, here, disagree.
Heterogeneity is quantified with Cochran's Q and I^2 so the reader can see
whether a common-effect pool is defensible at all.

Adult is reported in and out of the classification pool. It is a
data-dominated negative control by design (n_train ~ 27k, lambda driven to the
grid floor), so including a precisely measured null alongside the cases where a
prior could plausibly help conflates two different things.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

B = 200_000
RNG = np.random.default_rng(C.cfg.SEED)

GROUPS = [
    ("MSE", "regression (2 datasets)", ["Boston Housing", "Cancer"]),
    ("AUC", "classification (3 datasets)", ["Heart Disease", "Diabetes", "Adult"]),
    ("AUC", "classification, Adult excluded", ["Heart Disease", "Diabetes"]),
    ("LogLoss", "classification (3 datasets)", ["Heart Disease", "Diabetes", "Adult"]),
    ("LogLoss", "classification, Adult excluded", ["Heart Disease", "Diabetes"]),
]


def heterogeneity(means, variances):
    """Cochran's Q, its p-value, and I^2 for a fixed-effect pool."""
    from scipy.stats import chi2
    w = 1.0 / np.asarray(variances)
    m = np.asarray(means)
    pooled = float(np.sum(w * m) / np.sum(w))
    Q = float(np.sum(w * (m - pooled) ** 2))
    df = len(m) - 1
    p = float(1.0 - chi2.cdf(Q, df)) if df > 0 else np.nan
    I2 = float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0
    return Q, df, p, I2


def summarise(v):
    v = np.asarray(v)
    q = np.quantile(v, [0.025, 0.975])
    return dict(mean=float(v.mean()), q025=float(q[0]), q975=float(q[1]),
                frac_below=float(np.mean(v < 0)))


draws = pd.read_csv(C.OUT / "exp5_joint_draws.csv")
rows = []

for metric, label, datasets in GROUPS:
    for family in ("Prior-Mean", "Independent-Prior", "Correlated-Prior"):
        pools, means, variances = [], [], []
        ok = True
        for ds in datasets:
            v = draws[(draws.dataset == ds) & (draws.family == family)
                      & (draws.metric == metric)]["lift_pct"].values
            if v.size == 0:
                ok = False
                break
            pools.append(v)
            means.append(v.mean())
            variances.append(v.var(ddof=1))
        if not ok:
            continue

        # resample one draw per dataset, independently, B times
        picks = np.column_stack([RNG.choice(v, size=B, replace=True) for v in pools])

        equal = picks.mean(axis=1)
        w = 1.0 / np.asarray(variances)
        invvar = picks @ (w / w.sum())

        Q, df, pQ, I2 = heterogeneity(means, variances)
        for rule, vals in (("equal", equal), ("invvar", invvar)):
            s = summarise(vals)
            rows.append(dict(metric=metric, group=label, family=family, rule=rule,
                             k=len(datasets), Q=Q, Q_df=df, Q_p=pQ, I2=I2,
                             weights=np.round(w / w.sum(), 3).tolist(), **s))

frame = pd.DataFrame(rows)
C.write(frame, "exp6_pooled.csv")

print("=" * 118)
print("Pooled joint-uncertainty lift across datasets")
print("=" * 118)
print(f"{'metric':8s} {'group':32s} {'family':18s} {'rule':7s} {'mean':>7s} "
      f"{'95% interval':>18s} {'below':>6s} {'I^2':>6s} {'Q p':>7s}")
for _, r in frame.iterrows():
    ci = f"[{r.q025:+.2f}, {r.q975:+.2f}]"
    star = " *" if (r.q025 > 0 or r.q975 < 0) else "  "
    print(f"{r.metric:8s} {r.group:32s} {r.family:18s} {r.rule:7s} "
          f"{r['mean']:+6.2f} {ci:>18s}{star} {r.frac_below:5.1%} "
          f"{r.I2:5.0f}% {r.Q_p:7.4f}")

print("\nweights used by the inverse-variance rule (in dataset order):")
for _, r in frame[frame.rule == "invvar"].iterrows():
    print(f"  {r.metric:8s} {r.group:32s} {r.family:18s} {r.weights}")
