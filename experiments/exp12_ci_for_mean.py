"""Experiment 12: a confidence interval for the mean lift, not a prediction interval.

Reviewer 1's lead objection. The headline intervals pool S x B draws (S source
resplits, B bootstrap resamples of the OOD partition inside each) and report
the 2.5-97.5 percentile range. That is the spread of a *single future run* of
the pipeline -- a prediction interval. The Results prose says exactly this, but
the table captions call it a "95% confidence interval" and the Discussion turns
non-exclusion of zero into "do not reach standard thresholds for statistical
significance". Those are claims about a confidence interval for the mean, and
the two differ by whether the between-split component is divided by S.

With the split index now recorded (exp11) the decomposition is direct:

    sigma2_split   variance of the S per-split means        (source split + lambda)
    sigma2_tgt     mean within-split bootstrap variance     (target sampling)

For the *expected* lift, the source component averages over the S splits, but
the target component does not: all S splits are scored on the same realised
OOD partition, so there is only one draw of the target sample. Hence

    SE(mean) = sqrt(sigma2_split / S + sigma2_tgt)

and the prediction interval the paper currently reports corresponds instead to
sqrt(sigma2_split + sigma2_tgt), which is wider by construction.

Two estimands are reported, because the paper already argues both are
legitimate and they answer different questions:

  generalising      "what lift would a new deployment on a fresh sample from
                    this population give?"  -> SE above, target term at full width
  fixed-population  "did the prior help on this particular deployment
                    population, which is observed in full?"  -> target term
                    dropped, SE = sqrt(sigma2_split / S)

The second is the right frame for OOD partitions that are complete
subpopulations (every Hungarian and Swiss patient; every non-Southern county),
which is the caveat the paper already states in the Results.

Nothing here is a re-fit: it is a re-reading of the exp11 draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

Z = stats.norm.ppf(0.975)


def main():
    draws = pd.read_csv(C.OUT / "exp11_joint_draws.csv")
    joint = pd.read_csv(C.OUT / "exp11_joint.csv")

    rows = []
    for (ds, fam, metric), g in draws.groupby(["dataset", "family", "metric"],
                                              sort=False):
        per_split = g.groupby("split")["lift_pct"]
        split_means = per_split.mean().values
        S = len(split_means)
        var_split = float(np.var(split_means, ddof=1)) if S > 1 else 0.0
        var_tgt = float(per_split.var(ddof=1).mean())
        mean = float(g["lift_pct"].mean())

        se_gen = np.sqrt(var_split / S + var_tgt)
        se_fix = np.sqrt(var_split / S)

        src = joint[(joint.dataset == ds) & (joint.family == fam)
                    & (joint.metric == metric)]
        pi_lo = float(src.joint_q025.iloc[0]) if len(src) else np.nan
        pi_hi = float(src.joint_q975.iloc[0]) if len(src) else np.nan

        rows.append(dict(
            dataset=ds, family=fam, metric=metric, n_splits=S, n_draws=len(g),
            mean=mean, sd_split=np.sqrt(var_split), sd_target=np.sqrt(var_tgt),
            pred_lo=pi_lo, pred_hi=pi_hi,
            ci_gen_lo=mean - Z * se_gen, ci_gen_hi=mean + Z * se_gen,
            ci_fix_lo=mean - Z * se_fix, ci_fix_hi=mean + Z * se_fix,
            pred_excl0=bool(pi_lo > 0 or pi_hi < 0),
            ci_gen_excl0=bool((mean - Z * se_gen) > 0 or (mean + Z * se_gen) < 0),
            ci_fix_excl0=bool((mean - Z * se_fix) > 0 or (mean + Z * se_fix) < 0)))

    out = pd.DataFrame(rows)
    C.write(out, "exp12_ci_for_mean.csv")

    print("=" * 118)
    print("Prediction interval (as published) vs confidence interval for the MEAN lift")
    print("=" * 118)
    print(f"{'dataset':15s} {'family':18s} {'metric':8s} {'mean':>7s} "
          f"{'prediction 95% (published)':>28s} {'CI generalising':>22s} "
          f"{'CI fixed-pop':>21s}")
    for _, r in out.iterrows():
        f1 = "*" if r.pred_excl0 else " "
        f2 = "*" if r.ci_gen_excl0 else " "
        f3 = "*" if r.ci_fix_excl0 else " "
        print(f"{r.dataset:15s} {r.family:18s} {r.metric:8s} {r['mean']:+6.2f} "
              f"[{r.pred_lo:+7.2f}, {r.pred_hi:+7.2f}]{f1:2s}"
              f"  [{r.ci_gen_lo:+7.2f}, {r.ci_gen_hi:+7.2f}]{f2:2s}"
              f"  [{r.ci_fix_lo:+6.2f}, {r.ci_fix_hi:+6.2f}]{f3}")

    print()
    print("cells whose interval excludes zero:")
    for col, label in (("pred_excl0", "prediction interval (as published)"),
                       ("ci_gen_excl0", "CI for mean, generalising"),
                       ("ci_fix_excl0", "CI for mean, fixed population")):
        n = int(out[col].sum())
        print(f"  {label:38s} {n:2d} of {len(out)}")
        if n:
            for _, r in out[out[col]].iterrows():
                print(f"       {r.dataset} / {r.family} / {r.metric}  "
                      f"mean {r['mean']:+.2f}")

    print()
    print("variance split (how much of the width is target sampling):")
    for _, r in out.iterrows():
        tot = r.sd_split ** 2 + r.sd_target ** 2
        print(f"  {r.dataset:15s} {r.family:18s} {r.metric:8s} "
              f"sd_split {r.sd_split:6.2f}  sd_target {r.sd_target:6.2f}  "
              f"target share of variance {100 * r.sd_target ** 2 / max(tot, 1e-12):5.1f}%")


if __name__ == "__main__":
    main()
