# Results and what they change in the paper

Run 2026-09-14 on the thesis pipeline and its committed priors. Numbers below
are from `results/*.csv`; console output in `logs/`.

Summary in one line: **the regression and calibration claims survive the
placebo control, but none survives a full uncertainty accounting at 95%; the F1
claim and Heart Disease's AUC lift do not survive either.** See Addendum 3,
which supersedes the significance statements in sections 1-4 below.

---

## 1. Prior-only (λ→∞): the LLM coefficients with no data fitting

Coefficient vector frozen at the elicited prior mean, intercept fitted on the
source training pool. `results/exp1_prior_only.csv` has all 12 variants per
dataset; deployed Prior-Mean variant shown here.

| dataset | variant | vs. fitted baseline | paired Δ 95% CI |
|---|---|---|---|
| Cancer | Opus 4.5 | **+12.8%** MSE | [−94.5, −33.4] excludes 0 |
| Diabetes | Opus 4.5 | +0.8% AUC | [−0.0065, +0.0200] |
| Heart Disease | Opus 4.5 | +0.9% AUC | [−0.0094, +0.0250] |
| Adult | GPT-5-mini | −1.7% AUC | [−0.0174, −0.0126] |
| Boston Housing | Gemini Pro | −99.9% MSE | [+28.0, +67.3] |

Best / worst variant per dataset (OOD lift over baseline):

| dataset | best | worst |
|---|---|---|
| Boston Housing | Opus 4.5 −2.1% | Gemini Flash **−128,060%** |
| Cancer | Opus 4.5 +12.8% | Haiku 4.5 (R) −106.9% |
| Heart Disease | Opus 4.5 +0.9% | Haiku 4.5 −9.2% |
| Diabetes | GPT-5-mini +2.0% | Gemini Flash +0.8% |
| Adult | Opus 4.5 −0.8% | Haiku 4.5 (R) −3.9% |

**For the paper.** On three of five datasets a zero-data LLM prior is
competitive with a data-fitted regularised model on the OOD set, and on Cancer
it beats every fitted model in the study — CV is *under*-using the prior there,
because λ is tuned on source data where the prior looks worse than it is OOD.
The Gemini Flash Boston figure quantifies the single-draw outlier described in
the thesis (§ "What the LLMs Actually Know") and turns it from an anecdote into
an argument for robust prior estimation.

---

## 2. Threshold recalibration: the F1 gain is an operating-point artefact

Every model, baseline included, gets a threshold chosen by 5-fold CV on the
source training pool. OOD data never enters the choice.

### Heart Disease — prevalence 0.459 (ID train) → 0.530 (OOD)

| family | F1@0.5 | ΔF1@0.5 95% CI | F1@CV-thr | ΔF1@CV-thr 95% CI |
|---|---|---|---|---|
| Baseline | 0.7696 | — | **0.8121** | — |
| Prior-Mean | 0.8173 | [+0.0165, +0.0801] * | 0.8161 | [−0.0243, +0.0312] |
| Independent-Prior | 0.8020 | [+0.0034, +0.0611] * | 0.8132 | [−0.0294, +0.0306] |
| Correlated-Prior | 0.8067 | [+0.0098, +0.0669] * | 0.8178 | [−0.0228, +0.0335] |
| Mixed | 0.8173 | [+0.0165, +0.0801] * | 0.8161 | [−0.0243, +0.0312] |

### Diabetes — prevalence 0.212 → 0.495

| family | F1@0.5 | ΔF1@0.5 95% CI | F1@CV-thr | ΔF1@CV-thr 95% CI |
|---|---|---|---|---|
| Baseline | 0.5580 | — | **0.7236** | — |
| Prior-Mean | 0.6472 | [+0.0394, +0.1427] * | 0.7243 | [−0.0256, +0.0291] |
| Independent-Prior | 0.6471 | [+0.0399, +0.1425] * | 0.7021 | [−0.0534, +0.0071] |
| Correlated-Prior | 0.6429 | [+0.0346, +0.1375] * | 0.7243 | [−0.0257, +0.0291] |
| Mixed | 0.6429 | [+0.0346, +0.1375] * | 0.7136 | [−0.0372, +0.0195] |

### Adult — prevalence 0.218 → 0.288

Baseline F1 0.6685 → 0.6786 with recalibration; all prior families within
±0.002 of baseline at either threshold. No change to the thesis's null.

**8 of 8 significant F1 gains lose significance.** The mechanism is visible in
the selected thresholds: CV picks **0.33 (Heart), 0.25 (Diabetes), 0.33
(Adult)** for *every* model including the baseline. Threshold 0.5 was never
appropriate for these OOD sets. Recalibration alone gains the baseline +0.166
F1 on Diabetes, nearly double the +0.089 the prior was credited with; and
Independent-Prior ends up *behind* the recalibrated baseline (0.7021 vs 0.7236).

### The claim that replaces it: log loss, 8 of 8 significant

| dataset | family | LogLoss | ΔLogLoss 95% CI | Brier |
|---|---|---|---|---|
| Heart Disease | Baseline | 0.4709 | — | 0.1522 |
| | Prior-Mean | 0.4327 | [−0.0726, −0.0032] * | 0.1372 |
| | Independent-Prior | 0.4407 | [−0.0548, −0.0036] * | 0.1411 |
| | Correlated-Prior | 0.4331 | [−0.0635, −0.0102] * | 0.1381 |
| | Mixed | 0.4327 | [−0.0726, −0.0032] * | 0.1372 |
| Diabetes | Baseline | 0.6327 | — | 0.2201 |
| | Prior-Mean | 0.6025 | [−0.0462, −0.0132] * | 0.2061 |
| | Independent-Prior | 0.6047 | [−0.0430, −0.0118] * | 0.2071 |
| | Correlated-Prior | 0.6034 | [−0.0451, −0.0129] * | 0.2067 |
| | Mixed | 0.6029 | [−0.0452, −0.0134] * | 0.2063 |

AUC CIs all cross zero on both datasets, as the thesis reports.

**For the paper.** Classification claim becomes: *the priors improve probability
quality, not ranking and not fixed-threshold decisions.* Narrower than the
thesis, fully supported, threshold-free, and it needs no defence. Drop the F1
and sensitivity tables, or keep them only alongside the recalibrated baseline.

Caveat for writing: Adult's log-loss CIs are flagged significant at
[+0.0000, +0.0000]. Statistically correct, substantively noise — report effect
sizes there rather than significance stars.

---

## 3. Placebo control: the prior's content, not its scale

Two structure-preserving corruptions, λ re-tuned by CV on every draw so each
placebo gets the same tuning advantage as the real prior:

- **permuted** — the whole prior (mean, variances, covariance) re-indexed onto
  a random permutation of the features: same values, same correlation
  structure, wrong features
- **signflip** — coefficient signs flipped at random; precision transformed as
  `D P D` with `D = diag(s)`, so magnitudes and correlation magnitudes survive

200 draws for regression, 100 for classification. p is the empirical
`(#{placebo ≥ real} + 1) / (n + 1)`; 0.005 is the floor at 200 draws.

| dataset | family | metric | real | placebo mean | perm p | signflip p |
|---|---|---|---|---|---|---|
| Boston Housing | Prior-Mean | MSE | +8.34% | −0.45% / −0.94% | 0.005 | 0.005 |
| Boston Housing | Independent | MSE | +15.78% | −20.18% / −15.04% | 0.020 | 0.010 |
| Boston Housing | Correlated | MSE | +7.55% | −0.06% / −1.93% | 0.005 | 0.005 |
| Cancer | Prior-Mean | MSE | +3.90% | −2.89% / −2.23% | 0.005 | 0.015 |
| Cancer | Independent | MSE | +2.78% | −2.57% / −0.77% | 0.005 | 0.010 |
| Cancer | Correlated | MSE | +2.12% | −2.59% / −0.18% | 0.010 | **0.110** |
| Heart Disease | Prior-Mean | AUC | +0.93% | −2.99% / −2.82% | 0.040 | 0.010 |
| Heart Disease | Prior-Mean | LogLoss | +8.10% | −11.42% / −9.74% | 0.030 | 0.010 |
| Diabetes | Prior-Mean | AUC | +0.85% | −0.55% / −0.37% | 0.010 | **0.079** |
| Diabetes | Prior-Mean | LogLoss | +4.77% | +2.48% / +2.37% | 0.020 | 0.049 |

**14 of 16 cells at p < 0.05.** Two exceptions, both sign-flip: Cancer /
Correlated-Prior (p = 0.11) and Diabetes / Prior-Mean on AUC (p = 0.079).
Report them — they are an argument that Prior-Mean is the sound default and
that the covariance families buy less than their complexity suggests.

Artefact to avoid in the write-up: Boston / Independent's permuted placebo
range runs to −153%, so its mean (−20.2%) is dragged by a few catastrophic
draws. Use the median or the quantile range for that row, not the mean.

**For the paper.** This is the control that converts "the prior helps" into
"the prior's domain content helps". It also partly answers the contamination
objection: whatever the LLM is drawing on, it is feature-specific and
direction-specific, not a generic non-zero shrinkage target.

---

## 4. Resplit stability: three corrections to the headline table

Full deployment repeated over ID resplits with λ re-tuned from scratch. The OOD
partition is fixed by the domain split, so this varies exactly the component
the evaluation-set bootstrap cannot see.

| dataset | family | thesis (1 split) | resplit mean | sd | 2.5–97.5 pct | % below baseline |
|---|---|---|---|---|---|---|
| Boston Housing | Prior-Mean | +8.3% | +6.51% | 2.88 | [+1.29, +10.81] | 2% |
| Boston Housing | Independent | **+15.8%** | +11.47% | 3.68 | [+4.91, +19.73] | 0% |
| Boston Housing | Correlated | +7.6% | +5.26% | 2.25 | [+1.27, +9.94] | 2% |
| Boston Housing | Mixed | +8.3% | +6.51% | 2.88 | [+1.29, +10.81] | 2% |
| Cancer | Prior-Mean | **+3.9%** | +2.37% | 2.05 | [−0.30, +7.25] | 12% |
| Cancer | Independent | +2.8% | +1.13% | 1.08 | [−1.11, +3.33] | 10% |
| Cancer | Correlated | +2.1% | +0.91% | 1.09 | [−0.89, +3.17] | 16% |
| Cancer | Mixed | +2.7% | +1.65% | 1.43 | [−0.20, +5.10] | 12% |
| Heart Disease | Prior-Mean | **+0.9%** | **+0.08%** | 0.68 | [−0.76, +1.50] | **57%** |
| Heart Disease | Independent | +0.8% | −0.07% | 0.64 | [−0.79, +1.20] | 63% |
| Heart Disease | Correlated | +0.9% | +0.18% | 0.74 | [−0.61, +1.81] | 50% |
| Diabetes | Prior-Mean | +0.8% | +1.40% | 0.66 | [+0.26, +2.62] | **0%** |
| Diabetes | Independent | +1.1% | +1.78% | 0.77 | [+0.59, +3.15] | 0% |
| Diabetes | Correlated | +1.0% | +1.81% | 0.78 | [+0.59, +3.20] | 0% |
| Adult | all four | ≈0 | −0.01% | ≤0.02 | [−0.05, +0.00] | — |

n = 50 (Boston, Cancer), 30 (Heart, Diabetes), 10 (Adult). Regression is MSE
lift, classification AUC lift.

Three changes:

1. **Heart Disease's AUC lift is noise.** 57% of resplits fall below baseline,
   median −0.10%. The +0.9% in the thesis summary table came from one lucky
   split and should be removed. Heart's real contribution is the calibration
   gain (§2), which is significant and placebo-controlled.
2. **Diabetes is stronger than the thesis claims.** All four families positive
   in 100% of resplits with the 2.5th percentile above zero. Not a
   contradiction of the single-split bootstrap: that measures evaluation-set
   noise (n_OOD = 372), this measures training noise, and the lift is small but
   *systematic*. Reporting both uncertainty sources is a genuine methodological
   point.
3. **Boston and Cancer are mildly optimistic.** Boston survives comfortably
   (never below baseline, 2.5th pct +4.9% for Independent). Cancer is the weak
   one: 10–16% of resplits below baseline and CIs straddling zero for every
   family except Prior-Mean.

Adult confirms the safe-failure mode: −0.01% consistently. A null, not harm.

---

## Consolidated implication for the paper

**Claims that survive, with controls:**

- OOD MSE reduction on Boston Housing — placebo-controlled, resplit-stable
- OOD MSE reduction on Cancer — placebo-controlled, resplit-stable for
  Prior-Mean only
- Calibration (log loss, Brier) gains on Heart Disease and Diabetes — 8/8
  significant, placebo-controlled
- OOD AUC gain on Diabetes — small but systematic across resplits
- The selection-step finding — from the existing 48-cell sweep, no new runs
- Safe-failure on Adult — resplit-confirmed
- Prior-only competitiveness, and Cancer's +12.8% with no data at all

**Claims to drop or reframe:**

- F1 and sensitivity gains at threshold 0.5 — artefact of label shift
- Heart Disease's AUC lift — within resplit noise
- Cancer's covariance families as reliable winners — weaker than one split
  suggested

**Still unresolved, and not fixable without API budget:** dataset
contamination. Four of five datasets are near-certainly in LLM training data.
The placebo test rules out the weakest alternative explanation but cannot
separate domain knowledge from dataset recall. Recommended mitigation is
framing — "LLM priors as a practical regularisation target with a selection
procedure that makes them safe" rather than "LLMs know epidemiology".

**Separate decision outstanding:** Boston Housing supplies the largest lift and
is the dataset deprecated over the `B` feature (removed from scikit-learn in
1.2). Keeping it as the headline is the main remaining publication risk.


---

## Addendum (re-run with the source pipeline's optimiser convention)

The first pass of exp2/exp4 fitted every logistic family from a seeded random
start. The source pipeline instead starts **Correlated-Prior from zero** and the
other families from a seeded `0.01*N(0,1)` draw (`llmpriors/README.md`).
`common.start_seed_for()` now matches that, and exp2 was re-run; the CSVs and
logs here are from the corrected run.

What changed, all in the Correlated-Prior rows only:

| cell | before | after |
|---|---|---|
| Heart Disease, F1 @ 0.5 | 0.8048 | **0.8067** |
| Diabetes, F1 @ 0.5 | 0.6384 | **0.6429** |
| Diabetes, LogLoss | 0.6035 | **0.6034** |
| Adult, LogLoss | 0.3941 | **0.3942** |

No conclusion changes: every significance verdict in every table is identical.

**Worth knowing, and reported in the paper's Methods.** Neither convention
reproduces both of the thesis's published Correlated-Prior values: the
corrected one matches Heart Disease exactly (0.8067) but gives 0.6429 on
Diabetes where the thesis has 0.6384. The cause is that `Sigma_hat` from M=50
draws is ill-conditioned, so its inverse yields a penalty under which L-BFGS
terminates at slightly different points depending on the starting value and on
whether the fit is warm-started from the lambda grid walk (the deployment refit
in the original pipeline inherits the tuning path; a standalone refit does not).
Differences are third-decimal. This is a further argument for shrinking
`Sigma_hat` before inversion, e.g. Ledoit-Wolf.

**exp4 has since been re-run under the corrected convention** and the CSVs here
are from that run; `exp4_resplits_summary_oldconv.csv` keeps the previous output
for comparison. Exactly 3 of 20 cells changed, all of them classification
Correlated-Prior rows, and all by less than 0.01 percentage points:

| cell | mean lift before | after |
|---|---|---|
| Heart Disease, Correlated-Prior | +0.1841% | +0.1777% |
| Diabetes, Correlated-Prior | +1.8070% | +1.8069% |
| Adult, Correlated-Prior | −0.0138% | −0.0139% |

Boston Housing and Cancer are bit-identical, as expected: regression uses the
closed-form solve and never invokes L-BFGS, so the start convention cannot
affect it. At the two decimals reported in the paper only two numbers move:
the Heart Disease Correlated-Prior 2.5th percentile (−0.60 to −0.61) and the
Diabetes Correlated-Prior 97.5th percentile (+3.19 to +3.20). No verdict
changes anywhere.


---

## Addendum 2 — Boston Housing refitted without the `B` feature (15 Sep)

`B = 1000(Bk - 0.63)^2`, with Bk the proportion of Black residents, builds a
racial-segregation hypothesis into the feature's functional form and cannot be
inverted to recover Bk. It is why scikit-learn removed the dataset. We now drop
the column from the design matrix and the matching coordinate from every
elicited prior (`DROP_FEATURES` in `scripts/common.py`); the principal submatrix
of `Sigma_hat` is the marginal prior over the 11 features actually fitted. No
other dataset is affected. No new LLM calls were needed.

Because dropping a feature changes which (variant, family) cell wins,
`scripts/boston_reselect.py` re-runs the full 12x4 selection sweep for Boston.

**Deployment (single split).** Baseline ridge OOD MSE improves from 46.45 to
**44.10** without B, so the bar is higher as well.

| family | variant | lambda | OOD MSE | lift | paired Δ 95% CI |
|---|---|---|---|---|---|
| Prior-Mean | Gemini Pro | 251 | 42.92 | +2.7% | [−4.14, +1.45] — **not significant** |
| Independent-Prior | Opus 4.5 | 10.0 | 39.56 | +10.3% | [−8.81, −0.93] * |
| Correlated-Prior | Opus 4.5 | 15.8 | **39.18** | **+11.2%** | [−9.19, −1.35] * |
| Mixed | Gemini Pro | γ=1.00 | 42.92 | +2.7% | [−4.14, +1.45] — not significant |

**Resplits (n = 50).**

| family | single split | mean | sd | [2.5, 97.5] | below baseline |
|---|---|---|---|---|---|
| Prior-Mean | +2.7% | **−0.02%** | 4.02 | [−7.22, +6.55] | **42%** |
| Independent-Prior | +10.3% | +7.52% | 3.20 | [+2.66, +14.43] | 0% |
| Correlated-Prior | +11.2% | +7.94% | 3.48 | [+2.09, +15.27] | 0% |
| Mixed | +2.7% | −0.02% | 4.02 | [−7.22, +6.55] | 42% |

**Placebo — still passes, 6/6 cells.** Prior-Mean p = 0.005/0.005,
Independent-Prior 0.005/0.015, Correlated-Prior 0.010/0.010 (permuted/sign-flip).
At most 2 of 200 draws reach the real lift in any cell.

**Selection sweep.** Floor/ceiling across the 48 cells moves from
−1.6%/+17.2% to **−2.3%/+11.2%**. The deployed model coincides with the ceiling,
which is luck rather than design.

### What changed in the conclusions

1. **The headline family changes.** Correlated-Prior now leads (+11.2%), not
   Independent-Prior. The largest resplit average in the whole paper falls from
   +11.5% to **+7.9%**, so the abstract's "up to" figure changed.
2. **Prior-Mean stops working on Boston.** Not significant on the single split
   and ~0 across resplits, versus +6.5% with B. A meaningful share of the
   original Boston effect came from that one covariate — worth knowing, and an
   argument for having removed it.
3. **Nothing else moves.** Cancer, Heart Disease, Diabetes and Adult are
   untouched, and the classification results, placebo conclusions and threshold
   finding all stand.

### Validation of the re-implementation

The regenerated selection figure recomputes floor/ceiling for the three
unchanged datasets from scratch and reproduces the original values exactly:
Cancer −2.92/+6.11 (published −2.9/+6.1), Heart Disease −1.33/+1.67 (−1.3/+1.7),
Diabetes −0.47/+2.08 (−0.5/+2.1).

### Caveats

- The prompts described the **twelve**-feature problem, so the elicited
  coefficients are for a specification we no longer fit — a mild
  omitted-variable mismatch, documented in the paper's Discussion.
- Adult's floor/ceiling in the selection figure is carried over from the
  original run rather than recomputed (12x4x51 logistic fits on n ≈ 27k is
  hours of compute, and Adult is unaffected by the change).


---

## Addendum 3 - joint uncertainty accounting (exp5, 16 Sep)

### Why this was run

The author asked whether we should report only the resplit results, on the
grounds that they account for more sources of uncertainty (source split, lambda,
target) than the bootstrap tables (target only). **The premise was wrong in one
respect:** the resplits hold the OOD partition *fixed*, so they contain no
target-sample uncertainty at all. The two analyses cover **disjoint** sources:

| | source split | lambda | target sample |
|---|---|---|---|
| paired bootstrap (exp2, deployment tables) | - | - | yes |
| resplits (exp4) | yes | yes | - |
| **joint (exp5)** | **yes** | **yes** | **yes** |

Reporting resplit-only would therefore have *dropped* target uncertainty, which
is anti-conservative on OOD samples of 151-417 rows. `exp5_joint.py` nests the
two: for each resplit the model is refitted and lambda re-tuned, then the lift is
evaluated on 200 bootstrap resamples of the OOD set. Pooling gives S x 200
draws (10,000 / 6,000 / 2,000 depending on dataset).

### The headline result

**Under the joint accounting, 0 of 24 prior-informed cells have a 95% interval
excluding zero in the beneficial direction.** Nine cells do under resplit-only.
The only two joint intervals excluding zero are Adult's Independent-Prior, in
the direction of harm, at -0.01% AUC and -0.05% log loss - i.e. negligible.

| cell | resplit-only | joint | % of draws below 0 |
|---|---|---|---|
| Boston / Correlated | [+2.09, +15.27] | [−0.61, +18.87] | 3% |
| Boston / Independent | [+2.66, +14.43] | [−1.04, +18.23] | 4% |
| Boston / Prior-Mean | [−7.22, +6.55] | [−12.97, +10.54] | 48% |
| Cancer / Prior-Mean | [−0.30, +7.25] | [−0.38, +7.53] | 12% |
| Heart / Prior-Mean (LogLoss) | [+0.22, +36.53] | [−1.68, +39.30] | 8% |
| Heart / Correlated (LogLoss) | [+0.41, +36.69] | [−1.45, +39.43] | 7% |
| Diabetes / Correlated (AUC) | [+0.59, +3.20] | [−0.24, +4.14] | 4% |
| Diabetes / Independent (AUC) | [+0.59, +3.15] | [−0.24, +4.10] | 4% |

Full table in `results/exp5_joint.csv`; all 250k draws in
`exp5_joint_draws.csv` (8.9 MB).

### How to read it

Not "no effect". The directional evidence is real: Boston's structured families
are positive in 96-97% of joint draws and Diabetes AUC in 95-96%. These are
one-sided tail probabilities just short of the conventional threshold. The
honest statement is *consistent direction on four of five datasets, magnitude
unresolvable with OOD partitions of this size*. That is a statement about
statistical resolution, not about effect size.

One caveat that cuts the other way, stated in the paper rather than used to
rescue the claims: bootstrapping the OOD partition treats it as a sample from
P_T, which is right for the risk R_T and for "would this generalise to another
population like this one". But each OOD partition is a *complete*
subpopulation, not a random draw. For "did it help on this particular
deployment population" the target is fully observed, there is no target-sampling
uncertainty, and resplit-only is the relevant column. The paper reports joint as
the headline and keeps the resplit-only column visible.

### What survives untouched

- **The placebo control.** It is a permutation test on a fixed split, so the
  joint accounting does not bear on it. 14/16 cells at p < 0.05 stands, and it
  becomes the paper's firmest positive result.
- **The selection finding.** It is about spread across 48 candidates, not about
  significance of a mean.
- **The threshold/F1 correction.** A methodological result, unaffected.

### What changed in the paper

Abstract, the regression and classification subsections (now flagged as
target-only intervals), a new subsection "Accounting for all sources of
uncertainty" with Table 5, the Discussion opening, the Conclusion, and a new
limitation about target-sample size. Elicitation variability remains a fourth
source none of the intervals covers.


---

## Addendum 4 - pooling across datasets (exp6, 16 Sep)

The per-dataset joint intervals all contain zero while the point estimates are
consistently positive, so `exp6_pooled.py` aggregates them: one exp5 joint draw
per dataset, combined, 200,000 times. Two rules, because they disagree.

**Equal weights** (what the paper reports):

| task | family | pooled lift | 95% interval | below 0 |
|---|---|---|---|---|
| regression (2) | Correlated | +4.37% | **[+0.02, +9.98]** | 2% |
| regression (2) | Independent | +4.26% | [−0.11, +9.75] | 3% |
| regression (2) | Prior-Mean | +1.10% | [−5.53, +6.85] | 33% |
| classif. AUC (Heart+Diab) | Correlated | +1.00% | [−0.25, +2.53] | 6% |
| classif. LogLoss (Heart+Diab) | Correlated | +7.84% | [−0.31, +22.23] | 4% |

So **exactly one pooled interval excludes zero**, by +0.02 points on an estimate
of +4.37 - it touches zero rather than clearing it. Monte Carlo error is not the
issue: over 12 seeds the lower bound stays in [+0.002, +0.050], always > 0.

**Why we do not lean on it.** Inverse-variance weighting gives +1.24%
[−0.75, +3.61] for the same cell, a threefold difference in magnitude, because
Cancer is ~20x more precise than Boston and takes 95% of the weight. A
fixed-effect pool assumes a common effect, which our own analysis argues against
(the benefit depends on n and shift magnitude). Cochran's Q does not reject
homogeneity (I² 0-49%, all p > 0.16) but with k = 2-3 that test has no power, so
this is absence of evidence, not evidence of absence. Random effects is not
estimable at k <= 3.

**The Adult pathology, worth knowing.** Including Adult in an inverse-variance
classification pool gives it weight 1.000 (n_OOD = 14,936, effect ~ 0, variance
three orders of magnitude below the others). The pooled estimate becomes
-0.01% AUC with an interval excluding zero - formally significant *harm*,
produced entirely by the designed negative control. Adult is therefore excluded
from the pooled classification block, and the pathology is reported rather than
quietly avoided.

**Bottom line.** Pooling supports the direction, not the magnitude: every pooled
estimate is positive with 2-11% of draws below zero. Resolving it needs larger
target samples or more splits per dataset, not different weights.

### Paper restructure that followed

Target-only intervals are gone from the Results. Tables 2 and 3 now report the
three-source joint intervals plus a pooled block; the old "Accounting for all
sources of uncertainty" subsection is replaced by "What pooling across datasets
can and cannot show"; the resplit-only table moved to the appendix as
Supplementary Table for readers interested in the fixed-population question.


---

## Addendum 5 - what source-domain lambda selection costs (exp7, 17 Sep)

### The question

The author proposed that elicited priors should in principle help on Adult too,
since they correct a *bias* that is a property of the domain and independent of
n, but that with n_train ~ 27,000 the likelihood swamps the prior in the
posterior. The statistics is right. Two mechanisms were separated and tested.

1. **Bayesian asymptotics.** A prior of fixed precision adds O(1) to the
   log-posterior while the likelihood adds O(n), so its influence decays as
   1/n. Under shift the MLE converges to beta*_S, whose target bias does *not*
   shrink with n - so large n buys a precise estimate of the wrong quantity
   while silencing the only correction available. Correct as stated, but in our
   pipeline lambda is *tuned*, not fixed, so this does not bind automatically.
2. **The selection criterion.** lambda minimises source-domain CV error, and a
   prior that trades in-domain fit for out-of-domain transfer is strictly bad by
   that measure. This is the mechanism that actually binds here.

`exp7_lambda_oracle.py` sweeps the whole lambda grid per dataset and family,
recording OOD error at each point, and compares the CV-selected lambda with the
OOD-optimal one (an oracle, not achievable).

### Result: the mechanism is real, but Adult is the wrong exhibit

| dataset | family | cv lambda -> lift | oracle lambda -> lift | gap |
|---|---|---|---|---|
| **Cancer** | Prior-Mean | 158 -> +3.90% | 10,000 -> **+21.64%** | **+17.7** |
| **Cancer** | Independent | 158 -> +2.78% | 10,000 -> +13.07% | +10.3 |
| **Cancer** | Correlated | 158 -> +2.12% | 3,981 -> +9.38% | +7.3 |
| Boston | all three | - | - | 0.00-0.18 |
| Heart | all three | - | - | 0.01-0.06 |
| Diabetes | all three | - | - | 0.03-0.26 |
| **Adult** | all three, AUC | floor -> ~0% | **floor** -> ~0% | **0.00-0.02** |
| Adult | Prior-Mean, log loss | 1 -> -0.00% | 100 -> +0.38% | +0.38 |

**On Adult the OOD-optimal lambda is at the grid floor for every family on AUC**
- no prior weight would have helped. The null is genuine, not masked. Only log
loss offers a small interior gain (+0.38%).

**Cancer is the exhibit instead**, and note n_train there is only 1,106. So the
binding problem is *not* sample size - it is that CV optimises source risk.
That refines the author's diagnosis: right about the mechanism's existence,
wrong about which dataset displays it and about n being the cause.

### Literature added to the Discussion

- **shimodaira2000covariate** - importance-weighted log-likelihood; optimal
  weight is the density ratio P_T(x)/P_S(x) under covariate shift with a
  misspecified model.
- **sugiyama2007iwcv** - importance-weighted cross-validation. Directly
  substitutes for our selection rule and needs only *unlabelled* target
  features.
- **grunwald2017inconsistency** - Bayesian inference is inconsistent for
  misspecified *linear* models; the posterior concentrates on progressively
  worse fits as n grows. Repaired by a learning rate on the likelihood.
- **deheide2020safe** - the same construction for GLMs.
- **ibrahim2000power** - power priors; the mirror device, discounting external
  information relative to the current sample.

### Paper changes

Prior-only subsection now quantifies the oracle gap; the Adult paragraph is
rewritten around the two mechanisms and reports that its null is genuine; a new
future-work paragraph in Limitations sets out the design space with the five
references above.

---

## Addendum 6 — placebo control extended to all prior families (2026-09-21)

The placebo experiment originally ran all three families on the two regression
datasets but only Prior-Mean on Heart Disease and Diabetes. The restriction was
a cost guess in `exp3_placebo.py` ("regression is cheap (closed form), logistic
is not"), not a methodological choice. A timing check put one logistic placebo
draw at 0.75–1.7 s, so the full set costs ~15 min. `CELLS` now gives all three
families to all four datasets: **24 cells instead of 16**.

The per-dataset RNG is created fresh and consumed in family order with
`prior_mean` first, so **every previously reported cell reproduces bit-identically**
and this is purely additive. Verified against the pre-run CSV.

### Result: the two placebo types now separate clearly

| | rejected at p<0.05 |
|---|---|
| permuted (wrong features, values kept) | **11 / 12** |
| sign-flip (magnitudes kept, directions scrambled) | **7 / 12** |
| all cells | 18 / 24 |

Per dataset: Boston Housing 6/6, Cancer 5/6, Heart Disease 3/6, Diabetes 4/6.

The six exceptions, five of which are sign-flip:

| dataset | family | placebo | p |
|---|---|---|---|
| Cancer | Correlated-Prior | signflip | 0.109 |
| Heart Disease | Independent-Prior | permuted | 0.059 |
| Heart Disease | Independent-Prior | signflip | 0.228 |
| Heart Disease | Correlated-Prior | signflip | 0.089 |
| Diabetes | Prior-Mean | signflip | 0.079 |
| Diabetes | Correlated-Prior | signflip | 0.109 |

On OOD log loss (classification only) 9 of 12 reject; the exceptions are
Heart Disease Independent permuted (0.069) and sign-flip (0.535), and
Diabetes Independent permuted (0.059).

### What changed in the paper

The old claim was "14 of 16 cells at p<0.05" with two sign-flip exceptions
treated as incidental. With full coverage the sign-flip weakness is systematic,
so the claim was narrowed:

- **Survives:** the benefit is tied to *feature identity* — permuting which
  feature each elicited value belongs to destroys it in 11 of 12 cells.
- **Does not survive as a general claim:** direction-specificity. In 5 of 12
  sign-flip cells a prior with randomly reversed signs does about as well as
  the real one.
- "direction-specific" was therefore removed from the Discussion, and the
  Results now state explicitly what the control does and does not reach.

Figure `fig_placebo.pdf` regenerated with 24 rows; figsize reduced to
(6.6, 0.185·n·2+1.05) and y-limit headroom raised, because at the previous
aspect ratio the float was too tall to place and LaTeX deferred it ten pages
past its own text.

---

## Addendum 7 — placebo control switched from AUC to log loss (2026-09-21)

Prompted by the question "is there a specific reason AUC is used?". There was
not: the study tunes lambda on AUC (`tuning_score(..., "auc")`), and
`exp3_placebo.py` simply reported the metric it tuned on. That conflicts with
the paper's own classification finding — AUC is flat, log loss improves — so
the headline placebo control was validating the priors' content on the metric
the priors do not move.

Both metrics were already computed per cell; only the **per-draw** log-loss
values were missing, so the run was repeated with `logloss_lift_pct` recorded.
The summary reproduces the previous run bit-identically (verified cell by cell);
only the draws file gained columns.

### The metric changes the conclusion

Classification cells (12), same draws, two scorings:

| scoring | rejected p<0.05 | permuted | sign-flip | mean real lift |
|---|---|---|---|---|
| AUC | 7/12 | 5/6 | **2/6** | +0.91% |
| **log loss** | **9/12** | 4/6 | **5/6** | +6.08% |

Headline (MSE for regression, log loss for classification): **20 of 24**, with
**permuted 10/12 and sign-flip 10/12** — the asymmetry is gone.

The four exceptions group by *family*, not placebo type: Heart Disease
Independent-Prior under both corruptions (p=0.069, p=0.535), Diabetes
Independent-Prior permuted (p=0.059), Cancer Correlated-Prior sign-flip
(p=0.109). Per dataset: Boston 6/6, Cancer 5/6, Diabetes 5/6, Heart Disease 4/6.

### Consequence for the paper

Addendum 6 recorded a claim — "five of six exceptions are sign-flip, so the
benefit is about which coefficients are large more than which way they point" —
that was **an artefact of scoring on AUC**. Under log loss sign-flip is
rejected in 5 of 6 classification cells, and overall the two corruptions are
equally damaging. The original "feature-specific *and* direction-specific"
claim is therefore restored in the Results and Discussion, now at 24-cell
coverage instead of 16.

The Results also now state the AUC/log-loss contrast explicitly, as a caution:
a placebo or ablation scored on a threshold-free ranking metric can understate
an effect that is real in the predicted probabilities.

`fig_placebo.pdf` now plots MSE lift for the regression rows and log-loss lift
for the classification rows, on a **symlog** x-axis (linthresh 10) because
corrupted priors reach -639% on log loss and a linear axis collapsed every real
effect onto the zero line.

---

## Addendum 8 — referee-response experiments (2026-10-09)

Two simulated referee reports (a statistician and an LLM/ML reviewer) raised
findings that needed checking against the code and data rather than argued
about. Environment rebuilt from `llmpriors_for_paper.zip`; `boston_reselect.py`
reproduces the stored CSVs to ~1e-11, so the numbers below are comparable to
the published ones.

### exp8 — a real cross-family selection rule (`exp8_selection.py`)

The figure's "deployed pipeline" marker was `dep.lift_pct.max()` over families,
i.e. **selection on the OOD set**, with Cancer/Heart/Diabetes hard-coded at
their per-family maxima. The Methods define selection only *within* family, so
no cross-family rule existed. Defined one (best source CV over all 48 cells)
and applied it:

| dataset | floor | ceiling | cross-family (source CV) | OOD-best (plotted) | gap |
|---|---|---|---|---|---|
| Boston Housing | -2.27% | +11.15% | **+2.67%** | +11.15% | 8.48 |
| Cancer | -2.92% | +6.11% | **+2.12%** | +6.11% | 3.99 |
| Heart Disease | -1.33% | +1.67% | **+0.95%** | +1.67% | 0.72 |
| Diabetes | -0.47% | +2.08% | **+0.97%** | +2.08% | 1.11 |
| Adult | -0.82% | +0.01% | **-0.03%** | +0.01% | 0.04 |

Within-family deployed values reproduce Tables 2-3 exactly, so the per-family
tables are unaffected; only the single diamond and its narrative were wrong.

**Worse finding.** The CV margin between the winning cell and the runner-up is
far below the fold-level SE of the CV score itself:

| dataset | CV margin | fold SE | margin/SE |
|---|---|---|---|
| Boston | 0.00000 | 2.894 | 0.00 |
| Cancer | 0.036 | 24.765 | 0.00 |
| Heart | 0.0022 | 0.0187 | 0.12 |
| Diabetes | 0.00058 | 0.0328 | 0.02 |
| Adult | 0.00000 | 0.0016 | 0.00 |

Cross-family selection is effectively arbitrary. This bears on the claim that
selection is "the deployable contribution".

### exp9 — variance floor sensitivity (`exp9_epsilon.py`)

eps = 1e-8 does pin coefficients: Opus 4.5 (non-reasoning) returns
bit-identical draws for 4/11 Boston, 2/7 Diabetes, 2/18 Heart, 4/20 Adult
features, so the Independent-Prior weight is alpha/1e-8. But the results
survive:

| Boston | 1e-8 | 1e-4 | 1e-2 | 1e-1 | Ledoit-Wolf |
|---|---|---|---|---|---|
| Independent-Prior | +10.29% | +10.30% | +9.27% | +7.53% | - |
| Correlated-Prior | +11.15% | +11.15% | +9.70% | +7.69% | **+11.10%** |

Ledoit-Wolf shrinkage reproduces the headline. Caveat: in the LW arm the
per-fold priors were standardised with the full source-training scale rather
than each fold's own, which affects only that arm's lambda tuning (it selected
the same lambda = 15.85 as the jittered arm).

Conclusion: the floor is not generating the effect, but the Methods'
description of it is backwards, and the "factor of ten" spread sentence is
wrong (four values are exactly zero).

### exp10 — Heart Disease `fbs` ablation (`exp10_heart_fbs.py`)

The loader retained `fbs` after "an audit [that] found that dropping it
slightly hurt prior performance" — a feature decision made by looking at the
outcome. Re-running the full selection with and without it:

| | with fbs | without fbs |
|---|---|---|
| Prior-Mean log loss | +8.10% | +8.51% |
| Correlated-Prior log loss | +8.03% | +8.00% |
| cross-family log loss | +8.03% | +8.00% |

Robust. The audit's own conclusion does not reproduce — dropping `fbs` slightly
*helps* Prior-Mean. Restoring `ca`/`thal`/`slope` is impossible from stored
data: the elicited Heart priors cover 18 one-hot features and contain no
coordinates for them.

### Imputation "leaks" are vacuous

Boston has no missing values in any of the 12 features and Adult none in any
numeric column, so the median-fill branches never execute. The code would leak
if the data changed; no reported number is affected.

### In-domain vs out-of-domain lift (from exp8)

| dataset (Correlated-Prior) | ID lift | OOD lift |
|---|---|---|
| Boston Housing | +13.20% | +11.15% |
| Diabetes | +3.75% | +0.97% |
| Heart Disease | +1.72% | +0.95% |
| Cancer (Prior-Mean) | +0.09% | +3.90% |

On three of five datasets the in-domain lift exceeds the OOD lift. Only Cancer
is genuinely OOD-specific — and it is also the dataset least likely to be
contaminated.

### exp11/exp12 — prediction interval vs confidence interval for the mean

exp11 re-runs exp5 recording the split index (reproduces the published joint
intervals exactly). exp12 decomposes:

    sd_split  = sd of the S per-split means      (source split + lambda)
    sd_target = mean within-split bootstrap sd   (target sampling)

The published interval is the 2.5-97.5 percentile of the pooled S x B draws —
a **prediction interval for one future run**. A confidence interval for the
**mean** lift divides the source component by S but not the target component
(all splits share one realised OOD partition):

    SE = sqrt(sd_split^2 / S + sd_target^2)

| interval | cells excluding zero (of 32) |
|---|---|
| prediction interval (as published) | **2** (both Adult, harm) |
| CI for the mean, generalising | **14** (12 beneficial + 2 Adult harm) |
| CI for the mean, fixed population | **22** (18 beneficial + 4 Adult) |

Newly significant under the generalising CI: Boston Independent +7.40
[+0.57, +14.24] and Correlated +7.85 [+1.18, +14.51]; all four Cancer families;
all four Heart Disease log-loss cells (Prior-Mean +12.23 [+4.20, +20.25]);
Diabetes Independent and Correlated AUC.

Target sampling is about half the variance on Boston (48-53%) and Diabetes AUC
(53-57%), but only 0.7-4% on Cancer and 6-11% on the log-loss cells — which is
why the correction matters most where the OOD partition is small.
