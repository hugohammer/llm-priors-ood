# Datasets

Five datasets, each split on a real covariate into an in-domain (source) half
and an out-of-distribution (target) half. Provenance and terms differ, so they
are listed separately.

| Directory | Dataset | Source | Terms |
|---|---|---|---|
| `heart+disease/` | UCI Heart Disease (Cleveland, Hungary, Switzerland, Long Beach VA) | UCI ML Repository, id 45 | CC BY 4.0 |
| `adult/` | UCI Adult / Census Income | UCI ML Repository, id 2 | CC BY 4.0 |
| `diabetes/` | Pima Indians Diabetes | Smith et al. 1988; widely redistributed | public domain in practice |
| `cancer/` | County-level cancer mortality ("OLS Regression Challenge") | data.world/nrippner | **see note below** |
| `boston/` | Boston Housing, **B column removed** | Harrison & Rubinfeld 1978 | **see note below** |

## Boston Housing: the `B` column is present but never modelled

The dataset contains `B = 1000(Bk - 0.63)^2`, where `Bk` is the proportion of
Black residents in the tract. The transformation builds a racial-segregation
hypothesis into the feature itself and, because of the squaring, cannot be
inverted to recover the underlying proportion. It is the reason the dataset was
removed from scikit-learn, which deprecated `load_boston` in 1.0 and removed it
in 1.2.

The paper drops `B` from the design matrix and the matching coordinate from
every elicited prior, and fits the remaining eleven features. That exclusion is
implemented in the pipeline, not in the stored file: `DROP_FEATURES` in
`experiments/common.py` removes the column at load time and takes the principal
submatrix of the elicited covariance, which is the marginal prior over the
features actually fitted. The file is shipped unmodified so that the code path
is identical to the one that produced the published numbers, and so that the
exclusion is visible and auditable rather than silently baked into the data.

Note that the elicitation prompts described the twelve-feature problem, so the
stored Boston priors do contain a `B` coordinate. This is a mild
omitted-variable mismatch, discussed in the paper.

## Cancer: provenance, and why it is mirrored here

This county-level compilation was published openly by Noah Rippner on
data.world as the "OLS Regression Challenge", a public teaching dataset. It
aggregates three US federal government sources: the Census Bureau's American
Community Survey, Cancer.gov and ClinicalTrials.gov.

**The original is gone.** data.world has retired its Open Data Community, and
`data.world/nrippner/ols-regression-challenge` no longer resolves — the page
now returns a retirement notice stating that those datasets are "no longer
accessible or available for download". A mirror on Kaggle lists the licence as
"Unknown", and we have not been able to locate any licence statement for the
compilation.

**Why we include it anyway.** This is the dataset behind the paper's single
clearest result — the one case where the elicited prior improves out-of-domain
prediction without improving in-domain prediction. With the original host
defunct, removing the file would make that result impossible for anyone to
check.

The position we take is this. The underlying numbers are US federal government
outputs and are not subject to copyright. What could in principle carry rights
is the compilation itself — the selection of counties and variables and the
joins between sources — and we make no claim to it: credit belongs to Noah
Rippner, who released it publicly as an open teaching resource. We redistribute
it in good faith to preserve reproducibility of published work, and will remove
it immediately on request from the compiler or any rights holder.

If you are reusing this file for something else, the safe course is to go back
to the three upstream sources rather than to rely on this copy.

## Heart Disease preprocessing

The paper uses Cleveland as source and Hungary + Switzerland as target. Long
Beach VA is excluded for severe missingness across most clinical fields.
`ca`, `thal` and `slope` are dropped because they are 98%, 76% and 50% missing
in the target cohorts though near-complete in Cleveland; Switzerland records
cholesterol as 0 for every row, which is treated as missing. These choices are
implemented in `llmpriors/datasets.py` and discussed in the paper's Methods.
