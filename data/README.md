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

## Cancer: check the terms before relying on redistribution

This compilation was published on data.world as the "OLS Regression Challenge"
and aggregates the US Census American Community Survey, clinicaltrials.gov and
cancer.gov. It is included here because it is the dataset behind the paper's
most important single result and reproduction would otherwise be impossible.

Two caveats. We have not been able to confirm an explicit redistribution
licence for the compilation, as opposed to its public-domain upstream sources.
And data.world has announced the retirement of its Open Data Community
platform, so the original URL may not resolve indefinitely — which is part of
why the file is mirrored here.

If the terms turn out to prohibit redistribution, delete `cancer/` and replace
it with a download script; nothing else in the repository depends on the file
being present until a script is run.

## Heart Disease preprocessing

The paper uses Cleveland as source and Hungary + Switzerland as target. Long
Beach VA is excluded for severe missingness across most clinical fields.
`ca`, `thal` and `slope` are dropped because they are 98%, 76% and 50% missing
in the target cohorts though near-complete in Cleveland; Switzerland records
cholesterol as 0 for every row, which is treated as missing. These choices are
implemented in `llmpriors/datasets.py` and discussed in the paper's Methods.
