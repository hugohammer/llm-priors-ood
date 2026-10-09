# llmpriors

The thesis pipeline, written as a single readable package: elicit coefficient
priors from an LLM, fold them into regularised regression as structured
shrinkage, and evaluate under domain shift.

Start with `experiment.py` — it reads top to bottom as the protocol in Methods
§3.5, and everything else is called from there.

## Method → code

Each row is a section of the thesis and the code that implements it.

| Thesis | Code |
|---|---|
| §2.4 MAP estimation under a Gaussian prior | `estimators.py` |
| §3.2 Domain splits, preprocessing | `datasets.py`, `preprocessing.py` |
| §3.3 Prompt construction, sampling | `prompts.py`, `elicitation.py` |
| §3.3.3 Prior mean, variances, covariance | `priors.py` |
| §3.4 The four model families | `estimators.py` + the `families` dict in `experiment.py` |
| §3.4.5 Mixed model | `experiment.py`, the gamma loop |
| §3.5.1 5-fold CV, λ and γ tuning | `crossval.py` |
| §3.5.3 Evaluation metrics | `metrics.py` |
| §3.5.4 Bootstrap CIs | `bootstrap.py` |
| Fixed settings (grids, seed, thresholds) | `config.py` |

Cross-variant provider selection (§3.5.2) is not here: it operates across
completed runs rather than within one, so it sits outside this package.

## The one idea that makes it small

Every model in the study minimises

    L(beta) + 0.5 * alpha * (beta - mu)' P (beta - mu)

with `L` either squared error or the negative log-likelihood. The four families
are just different choices of `(mu, P)`:

| Family | mu | P |
|---|---|---|
| Ridge / L2-Logistic | 0 | I |
| Prior-Mean | mu_hat | I |
| Independent-Prior | mu_hat | diag(1 / (var_j + eps)) |
| Correlated-Prior | mu_hat | Sigma_hat^-1 |

So `estimators.py` contains two classes — one per loss — rather than one class
per family. The shared `0.5 * alpha` factor is what makes tuned alpha values
comparable across families in the results tables.

## Layout

```
config.py         grids, seed, jitter, output labels
datasets.py       the five datasets, their domain splits, feature sets
preprocessing.py  median impute -> standardise, fitted on training data only
priors.py         LLM samples -> mu_hat, variances, precision
estimators.py     the two MAP estimators
crossval.py       fold construction, per-fold prior rescaling, selection
metrics.py        MSE; AUC / F1 / Sensitivity / Specificity / Brier / LogLoss
bootstrap.py      marginal and paired-vs-baseline percentile CIs
experiment.py     run_regression / run_classification
elicitation.py    JSON schemas, provider calls, sampling loop
prompts.py        prompt assembly
prompt_content.py per-dataset prompt text
verify.py         reproduction check against results/
```

## Running it

Reproducing results needs no API calls — the elicited samples are committed
under `priors/`.

```python
from llmpriors import datasets, priors, experiment

data = datasets.load("heart_disease")
samples = priors.load_samples(
    "priors/heart_disease/anthropic__opus-4-5/prior_samples.json", max_samples=50)

experiment.run_classification(
    run_name="heart_opus", out_dir="out/heart_opus",
    X_source=data.X_source, y_source=data.y_source,
    X_ood=data.X_ood, y_ood=data.y_ood,
    feature_names=data.feature_names, samples=samples,
)
```

Writes `id_metrics.csv`, `ood_metrics.csv`, `bootstrap/`, `tuning/`,
`config.json` and `split_sizes.json` — the same layout as `results/`.

## Checking it

```bash
python -m llmpriors.verify
```

Re-runs each cell and compares the metrics against `results/`. Install
`requirements.txt` as pinned for exact agreement.

## Implementation details worth knowing

Not quirks — these change results, so they are fixed deliberately.

- **Preprocessing and prior rescaling happen inside each CV fold.** Priors are
  stored in raw feature units and multiplied by that fold's feature standard
  deviations, so the prior is on the same scale as the fold's design matrix.
- **Logistic tuning walks the alpha grid downwards on one model instance**, so
  each fit warm-starts from the previous, more regularised solution.
- **Correlated-Prior fits from a zero start; Prior-Mean and Independent-Prior
  from a seeded random start.** This affects the L-BFGS path.
- **Alpha enters the Independent-Prior precision as `alpha / (var + eps)`**, and
  the Correlated-Prior penalty as a factor outside the matrix product.
- **The sample covariance is jittered by 1e-6 before inversion**, so the
  precision matrix is well defined.
- **The classification baseline tunes on `alpha_equiv = 1/C`**, so its
  regularisation axis runs the same direction as the prior families'.
