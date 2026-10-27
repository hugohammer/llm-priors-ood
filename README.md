# LLM-elicited coefficient priors under domain shift

Code, elicited priors and results for the paper

> **LLM knowledge for improved out-of-distribution tabular prediction**
> S. Langseth, K. H. Hellton, H. L. Hammer

Everything reported in the paper can be reproduced from this repository
**without any LLM API calls**. The 50 sampled coefficient vectors for every
(dataset, provider, mode) combination are stored in `priors/`, so the elicitation step
— the only part that needs API access — never has to be repeated.

---

## What is here

| Path | Contents |
|---|---|
| `llmpriors/` | The pipeline package: elicitation, prior estimation, the four model families, cross-validation, metrics, bootstrap |
| `priors/` | 60 elicited prior bundles — 5 datasets x 12 LLM variants, 50 coefficient draws each |
| `data/` | The five datasets. See `data/README.md` for provenance and one important modification |
| `experiments/` | The follow-up analyses (`exp1`–`exp13`) and the figure scripts |
| `results/` | Every CSV and run log behind the tables and figures, plus `RESULTS.md`, a dated log of what was run and why |
| `figures/` | Generated figures as they appear in the paper |

`results/RESULTS.md` is the most useful file for a reader checking a specific
number: it records each analysis, the numbers it produced, and the addenda
documenting corrections made during review.

---

## Reproducing the paper

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd experiments
python exp8_selection.py        # 48-candidate sweep and selection rules
python exp11_joint_ci.py        # nested resplit x bootstrap draws
python exp12_ci_for_mean.py     # confidence vs prediction intervals
```

Each script writes to `experiments/out/` and prints the numbers that appear in
the paper. They are independent of one another except that `exp12` reads the
draws written by `exp11`.

| Script | Produces |
|---|---|
| `exp1_prior_only.py` | Prior-only predictor (lambda to infinity) |
| `exp2_threshold.py` | Threshold recalibration, Supplementary Note 2 |
| `exp3_placebo.py` | Permuted and sign-flip placebo controls |
| `exp4_resplits.py` | Source-split variability |
| `exp5_joint.py` | Nested resplit x bootstrap intervals |
| `exp6_pooled.py` | Cross-dataset pooling (reported but not used in the paper) |
| `exp7_lambda_oracle.py` | Cost of selecting lambda on the source domain |
| `exp8_selection.py` | Full 48-candidate grid, within- and cross-family selection, in-domain lifts |
| `exp9_epsilon.py` | Variance-floor sensitivity and Ledoit–Wolf shrinkage, Supplementary Note 3 |
| `exp10_heart_fbs.py` | Heart Disease ablation of the `fbs` feature |
| `exp11_joint_ci.py` | As `exp5` but recording the split index |
| `exp12_ci_for_mean.py` | Variance decomposition: CI for the mean vs prediction interval |
| `exp13_paired_margin.py` | Paired standard error of the selection margin between candidates |

### Environment

`requirements.txt` pins the versions used for the original runs. The
referee-response analyses (`exp8`–`exp13`) were run on newer versions
(numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1) and reproduced
the stored outputs to within 1e-11, so either environment works.

### Re-running the elicitation

Only needed if you want new priors. `llmpriors/elicitation.py` expects
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` in the environment
or a local `.env` file. No keys are committed to this repository.

---

## The twelve LLM variants

Six models from three providers, each queried with and without reasoning, at
temperature 0.8, 50 draws per combination:

| Provider | Models |
|---|---|
| OpenAI | GPT-5.1, GPT-5-mini |
| Anthropic | Claude Opus 4.5, Claude Haiku 4.5 |
| Google | Gemini 2.5 Pro, Gemini 2.5 Flash |

They are not twelve independent sources: two of the six are smaller siblings of
the others, and every variant received identical prompt text.

---

## Two things to know before using the priors

**The prompts contain author-written domain hints.** Each dataset's prompt
includes a short block stating the expected sign of the dominant coefficients,
and on some datasets their relative size. See
`llmpriors/prompt_content.py: signal_hints`. The elicited magnitudes, variances
and covariances are the model's; the directional content is partly ours. The
paper discusses what can and cannot be attributed to the elicitation as a
result.

**The variance floor is not cosmetic.** One variant returns bit-identical draws
for some features, giving an elicited variance of exactly zero. The
Independent-Prior penalty weight is then `lambda / epsilon`, which holds that
coefficient at the elicited value rather than shrinking toward it.
`exp9_epsilon.py` reports the sensitivity analysis.

---

## Citation

See `CITATION.cff`. Please cite the paper rather than this repository alone.

## Licence

Code is released under the MIT licence (`LICENSE`). The datasets in `data/`
carry their own terms — see `data/README.md`.
