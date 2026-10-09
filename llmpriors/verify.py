"""Check this package against the published results.

Runs two cells end to end and compares every model's evaluation metrics against
the corresponding CSV under `results/`. Exits non-zero on any disagreement.

    python -m llmpriors.verify

Metrics are compared at 1e-9, except LogLoss and Brier on the L-BFGS-fitted
prior families, which use a looser bound: those objectives are sensitive to the
scipy/numpy build, so the last decimals move between environments. Install
requirements.txt as pinned for exact agreement.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import datasets, experiment, priors

EXACT_TOLERANCE = 1e-9
CALIBRATION_TOLERANCE = 1e-3
PUBLISHED = Path("results/provider_sweep_n50_cv/runs")

CALIBRATION_METRICS = {"LogLoss", "Brier"}

_CLASSIFICATION_METRICS = ["AUC", "F1", "Acc", "Sensitivity", "Specificity", "LogLoss", "Brier"]

CASES = [
    ("boston_housing", "anthropic__opus-4-5", "regression", ["MSE"]),
    ("cancer", "anthropic__opus-4-5", "regression", ["MSE"]),
    ("heart_disease", "anthropic__opus-4-5", "classification", _CLASSIFICATION_METRICS),
    ("diabetes", "anthropic__opus-4-5", "classification", _CLASSIFICATION_METRICS),
    ("adult", "openai__gpt-5-mini", "classification", _CLASSIFICATION_METRICS),
]


def _run_case(dataset_name: str, variant: str, task: str, out_dir: Path) -> pd.DataFrame:
    data = datasets.load(dataset_name)
    samples = priors.load_samples(
        f"priors/{dataset_name}/{variant}/prior_samples.json", max_samples=50
    )
    runner = experiment.run_regression if task == "regression" else experiment.run_classification
    result = runner(
        run_name=f"verify_{dataset_name}",
        out_dir=out_dir,
        X_source=data.X_source, y_source=data.y_source,
        X_ood=data.X_ood, y_ood=data.y_ood,
        feature_names=data.feature_names,
        samples=samples,
    )
    return result.ood_metrics


def main() -> int:
    if not PUBLISHED.exists():
        print(f"published results not found at {PUBLISHED}", file=sys.stderr)
        return 2

    # Optional dataset filter: `python -m llmpriors.verify diabetes boston_housing`
    wanted = set(sys.argv[1:])
    cases = [c for c in CASES if not wanted or c[0] in wanted]

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for dataset_name, variant, task, metrics in cases:
            expected = pd.read_csv(PUBLISHED / variant / dataset_name / "ood_metrics.csv").set_index("Model")
            actual = _run_case(dataset_name, variant, task, Path(tmp) / dataset_name).set_index("Model")

            print(f"\n{dataset_name} / {variant}")
            for model in expected.index:
                if model not in actual.index:
                    print(f"  {model:<30} MISSING from clean package")
                    failures += 1
                    continue

                hard, soft, worst = [], [], 0.0
                for metric in metrics:
                    delta = abs(expected.loc[model, metric] - actual.loc[model, metric])
                    worst = max(worst, delta)
                    limit = (CALIBRATION_TOLERANCE if metric in CALIBRATION_METRICS
                             else EXACT_TOLERANCE)
                    if delta >= limit:
                        hard.append(f"{metric} {delta:.1e}")
                    elif delta > EXACT_TOLERANCE:
                        soft.append(f"{metric} {delta:.0e}")

                if hard:
                    failures += 1
                    print(f"  {model:<30} MISMATCH: {', '.join(hard)}")
                elif soft:
                    print(f"  {model:<30} ok, last decimals differ: {', '.join(soft)}")
                else:
                    print(f"  {model:<30} ok (max {worst:.1e})")

    print("\nPASS: clean package reproduces the published results"
          if not failures else f"\nFAIL: {failures} mismatch(es)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
