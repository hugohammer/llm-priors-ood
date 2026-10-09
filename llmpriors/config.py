"""Fixed experiment settings.

Every number here is part of the published configuration. Changing one changes
the results, so they live in one place rather than as defaults scattered across
call sites.
"""
from __future__ import annotations

import numpy as np

SEED = 42

#: Regularisation-strength grid, shared by every model family and dataset.
#: The thesis calls this lambda; the code and output CSVs call it alpha.
ALPHA_GRID = np.logspace(-4, 6, 51)

#: Prediction-mixing weight grid for the Mixed family.
GAMMA_GRID = np.linspace(0, 1, 101)

#: sklearn's LogisticRegression is parameterised by C. The baseline is tuned on
#: alpha_equiv = 1/C so its regularisation axis runs the same direction as the
#: prior families' alpha, over the same range.
C_GRID = np.logspace(-6, 4, 51)

N_FOLDS = 5
ID_TEST_FRACTION = 0.20

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI_LEVEL = 0.95
CLASSIFICATION_THRESHOLD = 0.5

#: Added to the diagonal of the LLM sample covariance before inversion, so the
#: precision matrix is well defined.
COVARIANCE_JITTER = 1e-6

#: Added to prior variances before inverting them into a diagonal precision.
VARIANCE_EPS = 1e-8

# Output labels. These appear in id_metrics.csv / ood_metrics.csv and are what
# the aggregation layer matches on, so they are fixed strings.
REGRESSION_LABELS = {
    "unregularised": "OLS",
    "baseline": "Ridge",
    "prior_mean": "Prior-Mean Ridge",
    "correlated": "Correlated-Prior Ridge",
    "independent": "Independent-Prior Ridge",
}
CLASSIFICATION_LABELS = {
    "unregularised": "OLS (reference)",
    "baseline": "L2 Logistic",
    "prior_mean": "Prior-Mean Logistic",
    "correlated": "Correlated-Prior Logistic",
    "independent": "Independent-Prior Logistic",
}


def mixed_label(gamma: float) -> str:
    """Label for the Mixed family, which carries its tuned gamma."""
    return f"Mixed (gamma={gamma:.2f})"
