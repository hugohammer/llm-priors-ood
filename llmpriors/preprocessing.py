"""Median imputation followed by standardisation.

Fitted on a training portion only and then applied to held-out data, so no
statistic from the ID test set or the OOD set reaches the fitted model. Used at
two levels: inside each cross-validation fold, and once on the full training
pool at deployment.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Imputer + scaler pair fitted as a unit."""

    def __init__(self) -> None:
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self._scaler.fit(self._imputer.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(self._imputer.transform(X))

    @property
    def scale(self) -> np.ndarray:
        """Per-feature standard deviations, used to rescale raw-unit priors."""
        return np.asarray(self._scaler.scale_, dtype=float)
