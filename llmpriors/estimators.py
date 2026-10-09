"""MAP estimators under a Gaussian coefficient prior.

Every model in the study minimises

    L(beta) + 0.5 * alpha * (beta - mu)^T P (beta - mu)

where L is squared error (regression) or the negative log-likelihood
(classification). The four families differ only in (mu, P):

    Ridge / L2-Logistic   mu = 0,      P = I
    Prior-Mean            mu = mu_hat, P = I
    Independent-Prior     mu = mu_hat, P = diag(1 / (var_j + eps))
    Correlated-Prior      mu = mu_hat, P = Sigma_hat^-1

So there are two estimators here, not eight: one per loss. The `0.5 * alpha`
factor is shared by all of them, which is what makes tuned alpha values
comparable across families in the reported tables.

The intercept is never penalised: it is appended as a leading column of ones
and the corresponding row/column of P is zeroed.

`precision` may be a matrix, in which case the estimator forms
`alpha * (P @ diff)`, or a callable `alpha -> P_eff` that has already folded
alpha in. The Independent-Prior family uses the callable form and the
Correlated-Prior family the matrix form, matching how each is defined.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

Precision = np.ndarray | Callable[[float], np.ndarray]

LBFGS_MAXITER = 5000
LBFGS_MAXITER_RETRY = 20000
_LOGIT_CLIP = 500.0


def _augment(mu: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extend (mu, P) with an unpenalised leading intercept term."""
    d = P.shape[0] + 1
    P_ext = np.zeros((d, d))
    P_ext[1:, 1:] = P
    mu_ext = np.concatenate([[0.0], mu])
    return mu_ext, P_ext


def _resolve(precision: Precision, alpha: float) -> tuple[np.ndarray, float]:
    """Return (P, outer_alpha).

    A callable precision has already absorbed alpha, so the outer factor is 1.
    A matrix precision has not, so alpha multiplies the product afterwards.
    """
    if callable(precision):
        return np.asarray(precision(alpha), dtype=float), 1.0
    return np.asarray(precision, dtype=float), float(alpha)


def _with_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


class GaussianPriorRidge:
    """Closed-form MAP for squared-error loss under N(mu, (1/alpha) P^-1).

    Minimises ||y - X beta||^2 + 0.5 * alpha * (beta - mu)^T P (beta - mu),
    whose stationary point solves

        (X^T X + 0.5*alpha*P) beta = X^T y + 0.5*alpha*P mu
    """

    def __init__(self, mu: np.ndarray, P: Precision, alpha: float = 1.0):
        self.mu = np.asarray(mu, dtype=float)
        self.P = P
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianPriorRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        P, outer_alpha = _resolve(self.P, self.alpha)
        mu_ext, P_ext = _augment(self.mu, P)
        Xw = _with_bias(X)

        half = 0.5 * outer_alpha
        beta = _solve(Xw.T @ Xw + half * P_ext, Xw.T @ y + half * (P_ext @ mu_ext))

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class GaussianPriorLogistic:
    """L-BFGS MAP for logistic loss under N(mu, (1/alpha) P^-1).

    Minimises -log L(beta) + 0.5 * alpha * (beta - mu)^T P (beta - mu).

    `alpha` is a plain attribute rather than a constructor-only value: the
    tuning loop walks the alpha grid downwards on a single instance so each fit
    warm-starts from the previous solution. Reassigning `alpha` and calling
    `fit` again is the supported way to do that, and the resulting path is part
    of the published configuration.

    `random_state` controls only the cold-start point: None starts at zero,
    an int starts at a seeded 0.01 * N(0, 1) draw.
    """

    def __init__(
        self,
        mu: np.ndarray,
        P: Precision,
        alpha: float = 1.0,
        random_state: int | None = None,
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.P = P
        self.alpha = float(alpha)
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.converged_: bool | None = None

    def _start_point(self, d: int) -> np.ndarray:
        if self.coef_ is not None and len(self.coef_) == d - 1:
            return np.concatenate([[float(self.intercept_)], self.coef_])
        if self.random_state is not None:
            np.random.seed(self.random_state)
            return 0.01 * np.random.randn(d)
        return np.zeros(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianPriorLogistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        P, alpha = _resolve(self.P, self.alpha)
        mu_ext, P_ext = _augment(self.mu, P)
        Xw = _with_bias(X)

        def objective(beta: np.ndarray) -> float:
            z = np.clip(Xw @ beta, -_LOGIT_CLIP, _LOGIT_CLIP)
            log_lik = np.sum(y * z - np.logaddexp(0, z))
            diff = beta - mu_ext
            return -log_lik + 0.5 * alpha * (diff @ P_ext @ diff)

        def gradient(beta: np.ndarray) -> np.ndarray:
            z = np.clip(Xw @ beta, -_LOGIT_CLIP, _LOGIT_CLIP)
            p = 1.0 / (1.0 + np.exp(-z))
            return -Xw.T @ (y - p) + alpha * (P_ext @ (beta - mu_ext))

        result = minimize(
            objective,
            self._start_point(Xw.shape[1]),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": LBFGS_MAXITER},
        )
        if not result.success:
            retry = minimize(
                objective,
                result.x,
                jac=gradient,
                method="L-BFGS-B",
                options={"maxiter": LBFGS_MAXITER_RETRY},
            )
            if retry.success or retry.nit > result.nit:
                result = retry

        self.converged_ = bool(result.success)
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(
            np.asarray(X, dtype=float) @ self.coef_ + self.intercept_,
            -_LOGIT_CLIP,
            _LOGIT_CLIP,
        )
        return 1.0 / (1.0 + np.exp(-z))
