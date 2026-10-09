"""LLM-elicited priors for regularised linear and logistic regression.

A clean reimplementation of the thesis pipeline. Every model is MAP estimation
under a Gaussian coefficient prior; the families differ only in the prior mean
and precision, both estimated from repeated LLM coefficient samples.

Typical use:

    from llmpriors import datasets, experiment, priors

    data = datasets.load("boston_housing")
    samples = priors.load_samples("priors/boston_housing/anthropic__opus-4-5/prior_samples.json",
                                  max_samples=50)
    experiment.run_regression(
        run_name="boston_opus",
        out_dir="out/boston_opus",
        X_source=data.X_source, y_source=data.y_source,
        X_ood=data.X_ood, y_ood=data.y_ood,
        feature_names=data.feature_names,
        samples=samples,
    )
"""
from . import bootstrap, config, crossval, datasets, estimators, experiment, metrics, priors
from .estimators import GaussianPriorLogistic, GaussianPriorRidge
from .experiment import run_classification, run_regression

__all__ = [
    "bootstrap", "config", "crossval", "datasets", "estimators", "experiment",
    "metrics", "priors",
    "GaussianPriorRidge", "GaussianPriorLogistic",
    "run_regression", "run_classification",
]
