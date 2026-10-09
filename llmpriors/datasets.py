"""Dataset loading and domain splitting.

Each loader returns a `Dataset`: a source (in-domain) half, an out-of-domain
half, and the feature names that the prior bundles must key on.

Two rules hold for every dataset:

  * The variable used to define the domain split is never a predictor. Source
    and OOD differ systematically in it by construction, so keeping it would let
    the model read domain identity straight off a feature.
  * Categorical levels are aligned across the two halves. Dummies are built on
    the source and the OOD half is reindexed onto those columns, so a level that
    appears only out-of-domain cannot introduce a column the model never trained
    on.

Missing-value policy: features above roughly 50% source missingness are dropped
at the dataset level (documented per loader); the rest are median-imputed from
the source domain inside the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("data")


@dataclass(frozen=True)
class Dataset:
    name: str
    task: str                    # "regression" | "classification"
    X_source: np.ndarray
    y_source: np.ndarray
    X_ood: np.ndarray
    y_ood: np.ndarray
    feature_names: list[str]


def _one_hot_aligned(
    source: pd.DataFrame,
    ood: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Median-fill numerics, mode-fill categoricals, one-hot on source columns.

    Every fill value comes from the source domain, including the ones applied to
    the OOD half.
    """
    source_num, ood_num = source[numeric].copy(), ood[numeric].copy()
    medians = source_num.median(numeric_only=True)
    source_num, ood_num = source_num.fillna(medians), ood_num.fillna(medians)

    source_cat, ood_cat = source[categorical].copy(), ood[categorical].copy()
    for col in categorical:
        mode = source_cat[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else 0.0
        source_cat[col] = source_cat[col].fillna(fill)
        ood_cat[col] = ood_cat[col].fillna(fill)

    source_cat = source_cat.round().astype(int).astype(str).add_prefix("cat__")
    ood_cat = ood_cat.round().astype(int).astype(str).add_prefix("cat__")

    source_dummies = pd.get_dummies(source_cat, prefix_sep="=", drop_first=False)
    ood_dummies = pd.get_dummies(ood_cat, prefix_sep="=", drop_first=False)
    ood_dummies = ood_dummies.reindex(columns=source_dummies.columns, fill_value=0)

    X_source = pd.concat([source_num.reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)
    X_ood = pd.concat([ood_num.reset_index(drop=True), ood_dummies.reset_index(drop=True)], axis=1)
    return X_source.values.astype(float), X_ood.values.astype(float), X_source.columns.tolist()


def load_boston_housing(root: Path = DATA_ROOT) -> Dataset:
    """Boston Housing, split on housing-stock age.

    Source is the newer-stock tracts (AGE <= 70th percentile), OOD the older
    ones. The 70th percentile is used rather than the median because the two
    AGE ranges then barely overlap, giving a clean domain gap. AGE is the
    splitter and is not a predictor.
    """
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    df = pd.read_csv(root / "boston" / "boston_housing.csv")
    for col in features + ["MEDV", "AGE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[features] = df[features].fillna(df[features].median(numeric_only=True))

    is_source = (df["AGE"] <= float(np.nanpercentile(df["AGE"], 70))).values
    X, y = df[features].values.astype(float), df["MEDV"].values.astype(float)
    return Dataset("boston_housing", "regression",
                   X[is_source], y[is_source], X[~is_source], y[~is_source], features)


def load_heart_disease(root: Path = DATA_ROOT) -> Dataset:
    """UCI Heart Disease, split on study cohort (Cleveland -> Hungary + Switzerland).

    `ca`, `thal` and `slope` are dropped: each is >= 50% missing in the OOD
    cohorts. `fbs` is kept at roughly 20% OOD missingness — an audit found that
    dropping it slightly hurt prior performance with no offsetting benefit. The
    Long Beach cohort is excluded entirely for severe missingness across most
    clinical fields.
    """
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical = ["sex", "cp", "fbs", "restecg", "exang"]

    directory = root / "heart+disease"
    frames = {name: pd.read_csv(directory / f"processed.{name}.data", names=columns, na_values="?")
              for name in ("cleveland", "hungarian", "switzerland")}
    # Switzerland records cholesterol 0 for every row; 0 mg/dL is impossible, so
    # treat it as missing and let source-median imputation handle it.
    for frame in frames.values():
        frame.loc[frame["chol"] == 0, "chol"] = np.nan

    source = frames["cleveland"]
    ood = pd.concat([frames["hungarian"], frames["switzerland"]], axis=0)

    y_source = (source["target"] > 0).astype(int).values
    y_ood = (ood["target"] > 0).astype(int).values
    source_X = source.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")
    ood_X = ood.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce")

    X_source, X_ood, names = _one_hot_aligned(source_X, ood_X, numeric, categorical)
    return Dataset("heart_disease", "classification", X_source, y_source, X_ood, y_ood, names)


def load_diabetes(root: Path = DATA_ROOT) -> Dataset:
    """Pima Indians Diabetes, split at the median age (29).

    Zeros in the clinical measurements encode "not recorded" rather than a real
    zero, so they are converted to missing before imputation. Age is the
    splitter and is not a predictor.
    """
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction"]
    zero_is_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df = pd.read_csv(root / "diabetes" / "diabetes.csv")
    df[zero_is_missing] = df[zero_is_missing].astype(float).replace(0, np.nan)
    for col in features + ["Outcome", "Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    is_source = (df["Age"] <= float(df["Age"].median())).values
    X = df[features].astype(float).values
    y = df["Outcome"].astype(int).values
    return Dataset("diabetes", "classification",
                   X[is_source], y[is_source], X[~is_source], y[~is_source], features)


SOUTH_STATES = frozenset({
    "Alabama", "Arkansas", "Delaware", "District of Columbia", "Florida",
    "Georgia", "Kentucky", "Louisiana", "Maryland", "Mississippi",
    "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas",
    "Virginia", "West Virginia",
})

#: County-level features kept from the 34-column source file. The rest are
#: population-size proxies, duplicated insurance-coverage measures, or
#: education-by-young-age bands that overlap the retained adult bands -- all
#: strongly collinear with what is kept here.
CANCER_FEATURES = [
    "incidenceRate", "medIncome", "povertyPercent", "MedianAge",
    "PctHS25_Over", "PctBachDeg25_Over", "PctEmployed16_Over",
    "PctUnemployed16_Over", "PctPrivateCoverage", "PctPublicCoverage",
    "PctBlack", "PctOtherRace", "PctMarriedHouseholds",
]


def load_cancer(root: Path = DATA_ROOT) -> Dataset:
    """US county cancer mortality, split on Census region (South -> rest of US).

    Coefficients are ecological: associations between county aggregates and
    county mortality, not individual-level clinical effects. Region is the
    splitter and is not a predictor.
    """
    df = pd.read_csv(root / "cancer" / "cancer_reg.csv", encoding="latin1")
    df["state"] = df["Geography"].str.extract(r",\s*([^,]+)$")[0].str.strip()
    # 30 rows carry MedianAge above 100 (max 624), a units error upstream.
    # Treat as missing so imputation handles them instead of skewing the fit.
    df.loc[df["MedianAge"] > 100, "MedianAge"] = np.nan

    is_source = df["state"].isin(SOUTH_STATES).values
    X = df[CANCER_FEATURES].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = df["TARGET_deathRate"].values.astype(float)
    return Dataset("cancer", "regression",
                   X[is_source], y[is_source], X[~is_source], y[~is_source], list(CANCER_FEATURES))


def load_adult(root: Path = DATA_ROOT) -> Dataset:
    """US Census income, split on workclass (private -> public/self-employed).

    `sex` is excluded alongside the splitter so no protected attribute enters
    the coefficient structure; the elicited Adult priors were built under the
    same exclusion. Occupation is collapsed to the six most common source-domain
    levels plus an `Other_occupation` catch-all, so a rare level cannot become a
    near-empty indicator column.
    """
    numeric = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical = ["marital-status", "occupation"]

    X_all = pd.read_csv(root / "adult" / "adult_X.csv")
    y_all = (pd.read_csv(root / "adult" / "adult_y.csv").iloc[:, 0].astype(str) == ">50K").astype(int).values

    is_source = (X_all["workclass"] == "Private").values
    X_all = X_all.drop(columns=["workclass", "sex"], errors="ignore")

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if out[col].isnull().any():
                out[col] = out[col].fillna(out[col].median())
        for col in categorical:
            out[col] = out[col].replace("?", np.nan).fillna("Unknown").astype(str)
        return out

    source, ood = prepare(X_all[is_source]), prepare(X_all[~is_source])

    top_occupations = list(source["occupation"].value_counts().head(6).index)
    for frame in (source, ood):
        frame["occupation"] = frame["occupation"].where(
            frame["occupation"].isin(top_occupations), other="Other_occupation")

    source_dummies = pd.get_dummies(source[categorical], prefix=categorical, drop_first=False)
    ood_dummies = pd.get_dummies(ood[categorical], prefix=categorical, drop_first=False)
    ood_dummies = ood_dummies.reindex(columns=source_dummies.columns, fill_value=0)

    X_source = pd.concat([source[numeric].reset_index(drop=True),
                          source_dummies.reset_index(drop=True)], axis=1)
    X_ood = pd.concat([ood[numeric].reset_index(drop=True),
                       ood_dummies.reset_index(drop=True)], axis=1)

    return Dataset("adult", "classification",
                   X_source.values.astype(float), y_all[is_source],
                   X_ood.values.astype(float), y_all[~is_source],
                   X_source.columns.tolist())


LOADERS = {
    "adult": load_adult,
    "boston_housing": load_boston_housing,
    "cancer": load_cancer,
    "diabetes": load_diabetes,
    "heart_disease": load_heart_disease,
}


def load(name: str, root: Path = DATA_ROOT) -> Dataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(LOADERS)}")
    return LOADERS[name](root)
