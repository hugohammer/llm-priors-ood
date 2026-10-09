"""Regenerate the two figures that originally came from the thesis aggregation
layer, so that Boston Housing reflects the removal of the B feature.

  fig_selection.*   per-dataset floor / ceiling / deployed OOD lift
  fig_cross_llm.*   prior-mean coefficient per feature for all 12 variants

Only Boston Housing changed, but both figures are single images, so both are
rebuilt in full. Cancer, Heart Disease and Diabetes are recomputed here from
the committed priors; Adult's selection sweep (12 variants x 4 families x 51
lambdas of logistic fits on n = 27k) is too costly to redo and is unaffected by
the change, so its three values are carried over from the original run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from make_figures import (BLUISH_GREEN, VERMILLION, GREY, FIG, save, DS_ORDER,
                          BLUE, ORANGE, SKY, PURPLE)

#: Carried over from the original sweep; unaffected by dropping B.
ADULT = dict(floor=-0.8, ceiling=-0.0, deployed=0.0)

VARIANT_STYLE = [
    ("GPT-5.1", BLUE, "o", False), ("GPT-5.1 (R)", BLUE, "o", True),
    ("GPT-5-mini", SKY, "o", False), ("GPT-5-mini (R)", SKY, "o", True),
    ("Opus 4.5", VERMILLION, "o", False), ("Opus 4.5 (R)", VERMILLION, "o", True),
    ("Haiku 4.5", ORANGE, "o", False), ("Haiku 4.5 (R)", ORANGE, "o", True),
    ("Gemini Pro", "#2E7D32", "o", False), ("Gemini Pro (R)", "#2E7D32", "o", True),
    ("Gemini Flash", "#4DB6AC", "o", False), ("Gemini Flash (R)", "#4DB6AC", "o", True),
]

PRETTY_FEATURE = {
    "TAX": "Property tax rate", "RM": "Avg rooms per dwelling",
    "LSTAT": "Lower-SES population (%)", "ZN": "Residential zoning (%)",
    "PTRATIO": "Pupil-teacher ratio", "NOX": "Nitric oxides (ppm)",
    "CRIM": "Per-capita crime rate", "DIS": "Distance to employment",
    "INDUS": "Non-retail business (%)", "CHAS": "Bounds Charles River",
    "RAD": "Highway accessibility",
    "oldpeak": "ST depression (exercise)", "thalach": "Max heart rate",
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "Glucose": "Plasma glucose (2h OGTT)", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes pedigree", "Pregnancies": "Pregnancies",
    "Insulin": "2h serum insulin", "BloodPressure": "Diastolic BP (mm Hg)",
    "SkinThickness": "Triceps skin fold (mm)",
}


# ---------------------------------------------------------------------------
def sweep_regression(dataset):
    """Lift of every (variant, family) cell at its CV-tuned lambda."""
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED,
                                    stratify=False)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=False)
    pre = C.Preprocessor().fit(Xtr)
    A, O = pre.transform(Xtr), pre.transform(d.X_ood)

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    a_b, _ = C.tune_alpha_baseline_regression(packs0)
    base = C.mse(d.y_ood, C.fit_baseline_ridge(A, ytr, a_b).predict(O))

    lifts = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(dataset, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
        for fam in ("prior_mean", "independent", "correlated"):
            a, _ = C.tune_alpha_regression(packs, fam)
            m = C.fit_prior_ridge(A, ytr, pq.mean, C.precision_for(fam, pq), a)
            lifts.append(100.0 * (base - C.mse(d.y_ood, m.predict(O))) / base)
    return np.array(lifts)


def sweep_classification(dataset):
    from sklearn.metrics import roc_auc_score
    d = C.load(dataset)
    Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source, seed=C.cfg.SEED,
                                    stratify=True)
    folds = C.make_folds(Xtr, ytr, seed=C.cfg.SEED, stratify=True)
    pre = C.Preprocessor().fit(Xtr)
    A, O = pre.transform(Xtr), pre.transform(d.X_ood)

    packs0 = C.prepare(Xtr, ytr, folds, C.load_samples(dataset, "Opus 4.5"),
                       d.feature_names)
    a_b, _ = C.tune_alpha_baseline_classification(packs0)
    base = float(roc_auc_score(
        d.y_ood, C.fit_baseline_logistic(A, ytr, a_b).predict_proba(O)[:, 1]))

    lifts = []
    for variant in C.VARIANT_DIR:
        samples = C.load_samples(dataset, variant)
        packs = C.prepare(Xtr, ytr, folds, samples, d.feature_names)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)
        for fam in ("prior_mean", "independent", "correlated"):
            a, _ = C.tune_alpha_classification(packs, fam)
            m = C.fit_prior_logistic(A, ytr, pq.mean, C.precision_for(fam, pq), a,
                                     seed=C.start_seed_for(fam))
            auc = float(roc_auc_score(d.y_ood, m.predict_proba(O)))
            lifts.append(100.0 * (auc - base) / base)
    return np.array(lifts)


def fig_selection():
    dep = pd.read_csv(C.OUT / "boston_nob_deployment.csv")
    stats = {"Boston Housing": dict(floor=None, ceiling=None,
                                    deployed=float(dep.lift_pct.max()))}
    for ds, pretty in (("boston_housing", "Boston Housing"), ("cancer", "Cancer"),
                       ("heart_disease", "Heart Disease"), ("diabetes", "Diabetes")):
        lifts = (sweep_regression(ds) if C.TASK[ds] == "regression"
                 else sweep_classification(ds))
        s = stats.setdefault(pretty, {})
        s["floor"], s["ceiling"] = float(lifts.min()), float(lifts.max())
        s.setdefault("deployed", None)
        print(f"  {pretty:15s} floor {s['floor']:+6.2f}  ceiling {s['ceiling']:+6.2f}")
    # deployed values for the three unchanged datasets, from the paper's tables
    stats["Cancer"]["deployed"] = 3.9
    stats["Heart Disease"]["deployed"] = 0.9
    stats["Diabetes"]["deployed"] = 1.1
    stats["Adult"] = dict(floor=ADULT["floor"], ceiling=ADULT["ceiling"],
                          deployed=ADULT["deployed"])

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, ds in enumerate(DS_ORDER):
        s = stats[ds]
        ax.plot([i, i], [s["floor"], s["ceiling"]], color=GREY, lw=1.6, zorder=2)
        ax.plot([i - 0.17, i + 0.17], [s["ceiling"]] * 2, color=BLUISH_GREEN, lw=3,
                solid_capstyle="butt", zorder=3)
        ax.plot([i - 0.17, i + 0.17], [s["floor"]] * 2, color=VERMILLION, lw=3,
                solid_capstyle="butt", zorder=3)
        ax.plot([i], [s["deployed"]], "D", ms=9, color="black", zorder=4)
        ax.annotate(f"{s['ceiling']:+.1f}%", (i + 0.21, s["ceiling"]),
                    color=BLUISH_GREEN, fontsize=8, va="center")
        ax.annotate(f"{s['floor']:+.1f}%", (i + 0.21, s["floor"]),
                    color=VERMILLION, fontsize=8, va="center")
        # deployed can coincide with the ceiling, so label it on the left
        ax.annotate(f"{s['deployed']:+.1f}%", (i - 0.22, s["deployed"]),
                    fontweight="bold", fontsize=8, va="center", ha="right")

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels(DS_ORDER)
    ax.set_xlim(-0.72, len(DS_ORDER) - 0.30)
    ax.set_ylabel("OOD lift over baseline (%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=BLUISH_GREEN, lw=3, label="best achievable (ceiling)"),
        Line2D([], [], color=VERMILLION, lw=3, label="worst achievable (floor)"),
        Line2D([], [], marker="D", ls="", color="black", label="deployed pipeline"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig_selection")


def fig_cross_llm():
    panels = [("cancer", "Cancer", None),
              ("boston_housing", "Boston Housing", None),
              ("heart_disease", "Heart Disease (numeric features)",
               ["oldpeak", "thalach", "age", "trestbps", "chol"]),
              ("diabetes", "Diabetes", None)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    for ax, (ds, title, subset) in zip(axes.ravel(), panels):
        d = C.load(ds)
        pre = C.Preprocessor().fit(d.X_source)
        feats = subset or d.feature_names
        means = {}
        for label, *_ in VARIANT_STYLE:
            pq = C.priors.prior_quantities(C.load_samples(ds, label),
                                           d.feature_names, pre.scale)
            means[label] = dict(zip(d.feature_names, pq.mean))
        order = sorted(feats, key=lambda f: -abs(np.median([means[v][f] for v in means])))
        ypos = np.arange(len(order))[::-1]

        # A handful of single draws (notably Gemini Flash on Boston's property
        # tax rate) sit orders of magnitude from the cluster and would flatten
        # every other feature, so the axis is clipped to the bulk and the
        # off-panel values are annotated at the boundary.
        allv = np.array([means[v][f] for v in means for f in order])
        lo_b, hi_b = np.percentile(allv, [2, 98])
        pad = 0.30 * max(hi_b - lo_b, 1e-9)
        xlo, xhi = min(lo_b - pad, 0.0), max(hi_b + pad, 0.0)

        off_count = {}   # (feature, side) -> how many labels already placed
        for label, colour, marker, reasoning in VARIANT_STYLE:
            for f, y in zip(order, ypos):
                x = means[label][f]
                if xlo <= x <= xhi:
                    ax.plot([x], [y], marker, ms=5,
                            mfc="none" if reasoning else colour, mec=colour,
                            mew=1.2, ls="", alpha=0.9)
                else:
                    side = "l" if x < xlo else "r"
                    k = off_count.get((f, side), 0)
                    off_count[(f, side)] = k + 1
                    ax.annotate(f"{label} {x:.0f}", (xlo if side == "l" else xhi, y),
                                fontsize=6, color=colour, va="center",
                                ha="left" if side == "l" else "right",
                                xytext=(3 if side == "l" else -3, 7 - 7 * k),
                                textcoords="offset points")
            ax.plot([], [], marker, ms=5, mfc="none" if reasoning else colour,
                    mec=colour, mew=1.2, ls="", label=label)
        for f, y in zip(order, ypos):
            mu = np.mean([means[v][f] for v in means])
            if xlo <= mu <= xhi:
                ax.plot([mu], [y], "|", ms=12, color=GREY, mew=2, zorder=1)

        ax.set_xlim(xlo, xhi)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([PRETTY_FEATURE.get(f, f) for f in order], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("prior-mean coefficient (source-standardised)", fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cross-LLM coefficient agreement (12 frontier variants)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, "fig_cross_llm")


if __name__ == "__main__":
    print("selection sweep")
    fig_selection()
    print("cross-LLM panels")
    fig_cross_llm()
