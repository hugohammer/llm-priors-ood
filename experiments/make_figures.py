"""Figures for the paper, from the follow-up experiment outputs.

Okabe-Ito palette throughout, to match the thesis figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
OUT = C.OUT

VERMILLION = "#D55E00"
BLUISH_GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GREY = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

DS_ORDER = ["Boston Housing", "Cancer", "Heart Disease", "Diabetes", "Adult"]
FAM_ORDER = ["Prior-Mean", "Independent-Prior", "Correlated-Prior", "Mixed"]
FAM_COLOR = {"Prior-Mean": BLUE, "Independent-Prior": ORANGE,
             "Correlated-Prior": BLUISH_GREEN, "Mixed": PURPLE}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure: resplit stability
# ---------------------------------------------------------------------------
def fig_resplits():
    raw = pd.read_csv(OUT / "exp4_resplits_raw.csv")
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.7), sharey=False)

    for ax, ds in zip(axes, DS_ORDER):
        g = raw[raw["dataset"] == ds]
        data, colors = [], []
        for fam in FAM_ORDER:
            data.append(g[g["family"] == fam]["lift_pct"].values)
            colors.append(FAM_COLOR[fam])

        parts = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", lw=1.2),
                           whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
        for patch, c in zip(parts["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(GREY)

        for i, (vals, c) in enumerate(zip(data, colors), start=1):
            jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.22
            ax.plot(i + jit, vals, ".", ms=2.0, color=c, alpha=0.5, zorder=3)

        ax.axhline(0, color="black", lw=0.8, ls="-", zorder=1)
        ax.set_title(ds)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["PM", "IP", "CP", "Mix"])
        n = int(g.groupby("family").size().iloc[0])
        ax.text(0.97, 0.97, f"n={n}", transform=ax.transAxes, fontsize=7,
                color=GREY, ha="right", va="top")

    axes[0].set_ylabel("OOD lift over baseline (%)")
    fig.tight_layout()
    save(fig, "fig_resplits")


# ---------------------------------------------------------------------------
# Figure: placebo control
# ---------------------------------------------------------------------------
def fig_placebo():
    path = OUT / "exp3_placebo_draws.csv"
    if not path.exists():
        print("  (skipping placebo figure: exp3_placebo_draws.csv not present yet)")
        return
    draws = pd.read_csv(path)
    summ = pd.read_csv(OUT / "exp3_placebo.csv")

    cells = (draws[["dataset", "family"]].drop_duplicates()
             .assign(_o=lambda d: d["dataset"].map({k: i for i, k in enumerate(DS_ORDER)}))
             .sort_values(["_o", "family"]))
    # Headline metric per task: MSE reduction for the regression datasets and
    # log-loss improvement for the classification ones, matching the main
    # results. AUC is deliberately not used here -- it is the metric the priors
    # leave flat, so a placebo test on it is the weakest available.
    def metric_cols(task):
        if task == "regression":
            return "lift_pct", "real_lift_pct", "empirical_p"
        return "logloss_lift_pct", "real_logloss_lift_pct", "empirical_p_logloss"

    # Corrupted priors can be catastrophically bad on log loss (to -639%), so a
    # linear axis would collapse every real effect onto the zero line.
    XMIN, XMAX = -800.0, 20.0

    labels, ypos = [], []
    # 24 rows since the classification datasets gained Independent/Correlated;
    # keep the aspect close to square so it stays legible at \linewidth.
    fig, ax = plt.subplots(figsize=(6.6, 0.185 * len(cells) * 2 + 1.05))

    y = 0.0
    for _, cell in cells.iterrows():
        ds, fam = cell["dataset"], cell["family"]
        for kind, marker in (("permuted", "o"), ("signflip", "s")):
            sub = draws[(draws["dataset"] == ds) & (draws["family"] == fam)
                        & (draws["placebo"] == kind)]
            if sub.empty:
                continue
            vcol, rcol, pcol = metric_cols(sub["task"].iloc[0])
            v = sub[vcol].values
            real = float(sub[rcol].iloc[0])
            p = float(summ[(summ["dataset"] == ds) & (summ["family"] == fam)
                           & (summ["placebo"] == kind)][pcol].iloc[0])

            lo, hi = np.quantile(v, [0.025, 0.975])
            ax.plot([max(lo, XMIN), min(hi, XMAX)], [y, y], color=GREY, lw=1.2,
                    solid_capstyle="butt", zorder=2)
            vis = v[(v >= XMIN) & (v <= XMAX)]
            jit = (np.random.default_rng(abs(int(y * 7))).random(len(vis)) - 0.5) * 0.34
            ax.plot(vis, y + jit, ".", ms=1.8, color=VERMILLION, alpha=0.45, zorder=3)
            ax.plot([np.median(v)], [y], "|", ms=9, color=VERMILLION, mew=1.6, zorder=4)
            ax.plot([real], [y], marker, ms=6, color=BLUISH_GREEN,
                    markeredgecolor="black", mew=0.6, zorder=5)

            ax.text(1.005, y, f"$p$={p:.3f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5,
                    color="black" if p < 0.05 else VERMILLION)
            labels.append(f"{ds} · {fam.replace('-Prior','')} · {kind}")
            ypos.append(y)
            y -= 1.0

    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.5, 3.0)
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([-100, -10, 0, 10])
    ax.set_xticklabels(["-100", "-10", "0", "10"])
    ax.set_xlabel("OOD lift over baseline (%), symlog scale: "
                  "MSE reduction / log-loss improvement")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUISH_GREEN, markeredgecolor="black",
               mew=0.6, label="real prior (permuted row)"),
        Line2D([], [], marker="s", ls="", color=BLUISH_GREEN, markeredgecolor="black",
               mew=0.6, label="real prior (sign-flip row)"),
        Line2D([], [], marker=".", ls="", color=VERMILLION, label="placebo draws"),
        Line2D([], [], color=GREY, lw=1.2, label="placebo 2.5–97.5 pct"),
    ], loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=2,
        fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_placebo")


# ---------------------------------------------------------------------------
# Figure: the threshold artefact
# ---------------------------------------------------------------------------
def fig_threshold():
    from sklearn.metrics import f1_score

    grid = np.linspace(0.02, 0.98, 97)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    for ax, dataset in zip(axes, ["heart_disease", "diabetes"]):
        d = C.load(dataset)
        Xtr, _, ytr, _ = C.split_source(d.X_source, d.y_source,
                                        seed=C.cfg.SEED, stratify=True)
        pre = C.Preprocessor().fit(Xtr)
        A, O = pre.transform(Xtr), pre.transform(d.X_ood)
        y_ood = np.asarray(d.y_ood).astype(int)

        variant, a_pm = C.DEPLOYED[dataset]["prior_mean"]
        samples = C.load_samples(dataset, variant)
        pq = C.priors.prior_quantities(samples, d.feature_names, pre.scale)

        p_base = C.fit_baseline_logistic(
            A, ytr, C.DEPLOYED[dataset]["baseline"][1]).predict_proba(O)[:, 1]
        p_pm = C.fit_prior_logistic(A, ytr, pq.mean, pq.identity, a_pm).predict_proba(O)

        f_base = [f1_score(y_ood, (p_base >= t).astype(int), zero_division=0) for t in grid]
        f_pm = [f1_score(y_ood, (p_pm >= t).astype(int), zero_division=0) for t in grid]

        ax.plot(grid, f_base, color=BLUE, lw=1.6, label="L2 baseline")
        ax.plot(grid, f_pm, color=VERMILLION, lw=1.6, label="Prior-Mean")

        row = pd.read_csv(OUT / "exp2_threshold.csv")
        row = row[row["dataset"] == C.PRETTY[dataset]]
        t_b = float(row[row["family"] == "Baseline"]["cv_threshold"].iloc[0])
        t_p = float(row[row["family"] == "Prior-Mean"]["cv_threshold"].iloc[0])

        ax.axvline(0.5, color="black", lw=1.0, ls=":", zorder=1)
        ax.text(0.52, 0.30, "fixed 0.5\n(as reported)", transform=ax.transAxes,
                fontsize=7.5, color="black", va="center")
        ax.plot([t_b], [np.interp(t_b, grid, f_base)], "o", color=BLUE,
                markeredgecolor="black", mew=0.6, ms=6, zorder=5)
        ax.plot([t_p], [np.interp(t_p, grid, f_pm)], "o", color=VERMILLION,
                markeredgecolor="black", mew=0.6, ms=6, zorder=5)
        ax.plot([0.5], [np.interp(0.5, grid, f_base)], "x", color=BLUE, ms=6,
                mew=1.6, zorder=5)
        ax.plot([0.5], [np.interp(0.5, grid, f_pm)], "x", color=VERMILLION, ms=6,
                mew=1.6, zorder=5)

        ax.set_title(f"{C.PRETTY[dataset]}   "
                     f"(prevalence {np.mean(ytr):.2f}$\\rightarrow${y_ood.mean():.2f})")
        ax.set_xlabel("classification threshold")
        ax.set_xlim(0, 1)
        ax.legend(loc="lower left", frameon=False)

    axes[0].set_ylabel("OOD $F_1$")
    fig.tight_layout()
    save(fig, "fig_threshold")


# ---------------------------------------------------------------------------
# Figure: prior-only competitiveness
# ---------------------------------------------------------------------------
def fig_prior_only():
    df = pd.read_csv(OUT / "exp1_prior_only.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    xs, xt = [], []
    for i, ds in enumerate(DS_ORDER):
        g = df[df["dataset"] == ds]
        v = g["pct_vs_baseline"].clip(lower=-30).values
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.30
        ax.plot(i + jit, v, "o", ms=4, color=SKY, alpha=0.75,
                markeredgecolor=GREY, mew=0.4,
                label="individual LLM variant" if i == 0 else None)
        dep = g[g["is_deployed_prior_mean"]]
        ax.plot([i], dep["pct_vs_baseline"].clip(lower=-30).values, "D", ms=7,
                color=BLUISH_GREEN, markeredgecolor="black", mew=0.7,
                label="deployed variant" if i == 0 else None, zorder=5)
        xs.append(i)
        xt.append(ds)
        clipped = int((g["pct_vs_baseline"] < -30).sum())
        if clipped:
            ax.text(i, -31.5, f"{clipped} below\naxis", ha="center", fontsize=6.5,
                    color=VERMILLION)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylim(-34, 18)
    ax.set_xticks(xs)
    ax.set_xticklabels(xt)
    ax.set_ylabel("OOD lift over fitted baseline (%)")
    ax.set_title("Prior-only ($\\lambda\\rightarrow\\infty$): the elicited coefficients "
                 "with no data fitting", pad=8)
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.5, -0.06,
             "Each point is one of the twelve elicited LLM variants, used as the "
             "coefficient vector with only the intercept fitted. Axis clipped at "
             "$-30\\%$; counts of off-axis points annotated.",
             ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_prior_only")


if __name__ == "__main__":
    print("building figures")
    fig_resplits()
    fig_threshold()
    fig_placebo()
