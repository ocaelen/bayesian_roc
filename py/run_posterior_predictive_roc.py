"""
Runnable figure script for posterior predictive ROC curves.

The code follows Algorithm alg:posterior-predictive-roc.

Python uses 0-based indices.
The paper uses bins k = 1,...,K.

Code notation:
    d_plus[k]    corresponds to D_{k+1}^+
    d_minus[k]   corresponds to D_{k+1}^-
    alpha_plus   corresponds to alpha^+
    alpha_minus  corresponds to alpha^-
    omega_plus   = alpha_plus + d_plus
    omega_minus  = alpha_minus + d_minus
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "bayesian_roc_matplotlib"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from roc_utils.posterior_predictive_roc_sampler import (
    fixed_uniform_grid,
    histograms_from_scores,
    linear_informative_priors,
    roc_from_histograms,
    sample_posterior_predictive_rocs,
    symmetric_dirichlet_priors,
)


NEGATIVE_COLOR = "#4C78A8"
POSITIVE_COLOR = "#F58518"
OBSERVED_COLOR = "#0072B2"
SAMPLE_COLOR = "0.15"
CHANCE_COLOR = "0.62"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def assert_pdf_created(output_path):
    """Assert that Matplotlib created a nonempty PDF at the requested path."""
    output_path = Path(output_path)
    assert output_path.is_file(), f"Expected PDF was not created: {output_path}"
    pdf_bytes = output_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-"), f"Invalid PDF header: {output_path}"
    assert b"%%EOF" in pdf_bytes[-1024:], f"Incomplete PDF trailer: {output_path}"


def style_axes(ax, horizontal_grid=False):
    """Apply the restrained publication style used by the prior figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)
    ax.grid(False)
    if horizontal_grid:
        ax.yaxis.grid(color="0.92", linewidth=0.45)
    ax.set_axisbelow(True)


def make_synthetic_scores(seed=13):
    """
    Create a simple binary classification problem.

    Replace this function by your own y_test and scores if you already have
    a trained classifier.
    """
    X, y = make_classification(
        n_samples=2500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.65, 0.35],
        class_sep=0.90,
        flip_y=0.06,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.45,
        stratify=y,
        random_state=seed,
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )

    classifier.fit(X_train, y_train)

    scores = classifier.predict_proba(X_test)[:, 1]

    return y_test, scores


def plot_score_histograms(gamma, d_plus, d_minus, output_path):
    """
    Plot observed class-conditional score histograms.

    The heights are

        d_k^+ / n_+
        d_k^- / n_-,

    so they estimate lambda_k^+ and lambda_k^-.
    """
    centers = 0.5 * (gamma[:-1] + gamma[1:])
    widths = np.diff(gamma)

    n_plus = d_plus.sum()
    n_minus = d_minus.sum()

    fig, ax = plt.subplots(figsize=(3.85, 2.55))

    negative_proportions = d_minus / n_minus
    positive_proportions = d_plus / n_plus
    bar_widths = 0.34 * widths
    bar_offsets = 0.19 * widths

    ax.bar(
        centers - bar_offsets,
        negative_proportions,
        width=bar_widths,
        color=NEGATIVE_COLOR,
        align="center",
        label="Negative class",
        zorder=2,
    )

    ax.bar(
        centers + bar_offsets,
        positive_proportions,
        width=bar_widths,
        color=POSITIVE_COLOR,
        align="center",
        label="Positive class",
        zorder=2,
    )

    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Class-conditional bin proportion")
    score_ticks = np.linspace(0.0, 1.0, 6)
    score_tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    ax.set_xticks(score_ticks, score_tick_labels)
    ax.set_xlim(0.0, 1.0)
    maximum_proportion = max(
        negative_proportions.max(),
        positive_proportions.max(),
    )
    y_tick_step = 0.05
    y_max = y_tick_step * np.ceil(maximum_proportion / y_tick_step)
    if y_max - maximum_proportion < 0.2 * y_tick_step:
        y_max += y_tick_step
    y_ticks = np.arange(0.0, y_max + 0.5 * y_tick_step, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(0.0, y_max)
    style_axes(ax, horizontal_grid=True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.4,
        borderaxespad=0,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    assert_pdf_created(output_path)


def plot_posterior_predictive_roc(
    fpr_observed,
    tpr_observed,
    fpr_samples,
    tpr_samples,
    output_path,
):
    """
    Plot the observed binned ROC curve and S posterior predictive ROC curves.

    The arrays are already ordered as R_K, R_{K-1}, ..., R_0, hence from
    (0,0) to (1,1).
    """
    fig, ax = plt.subplots(figsize=(2.55, 2.55))

    S = fpr_samples.shape[0]

    for s in range(S):
        ax.plot(
            fpr_samples[s],
            tpr_samples[s],
            linewidth=0.55,
            alpha=0.035,
            color=SAMPLE_COLOR,
            rasterized=True,
            zorder=1,
        )

    ax.plot(
        [],
        [],
        linewidth=1.0,
        color="0.55",
        label="Predictive samples",
    )

    observed_auc = np.trapezoid(
        tpr_observed,
        fpr_observed,
    )

    ax.plot(
        fpr_observed,
        tpr_observed,
        color=OBSERVED_COLOR,
        linewidth=1.55,
        marker="o",
        markersize=3.4,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Observed ROC",
        zorder=3,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (3.5, 2.5)),
        linewidth=0.85,
        color=CHANCE_COLOR,
        zorder=2,
    )

    ax.text(
        0.97,
        0.055,
        rf"$\mathrm{{AUC}}_{{\mathrm{{obs}}}}={observed_auc:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
    )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    roc_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    roc_tick_labels = ["0", "0.25", "0.50", "0.75", "1"]
    ax.set_xticks(roc_ticks, roc_tick_labels)
    ax.set_yticks(roc_ticks, roc_tick_labels)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.15),
        frameon=False,
        handlelength=1.25,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0,
        fontsize=8.0,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    assert_pdf_created(output_path)


def main():
    # ------------------------------------------------------------
    # 1. Synthetic example.
    #    Replace this block by your own y_test and scores.
    # ------------------------------------------------------------
    y_test, scores = make_synthetic_scores(seed=13)

    # ------------------------------------------------------------
    # 2. Fixed grid and observed histograms.
    # ------------------------------------------------------------
    K = 20
    gamma = fixed_uniform_grid(K)

    d_plus, d_minus = histograms_from_scores(
        y_true=y_test,
        scores=scores,
        gamma=gamma,
    )

    fpr_observed, tpr_observed = roc_from_histograms(
        d_plus=d_plus,
        d_minus=d_minus,
    )

    # ------------------------------------------------------------
    # 3. Prior.
    # ------------------------------------------------------------
    prior_mass = K

    alpha_plus, alpha_minus = symmetric_dirichlet_priors(
        K=K,
        prior_mass=prior_mass,
    )

    # Alternative prior from the draft:
    #
    # alpha_plus, alpha_minus = linear_informative_priors(
    #     K=K,
    #     prior_mass=prior_mass,
    # )

    # ------------------------------------------------------------
    # 4. Posterior predictive sampling.
    # ------------------------------------------------------------
    S = 500

    fpr_samples, tpr_samples = sample_posterior_predictive_rocs(
        d_plus=d_plus,
        d_minus=d_minus,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        S=S,
        seed=123,
    )

    # ------------------------------------------------------------
    # 5. Figures.
    # ------------------------------------------------------------
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)

    plot_score_histograms(
        gamma=gamma,
        d_plus=d_plus,
        d_minus=d_minus,
        output_path=figure_dir / "observed_score_histograms.pdf",
    )

    plot_posterior_predictive_roc(
        fpr_observed=fpr_observed,
        tpr_observed=tpr_observed,
        fpr_samples=fpr_samples,
        tpr_samples=tpr_samples,
        output_path=figure_dir / "posterior_predictive_roc_cloud.pdf",
    )


if __name__ == "__main__":
    main()
