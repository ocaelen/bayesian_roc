"""
Plot the three Dirichlet prior constructions from the manuscript.

For each prior, the script creates:

* a bar chart of the prior mean bin probabilities q_k^+ and q_k^-;
* the corresponding prior mean discretized ROC curve E[R_k].

One invocation generates all six PDF figures in ``py/figures``.
"""

from __future__ import annotations

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

from roc_utils.prior_profiles import (
    auc_calibrated_prior_bin_probabilities,
    expected_roc_from_bin_probabilities,
    increasing_decreasing_prior_bin_probabilities,
    symmetric_prior_bin_probabilities,
)


NUMBER_OF_BINS = 10

# For K=10, this is exactly halfway between the symmetric prior (AUC 0.50)
# and the fixed increasing/decreasing prior (AUC 0.80).
CALIBRATED_TARGET_AUC = 0.65

NEGATIVE_COLOR = "#4C78A8"
POSITIVE_COLOR = "#F58518"
ROC_COLOR = "#0072B2"
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


def build_prior_cases(K: int, calibrated_target_auc: float):
    """Construct and validate the three prior examples plotted by the script."""
    symmetric = symmetric_prior_bin_probabilities(K)
    increasing_decreasing = increasing_decreasing_prior_bin_probabilities(K)
    calibrated = auc_calibrated_prior_bin_probabilities(K, calibrated_target_auc)

    specifications = (
        (
            "Symmetric Dirichlet prior",
            "symmetric_dirichlet_prior",
            symmetric,
            0.5,
            r"a_{\mathcal{K}}",
        ),
        (
            "Increasing/decreasing priors",
            "increasing_decreasing_prior",
            increasing_decreasing,
            5 / 6 - 1 / (3 * K),
            r"a_{\mathcal{K}}",
        ),
        (
            "AUC-calibrated linear prior",
            "auc_calibrated_linear_prior",
            calibrated,
            calibrated_target_auc,
            r"a_0",
        ),
    )

    cases = []
    for (
        name,
        filename_prefix,
        (k, q_plus, q_minus),
        expected_auc,
        auc_symbol,
    ) in specifications:
        fpr, tpr, calculated_auc = expected_roc_from_bin_probabilities(
            q_plus,
            q_minus,
        )
        if not np.isclose(calculated_auc, expected_auc):
            raise RuntimeError(f"The expected AUC formula failed for {name}.")

        cases.append(
            {
                "name": name,
                "filename_prefix": filename_prefix,
                "k": k,
                "q_plus": q_plus,
                "q_minus": q_minus,
                "fpr": fpr,
                "tpr": tpr,
                "expected_auc": expected_auc,
                "auc_symbol": auc_symbol,
            }
        )

    assert_prior_cases_match_manuscript(
        cases,
        K=K,
        calibrated_target_auc=calibrated_target_auc,
    )
    return cases


def assert_prior_cases_match_manuscript(cases, K, calibrated_target_auc):
    """Assert that every plotted quantity equals its manuscript closed form."""
    assert len(cases) == 3, "Exactly three prior cases must be constructed."

    k = np.arange(1, K + 1, dtype=float)
    plot_index = np.arange(K + 1, dtype=float)

    symmetric_q = np.full(K, 1 / K, dtype=float)
    increasing_q_plus = 2 * k / (K * (K + 1))
    increasing_q_minus = 2 * (K - k + 1) / (K * (K + 1))

    assert 0.5 <= calibrated_target_auc < (5 * K + 2) / (6 * K), (
        "The calibrated target AUC must satisfy the manuscript feasibility bound."
    )
    calibrated_slope = (
        6 * (calibrated_target_auc - 0.5) / (K**2 - 1)
    )
    centered_k = k - (K + 1) / 2
    calibrated_q_plus = 1 / K + calibrated_slope * centered_k
    calibrated_q_minus = 1 / K - calibrated_slope * centered_k

    symmetric_fpr = plot_index / K
    symmetric_tpr = symmetric_fpr

    increasing_fpr = plot_index * (plot_index + 1) / (K * (K + 1))
    increasing_tpr = (
        plot_index * (2 * K - plot_index + 1) / (K * (K + 1))
    )

    calibrated_correction = (
        3
        * (2 * calibrated_target_auc - 1)
        * plot_index
        * (K - plot_index)
        / (2 * (K**2 - 1))
    )
    calibrated_fpr = plot_index / K - calibrated_correction
    calibrated_tpr = plot_index / K + calibrated_correction

    expected = {
        "symmetric_dirichlet_prior": (
            symmetric_q,
            symmetric_q,
            symmetric_fpr,
            symmetric_tpr,
            0.5,
        ),
        "increasing_decreasing_prior": (
            increasing_q_plus,
            increasing_q_minus,
            increasing_fpr,
            increasing_tpr,
            5 / 6 - 1 / (3 * K),
        ),
        "auc_calibrated_linear_prior": (
            calibrated_q_plus,
            calibrated_q_minus,
            calibrated_fpr,
            calibrated_tpr,
            calibrated_target_auc,
        ),
    }
    assert {case["filename_prefix"] for case in cases} == set(expected), (
        "The three requested prior cases are not uniquely represented."
    )

    for case in cases:
        name = case["name"]
        q_plus, q_minus, fpr, tpr, expected_auc = expected[
            case["filename_prefix"]
        ]
        assert np.allclose(
            case["q_plus"],
            q_plus,
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"{name}: q_plus does not match the manuscript formula."
        )
        assert np.allclose(
            case["q_minus"],
            q_minus,
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"{name}: q_minus does not match the manuscript formula."
        )
        assert np.allclose(
            case["fpr"],
            fpr,
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"{name}: FPR does not match the manuscript formula."
        )
        assert np.allclose(
            case["tpr"],
            tpr,
            rtol=1e-12,
            atol=1e-12,
        ), (
            f"{name}: TPR does not match the manuscript formula."
        )

        curve_auc = np.trapezoid(case["tpr"], case["fpr"])
        assert np.isclose(case["expected_auc"], expected_auc), (
            f"{name}: the stored expected AUC does not match the manuscript."
        )
        assert np.isclose(curve_auc, expected_auc), (
            f"{name}: the trapezoidal ROC area does not match the expected AUC."
        )

    if K == 10 and np.isclose(calibrated_target_auc, 0.65):
        symmetric, increasing_decreasing, calibrated = cases
        assert np.allclose(
            calibrated["q_plus"],
            0.5 * (symmetric["q_plus"] + increasing_decreasing["q_plus"]),
        ), "The calibrated q_plus profile must be the educational midpoint."
        assert np.allclose(
            calibrated["q_minus"],
            0.5 * (symmetric["q_minus"] + increasing_decreasing["q_minus"]),
        ), "The calibrated q_minus profile must be the educational midpoint."
        assert np.allclose(
            calibrated["fpr"],
            0.5 * (symmetric["fpr"] + increasing_decreasing["fpr"]),
        ), "The calibrated FPR coordinates must be the educational midpoint."
        assert np.allclose(
            calibrated["tpr"],
            0.5 * (symmetric["tpr"] + increasing_decreasing["tpr"]),
        ), "The calibrated TPR coordinates must be the educational midpoint."


def assert_pdf_created(output_path):
    """Assert that Matplotlib created a nonempty PDF at the requested path."""
    output_path = Path(output_path)
    assert output_path.is_file(), f"Expected PDF was not created: {output_path}"
    pdf_bytes = output_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-"), f"Invalid PDF header: {output_path}"
    assert b"%%EOF" in pdf_bytes[-1024:], f"Incomplete PDF trailer: {output_path}"


def style_axes(ax, horizontal_grid=False):
    """Apply a restrained publication style to a Matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)
    ax.grid(False)
    if horizontal_grid:
        ax.yaxis.grid(color="0.92", linewidth=0.45)
    ax.set_axisbelow(True)


def plot_bin_probabilities(k, q_plus, q_minus, y_max, y_ticks, output_path):
    """Plot q_k^+ and q_k^- as grouped bars on a shared vertical scale."""
    bar_width = 0.34
    bar_offset = 0.19
    fig, ax = plt.subplots(figsize=(3.85, 2.55))

    ax.bar(
        k - bar_offset,
        q_minus,
        width=bar_width,
        color=NEGATIVE_COLOR,
        label=r"Negative ($q_k^-$)",
        zorder=2,
    )
    ax.bar(
        k + bar_offset,
        q_plus,
        width=bar_width,
        color=POSITIVE_COLOR,
        label=r"Positive ($q_k^+$)",
        zorder=2,
    )

    ax.set_xlabel(r"Bin $k$")
    ax.set_ylabel("Prior mean probability")
    ax.set_xticks(k)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(0.4, k[-1] + 0.6)
    ax.set_ylim(0, y_max)
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


def plot_expected_roc(fpr, tpr, expected_auc, auc_symbol, output_path):
    """Plot one prior mean discretized ROC curve."""
    fig, ax = plt.subplots(figsize=(2.55, 2.55))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (3.5, 2.5)),
        linewidth=0.85,
        color=CHANCE_COLOR,
        zorder=1,
    )
    mean_is_chance = np.allclose(fpr, tpr)
    ax.plot(
        fpr,
        tpr,
        color=ROC_COLOR,
        linewidth=1.55,
        linestyle="none" if mean_is_chance else "-",
        marker="o",
        markersize=3.4,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )
    ax.text(
        0.97,
        0.055,
        rf"${auc_symbol}={expected_auc:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
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

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    assert_pdf_created(output_path)


def main():
    cases = build_prior_cases(
        K=NUMBER_OF_BINS,
        calibrated_target_auc=CALIBRATED_TARGET_AUC,
    )
    monotone_auc = 5 / 6 - 1 / (3 * NUMBER_OF_BINS)
    assert np.isclose(
        CALIBRATED_TARGET_AUC,
        (0.5 + monotone_auc) / 2,
        rtol=0.0,
        atol=1e-12,
    ), "The calibrated example must remain the midpoint of the first two AUCs."

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)

    # A common scale makes the three probability plots directly comparable.
    maximum_probability = max(
        max(case["q_plus"].max(), case["q_minus"].max()) for case in cases
    )
    y_tick_step = 0.05
    y_max = y_tick_step * np.ceil(maximum_probability / y_tick_step)
    if y_max - maximum_probability < 0.2 * y_tick_step:
        y_max += y_tick_step
    assert y_max > maximum_probability, (
        "The tallest probability bar must remain below the vertical axis limit."
    )
    y_ticks = np.arange(0, y_max + 0.5 * y_tick_step, y_tick_step)

    generated_paths = []
    for case in cases:
        probability_path = (
            figure_dir / f"{case['filename_prefix']}_bin_probabilities.pdf"
        )
        roc_path = figure_dir / f"{case['filename_prefix']}_expected_roc.pdf"
        generated_paths.extend([probability_path, roc_path])

        plot_bin_probabilities(
            k=case["k"],
            q_plus=case["q_plus"],
            q_minus=case["q_minus"],
            y_max=y_max,
            y_ticks=y_ticks,
            output_path=probability_path,
        )
        plot_expected_roc(
            fpr=case["fpr"],
            tpr=case["tpr"],
            expected_auc=case["expected_auc"],
            auc_symbol=case["auc_symbol"],
            output_path=roc_path,
        )

        print(f"{case['name']}: expected AUC = {case['expected_auc']:.3f}")
        print(f"Saved {probability_path}")
        print(f"Saved {roc_path}")

    assert len(generated_paths) == 6, "Exactly six figure paths must be generated."
    assert len(set(generated_paths)) == 6, "All six output paths must be distinct."
    assert all(path.is_file() for path in generated_paths), (
        "At least one expected figure file is missing."
    )
    print("All runtime assertions passed.")


if __name__ == "__main__":
    main()
