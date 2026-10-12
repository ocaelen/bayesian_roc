"""
Plot the class-conditional score distributions for the simulation scenarios.

One invocation creates the Easy, Medium, and Hard figures in ``py/figures``.
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
from scipy.stats import beta

from roc_utils.simulation_datasets import (
    SIMULATION_SCENARIOS,
    simulation_beta_parameters,
)


NEGATIVE_COLOR = "#4C78A8"
POSITIVE_COLOR = "#F58518"

_SCENARIO_FILENAMES = {
    "Easy": "simulation_easy_score_distributions.pdf",
    "Medium": "simulation_medium_score_distributions.pdf",
    "Hard": "simulation_hard_score_distributions.pdf",
}
SCENARIOS = tuple(
    (
        scenario,
        *simulation_beta_parameters(scenario),
        _SCENARIO_FILENAMES[scenario],
    )
    for scenario in SIMULATION_SCENARIOS
)

assert tuple(scenario for scenario, *_ in SCENARIOS) == SIMULATION_SCENARIOS


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


def style_axes(ax):
    """Apply the restrained publication style used by the other figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2.5)
    ax.grid(False)


def plot_score_distributions(
    positive_shape,
    negative_shape,
    output_path,
    show_y_labels,
    show_x_label,
    show_legend,
):
    """Plot one panel of the coordinated three-panel figure."""
    scores = np.linspace(0.0, 1.0, 1001)

    fig, ax = plt.subplots(figsize=(2.15, 2.55))

    ax.plot(
        scores,
        beta.pdf(scores, *negative_shape),
        color=NEGATIVE_COLOR,
        linewidth=1.55,
        label=r"$Y=0$",
    )
    ax.plot(
        scores,
        beta.pdf(scores, *positive_shape),
        color=POSITIVE_COLOR,
        linewidth=1.55,
        label=r"$Y=1$",
    )

    score_ticks = np.linspace(0.0, 1.0, 6)
    score_tick_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    density_ticks = np.arange(0.0, 3.01, 0.5)
    density_tick_labels = ["0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]

    ax.set_xticks(score_ticks, score_tick_labels)
    ax.set_yticks(density_ticks, density_tick_labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 3.0)
    if show_y_labels:
        ax.set_ylabel("Density")
    else:
        ax.tick_params(axis="y", labelleft=False)
    if show_x_label:
        ax.set_xlabel(r"Score $w$")
    style_axes(ax)
    if show_legend:
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

    # Fixed margins keep all three PDF page boxes and plotting areas identical.
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.18, top=0.82)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)

    center_panel = len(SCENARIOS) // 2
    for panel, (scenario, positive_shape, negative_shape, filename) in enumerate(
        SCENARIOS
    ):
        output_path = figure_dir / filename
        plot_score_distributions(
            positive_shape=positive_shape,
            negative_shape=negative_shape,
            output_path=output_path,
            show_y_labels=panel == 0,
            show_x_label=panel == center_panel,
            show_legend=panel == center_panel,
        )
        print(f"{scenario}: saved {output_path}")


if __name__ == "__main__":
    main()
