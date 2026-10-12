"""
Generate one simulated test/external score dataset from ``main.tex``.

The function in this module generates raw class-specific scores only. Grid
construction, histogramming, priors, posterior sampling, and evaluation all
belong to later stages of the experimental pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if not __debug__:
    raise RuntimeError(
        "simulation_datasets relies on runtime assertions; do not use python -O."
    )


SYNTHETIC_DESIGN_VERSION = 3

_BETA_PARAMETERS = {
    "Easy": ((3.75, 2.25), (2.25, 3.75)),
    "Medium": ((3.5, 2.5), (2.5, 3.5)),
    "Hard": ((3.25, 2.75), (2.75, 3.25)),
}
SIMULATION_SCENARIOS = tuple(_BETA_PARAMETERS)

assert SIMULATION_SCENARIOS == ("Easy", "Medium", "Hard")


@dataclass(frozen=True)
class SimulationDataset:
    """The four class-specific score samples from one replication."""

    positive_test_scores: np.ndarray
    negative_test_scores: np.ndarray
    positive_external_scores: np.ndarray
    negative_external_scores: np.ndarray


def simulation_beta_parameters(
    scenario: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the positive- and negative-class Beta shapes for a scenario."""

    assert isinstance(scenario, str)
    assert scenario in _BETA_PARAMETERS
    positive_parameters, negative_parameters = _BETA_PARAMETERS[scenario]
    assert all(parameter > 0.0 for parameter in positive_parameters)
    assert all(parameter > 0.0 for parameter in negative_parameters)
    return positive_parameters, negative_parameters


def generate_simulation_dataset(
    scenario: str,
    n_positive: int,
    n_negative: int,
    *,
    rng: np.random.Generator,
) -> SimulationDataset:
    """
    Generate one simulated test/external dataset from the manuscript.

    ``scenario`` must be exactly ``"Easy"``, ``"Medium"``, or ``"Hard"``.
    The same positive and negative counts are used for the test and external
    samples, as required by ``main.tex``. Across replications, the caller must
    use either one advancing generator or independent, non-overlapping random
    streams; it must never reuse the same initial generator state.

    For reproducibility, the four independent draws always occur in this
    order: positive test, negative test, positive external, negative external.
    """

    positive_parameters, negative_parameters = simulation_beta_parameters(
        scenario
    )
    assert isinstance(n_positive, (int, np.integer))
    assert not isinstance(n_positive, (bool, np.bool_))
    assert n_positive > 0
    assert isinstance(n_negative, (int, np.integer))
    assert not isinstance(n_negative, (bool, np.bool_))
    assert n_negative > 0
    assert callable(getattr(rng, "beta", None))

    positive_test_scores = np.asarray(
        rng.beta(*positive_parameters, size=n_positive),
        dtype=float,
    )
    negative_test_scores = np.asarray(
        rng.beta(*negative_parameters, size=n_negative),
        dtype=float,
    )
    positive_external_scores = np.asarray(
        rng.beta(*positive_parameters, size=n_positive),
        dtype=float,
    )
    negative_external_scores = np.asarray(
        rng.beta(*negative_parameters, size=n_negative),
        dtype=float,
    )

    expected_shapes = (
        (n_positive,),
        (n_negative,),
        (n_positive,),
        (n_negative,),
    )
    score_samples = (
        positive_test_scores,
        negative_test_scores,
        positive_external_scores,
        negative_external_scores,
    )
    for scores, expected_shape in zip(score_samples, expected_shapes):
        assert scores.shape == expected_shape
        assert np.all(np.isfinite(scores))
        assert np.all((0.0 <= scores) & (scores <= 1.0))

    assert positive_test_scores.size == positive_external_scores.size
    assert negative_test_scores.size == negative_external_scores.size

    return SimulationDataset(
        positive_test_scores=positive_test_scores,
        negative_test_scores=negative_test_scores,
        positive_external_scores=positive_external_scores,
        negative_external_scores=negative_external_scores,
    )
