"""Stratified empirical bootstrap on the manuscript's fixed score grid.

The only class probabilities are the observed frequencies d/n. There is no
prior, smoothing, or additional information about the score distributions.
"""

from __future__ import annotations

import numpy as np

from .posterior_predictive_roc_sampler import _check_histograms_and_priors

if not __debug__:
    raise RuntimeError(
        "bootstrap_predictive_roc_sampler relies on assertions; do not use python -O."
    )

PREDICTIVE_CORRECTION = np.sqrt(2.0)


def sample_stratified_bootstrap_histograms(d_plus, d_minus, S, *, rng):
    """Draw S classwise bootstrap histograms, each retaining its class count.

    Multinomial(n, d/n) is exactly the distribution of histogram counts after
    uniformly resampling the observed scores with replacement within a class.
    Returned arrays both have shape (S, K).
    """
    d_plus, d_minus = _check_histograms_and_priors(d_plus, d_minus)
    if not isinstance(S, (int, np.integer)) or isinstance(S, (bool, np.bool_)):
        raise TypeError("S must be an integer.")
    if S <= 0:
        raise ValueError("S must be positive.")
    assert isinstance(rng, np.random.Generator), "An explicit NumPy RNG is required."

    samples = []
    for observed in (d_plus, d_minus):
        n = int(observed.sum())
        probabilities = observed / n
        counts = rng.multinomial(n, probabilities, size=int(S))
        assert counts.shape == (int(S), observed.size)
        assert np.issubdtype(counts.dtype, np.integer)
        assert np.all(counts >= 0)
        assert np.all(counts.sum(axis=1) == n)
        assert np.all(counts[:, observed == 0] == 0), "Empty bins must remain empty."
        samples.append(counts)
    return tuple(samples)
