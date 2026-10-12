"""
Reusable core utilities for the binned Bayesian ROC model.

The implementation follows Algorithm alg:posterior-predictive-roc in
``main.tex`` and keeps one indexing convention throughout:

* the paper uses bins ``k = 1,...,K`` and ROC points ``R_0,...,R_K``;
* Python arrays use bins ``0,...,K-1``;
* returned ROC arrays are ordered as ``R_K, R_{K-1}, ..., R_0`` so they
  trace the curve from ``(0, 0)`` to ``(1, 1)``.
"""

from __future__ import annotations

import math

import numpy as np


def _as_1d_float(name, values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_1d_counts(name, values):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError(f"{name} must contain integer counts.")
    array = array.astype(np.int64, copy=False)
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative counts.")
    return array


def _check_histograms_and_priors(d_plus, d_minus, alpha_plus=None, alpha_minus=None):
    d_plus = _as_1d_counts("d_plus", d_plus)
    d_minus = _as_1d_counts("d_minus", d_minus)

    if d_plus.size == 0:
        raise ValueError("At least one score bin is required.")
    if d_plus.shape != d_minus.shape:
        raise ValueError("d_plus and d_minus must have the same length.")
    if d_plus.sum() <= 0 or d_minus.sum() <= 0:
        raise ValueError("Both classes must be present in the test sample.")

    if alpha_plus is None and alpha_minus is None:
        return d_plus, d_minus
    if alpha_plus is None or alpha_minus is None:
        raise ValueError("Both alpha_plus and alpha_minus must be provided together.")

    alpha_plus = _as_1d_float("alpha_plus", alpha_plus)
    alpha_minus = _as_1d_float("alpha_minus", alpha_minus)

    if alpha_plus.shape != d_plus.shape or alpha_minus.shape != d_minus.shape:
        raise ValueError("Prior vectors must have the same length as the histograms.")
    if np.any(alpha_plus <= 0.0) or np.any(alpha_minus <= 0.0):
        raise ValueError("Dirichlet prior parameters must be strictly positive.")

    return d_plus, d_minus, alpha_plus, alpha_minus


def validate_grid(gamma):
    """Validate and return a fixed score grid ``gamma_0,...,gamma_K``."""
    gamma = _as_1d_float("gamma", gamma)
    if gamma.size < 2:
        raise ValueError("gamma must contain at least two grid points.")
    if not np.isclose(gamma[0], 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
        gamma[-1],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("gamma must start at 0 and end at 1.")
    if np.any(np.diff(gamma) <= 0.0):
        raise ValueError("gamma must be strictly increasing.")
    return gamma


def fixed_uniform_grid(K):
    """Return the deterministic grid ``gamma_0,...,gamma_K`` on ``[0, 1]``."""
    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer.")
    return np.linspace(0.0, 1.0, int(K) + 1)


def histogram_from_scores(scores, gamma):
    """
    Count scores in the paper's bins.

    The bins are ``[gamma_0, gamma_1), ..., [gamma_{K-2}, gamma_{K-1})``
    and ``[gamma_{K-1}, gamma_K]``. Scores must be inside ``[0, 1]``.
    """
    gamma = validate_grid(gamma)
    scores = _as_1d_float("scores", scores)
    if np.any(scores < gamma[0]) or np.any(scores > gamma[-1]):
        raise ValueError("scores must lie in the closed interval [0, 1].")

    K = gamma.size - 1
    bin_index = np.digitize(scores, bins=gamma[1:-1], right=False)
    return np.bincount(bin_index, minlength=K).astype(np.int64, copy=False)


def histograms_from_scores(y_true, scores, gamma):
    """Compute the observed class-conditional histograms ``d^+`` and ``d^-``."""
    y_true = np.asarray(y_true)
    scores = _as_1d_float("scores", scores)
    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array.")
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    return (
        histogram_from_scores(scores[y_true == 1], gamma),
        histogram_from_scores(scores[y_true == 0], gamma),
    )


def symmetric_dirichlet_priors(K, prior_mass):
    """
    Symmetric Dirichlet prior with total class-specific mass ``prior_mass``.

    This matches ``alpha_k^+ = alpha_k^- = m / K`` in the manuscript.
    """
    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer.")
    if not np.isfinite(prior_mass) or prior_mass <= 0.0:
        raise ValueError("prior_mass must be strictly positive.")

    alpha = np.full(int(K), float(prior_mass) / int(K), dtype=float)
    return alpha.copy(), alpha.copy()


def linear_informative_priors(K, prior_mass):
    """
    Informative prior from Section ``Adding prior knowledge``.

    ``alpha_k^+ = 2mk / (K(K+1))`` and
    ``alpha_k^- = 2m(K-k+1) / (K(K+1))`` for paper indices
    ``k=1,...,K``.
    """
    if int(K) != K or K <= 0:
        raise ValueError("K must be a positive integer.")
    if not np.isfinite(prior_mass) or prior_mass <= 0.0:
        raise ValueError("prior_mass must be strictly positive.")

    K = int(K)
    k = np.arange(1, K + 1, dtype=float)
    scale = 2.0 * float(prior_mass) / (K * (K + 1))
    return scale * k, scale * k[::-1]


def right_tail_sums(bin_values):
    """
    For bin values ``(a_1,...,a_K)``, return tail sums for ``k=0,...,K``:

    ``(sum_{j=1}^K a_j, sum_{j=2}^K a_j, ..., a_K, 0)``.
    """
    bin_values = _as_1d_float("bin_values", bin_values)
    return np.r_[np.cumsum(bin_values[::-1])[::-1], 0.0]


def _right_tail_sums_2d(histogram_samples):
    histogram_samples = np.asarray(histogram_samples)
    if histogram_samples.ndim != 2:
        raise ValueError("histogram_samples must be a two-dimensional array.")

    tails = np.cumsum(histogram_samples[:, ::-1], axis=1)[:, ::-1]
    zeros = np.zeros((histogram_samples.shape[0], 1), dtype=tails.dtype)
    return np.concatenate([tails, zeros], axis=1)


def roc_from_histograms(d_plus, d_minus):
    """
    Compute binned ROC coordinates from observed histograms ``d^+`` and ``d^-``.

    Returned arrays have length ``K+1`` and are ordered
    ``R_K, R_{K-1}, ..., R_0``.
    """
    d_plus, d_minus = _check_histograms_and_priors(d_plus, d_minus)

    tpr = right_tail_sums(d_plus) / d_plus.sum()
    fpr = right_tail_sums(d_minus) / d_minus.sum()

    return fpr[::-1], tpr[::-1]


def rocs_from_histogram_samples(d_plus_samples, d_minus_samples):
    """
    Convert sampled class-specific histograms to sampled binned ROC curves.

    ``d_plus_samples`` and ``d_minus_samples`` must have shape ``(S, K)``.
    Returned arrays have shape ``(S, K+1)`` and are ordered from ``(0, 0)``
    to ``(1, 1)``.
    """
    d_plus_samples = np.asarray(d_plus_samples)
    d_minus_samples = np.asarray(d_minus_samples)

    if d_plus_samples.ndim != 2 or d_minus_samples.ndim != 2:
        raise ValueError("Histogram samples must be two-dimensional arrays.")
    if d_plus_samples.shape != d_minus_samples.shape:
        raise ValueError("Positive and negative histogram samples must match.")
    if d_plus_samples.shape[0] == 0 or d_plus_samples.shape[1] == 0:
        raise ValueError("Histogram sample arrays must not be empty.")
    if not np.all(np.isfinite(d_plus_samples)) or not np.all(
        np.isfinite(d_minus_samples)
    ):
        raise ValueError("Histogram samples must contain only finite values.")
    if not np.all(d_plus_samples == np.floor(d_plus_samples)) or not np.all(
        d_minus_samples == np.floor(d_minus_samples)
    ):
        raise ValueError("Histogram samples must contain integer counts.")
    if np.any(d_plus_samples < 0) or np.any(d_minus_samples < 0):
        raise ValueError("Histogram samples must contain non-negative counts.")

    n_plus = d_plus_samples.sum(axis=1)
    n_minus = d_minus_samples.sum(axis=1)
    if np.any(n_plus <= 0) or np.any(n_minus <= 0):
        raise ValueError("Both sampled class counts must be positive.")

    tpr = _right_tail_sums_2d(d_plus_samples) / n_plus[:, None]
    fpr = _right_tail_sums_2d(d_minus_samples) / n_minus[:, None]

    return fpr[:, ::-1], tpr[:, ::-1]


def posterior_predictive_mean_roc(d_plus, d_minus, alpha_plus, alpha_minus):
    """
    Exact posterior predictive mean ROC points from the manuscript.

    Since ``E[D_tilde_j^ell | d^ell] / n_ell = omega_j^ell / omega_.^ell``,
    the mean curve is obtained by replacing histogram counts by posterior
    mean bin probabilities before taking right-tail sums.
    """
    d_plus, d_minus, alpha_plus, alpha_minus = _check_histograms_and_priors(
        d_plus,
        d_minus,
        alpha_plus,
        alpha_minus,
    )

    omega_plus = alpha_plus + d_plus
    omega_minus = alpha_minus + d_minus

    tpr = right_tail_sums(omega_plus / omega_plus.sum())
    fpr = right_tail_sums(omega_minus / omega_minus.sum())

    return fpr[::-1], tpr[::-1]


def _multinomial_rows_from_probabilities(rng, n, probabilities):
    """
    Draw one multinomial vector for each row of ``probabilities``.

    NumPy's multinomial sampler uses one common probability vector per call.
    The sequential-binomial construction below is distributionally equivalent
    and vectorizes across posterior draws.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer.")
    if np.any(probabilities < -1e-12):
        raise ValueError("probabilities must be non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each probability row must sum to 1.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    sample_count, K = probabilities.shape
    counts = np.empty((sample_count, K), dtype=np.int64)
    remaining_counts = np.full(sample_count, int(n), dtype=np.int64)
    remaining_probability = np.ones(sample_count, dtype=float)

    for k in range(K - 1):
        conditional_probability = np.divide(
            probabilities[:, k],
            remaining_probability,
            out=np.zeros(sample_count, dtype=float),
            where=remaining_probability > 0.0,
        )
        conditional_probability = np.clip(conditional_probability, 0.0, 1.0)

        draws = rng.binomial(remaining_counts, conditional_probability)
        counts[:, k] = draws
        remaining_counts -= draws
        remaining_probability -= probabilities[:, k]
        remaining_probability = np.maximum(remaining_probability, 0.0)

    counts[:, -1] = remaining_counts
    return counts


def sample_posterior_predictive_histograms(
    d_plus,
    d_minus,
    alpha_plus,
    alpha_minus,
    S,
    seed=123,
    rng=None,
):
    """
    Draw posterior predictive class-specific histograms.

    This implements lines 7--10 of Algorithm alg:posterior-predictive-roc.
    """
    d_plus, d_minus, alpha_plus, alpha_minus = _check_histograms_and_priors(
        d_plus,
        d_minus,
        alpha_plus,
        alpha_minus,
    )
    if int(S) != S or S <= 0:
        raise ValueError("S must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng(seed)

    omega_plus = alpha_plus + d_plus
    omega_minus = alpha_minus + d_minus

    lambda_plus = rng.dirichlet(omega_plus, size=int(S))
    lambda_minus = rng.dirichlet(omega_minus, size=int(S))

    d_plus_tilde = _multinomial_rows_from_probabilities(
        rng,
        int(d_plus.sum()),
        lambda_plus,
    )
    d_minus_tilde = _multinomial_rows_from_probabilities(
        rng,
        int(d_minus.sum()),
        lambda_minus,
    )

    for label, draws, observed in (
        ("positive", d_plus_tilde, d_plus),
        ("negative", d_minus_tilde, d_minus),
    ):
        assert draws.shape == (int(S), observed.size), (
            f"Posterior predictive {label} histogram shape is {draws.shape}; "
            f"expected {(int(S), observed.size)}."
        )
        assert np.issubdtype(draws.dtype, np.integer), (
            f"Posterior predictive {label} histograms must contain integer counts."
        )
        assert np.all(draws >= 0), (
            f"Posterior predictive {label} histogram counts must be non-negative."
        )
        assert np.all(draws.sum(axis=1) == observed.sum()), (
            f"Posterior predictive {label} histograms must preserve the observed "
            f"class count {int(observed.sum())}."
        )

    return d_plus_tilde, d_minus_tilde


def sample_posterior_predictive_rocs(
    d_plus,
    d_minus,
    alpha_plus,
    alpha_minus,
    S,
    seed=123,
    rng=None,
):
    """
    Generate posterior predictive ROC curves by Monte Carlo.

    Returned arrays have shape ``(S, K+1)`` and columns ordered as
    ``R_K, R_{K-1}, ..., R_0``.
    """
    d_plus_tilde, d_minus_tilde = sample_posterior_predictive_histograms(
        d_plus=d_plus,
        d_minus=d_minus,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        S=S,
        seed=seed,
        rng=rng,
    )
    return rocs_from_histogram_samples(d_plus_tilde, d_minus_tilde)


def max_euclidean_deviations(fpr_samples, tpr_samples, fpr_center, tpr_center):
    """Return ``q_Euc`` for each sampled ROC curve."""
    fpr_samples = np.asarray(fpr_samples, dtype=float)
    tpr_samples = np.asarray(tpr_samples, dtype=float)
    fpr_center = _as_1d_float("fpr_center", fpr_center)
    tpr_center = _as_1d_float("tpr_center", tpr_center)

    if fpr_samples.shape != tpr_samples.shape:
        raise ValueError("fpr_samples and tpr_samples must have the same shape.")
    if fpr_samples.ndim != 2:
        raise ValueError("Sample arrays must be two-dimensional.")
    if fpr_samples.shape[0] == 0 or fpr_samples.shape[1] == 0:
        raise ValueError("Sampled ROC arrays must not be empty.")
    if fpr_samples.shape[1] != fpr_center.size or tpr_center.shape != fpr_center.shape:
        raise ValueError("Center arrays must match the sampled ROC length.")
    if not np.all(np.isfinite(fpr_samples)) or not np.all(np.isfinite(tpr_samples)):
        raise ValueError("Sampled ROC arrays must contain only finite values.")

    distances = np.hypot(fpr_samples - fpr_center[None, :], tpr_samples - tpr_center[None, :])
    return distances.max(axis=1)


def nearest_rank_quantile(values, probability):
    """
    Empirical nearest-rank quantile used by the manuscript algorithms.

    For ``S`` sorted values this returns rank ``ceil(probability * S)`` with
    one-based ranks, converted to a zero-based Python index.
    """
    values = _as_1d_float("values", values)
    if values.size == 0:
        raise ValueError("values must not be empty.")
    probability = float(probability)
    if not np.isfinite(probability):
        raise ValueError("probability must be finite.")
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("probability must lie in (0, 1].")

    scaled_rank = probability * values.size
    nearest_integer = round(scaled_rank)
    floating_tolerance = 8.0 * np.finfo(float).eps
    if math.isclose(
        scaled_rank,
        nearest_integer,
        rel_tol=floating_tolerance,
        abs_tol=floating_tolerance,
    ):
        scaled_rank = float(nearest_integer)

    index = max(0, min(values.size - 1, math.ceil(scaled_rank) - 1))
    return np.sort(values)[index]


def simultaneous_euclidean_radii(
    fpr_samples,
    tpr_samples,
    fpr_center,
    tpr_center,
    levels,
):
    """Compute simultaneous Euclidean credible-band radii for each level."""
    q_values = max_euclidean_deviations(
        fpr_samples,
        tpr_samples,
        fpr_center,
        tpr_center,
    )
    return {
        float(level): nearest_rank_quantile(q_values, float(level))
        for level in levels
    }


def curve_max_euclidean_deviation(fpr, tpr, fpr_center, tpr_center):
    """Maximum pointwise Euclidean deviation of one ROC curve from a center."""
    fpr = _as_1d_float("fpr", fpr)
    tpr = _as_1d_float("tpr", tpr)
    fpr_center = _as_1d_float("fpr_center", fpr_center)
    tpr_center = _as_1d_float("tpr_center", tpr_center)
    if fpr.shape != tpr.shape or fpr.shape != fpr_center.shape or fpr.shape != tpr_center.shape:
        raise ValueError("All ROC arrays must have the same shape.")
    if fpr.size == 0:
        raise ValueError("ROC arrays must not be empty.")
    return float(np.hypot(fpr - fpr_center, tpr - tpr_center).max())


def grid_auc(fpr, tpr):
    """
    Trapezoidal AUC for ROC arrays ordered from ``(0, 0)`` to ``(1, 1)``.

    The function accepts either one curve of shape ``(K+1,)`` or many curves
    of shape ``(S, K+1)``.

    The exact area is in [0, 1]. Floating-point summation can exceed a
    boundary by a few ulps even for a valid ROC. Check the computed area
    before correcting only those numerical excursions; values already
    inside [0, 1] are unchanged.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape.")
    if fpr.ndim not in (1, 2):
        raise ValueError("fpr and tpr must be one- or two-dimensional arrays.")
    if fpr.shape[-1] < 2 or (fpr.ndim == 2 and fpr.shape[0] == 0):
        raise ValueError("At least one ROC curve with two points is required.")
    if not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
        raise ValueError("fpr and tpr must contain only finite values.")
    if np.any(fpr < -1e-12) or np.any(fpr > 1.0 + 1e-12):
        raise ValueError("fpr values must lie in [0, 1].")
    if np.any(tpr < -1e-12) or np.any(tpr > 1.0 + 1e-12):
        raise ValueError("tpr values must lie in [0, 1].")
    if np.any(np.diff(fpr, axis=-1) < -1e-12):
        raise ValueError("fpr values must be ordered from 0 to 1.")
    if np.any(np.diff(tpr, axis=-1) < -1e-12):
        raise ValueError("tpr values must be ordered from 0 to 1.")
    if not np.allclose(fpr[..., 0], 0.0, rtol=0.0, atol=1e-12) or not np.allclose(
        tpr[..., 0],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ROC curves must start at (0, 0).")
    if not np.allclose(fpr[..., -1], 1.0, rtol=0.0, atol=1e-12) or not np.allclose(
        tpr[..., -1],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ROC curves must end at (1, 1).")
    auc = np.trapezoid(tpr, fpr, axis=-1)
    tolerance = 1e-12
    if (
        not np.all(np.isfinite(auc))
        or np.any(auc < -tolerance)
        or np.any(auc > 1.0 + tolerance)
    ):
        raise ValueError(
            "Trapezoidal AUC is non-finite or outside [0, 1] beyond numerical "
            f"tolerance {tolerance}: min={float(np.min(auc))!r}, "
            f"max={float(np.max(auc))!r}, ROC shape={fpr.shape}."
        )
    return np.clip(auc, 0.0, 1.0)


def equal_tailed_interval(values, level):
    """Nearest-rank equal-tailed interval used for grid-based AUC samples."""
    if level <= 0.0 or level >= 1.0:
        raise ValueError("level must lie in (0, 1).")
    delta = 1.0 - float(level)
    return (
        nearest_rank_quantile(values, delta / 2.0),
        nearest_rank_quantile(values, 1.0 - delta / 2.0),
    )
