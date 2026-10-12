"""Pure prior-profile calculations shared by figures and experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_distribution

if not __debug__:
    raise RuntimeError(
        "prior_profiles relies on runtime assertions; do not use python -O."
    )


def bin_indices(K: int) -> tuple[int, np.ndarray]:
    """Validate ``K`` and return the manuscript bin indices ``1,...,K``."""

    if (
        not isinstance(K, (int, np.integer))
        or isinstance(K, (bool, np.bool_))
        or K < 2
    ):
        raise ValueError("K must be an integer greater than or equal to 2.")
    K = int(K)
    return K, np.arange(1, K + 1, dtype=float)


def validate_prior_mean_probabilities(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two strictly positive class-specific probability vectors."""

    q_plus = np.asarray(q_plus, dtype=float)
    q_minus = np.asarray(q_minus, dtype=float)
    if q_plus.ndim != 1 or q_plus.shape != q_minus.shape or q_plus.size < 2:
        raise ValueError(
            "q_plus and q_minus must be equal-length one-dimensional vectors "
            "with at least two bins."
        )
    if not np.all(np.isfinite(q_plus)) or not np.all(np.isfinite(q_minus)):
        raise ValueError("Prior mean probabilities must be finite.")
    if np.any(q_plus <= 0.0) or np.any(q_minus <= 0.0):
        raise ValueError("Prior mean probabilities must be strictly positive.")
    if not np.isclose(q_plus.sum(), 1.0, rtol=0.0, atol=1e-12) or not np.isclose(
        q_minus.sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Each prior mean probability vector must sum to one.")
    return q_plus, q_minus


def _assert_mirrored_monotone_profile(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
) -> None:
    """Assert properties specific to the symmetric linear prior family."""

    assert np.allclose(q_plus, q_minus[::-1], rtol=0.0, atol=1e-12)
    assert np.all(np.diff(q_plus) >= -1e-12)
    assert np.all(np.diff(q_minus) <= 1e-12)


def symmetric_prior_bin_probabilities(
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``k`` and the symmetric prior means ``q_k^+=q_k^-=1/K``."""

    K, k = bin_indices(K)
    q = np.full(K, 1.0 / K, dtype=float)
    q_plus, q_minus = validate_prior_mean_probabilities(q, q.copy())
    _assert_mirrored_monotone_profile(q_plus, q_minus)
    return k, q_plus, q_minus


def historical_histogram_prior_bin_probabilities(
    d_plus_pr: np.ndarray,
    d_minus_pr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return historical profiles smoothed by one total pseudo-count per class.

    Each class is normalized separately as ``(d + 1/K) / (sum(d) + 1)``.
    Empty bins are allowed, but each historical class must be nonempty.
    The final concentration is supplied separately to
    :func:`dirichlet_parameters_from_bin_probabilities`; it is not the
    historical sample size or the smoothing pseudo-count.
    """

    counts_plus = np.asarray(d_plus_pr)
    counts_minus = np.asarray(d_minus_pr)
    if (
        counts_plus.ndim != 1
        or counts_plus.shape != counts_minus.shape
        or counts_plus.size < 2
    ):
        raise ValueError(
            "Historical histograms must be equal-length one-dimensional "
            "vectors with at least two bins."
        )
    K, k = bin_indices(counts_plus.size)
    profiles = []
    for counts in (counts_plus, counts_minus):
        if not (
            np.issubdtype(counts.dtype, np.integer)
            or np.issubdtype(counts.dtype, np.floating)
        ):
            raise ValueError("Historical histograms must contain numeric integer counts.")
        if (
            not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("Historical counts must be finite non-negative integers.")
        with np.errstate(over="ignore"):
            sample_size = float(counts.sum(dtype=float))
        if not np.isfinite(sample_size) or sample_size <= 0.0:
            raise ValueError("Each historical class must have a finite positive total.")
        profile = (counts.astype(float) + 1.0 / K) / (sample_size + 1.0)
        assert np.all(profile > 0.0)
        assert np.isclose(profile.sum(), 1.0, rtol=0.0, atol=1e-12)
        profiles.append(profile)
    q_plus, q_minus = validate_prior_mean_probabilities(*profiles)
    return k, q_plus, q_minus


def increasing_decreasing_prior_bin_probabilities(
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``k`` and the manuscript's increasing/decreasing prior means."""

    K, k = bin_indices(K)
    denominator = K * (K + 1)
    q_plus = 2.0 * k / denominator
    q_minus = 2.0 * (K - k + 1.0) / denominator
    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    _assert_mirrored_monotone_profile(q_plus, q_minus)
    return k, q_plus, q_minus


def auc_calibrated_prior_bin_probabilities(
    K: int,
    target_auc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mirrored linear prior with the requested expected AUC."""

    K, k = bin_indices(K)
    target_auc = float(target_auc)
    upper_auc = (5.0 * K + 2.0) / (6.0 * K)
    if not np.isfinite(target_auc) or not 0.5 <= target_auc < upper_auc:
        raise ValueError(
            f"target_auc must satisfy 0.5 <= target_auc < {upper_auc:.6g} "
            f"when K={K}."
        )

    centered_k = k - (K + 1.0) / 2.0
    slope = 6.0 * (target_auc - 0.5) / (K**2 - 1.0)
    q_plus = 1.0 / K + slope * centered_k
    q_minus = 1.0 / K - slope * centered_k
    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    _assert_mirrored_monotone_profile(q_plus, q_minus)
    assert np.isclose(
        prior_expected_grid_auc(q_plus, q_minus),
        target_auc,
        rtol=0.0,
        atol=1e-12,
    )
    return k, q_plus, q_minus


def beta_bin_probabilities(
    K: int,
    positive_shape: tuple[float, float],
    negative_shape: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize two Beta distributions on the fixed uniform ``K``-bin grid."""

    K, k = bin_indices(K)
    positive_shape = tuple(float(value) for value in positive_shape)
    negative_shape = tuple(float(value) for value in negative_shape)
    if len(positive_shape) != 2 or len(negative_shape) != 2:
        raise ValueError("Each Beta shape must contain exactly two parameters.")
    if not all(
        np.isfinite(value) and value > 0.0
        for value in positive_shape + negative_shape
    ):
        raise ValueError("Beta shape parameters must be finite and positive.")

    grid = np.linspace(0.0, 1.0, K + 1)
    q_plus = np.diff(beta_distribution.cdf(grid, *positive_shape))
    q_minus = np.diff(beta_distribution.cdf(grid, *negative_shape))
    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    if positive_shape == negative_shape[::-1]:
        assert np.allclose(q_plus, q_minus[::-1], rtol=0.0, atol=1e-12)
    return k, q_plus, q_minus


def prior_expected_grid_auc(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
) -> float:
    """Return the expected discretized AUC induced by two prior means."""

    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    auc = sum(
        q_minus[index]
        * (q_plus[index + 1 :].sum() + 0.5 * q_plus[index])
        for index in range(q_plus.size)
    )
    auc = float(auc)
    assert 0.0 <= auc <= 1.0
    return auc


def expected_roc_from_bin_probabilities(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the prior-mean ROC coordinates and their trapezoidal AUC."""

    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    fpr = np.r_[0.0, np.cumsum(q_minus[::-1])]
    tpr = np.r_[0.0, np.cumsum(q_plus[::-1])]

    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert np.allclose(
        [fpr[0], tpr[0], fpr[-1], tpr[-1]],
        [0.0, 0.0, 1.0, 1.0],
        rtol=0.0,
        atol=1e-12,
    )

    auc_from_profile = prior_expected_grid_auc(q_plus, q_minus)
    auc_from_curve = float(np.trapezoid(tpr, fpr))
    assert np.isclose(
        auc_from_profile,
        auc_from_curve,
        rtol=0.0,
        atol=1e-12,
    )
    return fpr, tpr, auc_from_curve


def dirichlet_parameters_from_bin_probabilities(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
    prior_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert prior mean probabilities to ``alpha=m*q`` parameters."""

    q_plus, q_minus = validate_prior_mean_probabilities(q_plus, q_minus)
    prior_mass = float(prior_mass)
    if not np.isfinite(prior_mass) or prior_mass <= 0.0:
        raise ValueError("prior_mass must be finite and strictly positive.")

    alpha_plus = prior_mass * q_plus
    alpha_minus = prior_mass * q_minus
    assert np.all(alpha_plus > 0.0)
    assert np.all(alpha_minus > 0.0)
    assert np.isclose(alpha_plus.sum(), prior_mass, rtol=0.0, atol=1e-12)
    assert np.isclose(alpha_minus.sum(), prior_mass, rtol=0.0, atol=1e-12)
    return alpha_plus, alpha_minus
