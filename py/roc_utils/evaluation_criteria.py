"""
Evaluation criteria used in the experiments from ``main.tex``.

Each function summarizes one experimental condition and one nominal level
across ``B`` replications. Curve construction, predictive sampling,
and grid-based AUC calculation remain the responsibility of the existing
utilities. The caller must preserve the condition, nominal level, grid,
class-count, and replication pairing associated with these summaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

if not __debug__:
    raise RuntimeError(
        "evaluation_criteria relies on runtime assertions; do not use python -O."
    )

_BOUNDARY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MeanEstimate:
    """Mean and Monte Carlo standard error of a scalar replication metric."""

    mean: float
    standard_error: float
    n_replications: int


def summarize_mean(values: Sequence[float] | np.ndarray) -> MeanEstimate:
    """Use the sample variance (ddof=1) for a mean over B replications."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and values.size >= 2
    assert np.all(np.isfinite(values))
    return MeanEstimate(
        mean=float(np.mean(values)),
        standard_error=float(np.std(values, ddof=1) / np.sqrt(values.size)),
        n_replications=int(values.size),
    )


def summarize_paired_differences(
    dip_values: Sequence[float] | np.ndarray,
    boot_values: Sequence[float] | np.ndarray,
) -> MeanEstimate:
    """Summarize DiP minus Boot, using already matched replication arrays.

    The caller validates the replication keys before providing these arrays.
    Converting to float before subtraction also handles Boolean indicators
    and unsigned integer timing values correctly.
    """
    dip_values = np.asarray(dip_values, dtype=float)
    boot_values = np.asarray(boot_values, dtype=float)
    assert dip_values.ndim == boot_values.ndim == 1
    assert dip_values.shape == boot_values.shape
    assert np.all(np.isfinite(dip_values)) and np.all(np.isfinite(boot_values))
    return summarize_mean(dip_values - boot_values)


@dataclass(frozen=True)
class CoverageEstimate:
    """External coverage estimate for one nominal level."""

    coverage: float
    standard_error: float
    n_replications: int


@dataclass(frozen=True)
class SharpnessEstimate:
    """Mean ROC/AUC sharpness and standard errors for one nominal level."""

    roc_radius_mean: float
    roc_radius_standard_error: float
    auc_interval_width_mean: float
    auc_interval_width_standard_error: float
    n_replications: int


def is_external_roc_covered(
    external_max_deviation: float,
    roc_radius: float,
) -> bool:
    """Return the inclusive simultaneous external ROC coverage event."""

    external_max_deviation = float(external_max_deviation)
    roc_radius = float(roc_radius)
    assert np.isfinite(external_max_deviation)
    assert np.isfinite(roc_radius)
    assert 0.0 <= external_max_deviation <= np.sqrt(2.0) + _BOUNDARY_TOLERANCE
    assert 0.0 <= roc_radius <= 2.0 + _BOUNDARY_TOLERANCE
    return bool(
        external_max_deviation <= roc_radius + _BOUNDARY_TOLERANCE
    )


def is_external_auc_covered(
    external_auc: float,
    auc_interval: Sequence[float] | np.ndarray,
) -> bool:
    """Return the inclusive external grid-AUC interval coverage event."""

    external_auc = float(external_auc)
    auc_interval = np.asarray(auc_interval, dtype=float)
    assert np.isfinite(external_auc)
    assert 0.0 <= external_auc <= 1.0
    assert auc_interval.shape == (2,)
    assert np.all(np.isfinite(auc_interval))
    lower = float(auc_interval[0])
    upper = float(auc_interval[1])
    assert 0.0 <= lower <= upper <= 1.0
    return bool(
        lower - _BOUNDARY_TOLERANCE
        <= external_auc
        <= upper + _BOUNDARY_TOLERANCE
    )


def _coverage_from_indicators(indicators: np.ndarray) -> float:
    """Return the mean of a non-empty one-dimensional Boolean array."""

    indicators = np.asarray(indicators)
    assert indicators.ndim == 1
    assert indicators.size > 0
    assert np.issubdtype(indicators.dtype, np.bool_)
    coverage = float(np.mean(indicators))
    assert 0.0 <= coverage <= 1.0
    return coverage


def _monte_carlo_standard_error(coverage: float, n_replications: int) -> float:
    """Return the plug-in binomial Monte Carlo standard error."""

    coverage = float(coverage)
    assert np.isfinite(coverage)
    assert 0.0 <= coverage <= 1.0
    assert isinstance(n_replications, (int, np.integer))
    assert not isinstance(n_replications, (bool, np.bool_))
    assert n_replications > 0
    return float(
        np.sqrt(coverage * (1.0 - coverage) / int(n_replications))
    )


def summarize_external_roc_coverage(
    external_max_deviations: Sequence[float] | np.ndarray,
    roc_radii: Sequence[float] | np.ndarray,
) -> CoverageEstimate:
    """
    Estimate simultaneous external ROC coverage and its Monte Carlo SE.

    ``external_max_deviations[b]`` is the maximum Euclidean distance between
    the external ROC and the test-derived center at corresponding grid points.
    ``roc_radii[b]`` is the test-derived simultaneous band radius. Both arrays
    refer to the same replications and one nominal level. The comparison
    uses the same 1e-12 numerical boundary tolerance as the existing
    per-replication coverage utility.
    """

    external_max_deviations = np.asarray(external_max_deviations, dtype=float)
    roc_radii = np.asarray(roc_radii, dtype=float)

    assert external_max_deviations.ndim == 1
    assert roc_radii.ndim == 1
    assert external_max_deviations.shape == roc_radii.shape
    assert external_max_deviations.size > 0
    assert np.all(np.isfinite(external_max_deviations))
    assert np.all(np.isfinite(roc_radii))
    assert np.all(external_max_deviations >= 0.0)
    assert np.all(roc_radii >= 0.0)
    assert np.all(external_max_deviations <= np.sqrt(2.0) + 1e-12)
    assert np.all(roc_radii <= 2.0 + 1e-12)

    indicators = np.asarray(
        [
            is_external_roc_covered(deviation, radius)
            for deviation, radius in zip(external_max_deviations, roc_radii)
        ],
        dtype=bool,
    )
    B = int(indicators.size)
    coverage = _coverage_from_indicators(indicators)
    standard_error = _monte_carlo_standard_error(coverage, B)

    assert np.isclose(coverage, np.count_nonzero(indicators) / B)
    assert np.isclose(
        standard_error,
        np.sqrt(coverage * (1.0 - coverage) / B),
    )
    assert 0.0 <= coverage <= 1.0
    assert standard_error >= 0.0

    return CoverageEstimate(
        coverage=coverage,
        standard_error=standard_error,
        n_replications=B,
    )


def summarize_external_auc_coverage(
    external_aucs: Sequence[float] | np.ndarray,
    auc_intervals: Sequence[Sequence[float]] | np.ndarray,
) -> CoverageEstimate:
    """
    Estimate external grid-based AUC coverage and its Monte Carlo SE.

    Each row of ``auc_intervals`` contains the lower and upper endpoints of the
    test-derived interval for the corresponding external discretized AUC.
    Interval endpoints are included in the coverage event, with the same
    1e-12 numerical tolerance as the existing per-replication utility.
    """

    external_aucs = np.asarray(external_aucs, dtype=float)
    auc_intervals = np.asarray(auc_intervals, dtype=float)

    assert external_aucs.ndim == 1
    assert auc_intervals.ndim == 2
    assert auc_intervals.shape == (external_aucs.size, 2)
    assert external_aucs.size > 0
    assert np.all(np.isfinite(external_aucs))
    assert np.all(np.isfinite(auc_intervals))
    assert np.all((0.0 <= external_aucs) & (external_aucs <= 1.0))

    lower = auc_intervals[:, 0]
    upper = auc_intervals[:, 1]
    assert np.all((0.0 <= lower) & (lower <= upper) & (upper <= 1.0))

    indicators = np.asarray(
        [
            is_external_auc_covered(external_auc, interval)
            for external_auc, interval in zip(external_aucs, auc_intervals)
        ],
        dtype=bool,
    )
    B = int(indicators.size)
    coverage = _coverage_from_indicators(indicators)
    standard_error = _monte_carlo_standard_error(coverage, B)

    assert np.isclose(coverage, np.count_nonzero(indicators) / B)
    assert np.isclose(
        standard_error,
        np.sqrt(coverage * (1.0 - coverage) / B),
    )
    assert 0.0 <= coverage <= 1.0
    assert standard_error >= 0.0

    return CoverageEstimate(
        coverage=coverage,
        standard_error=standard_error,
        n_replications=B,
    )


def summarize_sharpness(
    roc_radii: Sequence[float] | np.ndarray,
    auc_intervals: Sequence[Sequence[float]] | np.ndarray,
) -> SharpnessEstimate:
    """
    Summarize ROC-band and AUC-interval sharpness across replications.

    ROC sharpness is the common band radius. AUC sharpness is the interval
    width. The reported standard errors use the formula from ``main.tex``:
    ``sqrt(sum((x_b - mean(x))**2) / (B * (B - 1)))``.
    """

    roc_radii = np.asarray(roc_radii, dtype=float)
    auc_intervals = np.asarray(auc_intervals, dtype=float)

    assert roc_radii.ndim == 1
    assert auc_intervals.ndim == 2
    assert auc_intervals.shape == (roc_radii.size, 2)
    assert roc_radii.size >= 2
    assert np.all(np.isfinite(roc_radii))
    assert np.all(np.isfinite(auc_intervals))
    assert np.all((0.0 <= roc_radii) & (roc_radii <= 2.0 + 1e-12))

    lower = auc_intervals[:, 0]
    upper = auc_intervals[:, 1]
    assert np.all((0.0 <= lower) & (lower <= upper) & (upper <= 1.0))

    auc_widths = upper - lower
    B = int(roc_radii.size)

    roc_radius_mean = float(np.mean(roc_radii))
    auc_interval_width_mean = float(np.mean(auc_widths))
    roc_radius_standard_error = float(
        np.sqrt(
            np.sum((roc_radii - roc_radius_mean) ** 2)
            / (B * (B - 1))
        )
    )
    auc_interval_width_standard_error = float(
        np.sqrt(
            np.sum((auc_widths - auc_interval_width_mean) ** 2)
            / (B * (B - 1))
        )
    )

    assert np.isclose(
        roc_radius_standard_error,
        np.std(roc_radii, ddof=1) / np.sqrt(B),
    )
    assert np.isclose(
        auc_interval_width_standard_error,
        np.std(auc_widths, ddof=1) / np.sqrt(B),
    )
    assert roc_radius_mean >= 0.0
    assert auc_interval_width_mean >= 0.0
    assert roc_radius_standard_error >= 0.0
    assert auc_interval_width_standard_error >= 0.0

    return SharpnessEstimate(
        roc_radius_mean=roc_radius_mean,
        roc_radius_standard_error=roc_radius_standard_error,
        auc_interval_width_mean=auc_interval_width_mean,
        auc_interval_width_standard_error=auc_interval_width_standard_error,
        n_replications=B,
    )
