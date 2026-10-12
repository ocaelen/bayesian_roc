"""
Shared utilities for the experiment runners.

The two explicit constructors implement DiP-ROC and BootP-ROC. Both use the
same grid geometry, and external evaluation is separate from construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

if not __debug__:
    raise RuntimeError(
        "utilities_experiments relies on runtime assertions; do not use python -O."
    )

from .evaluation_criteria import (
    summarize_external_auc_coverage,
    summarize_external_roc_coverage,
    summarize_sharpness,
)
from .bootstrap_predictive_roc_sampler import (
    PREDICTIVE_CORRECTION,
    sample_stratified_bootstrap_histograms,
)
from .posterior_predictive_roc_sampler import (
    curve_max_euclidean_deviation,
    equal_tailed_interval,
    grid_auc,
    histogram_from_scores,
    posterior_predictive_mean_roc,
    roc_from_histograms,
    rocs_from_histogram_samples,
    sample_posterior_predictive_rocs,
    simultaneous_euclidean_radii,
)

_BOUNDARY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PredictiveBand:
    """ROC/AUC predictions used by either method's external validation step."""

    fpr_center: np.ndarray
    tpr_center: np.ndarray
    roc_radii: dict[float, float]
    auc_intervals: dict[float, tuple[float, float]]
    auc_samples: np.ndarray


@dataclass(frozen=True)
class TestExternalHistograms:
    """Class-specific test and external histograms on one fixed grid."""

    d_plus_test: np.ndarray
    d_minus_test: np.ndarray
    d_plus_external: np.ndarray
    d_minus_external: np.ndarray


@dataclass(frozen=True)
class ReplicationMeasurements:
    """Sufficient statistics retained from one experimental replication."""

    external_max_roc_deviation: float
    external_grid_auc: float
    roc_radii: dict[float, float]
    auc_intervals: dict[float, tuple[float, float]]


@dataclass(frozen=True)
class EvaluationCriteriaSummary:
    """Coverage and sharpness summaries for one nominal level."""

    roc_coverage: float
    roc_coverage_standard_error: float
    auc_coverage: float
    auc_coverage_standard_error: float
    roc_radius_mean: float
    roc_radius_standard_error: float
    auc_interval_width_mean: float
    auc_interval_width_standard_error: float
    n_replications: int


def assert_histogram_pair_for_same_grid(
    d_plus: np.ndarray,
    d_minus: np.ndarray,
    *,
    context: str,
) -> None:
    """Assert that class-specific histograms are valid and share one grid."""

    d_plus = np.asarray(d_plus)
    d_minus = np.asarray(d_minus)
    assert d_plus.ndim == 1, f"{context}: positive histogram must be one-dimensional."
    assert d_minus.ndim == 1, f"{context}: negative histogram must be one-dimensional."
    assert d_plus.shape == d_minus.shape, f"{context}: histograms must use the same grid."
    assert d_plus.size > 0, f"{context}: at least one score bin is required."
    assert np.all(np.isfinite(d_plus)), f"{context}: positive histogram must be finite."
    assert np.all(np.isfinite(d_minus)), f"{context}: negative histogram must be finite."
    assert np.all(d_plus >= 0), f"{context}: positive histogram cannot contain negative counts."
    assert np.all(d_minus >= 0), f"{context}: negative histogram cannot contain negative counts."
    assert np.all(d_plus == np.floor(d_plus)), (
        f"{context}: positive histogram must contain integer counts."
    )
    assert np.all(d_minus == np.floor(d_minus)), (
        f"{context}: negative histogram must contain integer counts."
    )
    assert int(np.sum(d_plus)) > 0, f"{context}: positive class must be present."
    assert int(np.sum(d_minus)) > 0, f"{context}: negative class must be present."


def _assert_test_external_histograms(
    histograms: TestExternalHistograms,
) -> None:
    """Assert the fixed-grid and equal-class-count replication invariants."""

    assert isinstance(histograms, TestExternalHistograms)
    assert_histogram_pair_for_same_grid(
        histograms.d_plus_test,
        histograms.d_minus_test,
        context="Observed test data",
    )
    assert_histogram_pair_for_same_grid(
        histograms.d_plus_external,
        histograms.d_minus_external,
        context="External validation data",
    )

    test_shape = np.asarray(histograms.d_plus_test).shape
    external_shape = np.asarray(histograms.d_plus_external).shape
    assert test_shape == external_shape, (
        "Observed test and external histograms must use the same fixed grid."
    )
    assert np.sum(histograms.d_plus_test) == np.sum(histograms.d_plus_external), (
        "Positive test and external class counts must be equal."
    )
    assert np.sum(histograms.d_minus_test) == np.sum(histograms.d_minus_external), (
        "Negative test and external class counts must be equal."
    )


def build_test_external_histograms(
    positive_test_scores: Sequence[float] | np.ndarray,
    negative_test_scores: Sequence[float] | np.ndarray,
    positive_external_scores: Sequence[float] | np.ndarray,
    negative_external_scores: Sequence[float] | np.ndarray,
    gamma: Sequence[float] | np.ndarray,
) -> TestExternalHistograms:
    """
    Bin four class-separated score samples on the same fixed grid.

    Test and external counts must match within each class, as required by the
    experimental protocol.
    """

    histograms = TestExternalHistograms(
        d_plus_test=histogram_from_scores(positive_test_scores, gamma),
        d_minus_test=histogram_from_scores(negative_test_scores, gamma),
        d_plus_external=histogram_from_scores(positive_external_scores, gamma),
        d_minus_external=histogram_from_scores(negative_external_scores, gamma),
    )
    _assert_test_external_histograms(histograms)

    assert int(np.sum(histograms.d_plus_test)) == len(positive_test_scores)
    assert int(np.sum(histograms.d_minus_test)) == len(negative_test_scores)
    assert int(np.sum(histograms.d_plus_external)) == len(positive_external_scores)
    assert int(np.sum(histograms.d_minus_external)) == len(negative_external_scores)

    for counts in (
        histograms.d_plus_test, histograms.d_minus_test,
        histograms.d_plus_external, histograms.d_minus_external,
    ):
        counts.setflags(write=False)
    return histograms


def assert_probability_roc_arrays(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    expected_length: int,
    context: str,
) -> None:
    """Assert that one ROC curve is ordered from (0,0) to (1,1)."""

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    assert fpr.shape == (expected_length,), f"{context}: unexpected FPR shape {fpr.shape}."
    assert tpr.shape == (expected_length,), f"{context}: unexpected TPR shape {tpr.shape}."
    assert np.all(np.isfinite(fpr)), f"{context}: FPR values must be finite."
    assert np.all(np.isfinite(tpr)), f"{context}: TPR values must be finite."
    assert np.all((-1e-12 <= fpr) & (fpr <= 1.0 + 1e-12)), (
        f"{context}: FPR values must lie in [0, 1]."
    )
    assert np.all((-1e-12 <= tpr) & (tpr <= 1.0 + 1e-12)), (
        f"{context}: TPR values must lie in [0, 1]."
    )
    assert np.all(np.diff(fpr) >= -1e-12), f"{context}: FPR must be nondecreasing."
    assert np.all(np.diff(tpr) >= -1e-12), f"{context}: TPR must be nondecreasing."
    assert np.isclose(fpr[0], 0.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE) and np.isclose(
        tpr[0], 0.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE,
    ), (
        f"{context}: ROC curve must start at (0, 0)."
    )
    assert np.isclose(fpr[-1], 1.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE) and np.isclose(
        tpr[-1], 1.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE,
    ), (
        f"{context}: ROC curve must end at (1, 1)."
    )


def normalize_levels(levels: Iterable[float]) -> tuple[float, ...]:
    """Validate and return nominal levels as a stable tuple of floats."""

    normalized = tuple(float(level) for level in levels)
    if not normalized:
        raise ValueError("At least one nominal level is required.")
    for level in normalized:
        if not 0.0 < level < 1.0:
            raise ValueError("Nominal levels must lie in (0, 1).")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Nominal levels must not contain duplicates.")
    return normalized


def _assert_nested_band_summaries(
    roc_radii: dict[float, float],
    auc_intervals: dict[float, tuple[float, float]],
    levels: tuple[float, ...],
) -> None:
    """Assert nesting when several nominal levels share the same draws."""

    ordered_levels = tuple(sorted(levels))
    for lower_level, higher_level in zip(
        ordered_levels[:-1],
        ordered_levels[1:],
    ):
        assert (
            roc_radii[lower_level]
            <= roc_radii[higher_level] + _BOUNDARY_TOLERANCE
        ), "ROC radii must be nondecreasing with the nominal level."

        lower_interval = auc_intervals[lower_level]
        higher_interval = auc_intervals[higher_level]
        assert (
            higher_interval[0]
            <= lower_interval[0] + _BOUNDARY_TOLERANCE
        ), "Higher-level AUC intervals must contain lower-level intervals."
        assert (
            lower_interval[1]
            <= higher_interval[1] + _BOUNDARY_TOLERANCE
        ), "Higher-level AUC intervals must contain lower-level intervals."


def build_posterior_predictive_band(
    d_plus: np.ndarray,
    d_minus: np.ndarray,
    alpha_plus: np.ndarray,
    alpha_minus: np.ndarray,
    S: int,
    levels: Iterable[float],
    rng: np.random.Generator,
) -> PredictiveBand:
    """
    Build ROC-band and AUC-interval summaries from one observed test sample.

    This function is the shared implementation of the Bayesian ROC procedure
    used by all experiment runners. It reuses
    ``sample_posterior_predictive_rocs`` to generate
    ``r_tilde_{K:0}^{(1)}, ..., r_tilde_{K:0}^{(S)}``.
    """

    assert isinstance(rng, np.random.Generator), (
        "rng must be a caller-supplied numpy.random.Generator."
    )
    if int(S) != S or S <= 0:
        raise ValueError("S must be a positive integer.")
    levels = normalize_levels(levels)
    assert_histogram_pair_for_same_grid(d_plus, d_minus, context="Observed test data")
    K = int(np.asarray(d_plus).size)
    assert np.asarray(alpha_plus).shape == (K,), "alpha_plus must match the number of bins."
    assert np.asarray(alpha_minus).shape == (K,), "alpha_minus must match the number of bins."

    fpr_center, tpr_center = posterior_predictive_mean_roc(
        d_plus=d_plus,
        d_minus=d_minus,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
    )
    fpr_samples, tpr_samples = sample_posterior_predictive_rocs(
        d_plus=d_plus,
        d_minus=d_minus,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        S=int(S),
        rng=rng,
    )
    assert_probability_roc_arrays(
        fpr_center,
        tpr_center,
        expected_length=K + 1,
        context="Posterior predictive mean ROC",
    )
    _assert_sampled_roc_arrays(fpr_samples, tpr_samples, int(S), K)

    roc_radii = simultaneous_euclidean_radii(
        fpr_samples=fpr_samples,
        tpr_samples=tpr_samples,
        fpr_center=fpr_center,
        tpr_center=tpr_center,
        levels=levels,
    )
    assert set(roc_radii) == set(levels), "One ROC radius is required per nominal level."
    assert all(
        0.0 <= radius <= np.sqrt(2.0) + _BOUNDARY_TOLERANCE
        for radius in roc_radii.values()
    ), "DiP-ROC radii must lie between zero and sqrt(2)."

    auc_samples = np.asarray(grid_auc(fpr_samples, tpr_samples), dtype=float)
    assert auc_samples.shape == (int(S),), "AUC samples must have length S."
    assert np.all(np.isfinite(auc_samples)), "AUC samples must be finite."
    assert np.all((0.0 <= auc_samples) & (auc_samples <= 1.0)), (
        f"DiP-ROC AUC samples must lie in [0, 1]: "
        f"min={auc_samples.min()!r}, max={auc_samples.max()!r}."
    )
    auc_intervals = {
        float(level): equal_tailed_interval(auc_samples, float(level))
        for level in levels
    }
    assert set(auc_intervals) == set(levels), "One AUC interval is required per level."
    for level, (lower, upper) in auc_intervals.items():
        assert 0.0 <= lower <= upper <= 1.0, (
            f"AUC interval for level {level:.2f} must lie inside [0, 1]."
        )
    _assert_nested_band_summaries(roc_radii, auc_intervals, levels)

    return PredictiveBand(
        fpr_center=fpr_center,
        tpr_center=tpr_center,
        roc_radii=roc_radii,
        auc_intervals=auc_intervals,
        auc_samples=auc_samples,
    )


def _assert_sampled_roc_arrays(fpr, tpr, S, K):
    """Shared deterministic checks for the two methods' simulated curves."""
    for values in (fpr, tpr):
        assert values.shape == (S, K + 1)
        assert np.all(np.isfinite(values))
        assert np.all(
            (-_BOUNDARY_TOLERANCE <= values)
            & (values <= 1.0 + _BOUNDARY_TOLERANCE)
        )
        assert np.all(np.diff(values, axis=1) >= -_BOUNDARY_TOLERANCE)
        assert np.allclose(values[:, 0], 0.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE)
        assert np.allclose(values[:, -1], 1.0, rtol=0.0, atol=_BOUNDARY_TOLERANCE)


def build_bootstrap_predictive_band(
    d_plus, d_minus, S, levels, rng,
) -> PredictiveBand:
    """BootP-ROC: empirical bootstrap with the fixed sqrt(2) future correction.

    Only observed histograms enter this construction. Simulated curves are
    ordinary bootstrap curves; the correction changes the radius and final
    AUC bounds, not the curves or their empirical center.
    """
    levels = normalize_levels(levels)
    assert_histogram_pair_for_same_grid(
        d_plus, d_minus, context="Observed test data",
    )
    K = int(np.asarray(d_plus).size)
    fpr_center, tpr_center = roc_from_histograms(d_plus, d_minus)
    d_plus_samples, d_minus_samples = sample_stratified_bootstrap_histograms(
        d_plus, d_minus, S, rng=rng,
    )
    fpr_samples, tpr_samples = rocs_from_histogram_samples(
        d_plus_samples, d_minus_samples,
    )
    assert_probability_roc_arrays(
        fpr_center, tpr_center, expected_length=K + 1, context="Observed empirical ROC",
    )
    _assert_sampled_roc_arrays(fpr_samples, tpr_samples, int(S), K)
    uncorrected_radii = simultaneous_euclidean_radii(
        fpr_samples, tpr_samples, fpr_center, tpr_center, levels,
    )
    roc_radii = {
        level: float(PREDICTIVE_CORRECTION * uncorrected_radii[level])
        for level in levels
    }
    assert all(
        0.0 <= radius <= 2.0 + _BOUNDARY_TOLERANCE
        for radius in roc_radii.values()
    )
    observed_auc = float(grid_auc(fpr_center, tpr_center))
    auc_samples = np.asarray(grid_auc(fpr_samples, tpr_samples), dtype=float)
    assert auc_samples.shape == (int(S),), "BootP-ROC AUC samples must have length S."
    assert np.all(np.isfinite(auc_samples)), "BootP-ROC AUC samples must be finite."
    assert np.all((0.0 <= auc_samples) & (auc_samples <= 1.0)), (
        f"BootP-ROC AUC samples must lie in [0, 1]: "
        f"min={auc_samples.min()!r}, max={auc_samples.max()!r}."
    )
    auc_intervals = {}
    for level in levels:
        lower, upper = equal_tailed_interval(auc_samples, level)
        lower = float(np.clip(
            observed_auc + PREDICTIVE_CORRECTION * (lower - observed_auc), 0.0, 1.0,
        ))
        upper = float(np.clip(
            observed_auc + PREDICTIVE_CORRECTION * (upper - observed_auc), 0.0, 1.0,
        ))
        assert 0.0 <= lower <= upper <= 1.0
        auc_intervals[level] = (lower, upper)
    _assert_nested_band_summaries(roc_radii, auc_intervals, levels)
    return PredictiveBand(
        fpr_center, tpr_center, roc_radii, auc_intervals, auc_samples,
    )


def _add_construction_failure_note(error, builder, phase, context, arguments):
    """Attach reproducible inputs to the original error, only on failure."""
    details = dict(context or {})
    details.update(
        phase=phase,
        builder=getattr(builder, "__name__", type(builder).__name__),
        S=arguments.get("S"),
        levels=arguments.get("levels"),
    )
    for name in ("d_plus", "d_minus", "alpha_plus", "alpha_minus"):
        if name in arguments:
            details[name] = np.asarray(arguments[name]).tolist()
    error.add_note(f"ROC construction diagnostics: {details!r}")


def warm_up_band(builder, *, diagnostic_context=None, **kwargs) -> None:
    """Run an unmeasured construction with a caller-supplied warm-up RNG."""
    try:
        with threadpool_limits(limits=1):
            builder(**kwargs)
    except Exception as error:
        _add_construction_failure_note(
            error, builder, "warmup", diagnostic_context, kwargs,
        )
        raise


def build_timed_band(
    builder, *, diagnostic_context=None, **kwargs,
) -> tuple[PredictiveBand, int]:
    """Time only construction, including both levels, using one native thread.

    Histogram preparation, RNG creation, thread-limit setup/restoration, and
    external evaluation are outside the measured interval.
    """
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter_ns()
            band = builder(**kwargs)
            elapsed = time.perf_counter_ns() - start
        assert isinstance(band, PredictiveBand), "The builder must return a PredictiveBand."
        assert isinstance(elapsed, int) and elapsed >= 0, (
            f"Construction time must be a non-negative integer: {elapsed!r}."
        )
    except Exception as error:
        _add_construction_failure_note(
            error, builder, "construction", diagnostic_context, kwargs,
        )
        raise
    return band, elapsed


def evaluate_replication(
    histograms: TestExternalHistograms,
    band: PredictiveBand,
) -> ReplicationMeasurements:
    """Evaluate an already constructed band; no simulation occurs here."""
    assert isinstance(band, PredictiveBand)
    levels = normalize_levels(band.roc_radii)
    assert set(band.auc_intervals) == set(levels)
    _assert_test_external_histograms(histograms)
    assert_probability_roc_arrays(
        band.fpr_center, band.tpr_center,
        expected_length=len(histograms.d_plus_test) + 1,
        context="Predictive band center",
    )

    fpr_external, tpr_external = roc_from_histograms(
        d_plus=histograms.d_plus_external,
        d_minus=histograms.d_minus_external,
    )
    K = int(np.asarray(histograms.d_plus_test).size)
    assert_probability_roc_arrays(
        fpr_external,
        tpr_external,
        expected_length=K + 1,
        context="External empirical ROC",
    )

    external_max_roc_deviation = curve_max_euclidean_deviation(
        fpr=fpr_external,
        tpr=tpr_external,
        fpr_center=band.fpr_center,
        tpr_center=band.tpr_center,
    )
    external_grid_auc = float(grid_auc(fpr_external, tpr_external))
    roc_radii = {
        level: float(band.roc_radii[level])
        for level in levels
    }
    auc_intervals = {
        level: (
            float(band.auc_intervals[level][0]),
            float(band.auc_intervals[level][1]),
        )
        for level in levels
    }

    assert (
        0.0
        <= external_max_roc_deviation
        <= np.sqrt(2.0) + _BOUNDARY_TOLERANCE
    )
    assert 0.0 <= external_grid_auc <= 1.0
    assert set(roc_radii) == set(levels)
    assert set(auc_intervals) == set(levels)
    assert all(
        0.0 <= radius <= 2.0 + _BOUNDARY_TOLERANCE
        for radius in roc_radii.values()
    )
    assert all(
        0.0 <= lower <= upper <= 1.0
        for lower, upper in auc_intervals.values()
    )
    _assert_nested_band_summaries(roc_radii, auc_intervals, levels)

    return ReplicationMeasurements(
        external_max_roc_deviation=float(external_max_roc_deviation),
        external_grid_auc=external_grid_auc,
        roc_radii=roc_radii,
        auc_intervals=auc_intervals,
    )


def summarize_replication_measurements(
    replications: Sequence[ReplicationMeasurements],
    levels: Iterable[float],
) -> dict[float, EvaluationCriteriaSummary]:
    """
    Summarize all three evaluation criteria for each nominal level.

    The measurements must come from one experimental condition in matching
    replication order. At least two replications are required because the
    sharpness standard errors use the sample variance.
    """

    levels = normalize_levels(levels)
    if len(replications) < 2:
        raise ValueError("At least two replications are required for sharpness.")

    for replication in replications:
        assert isinstance(replication, ReplicationMeasurements)
        assert set(replication.roc_radii) == set(levels), (
            "Every replication must contain one ROC radius per level."
        )
        assert set(replication.auc_intervals) == set(levels), (
            "Every replication must contain one AUC interval per level."
        )
        _assert_nested_band_summaries(
            replication.roc_radii,
            replication.auc_intervals,
            levels,
        )

    summaries: dict[float, EvaluationCriteriaSummary] = {}
    B = len(replications)
    for level in levels:
        external_max_deviations = [
            replication.external_max_roc_deviation
            for replication in replications
        ]
        external_aucs = [
            replication.external_grid_auc
            for replication in replications
        ]
        roc_radii = [
            replication.roc_radii[level]
            for replication in replications
        ]
        auc_intervals = [
            replication.auc_intervals[level]
            for replication in replications
        ]

        roc_coverage = summarize_external_roc_coverage(
            external_max_deviations,
            roc_radii,
        )
        auc_coverage = summarize_external_auc_coverage(
            external_aucs,
            auc_intervals,
        )
        sharpness = summarize_sharpness(
            roc_radii,
            auc_intervals,
        )
        assert (
            roc_coverage.n_replications
            == auc_coverage.n_replications
            == sharpness.n_replications
            == B
        )

        summaries[level] = EvaluationCriteriaSummary(
            roc_coverage=roc_coverage.coverage,
            roc_coverage_standard_error=roc_coverage.standard_error,
            auc_coverage=auc_coverage.coverage,
            auc_coverage_standard_error=auc_coverage.standard_error,
            roc_radius_mean=sharpness.roc_radius_mean,
            roc_radius_standard_error=sharpness.roc_radius_standard_error,
            auc_interval_width_mean=sharpness.auc_interval_width_mean,
            auc_interval_width_standard_error=(
                sharpness.auc_interval_width_standard_error
            ),
            n_replications=B,
        )

    return summaries
