"""Run the manuscript's main evaluation on synthetic data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Real

import numpy as np

from roc_utils.experiment_provenance import (
    assert_sources_unchanged,
    collect_run_metadata,
    publish_run_metadata,
)

if not __debug__:
    raise RuntimeError(
        "This experiment relies on runtime assertions; do not use python -O."
    )

from roc_utils.experiment_results import (
    MEASUREMENT_RAW_FIELDNAMES,
    RawLevelMeasurements,
    assert_complete_level_bundle,
    assert_method_metadata,
    parse_canonical_int,
    parse_finite_float,
    parse_raw_level_measurements,
    raw_level_measurement_fields,
    raw_level_measurements_from_replication,
    validate_positive_integer,
)
from roc_utils.posterior_predictive_roc_sampler import (
    fixed_uniform_grid,
    histogram_from_scores,
    roc_from_histograms,
    grid_auc,
    symmetric_dirichlet_priors,
)
from roc_utils.prior_profiles import (
    auc_calibrated_prior_bin_probabilities,
    dirichlet_parameters_from_bin_probabilities,
    historical_histogram_prior_bin_probabilities,
    prior_expected_grid_auc,
)
from roc_utils.semantic_rng import (
    RNG_ALGORITHM,
    make_semantic_rng,
    semantic_stream_id,
)
from roc_utils.simulation_datasets import (
    SYNTHETIC_DESIGN_VERSION,
    SIMULATION_SCENARIOS,
    generate_simulation_dataset,
    simulation_beta_parameters,
)
from roc_utils.utilities_experiments import (
    ReplicationMeasurements,
    PredictiveBand,
    TestExternalHistograms,
    build_test_external_histograms,
    evaluate_replication,
    build_posterior_predictive_band,
    build_timed_band,
    warm_up_band,
)


EXPERIMENT_NAME = "class_counts_and_imbalance"
RESULTS_DIRECTORY_NAME = "main_evaluation_on_synthetic_data"
MAIN_PROTOCOL_VERSION = 8
RAW_SCHEMA_VERSION = f"class_counts_raw_v{MAIN_PROTOCOL_VERSION}"

N_BINS = 10
HISTORICAL_N_POSITIVE = 500
HISTORICAL_N_NEGATIVE = 500
HISTORICAL_AUC_BOUNDS = (0.5, 0.86)
NOMINAL_LEVELS = (0.80, 0.95)
OFFICIAL_REPLICATIONS = 10_000
OFFICIAL_SAMPLES = 4000

_PROGRESS_INTERVAL = 1000
_PROGRESS_FIELDS = ("roc_covered", "auc_covered", "roc_radius", "auc_width")

# Prescribed in main.tex and recorded in every CSV row.
DEFAULT_MASTER_SEED = 20250731

CLASS_COUNT_PAIRS = (
    (20, 20),
    (100, 100),
    (500, 500),
    (20, 500),
    (500, 20),
)

_EXPERIMENT_STREAM_KEY = 1
_DATA_STREAM_ROLE = 1
_POSTERIOR_STREAM_ROLE = 2
_WARMUP_STREAM_ROLE = 4
_HISTORICAL_DATA_STREAM_ROLE = 5
_HISTORICAL_POSTERIOR_STREAM_ROLE = 6
METHODS = ("DiP-ROC",)
PRIOR_MASSES = (1.0, 5.0, 20.0)
PRIOR_FAMILIES = ("uniform", "linear", "histograms")
# Name, family, concentration, profile and fixed RNG role, in reporting order.
# Roles identify scientific streams; they must never follow list positions.
_SETTING_DEFINITIONS = (
    ("dip_uniform_m1", "uniform", 1.0, "symmetric", 2),
    ("dip_uniform_m5", "uniform", 5.0, "symmetric", 7),
    ("dip_uniform_m20", "uniform", 20.0, "symmetric", 11),
    ("dip_linear_m1", "linear", 1.0, "historical_auc_linear", 6),
    ("dip_linear_m5", "linear", 5.0, "historical_auc_linear", 8),
    ("dip_linear_m20", "linear", 20.0, "historical_auc_linear", 12),
    ("dip_histograms_m1", "histograms", 1.0, "historical_histograms", 9),
    ("dip_histograms_m5", "histograms", 5.0, "historical_histograms", 10),
    ("dip_histograms_m20", "histograms", 20.0, "historical_histograms", 13),
)
SETTINGS = tuple(row[0] for row in _SETTING_DEFINITIONS)
SETTING_FAMILIES = {row[0]: row[1] for row in _SETTING_DEFINITIONS}
SETTING_MASSES = {row[0]: row[2] for row in _SETTING_DEFINITIONS}
SETTING_METHODS = dict.fromkeys(SETTINGS, "DiP-ROC")
SETTING_LABELS = {
    setting: f"DiP-ROC ({SETTING_FAMILIES[setting]}, m={SETTING_MASSES[setting]:g})"
    for setting in SETTINGS
}
SETTING_PROFILES = {row[0]: row[3] for row in _SETTING_DEFINITIONS}
_SETTING_STREAM_ROLES = {row[0]: row[4] for row in _SETTING_DEFINITIONS}
_INFORMED_SETTINGS = tuple(setting for setting in SETTINGS
                           if SETTING_FAMILIES[setting] in ("linear", "histograms"))


def _assert_setting_definitions() -> None:
    """Fail before sampling if the manuscript settings or fixed streams drift."""
    assert PRIOR_MASSES == (1.0, 5.0, 20.0)
    assert PRIOR_FAMILIES == ("uniform", "linear", "histograms")
    assert SETTINGS == tuple(
        f"dip_{family}_m{mass:g}" for family in PRIOR_FAMILIES for mass in PRIOR_MASSES
    )
    assert len(SETTINGS) == len(set(SETTINGS)) == 9
    assert METHODS == ("DiP-ROC",)
    for mapping in (SETTING_FAMILIES, SETTING_MASSES, SETTING_METHODS,
                    SETTING_LABELS, SETTING_PROFILES, _SETTING_STREAM_ROLES):
        assert set(mapping) == set(SETTINGS), "Incomplete setting metadata."
    profiles = dict(zip(PRIOR_FAMILIES, ("symmetric", "historical_auc_linear", "historical_histograms"), strict=True))
    for family in PRIOR_FAMILIES:
        for mass in PRIOR_MASSES:
            setting = f"dip_{family}_m{mass:g}"
            assert SETTING_FAMILIES[setting] == family
            assert SETTING_MASSES[setting] == mass
            assert SETTING_PROFILES[setting] == profiles[family]
            assert SETTING_METHODS[setting] == "DiP-ROC"
    assert _SETTING_STREAM_ROLES == {
        "dip_uniform_m1": 2, "dip_uniform_m5": 7, "dip_uniform_m20": 11,
        "dip_linear_m1": 6, "dip_linear_m5": 8, "dip_linear_m20": 12,
        "dip_histograms_m1": 9, "dip_histograms_m5": 10, "dip_histograms_m20": 13,
    }, "Scientific stream roles must remain independent of reporting order."
    assert len(set(_SETTING_STREAM_ROLES.values())) == len(SETTINGS)
    assert _INFORMED_SETTINGS == tuple(s for s in SETTINGS if SETTING_FAMILIES[s] in ("linear", "histograms"))
    assert len(_INFORMED_SETTINGS) == 6


_assert_setting_definitions()

_HISTORICAL_FIELDS = (
    "historical_n_positive", "historical_n_negative", "historical_stream_id",
    "historical_grid_auc", "historical_auc_adjusted",
    "historical_counts_positive", "historical_counts_negative",
)

PY_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = (
    PY_DIR
    / "results"
    / "current_experiments"
    / RESULTS_DIRECTORY_NAME
    / f"raw_replications_v{MAIN_PROTOCOL_VERSION}.csv"
)

assert DEFAULT_OUTPUT_CSV.parent.name == RESULTS_DIRECTORY_NAME


@dataclass(frozen=True)
class ClassCountCondition:
    """One score scenario and one ordered pair of class counts."""

    index: int
    score_scenario: str
    n_positive: int
    n_negative: int

    def __post_init__(self) -> None:
        assert type(self.index) is int
        assert self.index >= 0
        assert self.score_scenario in SIMULATION_SCENARIOS
        assert type(self.n_positive) is int
        assert type(self.n_negative) is int
        assert self.n_positive > 0
        assert self.n_negative > 0

    @property
    def condition_id(self) -> str:
        """Return a stable identifier that preserves imbalance direction."""

        scenario = self.score_scenario.lower()
        return (
            f"{scenario}_n_positive_{self.n_positive}"
            f"_n_negative_{self.n_negative}"
        )


CLASS_COUNT_CONDITIONS = tuple(
    ClassCountCondition(
        index=condition_index,
        score_scenario=scenario,
        n_positive=n_positive,
        n_negative=n_negative,
    )
    for condition_index, (scenario, (n_positive, n_negative)) in enumerate(
        (scenario, counts)
        for scenario in SIMULATION_SCENARIOS
        for counts in CLASS_COUNT_PAIRS
    )
)

assert SIMULATION_SCENARIOS == ("Easy", "Medium", "Hard")
assert CLASS_COUNT_PAIRS == (
    (20, 20),
    (100, 100),
    (500, 500),
    (20, 500),
    (500, 20),
)
assert len(CLASS_COUNT_CONDITIONS) == 15
assert tuple(condition.index for condition in CLASS_COUNT_CONDITIONS) == tuple(
    range(15)
)
assert len(
    {condition.condition_id for condition in CLASS_COUNT_CONDITIONS}
) == len(CLASS_COUNT_CONDITIONS)


RAW_FIELDNAMES = (
    "schema_version",
    "experiment",
    "method",
    "setting",
    "condition_index",
    "condition_id",
    "replication",
    "level",
    "score_scenario",
    "positive_beta_a",
    "positive_beta_b",
    "negative_beta_a",
    "negative_beta_b",
    "n_positive",
    "n_negative",
    "K",
    "S",
    "B",
    "prior_profile",
    "prior_mass",
    "prior_expected_grid_auc",
    *_HISTORICAL_FIELDS,
    "rng_algorithm",
    "master_seed",
    "data_stream_id",
    "simulation_stream_id",
    "computation_time_ns",
    "execution_position",
    "timing_threads",
    *MEASUREMENT_RAW_FIELDNAMES,
)


def _validate_conditions(
    conditions: Iterable[ClassCountCondition],
) -> tuple[ClassCountCondition, ...]:
    """Validate a non-empty, duplicate-free subset of the official design."""

    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("At least one class-count condition is required.")
    for condition in conditions:
        if not isinstance(condition, ClassCountCondition):
            raise TypeError("conditions must contain ClassCountCondition values.")
        if condition not in CLASS_COUNT_CONDITIONS:
            raise ValueError("Every condition must belong to the official design.")
    condition_indices = [condition.index for condition in conditions]
    if len(set(condition_indices)) != len(condition_indices):
        raise ValueError("conditions must not contain duplicates.")
    return tuple(sorted(conditions, key=lambda condition: condition.index))


def _stream_keys(
    condition: ClassCountCondition,
    replication: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return independent semantic keys for data and posterior sampling."""

    data_key = (
        _EXPERIMENT_STREAM_KEY,
        _DATA_STREAM_ROLE,
        condition.index,
        replication,
    )
    posterior_key = (
        _EXPERIMENT_STREAM_KEY,
        _POSTERIOR_STREAM_ROLE,
        condition.index,
        replication,
    )
    assert data_key != posterior_key
    return data_key, posterior_key


def _simulation_stream_key(
    condition: ClassCountCondition, replication: int, setting: str,
) -> tuple[int, ...]:
    """Preserve each setting's scientific stream across protocol revisions."""
    assert setting in SETTINGS
    return (_EXPERIMENT_STREAM_KEY, _SETTING_STREAM_ROLES[setting], condition.index, replication)


def _historical_stream_key(condition, replication):
    return (_EXPERIMENT_STREAM_KEY, _HISTORICAL_DATA_STREAM_ROLE, condition.index, replication)


def _warmup_stream_key(condition_index: int, setting: str) -> tuple[int, ...]:
    assert setting in SETTINGS
    return (_EXPERIMENT_STREAM_KEY, _WARMUP_STREAM_ROLE,
            _SETTING_STREAM_ROLES[setting], condition_index, 0)


def _execution_order(replication: int) -> tuple[str, ...]:
    assert isinstance(replication, int) and replication >= 0
    offset = replication % len(SETTINGS)
    return SETTINGS[offset:] + SETTINGS[:offset]


def restrict_historical_auc(value: float) -> float:
    """Apply the prespecified restriction to a valid historical grid AUC."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("historical_grid_auc must be a finite scalar in [0, 1].")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("historical_grid_auc must be a finite scalar in [0, 1].")
    return min(HISTORICAL_AUC_BOUNDS[1], max(HISTORICAL_AUC_BOUNDS[0], value))


def historical_auc_from_histograms(d_plus, d_minus) -> float:
    """Canonical trapezoidal grid AUC, including half credit for within-bin ties."""
    counts = []
    for values in (d_plus, d_minus):
        values = np.asarray(values)
        assert values.shape == (N_BINS,) and values.dtype.kind in "iu"
        assert np.all(values >= 0) and int(values.sum()) > 0
        counts.append(values)
    fpr, tpr = roc_from_histograms(*counts)
    trapezoidal_auc = float(grid_auc(fpr, tpr))
    twice_wins, negatives_below = 0, 0
    for positive_count, negative_count in zip(*counts, strict=True):
        twice_wins += int(positive_count) * (2 * negatives_below + int(negative_count))
        negatives_below += int(negative_count)
    total_pairs = int(counts[0].sum()) * int(counts[1].sum())
    assert 0 <= twice_wins <= 2 * total_pairs
    auc = twice_wins / (2 * total_pairs)
    assert np.isclose(auc, trapezoidal_auc, rtol=0, atol=1e-12)
    assert 0.0 <= auc <= 1.0
    return auc


@dataclass(frozen=True)
class HistoricalSummary:
    """Observed counts and their empirical AUC; never pooled with the test data."""

    d_plus: np.ndarray
    d_minus: np.ndarray
    grid_auc: float = field(init=False)

    def __post_init__(self) -> None:
        auc = historical_auc_from_histograms(self.d_plus, self.d_minus)
        for name in ("d_plus", "d_minus"):
            counts = np.array(getattr(self, name), dtype=np.int64, copy=True)
            counts.setflags(write=False)
            object.__setattr__(self, name, counts)
        object.__setattr__(self, "grid_auc", auc)


def historical_summary_from_scores(positive_scores, negative_scores, gamma) -> HistoricalSummary:
    """Prepare historical histograms and grid AUC outside construction timing."""
    np.testing.assert_array_equal(gamma, fixed_uniform_grid(N_BINS))
    histograms = []
    for scores in (positive_scores, negative_scores):
        scores = np.asarray(scores, dtype=float)
        assert scores.ndim == 1 and scores.size > 0
        assert np.all(np.isfinite(scores)) and np.all((0 <= scores) & (scores <= 1))
        counts = histogram_from_scores(scores, gamma)
        assert counts.shape == (N_BINS,) and int(counts.sum()) == scores.size
        histograms.append(counts)
    return HistoricalSummary(*histograms)


def historical_auc_from_scores(positive_scores, negative_scores, gamma) -> float:
    """Return the unsmoothed historical grid AUC, including empty observed bins."""
    return historical_summary_from_scores(positive_scores, negative_scores, gamma).grid_auc


def _generate_historical_summary(condition, replication, *, gamma, master_seed) -> HistoricalSummary:
    """Draw one independent historical sample, positive class first."""
    key = _historical_stream_key(condition, replication)
    context = {
        "experiment": EXPERIMENT_NAME, "phase": "historical_preparation",
        "stage": "historical_preparation", "settings": _INFORMED_SETTINGS,
        "method": "DiP-ROC", "condition_id": condition.condition_id,
        "condition_index": condition.index, "replication_zero_based": replication,
        "K": N_BINS, "master_seed": master_seed,
        "historical_stream_id": semantic_stream_id(master_seed, *key),
        "data_stream_id": semantic_stream_id(master_seed, *_stream_keys(condition, replication)[0]),
        "historical_n_positive": HISTORICAL_N_POSITIVE,
        "historical_n_negative": HISTORICAL_N_NEGATIVE,
    }
    try:
        historical_rng = make_semantic_rng(master_seed, *key)
        positive_beta, negative_beta = simulation_beta_parameters(condition.score_scenario)
        positive = historical_rng.beta(*positive_beta, size=HISTORICAL_N_POSITIVE)
        negative = historical_rng.beta(*negative_beta, size=HISTORICAL_N_NEGATIVE)
        assert positive.shape == (HISTORICAL_N_POSITIVE,)
        assert negative.shape == (HISTORICAL_N_NEGATIVE,)
        return historical_summary_from_scores(positive, negative, gamma)
    except Exception as error:
        for name in ("positive", "negative"):
            if name in locals():
                try:
                    values = np.asarray(locals()[name])
                    context[name + "_shape"] = values.shape
                    if (values.ndim == 1 and np.all(np.isfinite(values))
                            and np.all((0 <= values) & (values <= 1))):
                        context[name + "_histogram"] = np.histogram(values, gamma)[0].tolist()
                except Exception as diagnostic_error:
                    context[name + "_diagnostics_unavailable"] = type(diagnostic_error).__name__
        error.add_note(f"Historical preparation diagnostics: {context!r}")
        raise


def build_uniform_predictive_band(d_plus, d_minus, prior_mass, S, levels, rng) -> PredictiveBand:
    """Construct uniform parameters within the same timer as the prediction."""
    assert np.asarray(d_plus).shape == np.asarray(d_minus).shape == (N_BINS,)
    alpha_plus, alpha_minus = symmetric_dirichlet_priors(N_BINS, prior_mass)
    return build_posterior_predictive_band(
        d_plus=d_plus, d_minus=d_minus, alpha_plus=alpha_plus,
        alpha_minus=alpha_minus, S=S, levels=levels, rng=rng,
    )


def build_linear_predictive_band(
    d_plus, d_minus, historical_grid_auc, prior_mass, S, levels, rng,
) -> PredictiveBand:
    """Restrict and calibrate one historical AUC, then run unchanged DiP-ROC."""
    assert np.asarray(d_plus).shape == np.asarray(d_minus).shape == (N_BINS,)
    target = restrict_historical_auc(historical_grid_auc)
    _, q_plus, q_minus = auc_calibrated_prior_bin_probabilities(N_BINS, target)
    np.testing.assert_allclose(q_plus, q_minus[::-1], rtol=0, atol=1e-12)
    assert np.isclose(prior_expected_grid_auc(q_plus, q_minus), target, rtol=0, atol=1e-12)
    alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(q_plus, q_minus, prior_mass)
    return build_posterior_predictive_band(
        d_plus=d_plus, d_minus=d_minus, alpha_plus=alpha_plus,
        alpha_minus=alpha_minus, S=S, levels=levels, rng=rng,
    )


def build_histogram_predictive_band(
    d_plus, d_minus, historical_d_plus, historical_d_minus, prior_mass, S, levels, rng,
) -> PredictiveBand:
    """Smooth, normalize and scale historical counts within the prediction timer."""
    assert np.asarray(d_plus).shape == np.asarray(d_minus).shape == (N_BINS,)
    assert np.asarray(historical_d_plus).shape == np.asarray(historical_d_minus).shape == (N_BINS,)
    _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(historical_d_plus, historical_d_minus)
    alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(q_plus, q_minus, prior_mass)
    return build_posterior_predictive_band(
        d_plus=d_plus, d_minus=d_minus, alpha_plus=alpha_plus,
        alpha_minus=alpha_minus, S=S, levels=levels, rng=rng,
    )


def _band_diagnostic_context(condition, replication, setting, simulation_key,
                             master_seed, historical_summary):
    """Identify the actual construction without consuming random draws."""
    context = {
        "experiment": EXPERIMENT_NAME, "protocol": MAIN_PROTOCOL_VERSION,
        "stage": "band_construction",
        "method": SETTING_METHODS[setting], "setting": setting,
        "prior_family": SETTING_FAMILIES[setting], "prior_mass": SETTING_MASSES[setting],
        "condition_id": condition.condition_id, "condition_index": condition.index,
        "replication_zero_based": replication, "K": N_BINS, "master_seed": master_seed,
        "execution_position": _execution_order(replication).index(setting) + 1,
        "data_stream_id": semantic_stream_id(master_seed, *_stream_keys(condition, replication)[0]),
        "simulation_stream_id": semantic_stream_id(master_seed, *simulation_key),
    }
    if setting in _INFORMED_SETTINGS:
        context.update(
            historical_stream_id=semantic_stream_id(master_seed, *_historical_stream_key(condition, replication)),
            historical_grid_auc=historical_summary.grid_auc,
        )
        if SETTING_FAMILIES[setting] == "linear":
            context["restricted_target"] = min(HISTORICAL_AUC_BOUNDS[1],
                max(HISTORICAL_AUC_BOUNDS[0], historical_summary.grid_auc))
        else:
            context.update(historical_counts_positive=historical_summary.d_plus.tolist(),
                           historical_counts_negative=historical_summary.d_minus.tolist())
    return context


def _evaluate_methods(
    condition: ClassCountCondition, replication: int, histograms: TestExternalHistograms,
    *, historical_summary: HistoricalSummary, n_samples: int, master_seed: int,
) -> dict[str, tuple]:
    """Construct all nine predictions before inspecting any external outcome."""
    assert isinstance(historical_summary, HistoricalSummary)
    builders = {"uniform": build_uniform_predictive_band, "linear": build_linear_predictive_band,
                "histograms": build_histogram_predictive_band}
    for counts in (histograms.d_plus_test, histograms.d_minus_test, historical_summary.d_plus, historical_summary.d_minus):
        assert not counts.flags.writeable, "Shared observed histograms must remain immutable."
    arguments, keys = {}, {}
    for setting in SETTINGS:
        keys[setting] = _simulation_stream_key(condition, replication, setting)
        arguments[setting] = dict(
            d_plus=histograms.d_plus_test, d_minus=histograms.d_minus_test,
            S=n_samples, levels=NOMINAL_LEVELS,
            rng=make_semantic_rng(master_seed, *keys[setting]),
        )
        family = SETTING_FAMILIES[setting]
        arguments[setting]["prior_mass"] = SETTING_MASSES[setting]
        if family == "linear":
            arguments[setting]["historical_grid_auc"] = historical_summary.grid_auc
        elif family == "histograms":
            arguments[setting].update(historical_d_plus=historical_summary.d_plus,
                                      historical_d_minus=historical_summary.d_minus)
    if replication == 0:
        for setting in SETTINGS:
            warm_arguments = dict(arguments[setting])
            warm_key = _warmup_stream_key(condition.index, setting)
            warm_arguments["rng"] = make_semantic_rng(master_seed, *warm_key)
            warm_up_band(builders[SETTING_FAMILIES[setting]], diagnostic_context=_band_diagnostic_context(
                condition, replication, setting, warm_key, master_seed, historical_summary,
            ), **warm_arguments)
    predictions = {}
    for position, setting in enumerate(_execution_order(replication), start=1):
        band, elapsed_ns = build_timed_band(builders[SETTING_FAMILIES[setting]],
            diagnostic_context=_band_diagnostic_context(
                condition, replication, setting, keys[setting], master_seed, historical_summary,
            ), **arguments[setting])
        predictions[setting] = (band, elapsed_ns, position)
    assert set(predictions) == set(SETTINGS), "Every prior setting must be constructed before evaluation."
    assert {value[2] for value in predictions.values()} == set(range(1, len(SETTINGS) + 1))
    results = {}
    for setting in SETTINGS:
        try:
            measurements = evaluate_replication(histograms, predictions[setting][0])
        except Exception as error:
            context = _band_diagnostic_context(condition, replication, setting,
                keys[setting], master_seed, historical_summary)
            context.update(stage="external_evaluation", S=n_samples)
            error.add_note(f"External evaluation diagnostics: {context!r}")
            raise
        results[setting] = (measurements, keys[setting], predictions[setting][1], predictions[setting][2])
    return results


def _evaluate_one_replication(
    condition: ClassCountCondition, replication: int, *, gamma: np.ndarray,
    n_samples: int, master_seed: int,
) -> tuple[dict[str, tuple], tuple[int, ...], HistoricalSummary]:
    """Preserve the four test/external draws; prepare independent history once."""
    np.testing.assert_array_equal(gamma, fixed_uniform_grid(N_BINS))
    data_key = _stream_keys(condition, replication)[0]
    try:
        data_rng = make_semantic_rng(master_seed, *data_key)
        dataset = generate_simulation_dataset(condition.score_scenario,
            condition.n_positive, condition.n_negative, rng=data_rng)
        assert dataset.positive_test_scores.shape == (condition.n_positive,)
        assert dataset.positive_external_scores.shape == (condition.n_positive,)
        assert dataset.negative_test_scores.shape == (condition.n_negative,)
        assert dataset.negative_external_scores.shape == (condition.n_negative,)
        histograms = build_test_external_histograms(
            dataset.positive_test_scores, dataset.negative_test_scores,
            dataset.positive_external_scores, dataset.negative_external_scores, gamma,
        )
        assert int(histograms.d_plus_test.sum()) == int(histograms.d_plus_external.sum()) == condition.n_positive
        assert int(histograms.d_minus_test.sum()) == int(histograms.d_minus_external.sum()) == condition.n_negative
    except Exception as error:
        context = dict(stage="test_external_preparation", protocol=MAIN_PROTOCOL_VERSION,
                       condition_id=condition.condition_id, condition_index=condition.index,
                       replication_zero_based=replication, master_seed=master_seed,
                       data_stream_id=semantic_stream_id(master_seed, *data_key),
                       n_positive=condition.n_positive, n_negative=condition.n_negative,
                       K=N_BINS, S=n_samples)
        error.add_note(f"Test/external preparation diagnostics: {context!r}")
        raise
    history = _generate_historical_summary(condition, replication, gamma=gamma, master_seed=master_seed)
    assert int(history.d_plus.sum()) == HISTORICAL_N_POSITIVE
    assert int(history.d_minus.sum()) == HISTORICAL_N_NEGATIVE
    results = _evaluate_methods(condition, replication, histograms,
        historical_summary=history, n_samples=n_samples, master_seed=master_seed)
    return results, data_key, history


def _historical_metadata(setting: str, history: HistoricalSummary | None) -> dict[str, object]:
    """Prepare recorded prior summaries after construction; never cache a timed prior."""
    fields = dict.fromkeys(_HISTORICAL_FIELDS, "")
    family = SETTING_FAMILIES[setting]
    if family in ("linear", "histograms"):
        assert isinstance(history, HistoricalSummary)
        assert int(history.d_plus.sum()) == HISTORICAL_N_POSITIVE
        assert int(history.d_minus.sum()) == HISTORICAL_N_NEGATIVE
        fields.update(historical_n_positive=HISTORICAL_N_POSITIVE,
                      historical_n_negative=HISTORICAL_N_NEGATIVE,
                      historical_grid_auc=history.grid_auc)
        if family == "linear":
            expected_auc = restrict_historical_auc(history.grid_auc)
            fields["historical_auc_adjusted"] = int(expected_auc != history.grid_auc)
        else:
            # Metadata is recomputed outside timing, not shared with a constructor.
            _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(history.d_plus, history.d_minus)
            expected_auc = prior_expected_grid_auc(q_plus, q_minus)
            fields["historical_counts_positive"] = json.dumps(history.d_plus.tolist(), separators=(",", ":"))
            fields["historical_counts_negative"] = json.dumps(history.d_minus.tolist(), separators=(",", ":"))
    else:
        assert history is None
        assert family == "uniform"
        expected_auc = 0.5
    return dict(prior_expected_grid_auc=expected_auc, **fields)


def _raw_rows_for_replication(
    condition: ClassCountCondition, replication: int, measurements: ReplicationMeasurements,
    *, n_replications: int, n_samples: int, master_seed: int, data_key: tuple[int, ...],
    simulation_key: tuple[int, ...], setting: str,
    historical_summary: HistoricalSummary | None = None,
    computation_time_ns: int, execution_position: int,
) -> tuple[dict[str, object], ...]:
    """Flatten one prediction into two full-precision rows, one per nominal level."""
    assert setting in SETTINGS
    assert data_key == _stream_keys(condition, replication)[0]
    assert simulation_key == _simulation_stream_key(condition, replication, setting)
    history_fields = _historical_metadata(setting, historical_summary)
    if setting in _INFORMED_SETTINGS:
        history_fields["historical_stream_id"] = semantic_stream_id(master_seed,
            *_historical_stream_key(condition, replication))
    positive_beta, negative_beta = simulation_beta_parameters(condition.score_scenario)
    data_stream_id = semantic_stream_id(master_seed, *data_key)
    simulation_stream_id = semantic_stream_id(master_seed, *simulation_key)
    assert data_stream_id != simulation_stream_id
    records = raw_level_measurements_from_replication(measurements, NOMINAL_LEVELS)
    computation_time_ns = validate_positive_integer("computation_time_ns", computation_time_ns, minimum=0)
    execution_position = validate_positive_integer("execution_position", execution_position, minimum=1)
    assert execution_position == _execution_order(replication).index(setting) + 1
    rows = []
    for record in records:
        row: dict[str, object] = {
            "schema_version": RAW_SCHEMA_VERSION, "experiment": EXPERIMENT_NAME,
            "method": SETTING_METHODS[setting], "setting": setting,
            "condition_index": condition.index, "condition_id": condition.condition_id,
            "replication": replication, "level": record.level, "score_scenario": condition.score_scenario,
            "positive_beta_a": positive_beta[0], "positive_beta_b": positive_beta[1],
            "negative_beta_a": negative_beta[0], "negative_beta_b": negative_beta[1],
            "n_positive": condition.n_positive, "n_negative": condition.n_negative,
            "K": N_BINS, "S": n_samples, "B": n_replications,
            "prior_profile": SETTING_PROFILES[setting],
            "prior_mass": SETTING_MASSES[setting],
            **history_fields,
            "rng_algorithm": RNG_ALGORITHM, "master_seed": master_seed,
            "data_stream_id": data_stream_id, "simulation_stream_id": simulation_stream_id,
            "computation_time_ns": computation_time_ns, "execution_position": execution_position,
            "timing_threads": 1,
        }
        row.update(raw_level_measurement_fields(record))
        assert tuple(row) == RAW_FIELDNAMES
        rows.append(row)
    assert len(rows) == len(NOMINAL_LEVELS)
    return tuple(rows)


def _parse_historical_counts(row, field_name, expected_total) -> np.ndarray:
    values = json.loads(row[field_name])
    assert isinstance(values, list) and len(values) == N_BINS
    assert all(type(value) is int and value >= 0 for value in values)
    assert sum(values) == expected_total
    assert row[field_name] == json.dumps(values, separators=(",", ":")), "Non-canonical histogram counts."
    return np.asarray(values, dtype=np.int64)


def _validate_replication_bundle(rows, records) -> None:
    """Validate a complete paired replication, then let the reader discard it."""
    expected_keys = {(setting, level) for setting in SETTINGS for level in NOMINAL_LEVELS}
    assert set(rows) == set(records) == expected_keys, "Incomplete paired replication."
    common_history = None
    histogram_counts = None
    for setting in SETTINGS:
        selected = [rows[setting, level] for level in NOMINAL_LEVELS]
        assert_complete_level_bundle([records[setting, level] for level in NOMINAL_LEVELS], NOMINAL_LEVELS)
        stable_fields = ("data_stream_id", "simulation_stream_id", "computation_time_ns",
                         "execution_position", "timing_threads", "prior_expected_grid_auc", *_HISTORICAL_FIELDS)
        for field_name in stable_fields:
            assert selected[0][field_name] == selected[1][field_name], f"Inconsistent {field_name} across levels."
        if setting in _INFORMED_SETTINGS:
            row = selected[0]
            history = tuple(row[name] for name in ("historical_n_positive", "historical_n_negative",
                                                   "historical_stream_id", "historical_grid_auc"))
            if common_history is None:
                common_history = history
            assert common_history == history, "Historically informed settings must share one historical sample."
            if SETTING_FAMILIES[setting] == "histograms":
                counts = (row["historical_counts_positive"], row["historical_counts_negative"])
                if histogram_counts is None:
                    histogram_counts = counts
                assert histogram_counts == counts, "Histogram settings must share the same observed counts."
    for level in NOMINAL_LEVELS:
        selected = [rows[setting, level] for setting in SETTINGS]
        reference = selected[0]
        # Every row's condition, counts and streams are also checked individually.
        # External ROC deviations can differ because each setting has its own center.
        for row in selected[1:]:
            for field_name in ("experiment", "condition_id", "data_stream_id", "rng_algorithm", "master_seed"):
                assert row[field_name] == reference[field_name], f"Unpaired {field_name}."
            for field_name in ("replication", "K", "S", "B", "n_positive", "n_negative"):
                assert parse_canonical_int(row, field_name) == parse_canonical_int(reference, field_name), f"Unpaired {field_name}."
            assert parse_finite_float(row, "external_grid_auc") == parse_finite_float(reference, "external_grid_auc"), "Unpaired external AUC."
        assert len({row["simulation_stream_id"] for row in selected}) == len(SETTINGS)
        assert len({row["execution_position"] for row in selected}) == len(SETTINGS)


def validate_raw_results_csv(
    csv_path: str | Path, *, conditions: Sequence[ClassCountCondition] = CLASS_COUNT_CONDITIONS,
    n_replications: int = OFFICIAL_REPLICATIONS, n_samples: int = OFFICIAL_SAMPLES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> int:
    """Validate every row with a compact seen mask and temporary replication bundles."""
    conditions = _validate_conditions(conditions)
    n_replications = validate_positive_integer("n_replications", n_replications, minimum=1)
    n_samples = validate_positive_integer("n_samples", n_samples, minimum=2)
    make_semantic_rng(master_seed, _EXPERIMENT_STREAM_KEY, 0, 0, 0)
    master_seed = int(master_seed)
    condition_lookup = {c.condition_id: (slot, c) for slot, c in enumerate(conditions)}
    seen = np.zeros((len(conditions), n_replications, len(SETTINGS), len(NOMINAL_LEVELS)), dtype=bool)
    pending = {}
    row_count = 0
    row = {}
    try:
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
                raise ValueError(f"Expected the v{MAIN_PROTOCOL_VERSION} synthetic comparison CSV header; earlier protocols cannot be reused.")
            for row in reader:
                assert set(row) == set(RAW_FIELDNAMES)
                if row["schema_version"] != RAW_SCHEMA_VERSION:
                    raise ValueError(f"Expected schema {RAW_SCHEMA_VERSION!r}, found {row['schema_version']!r}; "
                                     "old or mixed synthetic protocol versions cannot be combined.")
                assert row["experiment"] == EXPERIMENT_NAME
                setting = row["setting"]
                assert setting in SETTINGS
                assert row["method"] == SETTING_METHODS[setting]
                assert_method_metadata(row)
                family = SETTING_FAMILIES[setting]
                assert row["prior_profile"] == SETTING_PROFILES[setting]
                assert parse_finite_float(row, "prior_mass") == SETTING_MASSES[setting]
                if setting in _INFORMED_SETTINGS:
                    assert parse_canonical_int(row, "historical_n_positive") == HISTORICAL_N_POSITIVE
                    assert parse_canonical_int(row, "historical_n_negative") == HISTORICAL_N_NEGATIVE
                    historical_auc = parse_finite_float(row, "historical_grid_auc")
                    assert 0.0 <= historical_auc <= 1.0
                    if family == "linear":
                        target = restrict_historical_auc(historical_auc)
                        assert parse_finite_float(row, "prior_expected_grid_auc") == target
                        assert parse_canonical_int(row, "historical_auc_adjusted") == int(target != historical_auc)
                        assert row["historical_counts_positive"] == row["historical_counts_negative"] == ""
                    else:
                        assert row["historical_auc_adjusted"] == ""
                        d_plus = _parse_historical_counts(row, "historical_counts_positive", HISTORICAL_N_POSITIVE)
                        d_minus = _parse_historical_counts(row, "historical_counts_negative", HISTORICAL_N_NEGATIVE)
                        assert historical_auc == historical_auc_from_histograms(d_plus, d_minus)
                        q_plus = (d_plus + 1.0 / N_BINS) / (HISTORICAL_N_POSITIVE + 1)
                        q_minus = (d_minus + 1.0 / N_BINS) / (HISTORICAL_N_NEGATIVE + 1)
                        expected_auc = float(np.dot(q_plus, np.cumsum(q_minus) - 0.5 * q_minus))
                        assert np.isclose(parse_finite_float(row, "prior_expected_grid_auc"), expected_auc, rtol=0, atol=1e-12)
                else:
                    assert all(row[name] == "" for name in _HISTORICAL_FIELDS)
                    assert family == "uniform"
                    assert parse_finite_float(row, "prior_expected_grid_auc") == 0.5
                assert row["rng_algorithm"] == RNG_ALGORITHM
                condition_id = row["condition_id"]
                assert condition_id in condition_lookup
                condition_slot, condition = condition_lookup[condition_id]
                replication = parse_canonical_int(row, "replication")
                assert 0 <= replication < n_replications
                measurements = parse_raw_level_measurements(row)
                level = measurements.level
                assert level in NOMINAL_LEVELS
                assert parse_canonical_int(row, "condition_index") == condition.index
                assert row["score_scenario"] == condition.score_scenario
                for name, expected in (("n_positive", condition.n_positive), ("n_negative", condition.n_negative),
                                       ("K", N_BINS), ("S", n_samples), ("B", n_replications), ("master_seed", master_seed)):
                    assert parse_canonical_int(row, name) == expected
                positive_beta, negative_beta = simulation_beta_parameters(condition.score_scenario)
                recorded_beta = tuple(parse_finite_float(row, name) for name in (
                    "positive_beta_a", "positive_beta_b", "negative_beta_a", "negative_beta_b"))
                if recorded_beta != (*positive_beta, *negative_beta):
                    raise ValueError(f"Beta metadata {recorded_beta!r} does not match synthetic design "
                                     f"v{SYNTHETIC_DESIGN_VERSION}: {(*positive_beta, *negative_beta)!r}; "
                                     "old or mixed score scenarios cannot be reused.")
                data_key = _stream_keys(condition, replication)[0]
                simulation_key = _simulation_stream_key(condition, replication, setting)
                assert row["data_stream_id"] == semantic_stream_id(master_seed, *data_key)
                assert row["simulation_stream_id"] == semantic_stream_id(master_seed, *simulation_key)
                assert row["data_stream_id"] != row["simulation_stream_id"]
                if setting in _INFORMED_SETTINGS:
                    assert row["historical_stream_id"] == semantic_stream_id(master_seed, *_historical_stream_key(condition, replication))
                    assert row["historical_stream_id"] not in (row["data_stream_id"], row["simulation_stream_id"])
                assert parse_canonical_int(row, "execution_position") == _execution_order(replication).index(setting) + 1
                key = (condition_slot, replication, SETTINGS.index(setting), NOMINAL_LEVELS.index(level))
                assert not seen[key], "Duplicate setting/configuration/replication/level."
                seen[key] = True
                rows, records = pending.setdefault((condition_slot, replication), ({}, {}))
                rows[setting, level] = row
                records[setting, level] = measurements
                if len(rows) == len(SETTINGS) * len(NOMINAL_LEVELS):
                    _validate_replication_bundle(rows, records)
                    del pending[condition_slot, replication]
                row_count += 1
    except Exception as error:
        context = {name: row.get(name) for name in (
            "setting", "condition_id", "replication", "level", "master_seed",
            "data_stream_id", "simulation_stream_id", "historical_stream_id",
        )}
        context.update(stage="csv_validation", protocol=MAIN_PROTOCOL_VERSION,
                       csv_path=str(csv_path), csv_line=row_count + 2)
        error.add_note(f"CSV validation diagnostics: {context!r}")
        raise
    assert np.all(seen), "Missing settings, replications or nominal levels."
    assert not pending
    assert row_count == seen.size
    return row_count


def _print_progress(condition, completed, n_replications, totals, time_totals, historical_totals):
    """Report cumulative values from the written rows, outside all timing."""
    print(
        f"Progress {condition.condition_id}: {completed}/{n_replications} "
        f"replications ({100.0 * completed / n_replications:.1f}%); "
        f"last replication index={completed - 1} (zero-based).",
        flush=True,
    )
    for setting in SETTINGS:
        print(
            f"  {SETTING_LABELS[setting]}: mean construction time="
            f"{time_totals[setting] / completed / 1e6:.3f} ms "
            "(both levels together)",
            flush=True,
        )
        for level in NOMINAL_LEVELS:
            values = totals[setting, level]
            print(
                f"    level={level:.0%}: "
                f"ROC coverage={100.0 * values['roc_covered'] / completed:.2f}%, "
                f"AUC coverage={100.0 * values['auc_covered'] / completed:.2f}%, "
                f"mean radius={values['roc_radius'] / completed:.6f}, "
                f"mean AUC width={values['auc_width'] / completed:.6f}",
                flush=True,
            )

    print(
        f"  Historical AUC: mean={historical_totals['auc_sum'] / completed:.6f}, "
        f"range=[{historical_totals['auc_min']:.6f}, {historical_totals['auc_max']:.6f}]; "
        f"target: mean={historical_totals['target_sum'] / completed:.6f}, "
        f"range=[{historical_totals['target_min']:.6f}, {historical_totals['target_max']:.6f}]; "
        f"adjusted={historical_totals['adjusted']}/{completed} "
        f"({100 * historical_totals['adjusted'] / completed:.2f}%).",
        flush=True,
    )


def _protect_previous_results(output_csv: Path) -> None:
    """Keep new runs out of previous-protocol files, including custom paths."""
    resolved = output_csv.resolve()
    previous_csv_names = {"raw_replications.csv", *(f"raw_replications_v{version}.csv"
                          for version in range(1, MAIN_PROTOCOL_VERSION))}
    previous_table_names = {"latex_tables", *(f"latex_tables_v{version}"
                            for version in range(1, MAIN_PROTOCOL_VERSION))}
    if any(
        path.name.lower() in previous_csv_names
        or any(part.lower() in previous_table_names for part in path.parts)
        for path in (output_csv, resolved)
    ):
        raise ValueError(f"Previous output paths are protected; choose a v{MAIN_PROTOCOL_VERSION} CSV path.")
    if resolved.is_file():
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                recorded_schema = row.get("schema_version")
                if recorded_schema and recorded_schema != RAW_SCHEMA_VERSION:
                    raise ValueError(
                        f"Existing CSV belongs to {recorded_schema!r}; previous or other protocols "
                        f"are protected even with overwrite=True. Choose a separate v{MAIN_PROTOCOL_VERSION} CSV."
                    )


def _collect_experiment_metadata(conditions, n_replications, n_samples, master_seed):
    """Capture reproducibility details before scientific sampling, outside timing."""
    source_paths = [Path(__file__), *(
        PY_DIR / "roc_utils" / f"{name}.py" for name in (
            "__init__", "experiment_provenance", "experiment_results",
            "evaluation_criteria", "utilities_experiments", "prior_profiles",
            "posterior_predictive_roc_sampler", "bootstrap_predictive_roc_sampler",
            "simulation_datasets", "semantic_rng",
        )
    )]
    configuration = {
        "experiment": EXPERIMENT_NAME, "main_protocol_version": MAIN_PROTOCOL_VERSION,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "synthetic_design_version": SYNTHETIC_DESIGN_VERSION,
        "master_seed": master_seed, "rng_algorithm": RNG_ALGORITHM,
        "B": n_replications, "S": n_samples, "K": N_BINS,
        "nominal_levels": NOMINAL_LEVELS,
        "historical_class_counts": [HISTORICAL_N_POSITIVE, HISTORICAL_N_NEGATIVE],
        "linear_auc_restriction": HISTORICAL_AUC_BOUNDS,
        "settings": [dict(setting=setting, method=SETTING_METHODS[setting],
                          family=SETTING_FAMILIES[setting], mass=SETTING_MASSES[setting])
                     for setting in SETTINGS],
        "conditions": [dict(index=c.index, identifier=c.condition_id,
                            score_scenario=c.score_scenario,
                            n_positive=c.n_positive, n_negative=c.n_negative,
                            beta_parameters=simulation_beta_parameters(c.score_scenario))
                       for c in conditions],
        "stream_key_templates": {
            "data": [1, _DATA_STREAM_ROLE, "condition_index", "replication"],
            "history": [1, _HISTORICAL_DATA_STREAM_ROLE, "condition_index", "replication"],
            "simulations": {setting: [1, role, "condition_index", "replication"]
                            for setting, role in _SETTING_STREAM_ROLES.items()},
            "warmup": {setting: [1, _WARMUP_STREAM_ROLE, role, "condition_index", 0]
                       for setting, role in _SETTING_STREAM_ROLES.items()},
        },
        "replication_indexing": "zero-based",
        "data_draw_order": ["positive_test", "negative_test", "positive_external", "negative_external"],
        "historical_draw_order": ["positive", "negative"],
        "execution_order": f"reporting order rotated left by replication modulo {len(SETTINGS)}",
        "timing_scope": "prior construction and complete predictive bands for both levels",
    }
    return collect_run_metadata(source_root=PY_DIR.parent, source_paths=source_paths,
                                configuration=configuration)


def run_experiment(
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replications: int = OFFICIAL_REPLICATIONS,
    n_samples: int = OFFICIAL_SAMPLES,
    conditions: Sequence[ClassCountCondition] = CLASS_COUNT_CONDITIONS,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Run the experiment and atomically write its authoritative raw CSV.

    The defaults are the official manuscript design. Smaller ``B`` and ``S``
    values and condition subsets are exposed only for deterministic tests and
    short pilot runs; every row records the values actually used.
    """

    _assert_setting_definitions()
    output_csv = Path(output_csv)
    _protect_previous_results(output_csv)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("output_csv must have a .csv suffix.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be Boolean.")
    if not isinstance(show_progress, bool):
        raise TypeError("show_progress must be Boolean.")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results: {output_csv}"
        )

    conditions = _validate_conditions(conditions)
    n_replications = validate_positive_integer(
        "n_replications",
        n_replications,
        minimum=1,
    )
    n_samples = validate_positive_integer(
        "n_samples",
        n_samples,
        minimum=2,
    )
    make_semantic_rng(master_seed, _EXPERIMENT_STREAM_KEY, 0, 0, 0)
    master_seed = int(master_seed)

    metadata = _collect_experiment_metadata(conditions, n_replications, n_samples, master_seed)
    gamma = fixed_uniform_grid(N_BINS)
    np.testing.assert_array_equal(gamma, np.linspace(0.0, 1.0, 11))

    expected_rows = (
        len(conditions) * n_replications * len(NOMINAL_LEVELS) * len(SETTINGS)
    )
    if show_progress:
        print(
            f"Synthetic comparison v{MAIN_PROTOCOL_VERSION} (score design v{SYNTHETIC_DESIGN_VERSION}): {len(conditions)} conditions; "
            f"B={n_replications}, S={n_samples} per setting, K={N_BINS}, "
            f"levels={NOMINAL_LEVELS}, master_seed={master_seed}.",
            flush=True,
        )
        print(
            f"Python {platform.python_version()}, NumPy {np.__version__}, "
            f"RNG={RNG_ALGORITHM}; construction threads=1. "
            f"Historical sample: {HISTORICAL_N_POSITIVE}+{HISTORICAL_N_NEGATIVE} scores; "
            f"linear AUC restriction={HISTORICAL_AUC_BOUNDS}; "
            "uniform, linear and histogram priors each use m=1, m=5 and m=20 per class.",
            flush=True,
        )
        print("Reporting order: " + "; ".join(SETTING_LABELS[s] for s in SETTINGS), flush=True)
        print(f"Each replication constructs {len(SETTINGS)} settings once for both levels; "
              f"execution order rotates through {len(SETTINGS)} positions. Existing timings are not reused.", flush=True)
        print(
            "Reported values are cumulative within each condition. "
            f"The CSV will be published after validation of all {expected_rows} rows: "
            f"{output_csv}",
            flush=True,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    written_rows = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_csv.name}.",
            suffix=".tmp",
            dir=output_csv.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDNAMES)
            writer.writeheader()

            for condition_number, condition in enumerate(conditions, start=1):
                if show_progress:
                    totals = {
                        (setting, level): dict.fromkeys(_PROGRESS_FIELDS, 0.0)
                        for setting in SETTINGS for level in NOMINAL_LEVELS
                    }
                    time_totals = dict.fromkeys(SETTINGS, 0)
                    historical_totals = dict(auc_sum=0., target_sum=0., adjusted=0,
                        auc_min=float("inf"), auc_max=float("-inf"),
                        target_min=float("inf"), target_max=float("-inf"))
                    positive_beta, negative_beta = simulation_beta_parameters(
                        condition.score_scenario,
                    )
                    print(
                        f"Starting condition {condition_number}/{len(conditions)}: "
                        f"{condition.condition_id}; "
                        f"Beta+={positive_beta}, Beta-={negative_beta}. "
                        "Shared test/external class counts; separate unmeasured "
                        "warm-ups before the first replication.",
                        flush=True,
                    )
                for replication in range(n_replications):
                    results, data_key, history = _evaluate_one_replication(
                        condition, replication, gamma=gamma,
                        n_samples=n_samples, master_seed=master_seed,
                    )
                    for setting in SETTINGS:
                        measurements, simulation_key, elapsed_ns, position = results[setting]
                        try:
                            rows = _raw_rows_for_replication(
                                condition, replication, measurements,
                                n_replications=n_replications, n_samples=n_samples,
                                master_seed=master_seed, data_key=data_key,
                                simulation_key=simulation_key, setting=setting,
                                historical_summary=history if setting in _INFORMED_SETTINGS else None,
                                computation_time_ns=elapsed_ns, execution_position=position,
                            )
                            writer.writerows(rows)
                        except Exception as error:
                            context = _band_diagnostic_context(
                                condition, replication, setting, simulation_key, master_seed, history)
                            context.update(stage="csv_serialization", S=n_samples,
                                           output_csv=str(output_csv))
                            error.add_note(f"CSV serialization diagnostics: {context!r}")
                            raise
                        written_rows += len(rows)
                        if show_progress:
                            # Reuse the exact coverage flags and final widths
                            # written to the CSV; count each construction once.
                            time_totals[setting] += elapsed_ns
                            for row in rows:
                                for field in _PROGRESS_FIELDS:
                                    totals[setting, row["level"]][field] += row[field]

                    if show_progress:
                        historical_auc = history.grid_auc
                        target = restrict_historical_auc(historical_auc)
                        historical_totals["adjusted"] += int(target != historical_auc)
                        for name, value in (("auc", historical_auc), ("target", target)):
                            historical_totals[name + "_sum"] += value
                            historical_totals[name + "_min"] = min(historical_totals[name + "_min"], value)
                            historical_totals[name + "_max"] = max(historical_totals[name + "_max"], value)
                    completed = replication + 1
                    if show_progress and (
                        completed in (1, n_replications)
                        or completed % _PROGRESS_INTERVAL == 0
                    ):
                        _print_progress(
                            condition, completed, n_replications, totals, time_totals, historical_totals,
                        )

            handle.flush()
            os.fsync(handle.fileno())

        assert written_rows == expected_rows
        if show_progress:
            print(
                f"Simulation complete. Validating all {written_rows} raw rows "
                "and paired results before publishing the CSV...",
                flush=True,
            )
        validated_rows = validate_raw_results_csv(
            temporary_path,
            conditions=conditions,
            n_replications=n_replications,
            n_samples=n_samples,
            master_seed=master_seed,
        )
        assert validated_rows == written_rows
        if output_csv.exists() and not overwrite:
            raise FileExistsError(
                f"Output appeared while the experiment was running: {output_csv}"
            )
        _protect_previous_results(output_csv)
        assert_sources_unchanged(metadata, source_root=PY_DIR.parent)
        metadata_path = publish_run_metadata(temporary_path, output_csv, metadata)
        if overwrite:
            os.replace(temporary_path, output_csv)
        else:
            # Same-directory hard-link publication is atomic and refuses an
            # output created by another process after the existence checks.
            os.link(temporary_path, output_csv)
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    assert output_csv.is_file()
    if show_progress:
        print(f"Saved {written_rows} validated raw rows to {output_csv}", flush=True)
        print(f"Saved reproducibility metadata to {metadata_path}", flush=True)
    return output_csv


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the deliberately small command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the official Main evaluation on synthetic data from main.tex "
            "and save full-precision raw replication results."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"Master PCG64 seed (default: {DEFAULT_MASTER_SEED}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV after validating the new file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the exact official design specified in main.tex."""

    arguments = _build_argument_parser().parse_args(argv)
    assert SIMULATION_SCENARIOS == ("Easy", "Medium", "Hard")
    assert CLASS_COUNT_PAIRS == (
        (20, 20),
        (100, 100),
        (500, 500),
        (20, 500),
        (500, 20),
    )
    assert len(CLASS_COUNT_CONDITIONS) == 15
    assert N_BINS == 10
    assert (HISTORICAL_N_POSITIVE, HISTORICAL_N_NEGATIVE) == (500, 500), (
        "The manuscript's historical sample has 500 observations per class."
    )
    assert HISTORICAL_AUC_BOUNDS == (0.5, 0.86), "The linear AUC restriction must be [0.5, 0.86]."
    _assert_setting_definitions()
    assert NOMINAL_LEVELS == (0.80, 0.95)
    assert OFFICIAL_REPLICATIONS == 10_000
    assert OFFICIAL_SAMPLES == 4000
    return run_experiment(
        arguments.output,
        master_seed=arguments.master_seed,
        n_replications=OFFICIAL_REPLICATIONS,
        n_samples=OFFICIAL_SAMPLES,
        conditions=CLASS_COUNT_CONDITIONS,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
