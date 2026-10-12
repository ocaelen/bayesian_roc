"""Run the manuscript's sensitivity analysis for prior specification."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError(
        "This experiment relies on runtime assertions; do not use python -O."
    )

from roc_utils.experiment_results import (
    MEASUREMENT_RAW_FIELDNAMES,
    RawLevelMeasurements,
    assert_complete_level_bundle,
    assert_method_metadata,
    assert_paired_method_rows,
    parse_canonical_int,
    parse_finite_float,
    parse_raw_level_measurements,
    raw_level_measurement_fields,
    raw_level_measurements_from_replication,
    validate_positive_integer,
)
from roc_utils.posterior_predictive_roc_sampler import fixed_uniform_grid
from roc_utils.prior_profiles import (
    auc_calibrated_prior_bin_probabilities,
    beta_bin_probabilities,
    dirichlet_parameters_from_bin_probabilities,
    prior_expected_grid_auc,
    symmetric_prior_bin_probabilities,
    validate_prior_mean_probabilities,
)
from roc_utils.semantic_rng import (
    RNG_ALGORITHM,
    make_semantic_rng,
    semantic_stream_id,
)
from roc_utils.simulation_datasets import (
    SYNTHETIC_DESIGN_VERSION,
    generate_simulation_dataset,
    simulation_beta_parameters,
)
from roc_utils.utilities_experiments import (
    ReplicationMeasurements,
    TestExternalHistograms,
    build_test_external_histograms,
    evaluate_replication,
    build_posterior_predictive_band,
    build_bootstrap_predictive_band,
    build_timed_band,
    warm_up_band,
)


EXPERIMENT_NAME = "prior_shape_concentration_and_misspecification"
RESULTS_DIRECTORY_NAME = "sensitivity_to_prior_specification"
RAW_SCHEMA_VERSION = f"prior_shape_raw_v{SYNTHETIC_DESIGN_VERSION}"

SCORE_SCENARIO = "Medium"
N_POSITIVE = 50
N_NEGATIVE = 50
N_BINS = 10
PRIOR_PROFILES = ("oracle", "auc_compatible", "symmetric")
PRIOR_MASSES = (1, 20, 50)

# Both nominal levels are specified by the common experimental protocol.
NOMINAL_LEVELS = (0.80, 0.95)
OFFICIAL_REPLICATIONS = 10_000
OFFICIAL_SAMPLES = 4000
DEFAULT_MASTER_SEED = 20250731

_EXPERIMENT_STREAM_KEY = 2
_DATA_STREAM_ROLE = 1
_POSTERIOR_STREAM_ROLE = 2
_BOOTSTRAP_STREAM_ROLE = 3
_WARMUP_STREAM_ROLE = 4
METHODS = ("DiP-ROC", "BootP-ROC")
BOOTSTRAP_CONDITION_ID = "bootstrap"
BOOTSTRAP_CONDITION_INDEX = 9

PY_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = (
    PY_DIR
    / "results"
    / "current_experiments"
    / RESULTS_DIRECTORY_NAME
    / f"raw_replications_v{SYNTHETIC_DESIGN_VERSION}.csv"
)

assert DEFAULT_OUTPUT_CSV.parent.name == RESULTS_DIRECTORY_NAME


@dataclass(frozen=True)
class PriorCondition:
    """One prior mean profile and one total Dirichlet concentration."""

    index: int
    prior_profile: str
    prior_mass: int

    def __post_init__(self) -> None:
        assert type(self.index) is int
        assert self.index >= 0
        assert self.prior_profile in PRIOR_PROFILES
        assert type(self.prior_mass) is int
        assert self.prior_mass > 0
        assert self.prior_mass in PRIOR_MASSES

    @property
    def condition_id(self) -> str:
        """Return the stable identifier of this prior configuration."""

        return f"{self.prior_profile}_m_{self.prior_mass}"


@dataclass(frozen=True)
class PriorProfileDefinition:
    """One immutable pair of prior mean vectors and its expected grid AUC."""

    name: str
    q_plus: np.ndarray
    q_minus: np.ndarray
    expected_grid_auc: float

    def __post_init__(self) -> None:
        assert self.name in PRIOR_PROFILES
        validated_plus, validated_minus = validate_prior_mean_probabilities(
            self.q_plus,
            self.q_minus,
        )
        assert validated_plus.shape == (N_BINS,)
        assert validated_minus.shape == (N_BINS,)
        assert not self.q_plus.flags.writeable
        assert not self.q_minus.flags.writeable
        assert np.isclose(
            prior_expected_grid_auc(self.q_plus, self.q_minus),
            self.expected_grid_auc,
            rtol=0.0,
            atol=1e-12,
        )


PRIOR_CONDITION_SETTINGS = (
    ("oracle", 1),
    ("oracle", 20),
    ("auc_compatible", 1),
    ("auc_compatible", 20),
    ("symmetric", 1),
    ("symmetric", 20),
    ("oracle", 50),
    ("auc_compatible", 50),
    ("symmetric", 50),
)

# Keep the six m in {1, 20} condition indices and their semantic posterior
# streams unchanged; the replacement m=50 conditions occupy indices 6--8.
PRIOR_CONDITIONS = tuple(
    PriorCondition(condition_index, profile, prior_mass)
    for condition_index, (profile, prior_mass) in enumerate(
        PRIOR_CONDITION_SETTINGS
    )
)

assert PRIOR_PROFILES == ("oracle", "auc_compatible", "symmetric")
assert PRIOR_MASSES == (1, 20, 50)
assert PRIOR_CONDITION_SETTINGS[:6] == (
    ("oracle", 1),
    ("oracle", 20),
    ("auc_compatible", 1),
    ("auc_compatible", 20),
    ("symmetric", 1),
    ("symmetric", 20),
)
assert PRIOR_CONDITION_SETTINGS[6:] == (
    ("oracle", 50),
    ("auc_compatible", 50),
    ("symmetric", 50),
)
assert set(PRIOR_CONDITION_SETTINGS) == {
    (profile, prior_mass)
    for profile in PRIOR_PROFILES
    for prior_mass in PRIOR_MASSES
}
assert len(PRIOR_CONDITIONS) == 9
assert tuple(condition.index for condition in PRIOR_CONDITIONS) == tuple(
    range(9)
)
assert len({condition.condition_id for condition in PRIOR_CONDITIONS}) == 9


RAW_FIELDNAMES = (
    "schema_version",
    "experiment",
    "method",
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
    "true_grid_auc",
    "prior_expected_grid_auc",
    "rng_algorithm",
    "master_seed",
    "data_stream_id",
    "simulation_stream_id",
    "computation_time_ns",
    "execution_position",
    "timing_threads",
    *MEASUREMENT_RAW_FIELDNAMES,
)


def _immutable_probability_vector(values: np.ndarray) -> np.ndarray:
    """Return a read-only copy of one prior mean probability vector."""

    values = np.asarray(values, dtype=float).copy()
    values.setflags(write=False)
    assert not values.flags.writeable
    return values


def build_prior_profiles() -> dict[str, PriorProfileDefinition]:
    """Construct and cross-check the manuscript's three prior mean profiles."""

    positive_shape, negative_shape = simulation_beta_parameters(SCORE_SCENARIO)
    assert positive_shape == (3.5, 2.5)
    assert negative_shape == (2.5, 3.5)
    k_oracle, oracle_plus, oracle_minus = beta_bin_probabilities(
        N_BINS,
        positive_shape,
        negative_shape,
    )
    true_grid_auc = prior_expected_grid_auc(oracle_plus, oracle_minus)
    k_calibrated, calibrated_plus, calibrated_minus = (
        auc_calibrated_prior_bin_probabilities(
            N_BINS,
            true_grid_auc,
        )
    )
    k_symmetric, symmetric_plus, symmetric_minus = (
        symmetric_prior_bin_probabilities(N_BINS)
    )

    np.testing.assert_array_equal(k_oracle, np.arange(1.0, 11.0))
    np.testing.assert_array_equal(k_calibrated, k_oracle)
    np.testing.assert_array_equal(k_symmetric, k_oracle)
    assert np.isclose(
        true_grid_auc,
        0.7298989020330722,
        rtol=0.0,
        atol=1e-12,
    )
    assert true_grid_auc < 13.0 / 15.0
    calibrated_auc = prior_expected_grid_auc(
        calibrated_plus,
        calibrated_minus,
    )
    symmetric_auc = prior_expected_grid_auc(
        symmetric_plus,
        symmetric_minus,
    )
    assert np.isclose(calibrated_auc, true_grid_auc, rtol=0.0, atol=1e-12)
    assert np.isclose(symmetric_auc, 0.5, rtol=0.0, atol=1e-12)
    assert not np.allclose(oracle_plus, calibrated_plus, rtol=0.0, atol=1e-6)
    assert not np.allclose(oracle_minus, calibrated_minus, rtol=0.0, atol=1e-6)
    assert np.allclose(oracle_plus, oracle_minus[::-1], rtol=0.0, atol=1e-12)
    assert np.allclose(
        calibrated_plus,
        calibrated_minus[::-1],
        rtol=0.0,
        atol=1e-12,
    )

    profiles = {
        "oracle": PriorProfileDefinition(
            name="oracle",
            q_plus=_immutable_probability_vector(oracle_plus),
            q_minus=_immutable_probability_vector(oracle_minus),
            expected_grid_auc=true_grid_auc,
        ),
        "auc_compatible": PriorProfileDefinition(
            name="auc_compatible",
            q_plus=_immutable_probability_vector(calibrated_plus),
            q_minus=_immutable_probability_vector(calibrated_minus),
            expected_grid_auc=true_grid_auc,
        ),
        "symmetric": PriorProfileDefinition(
            name="symmetric",
            q_plus=_immutable_probability_vector(symmetric_plus),
            q_minus=_immutable_probability_vector(symmetric_minus),
            expected_grid_auc=0.5,
        ),
    }
    assert tuple(profiles) == PRIOR_PROFILES
    return profiles


def _build_condition_priors(
    profiles: dict[str, PriorProfileDefinition],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Construct all alpha=m*q prior pairs once per experiment run."""

    assert tuple(profiles) == PRIOR_PROFILES
    priors = {}
    for condition in PRIOR_CONDITIONS:
        profile = profiles[condition.prior_profile]
        alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(
            profile.q_plus,
            profile.q_minus,
            condition.prior_mass,
        )
        np.testing.assert_allclose(
            alpha_plus,
            condition.prior_mass * profile.q_plus,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            alpha_minus,
            condition.prior_mass * profile.q_minus,
            rtol=0.0,
            atol=1e-15,
        )
        assert np.isclose(
            np.sum(alpha_plus),
            condition.prior_mass,
            rtol=0.0,
            atol=1e-12,
        )
        assert np.isclose(
            np.sum(alpha_minus),
            condition.prior_mass,
            rtol=0.0,
            atol=1e-12,
        )
        priors[condition.index] = (alpha_plus, alpha_minus)

    assert set(priors) == set(range(len(PRIOR_CONDITIONS)))
    conditions_by_setting = {
        (condition.prior_profile, condition.prior_mass): condition
        for condition in PRIOR_CONDITIONS
    }
    assert len(conditions_by_setting) == len(PRIOR_CONDITIONS)
    for profile_name in PRIOR_PROFILES:
        mass_one_condition = conditions_by_setting[(profile_name, 1)]
        mass_one = priors[mass_one_condition.index]
        for prior_mass in PRIOR_MASSES:
            condition = conditions_by_setting[(profile_name, prior_mass)]
            alpha_plus, alpha_minus = priors[condition.index]
            np.testing.assert_allclose(
                alpha_plus,
                prior_mass * mass_one[0],
                rtol=0.0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                alpha_minus,
                prior_mass * mass_one[1],
                rtol=0.0,
                atol=1e-14,
            )
    return priors


def _validate_conditions(
    conditions: Iterable[PriorCondition],
) -> tuple[PriorCondition, ...]:
    """Validate a non-empty, duplicate-free subset of the official design."""

    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("At least one prior condition is required.")
    for condition in conditions:
        if not isinstance(condition, PriorCondition):
            raise TypeError("conditions must contain PriorCondition values.")
        if condition not in PRIOR_CONDITIONS:
            raise ValueError("Every condition must belong to the official design.")
    indices = [condition.index for condition in conditions]
    if len(set(indices)) != len(indices):
        raise ValueError("conditions must not contain duplicates.")
    return tuple(sorted(conditions, key=lambda condition: condition.index))


def _data_stream_key(replication: int) -> tuple[int, ...]:
    """Return the data key shared by all priors in one replication."""

    return (_EXPERIMENT_STREAM_KEY, _DATA_STREAM_ROLE, replication)


def _posterior_stream_key(
    condition: PriorCondition,
    replication: int,
) -> tuple[int, ...]:
    """Return one condition-specific posterior Monte Carlo key."""

    return (
        _EXPERIMENT_STREAM_KEY,
        _POSTERIOR_STREAM_ROLE,
        condition.index,
        replication,
    )


def _bootstrap_stream_key(replication: int) -> tuple[int, ...]:
    """One bootstrap stream shared across all prior comparisons."""
    return (_EXPERIMENT_STREAM_KEY, _BOOTSTRAP_STREAM_ROLE, replication)


def _warmup_stream_key(condition_index: int, method: str) -> tuple[int, ...]:
    assert method in METHODS
    role = (
        _POSTERIOR_STREAM_ROLE if method == "DiP-ROC" else _BOOTSTRAP_STREAM_ROLE
    )
    return (_EXPERIMENT_STREAM_KEY, _WARMUP_STREAM_ROLE, role, condition_index, 0)


def _band_diagnostic_context(condition, replication, method, simulation_key, master_seed):
    """Identify a prior setting or the shared bootstrap without random draws."""
    return {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "condition_id": condition.condition_id if condition is not None else BOOTSTRAP_CONDITION_ID,
        "condition_index": condition.index if condition is not None else BOOTSTRAP_CONDITION_INDEX,
        "replication_zero_based": replication,
        "K": N_BINS,
        "master_seed": master_seed,
        "data_stream_id": semantic_stream_id(master_seed, *_data_stream_key(replication)),
        "simulation_stream_id": semantic_stream_id(master_seed, *simulation_key),
    }


def _warm_up_conditions(histograms, conditions, priors, *, n_samples, master_seed):
    """Warm each distinct construction without advancing any inference stream."""
    common = dict(d_plus=histograms.d_plus_test, d_minus=histograms.d_minus_test,
                  S=n_samples, levels=NOMINAL_LEVELS)
    for condition in conditions:
        alpha_plus, alpha_minus = priors[condition.index]
        warm_key = _warmup_stream_key(condition.index, "DiP-ROC")
        warm_up_band(build_posterior_predictive_band, **common,
                     alpha_plus=alpha_plus, alpha_minus=alpha_minus,
                     rng=make_semantic_rng(master_seed, *warm_key),
                     diagnostic_context=_band_diagnostic_context(
                         condition, 0, "DiP-ROC", warm_key, master_seed))
    warm_key = _warmup_stream_key(BOOTSTRAP_CONDITION_INDEX, "BootP-ROC")
    warm_up_band(build_bootstrap_predictive_band, **common,
                 rng=make_semantic_rng(master_seed, *warm_key),
                 diagnostic_context=_band_diagnostic_context(
                     None, 0, "BootP-ROC", warm_key, master_seed))


def _generate_replication_histograms(
    replication: int,
    *,
    gamma: np.ndarray,
    master_seed: int,
) -> tuple[TestExternalHistograms, tuple[int, ...]]:
    """Generate and histogram one paired Medium test/external realization."""

    np.testing.assert_array_equal(gamma, fixed_uniform_grid(N_BINS))
    data_key = _data_stream_key(replication)
    dataset = generate_simulation_dataset(
        SCORE_SCENARIO,
        N_POSITIVE,
        N_NEGATIVE,
        rng=make_semantic_rng(master_seed, *data_key),
    )
    assert dataset.positive_test_scores.shape == (N_POSITIVE,)
    assert dataset.negative_test_scores.shape == (N_NEGATIVE,)
    assert dataset.positive_external_scores.shape == (N_POSITIVE,)
    assert dataset.negative_external_scores.shape == (N_NEGATIVE,)

    histograms = build_test_external_histograms(
        dataset.positive_test_scores,
        dataset.negative_test_scores,
        dataset.positive_external_scores,
        dataset.negative_external_scores,
        gamma,
    )
    assert int(np.sum(histograms.d_plus_test)) == N_POSITIVE
    assert int(np.sum(histograms.d_minus_test)) == N_NEGATIVE
    assert int(np.sum(histograms.d_plus_external)) == N_POSITIVE
    assert int(np.sum(histograms.d_minus_external)) == N_NEGATIVE
    return histograms, data_key


def _evaluate_prior_condition(
    condition: PriorCondition,
    replication: int,
    histograms: TestExternalHistograms,
    alpha_plus: np.ndarray,
    alpha_minus: np.ndarray,
    *,
    n_samples: int,
    master_seed: int,
) -> tuple[ReplicationMeasurements, tuple[int, ...], int]:
    """Evaluate one prior on the shared histograms of one replication."""

    posterior_key = _posterior_stream_key(condition, replication)
    band, elapsed_ns = build_timed_band(
        build_posterior_predictive_band,
        diagnostic_context=_band_diagnostic_context(
            condition, replication, "DiP-ROC", posterior_key, master_seed,
        ),
        d_plus=histograms.d_plus_test, d_minus=histograms.d_minus_test,
        alpha_plus=alpha_plus, alpha_minus=alpha_minus,
        S=n_samples, levels=NOMINAL_LEVELS,
        rng=make_semantic_rng(master_seed, *posterior_key),
    )
    return evaluate_replication(histograms, band), posterior_key, elapsed_ns


def _evaluate_bootstrap_condition(replication, histograms, *, n_samples, master_seed):
    key = _bootstrap_stream_key(replication)
    band, elapsed_ns = build_timed_band(
        build_bootstrap_predictive_band,
        diagnostic_context=_band_diagnostic_context(
            None, replication, "BootP-ROC", key, master_seed,
        ),
        d_plus=histograms.d_plus_test, d_minus=histograms.d_minus_test,
        S=n_samples, levels=NOMINAL_LEVELS,
        rng=make_semantic_rng(master_seed, *key),
    )
    return evaluate_replication(histograms, band), key, elapsed_ns


def _raw_rows_for_condition_replication(
    condition: PriorCondition | None,
    replication: int,
    measurements: ReplicationMeasurements,
    profile: PriorProfileDefinition | None,
    *,
    true_grid_auc: float,
    n_replications: int,
    n_samples: int,
    master_seed: int,
    data_key: tuple[int, ...],
    simulation_key: tuple[int, ...],
    method: str,
    computation_time_ns: int,
    execution_position: int,
) -> tuple[dict[str, object], ...]:
    """Flatten one prior evaluation into one raw row per level."""

    assert method in METHODS
    if method == "DiP-ROC":
        assert condition is not None and profile is not None
        assert profile.name == condition.prior_profile
        assert simulation_key == _posterior_stream_key(condition, replication)
        condition_index, condition_id = condition.index, condition.condition_id
    else:
        assert condition is None and profile is None
        assert simulation_key == _bootstrap_stream_key(replication)
        condition_index, condition_id = BOOTSTRAP_CONDITION_INDEX, BOOTSTRAP_CONDITION_ID
    assert data_key == _data_stream_key(replication)
    positive_beta, negative_beta = simulation_beta_parameters(SCORE_SCENARIO)
    data_stream_id = semantic_stream_id(master_seed, *data_key)
    simulation_stream_id = semantic_stream_id(master_seed, *simulation_key)
    assert data_stream_id != simulation_stream_id
    level_records = raw_level_measurements_from_replication(
        measurements,
        NOMINAL_LEVELS,
    )

    computation_time_ns = validate_positive_integer(
        "computation_time_ns", computation_time_ns, minimum=0,
    )
    execution_position = validate_positive_integer(
        "execution_position", execution_position, minimum=1,
    )
    rows = []
    for record in level_records:
        row: dict[str, object] = {
            "schema_version": RAW_SCHEMA_VERSION,
            "experiment": EXPERIMENT_NAME,
            "method": method,
            "condition_index": condition_index,
            "condition_id": condition_id,
            "replication": replication,
            "level": record.level,
            "score_scenario": SCORE_SCENARIO,
            "positive_beta_a": positive_beta[0],
            "positive_beta_b": positive_beta[1],
            "negative_beta_a": negative_beta[0],
            "negative_beta_b": negative_beta[1],
            "n_positive": N_POSITIVE,
            "n_negative": N_NEGATIVE,
            "K": N_BINS,
            "S": n_samples,
            "B": n_replications,
            "prior_profile": condition.prior_profile if condition is not None else "",
            "prior_mass": condition.prior_mass if condition is not None else "",
            "true_grid_auc": true_grid_auc,
            "prior_expected_grid_auc": profile.expected_grid_auc if profile is not None else "",
            "rng_algorithm": RNG_ALGORITHM,
            "master_seed": master_seed,
            "data_stream_id": data_stream_id,
            "simulation_stream_id": simulation_stream_id,
            "computation_time_ns": computation_time_ns,
            "execution_position": execution_position,
            "timing_threads": 1,
        }
        row.update(raw_level_measurement_fields(record))
        assert tuple(row) == RAW_FIELDNAMES
        rows.append(row)

    assert len(rows) == len(NOMINAL_LEVELS)
    assert rows[0]["data_stream_id"] == rows[1]["data_stream_id"]
    assert rows[0]["simulation_stream_id"] == rows[1]["simulation_stream_id"]
    return tuple(rows)


def validate_raw_results_csv(
    csv_path: str | Path,
    *,
    conditions: Sequence[PriorCondition] = PRIOR_CONDITIONS,
    n_replications: int = OFFICIAL_REPLICATIONS,
    n_samples: int = OFFICIAL_SAMPLES,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> int:
    """Read back and validate every prior/replication/level raw row."""

    csv_path = Path(csv_path)
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
    profiles = build_prior_profiles()
    true_grid_auc = profiles["oracle"].expected_grid_auc
    conditions_by_id = {
        condition.condition_id: condition for condition in conditions
    }
    primary_keys: set[tuple[str, str, int, float]] = set()
    bundles: dict[
        tuple[str, str, int],
        dict[float, tuple[RawLevelMeasurements, str, str, str, str]],
    ] = {}
    row_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
            raise ValueError("Expected the v3 synthetic comparison CSV header; v1/v2 files cannot be reused.")
        for row in reader:
            row_count += 1
            assert set(row) == set(RAW_FIELDNAMES)
            assert_method_metadata(row)
            method = row["method"]
            assert method in METHODS
            if row["schema_version"] != RAW_SCHEMA_VERSION:
                raise ValueError(
                    f"Expected schema {RAW_SCHEMA_VERSION!r}, found {row['schema_version']!r}; "
                    "old or mixed synthetic protocol versions cannot be combined."
                )
            assert row["experiment"] == EXPERIMENT_NAME
            assert row["score_scenario"] == SCORE_SCENARIO
            assert row["rng_algorithm"] == RNG_ALGORITHM

            condition_id = row["condition_id"]
            if method == "DiP-ROC":
                assert condition_id in conditions_by_id
                condition = conditions_by_id[condition_id]
            else:
                assert condition_id == BOOTSTRAP_CONDITION_ID
                condition = None
            condition_index = parse_canonical_int(row, "condition_index")
            replication = parse_canonical_int(row, "replication")
            raw_measurements = parse_raw_level_measurements(row)
            level = raw_measurements.level
            assert condition_index == (condition.index if condition is not None else BOOTSTRAP_CONDITION_INDEX)
            order = (None, *conditions) if replication % 2 == 0 else (*conditions, None)
            assert parse_canonical_int(row, "execution_position") == order.index(condition) + 1
            assert 0 <= replication < n_replications
            assert level in NOMINAL_LEVELS
            if condition is not None:
                assert row["prior_profile"] == condition.prior_profile
                assert parse_canonical_int(row, "prior_mass") == condition.prior_mass
            assert parse_canonical_int(row, "n_positive") == N_POSITIVE
            assert parse_canonical_int(row, "n_negative") == N_NEGATIVE
            assert parse_canonical_int(row, "K") == N_BINS
            assert parse_canonical_int(row, "S") == n_samples
            assert parse_canonical_int(row, "B") == n_replications
            assert parse_canonical_int(row, "master_seed") == master_seed

            positive_beta, negative_beta = simulation_beta_parameters(
                SCORE_SCENARIO
            )
            recorded_beta = tuple(parse_finite_float(row, name) for name in (
                "positive_beta_a", "positive_beta_b", "negative_beta_a", "negative_beta_b",
            ))
            if recorded_beta != (*positive_beta, *negative_beta):
                raise ValueError(
                    f"Beta metadata {recorded_beta!r} does not match synthetic design "
                    f"v{SYNTHETIC_DESIGN_VERSION}: {(*positive_beta, *negative_beta)!r}; "
                    "old or mixed score scenarios cannot be reused."
                )
            assert parse_finite_float(row, "true_grid_auc") == true_grid_auc
            if condition is not None:
                assert parse_finite_float(row, "prior_expected_grid_auc") == (
                    profiles[condition.prior_profile].expected_grid_auc
                )

            data_key = _data_stream_key(replication)
            posterior_key = (_posterior_stream_key(condition, replication)
                             if condition is not None else _bootstrap_stream_key(replication))
            data_stream_id = semantic_stream_id(master_seed, *data_key)
            simulation_stream_id = semantic_stream_id(
                master_seed,
                *posterior_key,
            )
            assert row["data_stream_id"] == data_stream_id
            assert row["simulation_stream_id"] == simulation_stream_id
            assert data_stream_id != simulation_stream_id

            primary_key = (method, condition_id, replication, level)
            assert primary_key not in primary_keys
            primary_keys.add(primary_key)
            bundle_key = (method, condition_id, replication)
            level_bundle = bundles.setdefault(bundle_key, {})
            assert level not in level_bundle
            level_bundle[level] = (
                raw_measurements,
                data_stream_id,
                simulation_stream_id,
                row["computation_time_ns"], row["execution_position"],
            )

    evaluation_settings = tuple(("DiP-ROC", condition.condition_id) for condition in conditions) + (
        ("BootP-ROC", BOOTSTRAP_CONDITION_ID),
    )
    expected_primary_keys = {
        (method, condition_id, replication, level)
        for method, condition_id in evaluation_settings
        for replication in range(n_replications)
        for level in NOMINAL_LEVELS
    }
    assert primary_keys == expected_primary_keys
    assert row_count == (len(conditions) + 1) * n_replications * len(NOMINAL_LEVELS)
    assert len(bundles) == (len(conditions) + 1) * n_replications
    for level_bundle in bundles.values():
        assert_complete_level_bundle([entry[0] for entry in level_bundle.values()], NOMINAL_LEVELS)
        for index in range(1, 5):
            assert len({entry[index] for entry in level_bundle.values()}) == 1

    def pair_reference(method, replication, entry):
        record, data_id, simulation_id, _, _ = entry
        return {
            "experiment": EXPERIMENT_NAME, "method": method,
            "replication": str(replication), "level": str(record.level),
            "K": str(N_BINS), "S": str(n_samples), "B": str(n_replications),
            "n_positive": str(N_POSITIVE), "n_negative": str(N_NEGATIVE),
            "rng_algorithm": RNG_ALGORITHM, "master_seed": str(master_seed),
            "data_stream_id": data_id, "simulation_stream_id": simulation_id,
            "external_grid_auc": str(record.external_grid_auc),
        }

    for replication in range(n_replications):
        bootstrap = bundles[("BootP-ROC", BOOTSTRAP_CONDITION_ID, replication)][NOMINAL_LEVELS[0]]
        bootstrap_row = pair_reference("BootP-ROC", replication, bootstrap)
        simulation_ids = {bootstrap[2]}
        for condition in conditions:
            reference = bundles[("DiP-ROC", condition.condition_id, replication)][NOMINAL_LEVELS[0]]
            assert_paired_method_rows(pair_reference("DiP-ROC", replication, reference), bootstrap_row)
            assert reference[2] not in simulation_ids
            simulation_ids.add(reference[2])
        assert len(simulation_ids) == len(conditions) + 1
    return row_count


def _protect_previous_results(output_csv: Path) -> None:
    """Keep v3 runs out of previous-protocol files, including custom paths."""
    resolved = output_csv.resolve()
    if any(
        path.name.lower() in {"raw_replications.csv", "raw_replications_v1.csv", "raw_replications_v2.csv"}
        or any(part.lower() in {"latex_tables", "latex_tables_v1", "latex_tables_v2"}
               for part in path.parts)
        for path in (output_csv, resolved)
    ):
        raise ValueError("Previous v1/v2 output paths are protected; choose a v3 CSV path.")
    if resolved.is_file():
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle), None)
        recorded_schema = first.get("schema_version") if first else None
        if recorded_schema and recorded_schema != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"Existing CSV belongs to {recorded_schema!r}; previous or other protocols "
                "are protected even with overwrite=True. Choose a separate v3 CSV."
            )


def run_experiment(
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replications: int = OFFICIAL_REPLICATIONS,
    n_samples: int = OFFICIAL_SAMPLES,
    conditions: Sequence[PriorCondition] = PRIOR_CONDITIONS,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Run the paired prior experiment and atomically write its raw CSV."""

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

    gamma = fixed_uniform_grid(N_BINS)
    np.testing.assert_array_equal(gamma, np.linspace(0.0, 1.0, 11))
    profiles = build_prior_profiles()
    all_condition_priors = _build_condition_priors(profiles)
    condition_priors = {
        condition.index: all_condition_priors[condition.index]
        for condition in conditions
    }
    true_grid_auc = profiles["oracle"].expected_grid_auc

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

            for replication in range(n_replications):
                histograms, data_key = _generate_replication_histograms(
                    replication,
                    gamma=gamma,
                    master_seed=master_seed,
                )
                if replication == 0:
                    _warm_up_conditions(histograms, conditions, condition_priors,
                                        n_samples=n_samples, master_seed=master_seed)
                results = {}
                # Alternate the complete DiP block against the single bootstrap.
                order = (None, *conditions) if replication % 2 == 0 else (*conditions, None)
                for position, condition in enumerate(order, start=1):
                    if condition is None:
                        result = _evaluate_bootstrap_condition(
                            replication, histograms, n_samples=n_samples,
                            master_seed=master_seed)
                        results[BOOTSTRAP_CONDITION_ID] = (*result, position)
                    else:
                        alpha_plus, alpha_minus = condition_priors[condition.index]
                        result = _evaluate_prior_condition(
                            condition, replication, histograms, alpha_plus, alpha_minus,
                            n_samples=n_samples, master_seed=master_seed)
                        results[condition.condition_id] = (*result, position)
                simulation_ids = set()
                for condition in (*conditions, None):
                    method = "BootP-ROC" if condition is None else "DiP-ROC"
                    condition_id = BOOTSTRAP_CONDITION_ID if condition is None else condition.condition_id
                    measurements, key, elapsed_ns, position = results[condition_id]
                    simulation_id = semantic_stream_id(master_seed, *key)
                    assert simulation_id not in simulation_ids
                    assert simulation_id != semantic_stream_id(master_seed, *data_key)
                    simulation_ids.add(simulation_id)
                    rows = _raw_rows_for_condition_replication(
                        condition, replication, measurements,
                        None if condition is None else profiles[condition.prior_profile],
                        true_grid_auc=true_grid_auc, n_replications=n_replications,
                        n_samples=n_samples, master_seed=master_seed,
                        data_key=data_key, simulation_key=key, method=method,
                        computation_time_ns=elapsed_ns, execution_position=position,
                    )
                    writer.writerows(rows)
                    written_rows += len(rows)
                assert len(simulation_ids) == len(conditions) + 1

                if show_progress and (
                    (replication + 1) % 100 == 0
                    or replication + 1 == n_replications
                ):
                    print(
                        f"Completed {replication + 1}/{n_replications} "
                        "paired replications."
                    )

            handle.flush()
            os.fsync(handle.fileno())

        expected_rows = (
            (len(conditions) + 1) * n_replications * len(NOMINAL_LEVELS)
        )
        assert written_rows == expected_rows
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
        os.replace(temporary_path, output_csv)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    assert output_csv.is_file()
    if show_progress:
        print(f"Saved {written_rows} validated raw rows to {output_csv}")
    return output_csv


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the deliberately small official command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Sensitivity to prior specification experiment and save "
            "full-precision raw replication results."
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
    """Run the official paired method comparison design."""

    arguments = _build_argument_parser().parse_args(argv)
    assert SCORE_SCENARIO == "Medium"
    assert N_POSITIVE == N_NEGATIVE == 50
    assert N_BINS == 10
    assert PRIOR_PROFILES == ("oracle", "auc_compatible", "symmetric")
    assert PRIOR_MASSES == (1, 20, 50)
    assert len(PRIOR_CONDITIONS) == 9
    assert NOMINAL_LEVELS == (0.80, 0.95)
    assert OFFICIAL_REPLICATIONS == 10_000
    assert OFFICIAL_SAMPLES == 4000
    return run_experiment(
        arguments.output,
        master_seed=arguments.master_seed,
        n_replications=OFFICIAL_REPLICATIONS,
        n_samples=OFFICIAL_SAMPLES,
        conditions=PRIOR_CONDITIONS,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
