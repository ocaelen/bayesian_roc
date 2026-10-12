"""Run ten-bin real-data evaluation and the supplementary five/twenty-bin checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
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
    parse_canonical_int,
    parse_finite_float,
    parse_raw_level_measurements,
    raw_level_measurement_fields,
    raw_level_measurements_from_replication,
    validate_positive_integer,
)
from roc_utils.posterior_predictive_roc_sampler import (
    fixed_uniform_grid,
    symmetric_dirichlet_priors,
)
from roc_utils.real_datasets import (
    DEFAULT_REAL_DATA_DIR,
    REAL_DATASET_NAMES,
    LoadedRealDataset,
    PreparedRealDataset,
    RealDatasetMetadata,
    load_real_dataset,
    real_dataset_metadata,
    split_and_encode_real_dataset,
)
from roc_utils.real_model import (
    MODEL_C,
    MODEL_CLASS_WEIGHT,
    MODEL_FIT_INTERCEPT,
    MODEL_L1_RATIO,
    MODEL_MAX_ITER,
    MODEL_PENALTY,
    MODEL_SOLVER,
    MODEL_SPEC_VERSION,
    MODEL_TOL,
    NUMERIC_SCALING,
    SKLEARN_VERSION,
    RealDatasetScores,
    fit_logistic_regression_scores,
)
from roc_utils.semantic_rng import (
    RNG_ALGORITHM,
    make_semantic_rng,
    semantic_stream_id,
)
from roc_utils.utilities_experiments import (
    ReplicationMeasurements,
    TestExternalHistograms,
    build_test_external_histograms,
    evaluate_replication,
    build_posterior_predictive_band,
    build_timed_band,
    warm_up_band,
)


from roc_utils.experiment_provenance import (
    assert_sources_unchanged, collect_run_metadata, file_sha256,
    publish_run_metadata, read_run_metadata,
)


EXPERIMENT_NAME = "real_data"
REAL_PROTOCOL_VERSION = 3
RUN_METADATA_SCHEMA = "real_run_metadata_v1"
RESULTS_DIRECTORY_NAME = "empirical_evaluation_on_real_world_datasets"
RAW_SCHEMA_VERSION = "real_data_raw_v3"

N_BINS_VALUES = (5, 10, 20)
MAIN_N_BINS = 10
SUPPLEMENTARY_N_BINS_VALUES = (5, 20)
PRIOR_PROFILE = "symmetric"
PRIOR_MASS = 1.0
NOMINAL_LEVELS = (0.80, 0.95)
OFFICIAL_REPLICATIONS = 300
OFFICIAL_SAMPLES = 4000
DEFAULT_MASTER_SEED = 20250731

_EXPERIMENT_STREAM_KEY = 3
_DATA_STREAM_ROLE = 1
_POSTERIOR_STREAM_ROLE = 2
_WARMUP_STREAM_ROLE = 4
METHODS = ("DiP-ROC",)
_MODEL_CLASS_WEIGHT_CSV = "none"

PY_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = (
    PY_DIR
    / "results"
    / "current_experiments"
    / RESULTS_DIRECTORY_NAME
    / "raw_replications_v3.csv"
)

assert DEFAULT_OUTPUT_CSV.parent.name == RESULTS_DIRECTORY_NAME


@dataclass(frozen=True)
class RealDataCondition:
    """One pinned real dataset and one fixed uniform grid size."""

    index: int
    dataset_index: int
    dataset_name: str
    n_bins: int

    def __post_init__(self) -> None:
        assert type(self.index) is int
        assert self.index >= 0
        assert type(self.dataset_index) is int
        assert 0 <= self.dataset_index < len(REAL_DATASET_NAMES)
        assert self.dataset_name == REAL_DATASET_NAMES[self.dataset_index]
        assert type(self.n_bins) is int
        assert self.n_bins in N_BINS_VALUES

    @property
    def condition_id(self) -> str:
        """Return a stable readable dataset-grid identifier."""

        dataset_slug = self.dataset_name.casefold().replace(" ", "_")
        assert dataset_slug.replace("_", "").isalnum()
        return f"{dataset_slug}_k_{self.n_bins}"


REAL_DATA_CONDITIONS = tuple(
    RealDataCondition(
        index=condition_index,
        dataset_index=dataset_index,
        dataset_name=dataset_name,
        n_bins=n_bins,
    )
    for condition_index, (dataset_index, dataset_name, n_bins) in enumerate(
        (dataset_index, dataset_name, n_bins)
        for dataset_index, dataset_name in enumerate(REAL_DATASET_NAMES)
        for n_bins in N_BINS_VALUES
    )
)

assert len(REAL_DATASET_NAMES) == 12
assert N_BINS_VALUES == (5, 10, 20)
assert MAIN_N_BINS not in SUPPLEMENTARY_N_BINS_VALUES
assert tuple(sorted((MAIN_N_BINS, *SUPPLEMENTARY_N_BINS_VALUES))) == N_BINS_VALUES
assert len(REAL_DATA_CONDITIONS) == 36
assert tuple(condition.index for condition in REAL_DATA_CONDITIONS) == tuple(
    range(36)
)
assert len({condition.condition_id for condition in REAL_DATA_CONDITIONS}) == 36


RAW_FIELDNAMES = (
    "schema_version",
    "experiment",
    "method",
    "condition_index",
    "condition_id",
    "replication",
    "level",
    "dataset_index",
    "dataset_name",
    "dataset_slug",
    "openml_data_id",
    "openml_version",
    "openml_file_id",
    "md5_checksum",
    "n_source_rows",
    "n_complete_rows",
    "n_removed_rows",
    "n_train",
    "n_train_positive",
    "n_train_negative",
    "n_positive",
    "n_negative",
    "encoded_feature_count",
    "K",
    "S",
    "B",
    "prior_profile",
    "prior_mass",
    "model_spec_version",
    "sklearn_version",
    "model_penalty",
    "model_C",
    "model_l1_ratio",
    "model_solver",
    "model_fit_intercept",
    "model_class_weight",
    "model_max_iter",
    "model_tol",
    "numeric_scaling",
    "model_n_iter",
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
    conditions: Iterable[RealDataCondition],
) -> tuple[RealDataCondition, ...]:
    """Validate and canonically order a non-empty official-design subset."""

    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("At least one real-data condition is required.")
    for condition in conditions:
        if not isinstance(condition, RealDataCondition):
            raise TypeError("conditions must contain RealDataCondition values.")
        if condition not in REAL_DATA_CONDITIONS:
            raise ValueError("Every condition must belong to the official design.")
    indices = [condition.index for condition in conditions]
    if len(set(indices)) != len(indices):
        raise ValueError("conditions must not contain duplicates.")
    return tuple(sorted(conditions, key=lambda condition: condition.index))


def _conditions_by_dataset(
    conditions: Sequence[RealDataCondition],
) -> tuple[tuple[int, tuple[RealDataCondition, ...]], ...]:
    """Group a validated condition subset in canonical dataset order."""

    grouped = []
    for dataset_index in range(len(REAL_DATASET_NAMES)):
        dataset_conditions = tuple(
            condition
            for condition in conditions
            if condition.dataset_index == dataset_index
        )
        if dataset_conditions:
            grouped.append((dataset_index, dataset_conditions))
    assert sum(len(item[1]) for item in grouped) == len(conditions)
    return tuple(grouped)


def _split_counts(metadata: RealDatasetMetadata) -> tuple[int, int, int, int]:
    """Return train-positive, train-negative, held-out-positive/negative."""

    held_positive = metadata.n_positive // 5
    held_negative = metadata.n_negative // 5
    train_positive = metadata.n_positive - 2 * held_positive
    train_negative = metadata.n_negative - 2 * held_negative
    assert held_positive >= 1
    assert held_negative >= 1
    assert train_positive + 2 * held_positive == metadata.n_positive
    assert train_negative + 2 * held_negative == metadata.n_negative
    return train_positive, train_negative, held_positive, held_negative


def _data_stream_key(dataset_index: int, replication: int) -> tuple[int, ...]:
    """Return the split key shared by all K values in one replication."""

    return (
        _EXPERIMENT_STREAM_KEY,
        _DATA_STREAM_ROLE,
        dataset_index,
        replication,
    )


def _posterior_stream_key(
    condition: RealDataCondition,
    replication: int,
) -> tuple[int, ...]:
    """Return one dataset-grid-specific posterior Monte Carlo key."""

    return (
        _EXPERIMENT_STREAM_KEY,
        _POSTERIOR_STREAM_ROLE,
        condition.index,
        replication,
    )


def _simulation_stream_key(
    condition: RealDataCondition, replication: int, method: str,
) -> tuple[int, ...]:
    """Keep the original dataset-grid-specific DiP simulation stream."""
    assert method == "DiP-ROC"
    return _posterior_stream_key(condition, replication)


def _warmup_stream_key(condition_index: int, method: str) -> tuple[int, ...]:
    assert method == "DiP-ROC"
    return (_EXPERIMENT_STREAM_KEY, _WARMUP_STREAM_ROLE,
            _POSTERIOR_STREAM_ROLE, condition_index, 0)


def _execution_order(dataset_conditions, replication):
    """Rotate the selected ascending grids without changing stream identity."""
    assert type(replication) is int and replication >= 0
    ordered = tuple(sorted(dataset_conditions, key=lambda condition: condition.n_bins))
    assert ordered
    assert all(condition in REAL_DATA_CONDITIONS for condition in ordered)
    assert len({condition.dataset_index for condition in ordered}) == 1
    assert len({condition.n_bins for condition in ordered}) == len(ordered)
    offset = replication % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _band_diagnostic_context(condition, replication, method, simulation_key, master_seed):
    """Identify one dataset/grid construction without consuming random draws."""
    return {
        "experiment": EXPERIMENT_NAME,
        "real_protocol_version": REAL_PROTOCOL_VERSION,
        "method": method,
        "dataset_name": condition.dataset_name,
        "condition_id": condition.condition_id,
        "condition_index": condition.index,
        "replication_zero_based": replication,
        "K": condition.n_bins,
        "master_seed": master_seed,
        "data_stream_id": semantic_stream_id(
            master_seed, *_data_stream_key(condition.dataset_index, replication),
        ),
        "simulation_stream_id": semantic_stream_id(master_seed, *simulation_key),
    }


def _add_failure_note(error, phase, **context):
    """Add small reproducible context without replacing the original error."""
    error.add_note(f"Real-data diagnostics: {dict(phase=phase, **context)!r}")


def _evaluate_methods(
    condition: RealDataCondition,
    replication: int,
    histograms: TestExternalHistograms,
    alpha_plus: np.ndarray,
    alpha_minus: np.ndarray,
    *,
    n_samples: int,
    master_seed: int,
    execution_position: int = 1,
) -> dict[str, tuple]:
    """Construct DiP predictions before external evaluation, with one timing."""
    method = "DiP-ROC"
    key = _simulation_stream_key(condition, replication, method)
    arguments = dict(
        d_plus=histograms.d_plus_test, d_minus=histograms.d_minus_test,
        alpha_plus=alpha_plus, alpha_minus=alpha_minus,
        S=n_samples, levels=NOMINAL_LEVELS,
        rng=make_semantic_rng(master_seed, *key),
    )
    if replication == 0:
        warm_arguments = dict(arguments)
        warm_key = _warmup_stream_key(condition.index, method)
        warm_arguments["rng"] = make_semantic_rng(master_seed, *warm_key)
        warm_up_band(
            build_posterior_predictive_band,
            diagnostic_context=_band_diagnostic_context(
                condition, replication, method, warm_key, master_seed,
            ), **warm_arguments,
        )
    band, elapsed_ns = build_timed_band(
        build_posterior_predictive_band,
        diagnostic_context=_band_diagnostic_context(
            condition, replication, method, key, master_seed,
        ), **arguments,
    )
    try:
        measurements = evaluate_replication(histograms, band)
    except Exception as error:
        context = _band_diagnostic_context(condition, replication, method, key, master_seed)
        context.update({name: getattr(histograms, name).tolist() for name in (
            "d_plus_test", "d_minus_test", "d_plus_external", "d_minus_external",
        )})
        _add_failure_note(error, "external evaluation", **context)
        raise
    return {method: (measurements, key, elapsed_ns, execution_position)}


def _assert_loaded_source(
    dataset: LoadedRealDataset,
    metadata: RealDatasetMetadata,
) -> None:
    """Assert that a loaded source matches the pinned no-network metadata."""

    assert isinstance(dataset, LoadedRealDataset)
    assert dataset.name == metadata.name
    assert dataset.slug == metadata.slug
    assert dataset.openml_data_id == metadata.openml_data_id
    assert dataset.openml_version == metadata.openml_version
    assert dataset.openml_file_id == metadata.openml_file_id
    assert dataset.md5_checksum == metadata.md5_checksum
    assert dataset.n_source_rows == metadata.n_source_rows
    assert dataset.n_complete_rows == metadata.n_complete_rows
    assert dataset.n_removed_rows == metadata.n_removed_rows
    assert int(np.count_nonzero(dataset.y == 1)) == metadata.n_positive
    assert int(np.count_nonzero(dataset.y == 0)) == metadata.n_negative


def _assert_prepared_replication(
    dataset: PreparedRealDataset,
    metadata: RealDatasetMetadata,
    *,
    loaded_dataset: LoadedRealDataset | None = None,
) -> None:
    """Assert source identity, exact counts, and a leakage-free partition."""

    assert isinstance(dataset, PreparedRealDataset)
    assert dataset.name == metadata.name
    assert dataset.slug == metadata.slug
    assert dataset.openml_data_id == metadata.openml_data_id
    assert dataset.openml_version == metadata.openml_version
    assert dataset.openml_file_id == metadata.openml_file_id
    assert dataset.md5_checksum == metadata.md5_checksum
    assert dataset.n_source_rows == metadata.n_source_rows
    assert dataset.n_complete_rows == metadata.n_complete_rows
    assert dataset.n_removed_rows == metadata.n_removed_rows
    train_positive, train_negative, held_positive, held_negative = (
        _split_counts(metadata)
    )
    assert int(np.count_nonzero(dataset.y_train == 1)) == train_positive
    assert int(np.count_nonzero(dataset.y_train == 0)) == train_negative
    for y in (dataset.y_test, dataset.y_external):
        assert int(np.count_nonzero(y == 1)) == held_positive
        assert int(np.count_nonzero(y == 0)) == held_negative
    assert dataset.X_train.shape[1] == len(dataset.feature_names)
    assert dataset.X_test.shape[1] == len(dataset.feature_names)
    assert dataset.X_external.shape[1] == len(dataset.feature_names)
    assert len(dataset.feature_names) > 0

    partition = (
        (dataset.X_train, dataset.y_train, dataset.train_row_indices),
        (dataset.X_test, dataset.y_test, dataset.test_row_indices),
        (
            dataset.X_external,
            dataset.y_external,
            dataset.external_row_indices,
        ),
    )
    for X, y, source_indices in partition:
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(source_indices, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.size == source_indices.size
        assert source_indices.shape == (y.size,)
        assert np.all(np.isfinite(X))
        assert y.dtype == np.int64
        assert set(np.unique(y)) == {0, 1}
        assert source_indices.dtype == np.int64
        assert np.unique(source_indices).size == source_indices.size
        assert np.all(source_indices >= 0)
        assert np.all(source_indices < metadata.n_source_rows)

    all_source_indices = np.concatenate(
        tuple(source_indices for _, _, source_indices in partition)
    )
    assert all_source_indices.size == metadata.n_complete_rows
    assert np.unique(all_source_indices).size == metadata.n_complete_rows, (
        "Training, test, and external source rows must be pairwise disjoint."
    )

    if loaded_dataset is not None:
        _assert_loaded_source(loaded_dataset, metadata)
        assert dataset.source == loaded_dataset.source
        assert dataset.source_path == loaded_dataset.source_path
        np.testing.assert_array_equal(
            np.sort(all_source_indices),
            loaded_dataset.source_row_indices,
        )
        for _, y, source_indices in partition:
            complete_positions = np.searchsorted(
                loaded_dataset.source_row_indices,
                source_indices,
            )
            assert np.all(complete_positions < loaded_dataset.y.size)
            np.testing.assert_array_equal(
                loaded_dataset.source_row_indices[complete_positions],
                source_indices,
            )
            np.testing.assert_array_equal(
                loaded_dataset.y[complete_positions],
                y,
            )


def _assert_scores(
    scores: RealDatasetScores,
    metadata: RealDatasetMetadata,
) -> None:
    """Assert score counts against the pinned classwise floor split."""

    assert isinstance(scores, RealDatasetScores)
    _, _, held_positive, held_negative = _split_counts(metadata)
    assert scores.positive_test_scores.shape == (held_positive,)
    assert scores.positive_external_scores.shape == (held_positive,)
    assert scores.negative_test_scores.shape == (held_negative,)
    assert scores.negative_external_scores.shape == (held_negative,)
    # Recheck the shared arrays at orchestration boundaries: a frozen dataclass
    # alone cannot prevent accidental changes to its NumPy arrays.
    for values in (scores.positive_test_scores, scores.negative_test_scores,
                   scores.positive_external_scores, scores.negative_external_scores):
        assert not values.flags.writeable, "Held-out scores must remain immutable across grids."
        assert np.all(np.isfinite(values))
        assert np.all((0.0 <= values) & (values <= 1.0))


def _prepare_dataset_replication(
    loaded_dataset: LoadedRealDataset,
    metadata: RealDatasetMetadata,
    dataset_index: int,
    replication: int,
    *,
    master_seed: int,
) -> tuple[PreparedRealDataset, RealDatasetScores, tuple[int, ...]]:
    """Split, encode, fit, and score exactly once for one replication."""

    assert type(dataset_index) is int
    assert 0 <= dataset_index < len(REAL_DATASET_NAMES)
    assert metadata.name == REAL_DATASET_NAMES[dataset_index]
    assert loaded_dataset.name == metadata.name
    data_key = _data_stream_key(dataset_index, replication)
    phase = "split and encoding"
    try:
        prepared = split_and_encode_real_dataset(
            loaded_dataset, rng=make_semantic_rng(master_seed, *data_key),
        )
        _assert_prepared_replication(prepared, metadata, loaded_dataset=loaded_dataset)
        phase = "model fitting and scoring"
        scores = fit_logistic_regression_scores(prepared)
        _assert_scores(scores, metadata)
    except Exception as error:
        _add_failure_note(
            error, phase, dataset_name=metadata.name, dataset_index=dataset_index,
            replication_zero_based=replication, master_seed=master_seed,
            data_stream_id=semantic_stream_id(master_seed, *data_key),
            model_spec_version=MODEL_SPEC_VERSION,
        )
        raise
    return prepared, scores, data_key


def _evaluate_grid_condition(
    condition: RealDataCondition,
    replication: int,
    scores: RealDatasetScores,
    gamma: np.ndarray,
    alpha_plus: np.ndarray,
    alpha_minus: np.ndarray,
    *,
    n_samples: int,
    master_seed: int,
    execution_position: int = 1,
) -> dict[str, tuple]:
    """Evaluate one K using scores shared across the three grids."""

    assert gamma.shape == (condition.n_bins + 1,)
    np.testing.assert_array_equal(
        gamma,
        fixed_uniform_grid(condition.n_bins),
    )
    assert alpha_plus.shape == (condition.n_bins,)
    assert alpha_minus.shape == (condition.n_bins,)
    np.testing.assert_allclose(
        alpha_plus,
        np.full(condition.n_bins, 1.0 / condition.n_bins),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(alpha_plus, alpha_minus)
    try:
        histograms = build_test_external_histograms(
            scores.positive_test_scores, scores.negative_test_scores,
            scores.positive_external_scores, scores.negative_external_scores, gamma,
        )
    except Exception as error:
        _add_failure_note(error, "histogram preparation", **_band_diagnostic_context(
            condition, replication, "DiP-ROC",
            _posterior_stream_key(condition, replication), master_seed,
        ))
        raise
    results = _evaluate_methods(
        condition, replication, histograms, alpha_plus, alpha_minus,
        n_samples=n_samples, master_seed=master_seed,
        execution_position=execution_position,
    )
    return results


def _raw_rows_for_condition_replication(
    condition: RealDataCondition,
    replication: int,
    measurements: ReplicationMeasurements,
    metadata: RealDatasetMetadata,
    prepared: PreparedRealDataset,
    scores: RealDatasetScores,
    *,
    n_replications: int,
    n_samples: int,
    master_seed: int,
    data_key: tuple[int, ...],
    simulation_key: tuple[int, ...],
    method: str,
    computation_time_ns: int,
    execution_position: int,
) -> tuple[dict[str, object], ...]:
    """Flatten one grid evaluation into one full-precision row per level."""

    assert condition.dataset_name == metadata.name
    assert data_key == _data_stream_key(condition.dataset_index, replication)
    assert simulation_key == _simulation_stream_key(condition, replication, method)
    _assert_prepared_replication(prepared, metadata)
    _assert_scores(scores, metadata)
    train_positive, train_negative, held_positive, held_negative = (
        _split_counts(metadata)
    )
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
            "condition_index": condition.index,
            "condition_id": condition.condition_id,
            "replication": replication,
            "level": record.level,
            "dataset_index": condition.dataset_index,
            "dataset_name": metadata.name,
            "dataset_slug": metadata.slug,
            "openml_data_id": metadata.openml_data_id,
            "openml_version": metadata.openml_version,
            "openml_file_id": metadata.openml_file_id,
            "md5_checksum": metadata.md5_checksum,
            "n_source_rows": metadata.n_source_rows,
            "n_complete_rows": metadata.n_complete_rows,
            "n_removed_rows": metadata.n_removed_rows,
            "n_train": train_positive + train_negative,
            "n_train_positive": train_positive,
            "n_train_negative": train_negative,
            "n_positive": held_positive,
            "n_negative": held_negative,
            "encoded_feature_count": len(prepared.feature_names),
            "K": condition.n_bins,
            "S": n_samples,
            "B": n_replications,
            "prior_profile": PRIOR_PROFILE,
            "prior_mass": PRIOR_MASS,
            "model_spec_version": MODEL_SPEC_VERSION,
            "sklearn_version": SKLEARN_VERSION,
            "model_penalty": MODEL_PENALTY,
            "model_C": MODEL_C,
            "model_l1_ratio": MODEL_L1_RATIO,
            "model_solver": MODEL_SOLVER,
            "model_fit_intercept": int(MODEL_FIT_INTERCEPT),
            "model_class_weight": _MODEL_CLASS_WEIGHT_CSV,
            "model_max_iter": MODEL_MAX_ITER,
            "model_tol": MODEL_TOL,
            "numeric_scaling": NUMERIC_SCALING,
            "model_n_iter": scores.n_iterations,
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


def _validate_raw_results_csv(
    csv_path: str | Path,
    *,
    conditions: Sequence[RealDataCondition] = REAL_DATA_CONDITIONS,
    n_replications: int = OFFICIAL_REPLICATIONS,
    n_samples: int = OFFICIAL_SAMPLES,
    master_seed: int = DEFAULT_MASTER_SEED,
    metadata: dict,
    _diagnostic_context=None,
) -> int:
    """Read back and validate every real-data raw row and paired K bundle."""

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
    conditions_by_id = {
        condition.condition_id: condition for condition in conditions
    }
    metadata_by_index = {
        dataset_index: real_dataset_metadata(REAL_DATASET_NAMES[dataset_index])
        for dataset_index, _ in _conditions_by_dataset(conditions)
    }
    primary_keys: set[tuple[str, str, int, float]] = set()
    bundles: dict[
        tuple[str, str, int],
        dict[float, tuple[RawLevelMeasurements, str, str, int, int, str, str]],
    ] = {}
    selected_by_dataset = dict(_conditions_by_dataset(conditions))
    row_count = 0
    recorded_sklearn = metadata["software"]["scikit-learn"]

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == RAW_FIELDNAMES
        for row in reader:
            row_count += 1
            if _diagnostic_context is not None:
                _diagnostic_context.update(
                    csv_line=reader.line_num, condition_id=row.get("condition_id"),
                    replication_zero_based=row.get("replication"),
                    data_stream_id=row.get("data_stream_id"),
                )
            assert set(row) == set(RAW_FIELDNAMES)
            assert row["schema_version"] == RAW_SCHEMA_VERSION
            assert row["experiment"] == EXPERIMENT_NAME
            assert_method_metadata(row)
            method = row["method"]
            assert method in METHODS
            if method == "DiP-ROC":
                assert row["prior_profile"] == PRIOR_PROFILE
                assert parse_finite_float(row, "prior_mass") == PRIOR_MASS
            assert row["model_spec_version"] == MODEL_SPEC_VERSION
            assert row["sklearn_version"] == recorded_sklearn
            assert row["model_penalty"] == MODEL_PENALTY
            assert row["model_solver"] == MODEL_SOLVER
            assert row["model_class_weight"] == _MODEL_CLASS_WEIGHT_CSV
            assert row["numeric_scaling"] == NUMERIC_SCALING
            assert row["rng_algorithm"] == RNG_ALGORITHM

            condition_id = row["condition_id"]
            assert condition_id in conditions_by_id
            condition = conditions_by_id[condition_id]
            metadata = metadata_by_index[condition.dataset_index]
            condition_index = parse_canonical_int(row, "condition_index")
            dataset_index = parse_canonical_int(row, "dataset_index")
            replication = parse_canonical_int(row, "replication")
            measurements = parse_raw_level_measurements(row)
            level = measurements.level
            assert condition_index == condition.index
            assert dataset_index == condition.dataset_index
            assert 0 <= replication < n_replications
            expected_order = _execution_order(selected_by_dataset[dataset_index], replication)
            assert parse_canonical_int(row, "execution_position") == expected_order.index(condition) + 1
            assert level in NOMINAL_LEVELS
            assert row["dataset_name"] == metadata.name
            assert row["dataset_slug"] == metadata.slug
            assert row["md5_checksum"] == metadata.md5_checksum
            assert parse_canonical_int(row, "openml_data_id") == (
                metadata.openml_data_id
            )
            assert parse_canonical_int(row, "openml_version") == (
                metadata.openml_version
            )
            assert parse_canonical_int(row, "openml_file_id") == (
                metadata.openml_file_id
            )
            assert parse_canonical_int(row, "n_source_rows") == (
                metadata.n_source_rows
            )
            assert parse_canonical_int(row, "n_complete_rows") == (
                metadata.n_complete_rows
            )
            assert parse_canonical_int(row, "n_removed_rows") == (
                metadata.n_removed_rows
            )

            train_positive, train_negative, held_positive, held_negative = (
                _split_counts(metadata)
            )
            assert parse_canonical_int(row, "n_train") == (
                train_positive + train_negative
            )
            assert parse_canonical_int(row, "n_train_positive") == (
                train_positive
            )
            assert parse_canonical_int(row, "n_train_negative") == (
                train_negative
            )
            assert parse_canonical_int(row, "n_positive") == held_positive
            assert parse_canonical_int(row, "n_negative") == held_negative
            feature_count = parse_canonical_int(row, "encoded_feature_count")
            assert feature_count >= 1
            assert parse_canonical_int(row, "K") == condition.n_bins
            assert parse_canonical_int(row, "S") == n_samples
            assert parse_canonical_int(row, "B") == n_replications
            assert parse_finite_float(row, "model_C") == MODEL_C
            assert parse_finite_float(row, "model_l1_ratio") == MODEL_L1_RATIO
            assert parse_canonical_int(row, "model_fit_intercept") == int(
                MODEL_FIT_INTERCEPT
            )
            assert parse_canonical_int(row, "model_max_iter") == MODEL_MAX_ITER
            assert parse_finite_float(row, "model_tol") == MODEL_TOL
            model_n_iter = parse_canonical_int(row, "model_n_iter")
            assert 0 <= model_n_iter <= MODEL_MAX_ITER
            assert parse_canonical_int(row, "master_seed") == master_seed

            data_key = _data_stream_key(condition.dataset_index, replication)
            posterior_key = _simulation_stream_key(condition, replication, method)
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
                measurements,
                data_stream_id,
                simulation_stream_id,
                feature_count,
                model_n_iter,
                row["computation_time_ns"], row["execution_position"],
            )
    expected_primary_keys = {
        (method, condition.condition_id, replication, level)
        for condition in conditions
        for method in METHODS
        for replication in range(n_replications)
        for level in NOMINAL_LEVELS
    }
    assert primary_keys == expected_primary_keys
    expected_row_count = (
        len(conditions) * n_replications * len(NOMINAL_LEVELS) * len(METHODS)
    )
    assert row_count == expected_row_count
    assert len(bundles) == len(conditions) * n_replications * len(METHODS)

    dataset_references: dict[
        tuple[int, int],
        list[tuple[RealDataCondition, str, str, int, int]],
    ] = {}
    for condition in conditions:
        for method in METHODS:
            for replication in range(n_replications):
                level_bundle = bundles[(method, condition.condition_id, replication)]
                records = [entry[0] for entry in level_bundle.values()]
                assert_complete_level_bundle(records, NOMINAL_LEVELS)
                data_ids = {entry[1] for entry in level_bundle.values()}
                simulation_ids = {entry[2] for entry in level_bundle.values()}
                feature_counts = {entry[3] for entry in level_bundle.values()}
                model_iterations = {entry[4] for entry in level_bundle.values()}
                assert len(data_ids) == 1
                assert len(simulation_ids) == 1
                assert len(feature_counts) == 1
                assert len(model_iterations) == 1
                assert len({entry[5] for entry in level_bundle.values()}) == 1
                assert len({entry[6] for entry in level_bundle.values()}) == 1
                dataset_key = (condition.dataset_index, replication)
                dataset_references.setdefault(dataset_key, []).append(
                    (
                        condition,
                        next(iter(data_ids)),
                        next(iter(simulation_ids)),
                        next(iter(feature_counts)),
                        next(iter(model_iterations)),
                    )
                )

    selected_by_dataset = dict(_conditions_by_dataset(conditions))
    all_data_ids = set()
    all_simulation_ids = set()
    for (dataset_index, _), references in dataset_references.items():
        assert len(references) == len(selected_by_dataset[dataset_index]) * len(METHODS)
        data_ids = {reference[1] for reference in references}
        simulation_ids = {reference[2] for reference in references}
        feature_counts = {reference[3] for reference in references}
        model_iterations = {reference[4] for reference in references}
        assert len(data_ids) == 1
        assert len(simulation_ids) == len(references)
        assert len(feature_counts) == 1
        assert len(model_iterations) == 1
        all_data_ids.update(data_ids)
        all_simulation_ids.update(simulation_ids)

    expected_dataset_replications = (
        len(selected_by_dataset) * n_replications
    )
    assert len(all_data_ids) == expected_dataset_replications
    assert len(all_simulation_ids) == len(conditions) * n_replications * len(METHODS)
    assert all_data_ids.isdisjoint(all_simulation_ids)
    return row_count


def _source_paths():
    return [Path(__file__), *(PY_DIR / "roc_utils" / f"{name}.py" for name in (
        "__init__", "experiment_provenance", "experiment_results", "evaluation_criteria",
        "utilities_experiments", "posterior_predictive_roc_sampler",
        "bootstrap_predictive_roc_sampler", "real_datasets", "real_model", "semantic_rng",
    ))]


def _experiment_configuration(conditions, n_replications, n_samples, master_seed):
    """Describe the exact selected scientific design using JSON-native values."""
    configuration = {
        "experiment": EXPERIMENT_NAME, "real_protocol_version": REAL_PROTOCOL_VERSION,
        "raw_schema_version": RAW_SCHEMA_VERSION, "methods": METHODS,
        "master_seed": master_seed, "rng_algorithm": RNG_ALGORITHM,
        "B": n_replications, "S": n_samples, "K_values": N_BINS_VALUES,
        "nominal_levels": NOMINAL_LEVELS,
        "prior": {"profile": PRIOR_PROFILE, "mass_per_class": PRIOR_MASS},
        "model": {"spec_version": MODEL_SPEC_VERSION, "penalty": MODEL_PENALTY,
                  "C": MODEL_C, "l1_ratio": MODEL_L1_RATIO, "solver": MODEL_SOLVER,
                  "fit_intercept": MODEL_FIT_INTERCEPT, "class_weight": MODEL_CLASS_WEIGHT,
                  "max_iter": MODEL_MAX_ITER, "tol": MODEL_TOL,
                  "numeric_scaling": NUMERIC_SCALING},
        "datasets": [asdict(real_dataset_metadata(REAL_DATASET_NAMES[index]))
                     for index, _ in _conditions_by_dataset(conditions)],
        "conditions": [dict(index=c.index, identifier=c.condition_id,
                            dataset_index=c.dataset_index, dataset_name=c.dataset_name,
                            K=c.n_bins) for c in conditions],
        "stream_key_templates": {
            "data": [3, 1, "dataset_index", "replication"],
            "simulations": {"DiP-ROC": [3, 2, "condition_index", "replication"]},
            "warmup": {"DiP-ROC": [3, 4, 2, "condition_index", 0]},
        },
        "replication_indexing": "zero-based",
        "split_draw_order": "classes 0 then 1; permutation per class; test, external, remaining training",
        "execution_order": "selected ascending grids rotated left by replication modulo selected grid count per dataset; positions one-based",
        "warmup": "one full-budget construction per selected grid on replication-zero inputs; separate stream; discarded",
        "timing_scope": "one-thread complete DiP bands and AUC intervals for both levels; precomputed prior, RNG creation, training, histograms, external evaluation, thread-control setup/restoration and output excluded",
    }
    return json.loads(json.dumps(configuration, allow_nan=False))


def _collect_experiment_metadata(conditions, n_replications, n_samples, master_seed):
    return collect_run_metadata(
        source_root=PY_DIR.parent, source_paths=_source_paths(),
        configuration=_experiment_configuration(conditions, n_replications, n_samples, master_seed),
        metadata_schema=RUN_METADATA_SCHEMA,
    )


def validate_run_metadata(metadata, conditions, n_replications, n_samples, master_seed):
    """Validate recorded design without requiring the original software install."""
    conditions = _validate_conditions(conditions)
    n_replications = validate_positive_integer("n_replications", n_replications, minimum=1)
    n_samples = validate_positive_integer("n_samples", n_samples, minimum=2)
    make_semantic_rng(master_seed, _EXPERIMENT_STREAM_KEY, 0, 0, 0)
    master_seed = int(master_seed)
    assert isinstance(metadata, dict)
    assert metadata.get("metadata_schema") == RUN_METADATA_SCHEMA
    expected_configuration = _experiment_configuration(conditions, n_replications, n_samples, master_seed)
    assert json.dumps(metadata.get("configuration"), sort_keys=True, allow_nan=False) == json.dumps(
        expected_configuration, sort_keys=True, allow_nan=False,
    ), "Run metadata does not describe the requested real-data protocol."
    source_hashes = metadata.get("source_sha256")
    assert isinstance(source_hashes, dict)
    expected_sources = {str(path.relative_to(PY_DIR.parent)) for path in _source_paths()}
    assert set(source_hashes) == expected_sources
    assert all(isinstance(value, str) and len(value) == 64
               and all(character in "0123456789abcdef" for character in value)
               for value in source_hashes.values())
    software = metadata.get("software", {})
    assert all(isinstance(software.get(name), str) and software[name]
               for name in ("python", "numpy", "scipy", "threadpoolctl", "scikit-learn"))
    environment = metadata.get("timing_environment", {})
    assert type(environment.get("timing_threads")) is int and environment["timing_threads"] == 1
    assert environment.get("clock") == "time.perf_counter_ns"
    assert all(isinstance(environment.get(name), str) and environment[name]
               for name in ("operating_system", "architecture", "processor"))
    libraries = environment.get("native_libraries_under_timing_limit")
    assert isinstance(libraries, list)
    assert all(type(library.get("num_threads")) is int and library["num_threads"] == 1
               for library in libraries)


def validate_raw_results_csv(
    csv_path: str | Path, *, conditions=REAL_DATA_CONDITIONS,
    n_replications=OFFICIAL_REPLICATIONS, n_samples=OFFICIAL_SAMPLES,
    master_seed=DEFAULT_MASTER_SEED, metadata=None,
) -> int:
    """Validate complete v3 rows and their exact checksum-linked provenance.

    A supplied metadata dictionary is for staged, not-yet-published CSVs.
    Public readers omit it and must have the CSV's matching companion.
    """
    context = {"csv_path": str(csv_path)}
    supplied_metadata = metadata is not None
    try:
        conditions = _validate_conditions(conditions)
        n_replications = validate_positive_integer("n_replications", n_replications, minimum=1)
        n_samples = validate_positive_integer("n_samples", n_samples, minimum=2)
        make_semantic_rng(master_seed, _EXPERIMENT_STREAM_KEY, 0, 0, 0)
        master_seed = int(master_seed)
        if metadata is None:
            metadata = read_run_metadata(csv_path, metadata_schema=RUN_METADATA_SCHEMA)
        validate_run_metadata(metadata, conditions, n_replications, n_samples, master_seed)
        count = _validate_raw_results_csv(
            csv_path, conditions=conditions, n_replications=n_replications,
            n_samples=n_samples, master_seed=master_seed, metadata=metadata,
            _diagnostic_context=context,
        )
        if not supplied_metadata:
            assert file_sha256(csv_path) == metadata["raw_csv_sha256"], (
                "CSV changed during validation."
            )
        return count
    except Exception as error:
        _add_failure_note(error, "CSV validation", **context)
        raise


def _protect_previous_results(output_csv):
    """Keep new runs out of old-protocol and archived output locations."""
    resolved = output_csv.resolve()
    old_csv_names = {"raw_replications.csv", "raw_replications_v1.csv", "raw_replications_v2.csv"}
    old_directories = {"latex_tables", "latex_tables_v1", "latex_tables_v2", "archived_experiments", "back_up_legacy_pipeline"}
    for candidate in (output_csv, resolved):
        if (candidate.name.lower() in old_csv_names
                or any(part.lower() in old_directories for part in candidate.parts)):
            raise ValueError("Previous and archived outputs are protected; choose a separate v3 CSV.")
    if resolved.is_file():
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                schema = row.get("schema_version")
                if schema and schema != RAW_SCHEMA_VERSION:
                    raise ValueError(f"Existing {schema!r} results are protected even with overwrite=True.")


def _print_progress(dataset_name, completed, n_replications, statistics):
    print(f"{dataset_name}: completed {completed}/{n_replications} replications (v3 DiP-ROC).", flush=True)
    for condition_id, levels in statistics.items():
        for level, totals in levels.items():
            print(
                f"  {condition_id} level={level:.2f}: "
                f"ROC coverage={100 * totals['roc_covered'] / completed:.2f}%, "
                f"AUC coverage={100 * totals['auc_covered'] / completed:.2f}%, "
                f"mean radius={totals['roc_radius'] / completed:.6f}, "
                f"mean AUC width={totals['auc_width'] / completed:.6f}, "
                f"mean construction={totals['computation_time_ns'] / completed / 1_000_000:.3f} ms",
                flush=True,
            )


def run_experiment(
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    *,
    data_dir: str | Path = DEFAULT_REAL_DATA_DIR,
    master_seed: int = DEFAULT_MASTER_SEED,
    n_replications: int = OFFICIAL_REPLICATIONS,
    n_samples: int = OFFICIAL_SAMPLES,
    conditions: Sequence[RealDataCondition] = REAL_DATA_CONDITIONS,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Publish v3 results: K=10 for the main text, K=5/20 for the appendix.

    All grids reuse the same fitted classifier and scores. Their reporting roles
    do not change the experimental protocol, simulation streams, or CSV schema.
    """

    output_csv = Path(output_csv)
    data_dir = Path(data_dir)
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

    _protect_previous_results(output_csv)
    conditions = _validate_conditions(conditions)
    grouped_conditions = _conditions_by_dataset(conditions)
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

    metadata_record = _collect_experiment_metadata(conditions, n_replications, n_samples, master_seed)
    validate_run_metadata(metadata_record, conditions, n_replications, n_samples, master_seed)
    if show_progress:
        print(f"Real-data protocol v3: DiP-ROC, {len(conditions)} selected dataset-grid conditions, "
              f"B={n_replications}, S={n_samples}, levels={NOMINAL_LEVELS}, seed={master_seed}; "
              f"prior={PRIOR_PROFILE}, mass_per_class={PRIOR_MASS}, "
              f"expected_rows={len(conditions) * n_replications * len(NOMINAL_LEVELS)}; "
              f"model={MODEL_SPEC_VERSION}, C={MODEL_C}, solver={MODEL_SOLVER}, "
              f"intercept={MODEL_FIT_INTERCEPT}, class_weight={MODEL_CLASS_WEIGHT}, "
              f"tol={MODEL_TOL}, max_iter={MODEL_MAX_ITER}; output={output_csv}", flush=True)
        main_rows = sum(c.n_bins == MAIN_N_BINS for c in conditions) * n_replications * len(NOMINAL_LEVELS)
        supplementary_rows = sum(c.n_bins in SUPPLEMENTARY_N_BINS_VALUES for c in conditions) * n_replications * len(NOMINAL_LEVELS)
        assert main_rows + supplementary_rows == len(conditions) * n_replications * len(NOMINAL_LEVELS)
        print(f"Manuscript presentation: main K={MAIN_N_BINS} ({main_rows} rows); "
              f"supplementary K={SUPPLEMENTARY_N_BINS_VALUES} ({supplementary_rows} rows). "
              "All selected grids share each split, fitted classifier and scores; no historical information is used.", flush=True)
    grids = {
        n_bins: fixed_uniform_grid(n_bins)
        for n_bins in {condition.n_bins for condition in conditions}
    }
    priors = {
        n_bins: symmetric_dirichlet_priors(n_bins, PRIOR_MASS)
        for n_bins in grids
    }
    for n_bins, (alpha_plus, alpha_minus) in priors.items():
        np.testing.assert_allclose(
            alpha_plus,
            np.full(n_bins, 1.0 / n_bins),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(alpha_plus, alpha_minus)
        assert np.isclose(np.sum(alpha_plus), PRIOR_MASS)
        alpha_plus.setflags(write=False)
        alpha_minus.setflags(write=False)
        grids[n_bins].setflags(write=False)

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

            for dataset_index, dataset_conditions in grouped_conditions:
                dataset_name = REAL_DATASET_NAMES[dataset_index]
                metadata = real_dataset_metadata(dataset_name)
                if show_progress:
                    counts = _split_counts(metadata)
                    print(f"Loading {dataset_name}; OpenML={metadata.openml_data_id} "
                          f"version={metadata.openml_version} file={metadata.openml_file_id}; "
                          f"complete_cases={metadata.n_complete_rows}, "
                          f"training_class_counts={counts[:2]}, test/external_class_counts={counts[2:]}; "
                          f"selected grids {[c.n_bins for c in dataset_conditions]}.", flush=True)
                try:
                    loaded_dataset = load_real_dataset(dataset_name, data_dir=data_dir)
                    _assert_loaded_source(loaded_dataset, metadata)
                except Exception as error:
                    _add_failure_note(error, "source loading", dataset_name=dataset_name,
                                      dataset_index=dataset_index, master_seed=master_seed,
                                      data_dir=str(data_dir))
                    raise
                statistics = {c.condition_id: {level: {field: 0.0 for field in (
                    "roc_covered", "auc_covered", "roc_radius", "auc_width", "computation_time_ns",
                )} for level in NOMINAL_LEVELS} for c in dataset_conditions}

                for replication in range(n_replications):
                    prepared, scores, data_key = _prepare_dataset_replication(
                        loaded_dataset,
                        metadata,
                        dataset_index,
                        replication,
                        master_seed=master_seed,
                    )
                    data_id = semantic_stream_id(master_seed, *data_key)
                    simulation_ids = set()

                    rows_by_condition = {}
                    for position, condition in enumerate(_execution_order(dataset_conditions, replication), start=1):
                        alpha_plus, alpha_minus = priors[condition.n_bins]
                        results = _evaluate_grid_condition(
                            condition, replication, scores, grids[condition.n_bins],
                            alpha_plus, alpha_minus, n_samples=n_samples,
                            master_seed=master_seed, execution_position=position,
                        )
                        assert set(results) == {"DiP-ROC"}, "Only uniform DiP-ROC belongs to this experiment."
                        measurements, simulation_key, elapsed_ns, actual_position = results["DiP-ROC"]
                        assert actual_position == position
                        simulation_id = semantic_stream_id(master_seed, *simulation_key)
                        assert simulation_id not in simulation_ids and simulation_id != data_id
                        simulation_ids.add(simulation_id)
                        assert not grids[condition.n_bins].flags.writeable
                        assert not alpha_plus.flags.writeable and not alpha_minus.flags.writeable
                        try:
                            rows_by_condition[condition.index] = _raw_rows_for_condition_replication(
                                condition, replication, measurements, metadata, prepared, scores,
                                n_replications=n_replications, n_samples=n_samples,
                                master_seed=master_seed, data_key=data_key,
                                simulation_key=simulation_key, method="DiP-ROC",
                                computation_time_ns=elapsed_ns, execution_position=position,
                            )
                        except Exception as error:
                            _add_failure_note(error, "CSV serialization", **_band_diagnostic_context(
                                condition, replication, "DiP-ROC", simulation_key, master_seed,
                            ))
                            raise
                    assert set(rows_by_condition) == {c.index for c in dataset_conditions}
                    for condition in dataset_conditions:
                        rows = rows_by_condition[condition.index]
                        try:
                            writer.writerows(rows)
                        except Exception as error:
                            _add_failure_note(error, "CSV serialization", **_band_diagnostic_context(
                                condition, replication, "DiP-ROC",
                                _posterior_stream_key(condition, replication), master_seed,
                            ))
                            raise
                        written_rows += len(rows)
                        if show_progress:
                            for row in rows:
                                totals = statistics[condition.condition_id][row["level"]]
                                for field in totals:
                                    totals[field] += row[field]
                    assert len(simulation_ids) == len(dataset_conditions)
                    if show_progress and ((replication + 1) % 25 == 0
                                          or replication in (0, n_replications - 1)):
                        _print_progress(dataset_name, replication + 1, n_replications, statistics)

            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception as error:
                _add_failure_note(error, "CSV flush", csv_path=str(output_csv), written_rows=written_rows)
                raise

        expected_rows = (
            len(conditions) * n_replications * len(NOMINAL_LEVELS) * len(METHODS)
        )
        assert written_rows == expected_rows
        if show_progress:
            print(f"Validating {written_rows} raw rows and provenance before publication...", flush=True)
        validated_rows = validate_raw_results_csv(
            temporary_path,
            conditions=conditions,
            n_replications=n_replications,
            n_samples=n_samples,
            master_seed=master_seed, metadata=metadata_record,
        )
        assert validated_rows == written_rows
        if output_csv.exists() and not overwrite:
            raise FileExistsError(
                f"Output appeared while the experiment was running: {output_csv}"
            )
        try:
            _protect_previous_results(output_csv)
            assert_sources_unchanged(metadata_record, source_root=PY_DIR.parent)
            publish_run_metadata(temporary_path, output_csv, metadata_record)
            if overwrite:
                os.replace(temporary_path, output_csv)
            else:
                os.link(temporary_path, output_csv)
                temporary_path.unlink()
        except Exception as error:
            _add_failure_note(error, "publication", csv_path=str(output_csv),
                              written_rows=written_rows, master_seed=master_seed)
            raise
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    assert output_csv.is_file()
    if show_progress:
        print(f"Saved {written_rows} validated raw rows and provenance to {output_csv}", flush=True)
    return output_csv


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the deliberately small official command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run uniform DiP-ROC (m=1) on twelve real datasets: K=10 for the main "
            "text and K=5/20 for supplementary checks. Save full-precision raw "
            "replication results for the separate table generator. Missing pinned datasets "
            "are downloaded."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_REAL_DATA_DIR,
        help=f"Pinned dataset cache (default: {DEFAULT_REAL_DATA_DIR}).",
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
    """Run the exact official real-data design."""

    arguments = _build_argument_parser().parse_args(argv)
    assert REAL_PROTOCOL_VERSION == 3
    assert METHODS == ("DiP-ROC",)
    assert len(REAL_DATASET_NAMES) == 12
    assert N_BINS_VALUES == (5, 10, 20)
    assert MAIN_N_BINS == 10
    assert SUPPLEMENTARY_N_BINS_VALUES == (5, 20)
    assert PRIOR_PROFILE == "symmetric"
    assert PRIOR_MASS == 1.0
    assert NOMINAL_LEVELS == (0.80, 0.95)
    assert OFFICIAL_REPLICATIONS == 300
    assert OFFICIAL_SAMPLES == 4000
    assert len(REAL_DATA_CONDITIONS) == 36
    assert MODEL_SPEC_VERSION == (
        "logistic_l2_c1_lbfgs_numeric_standardization_v2"
    )
    assert MODEL_PENALTY == "l2"
    assert MODEL_C == 1.0
    assert MODEL_L1_RATIO == 0.0
    assert MODEL_SOLVER == "lbfgs"
    assert MODEL_FIT_INTERCEPT is True
    assert MODEL_CLASS_WEIGHT is None
    assert MODEL_MAX_ITER == 10_000
    assert MODEL_TOL == 1e-8
    assert NUMERIC_SCALING == "training_only_numeric_standardization"
    return run_experiment(
        arguments.output,
        data_dir=arguments.data_dir,
        master_seed=arguments.master_seed,
        n_replications=OFFICIAL_REPLICATIONS,
        n_samples=OFFICIAL_SAMPLES,
        conditions=REAL_DATA_CONDITIONS,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
