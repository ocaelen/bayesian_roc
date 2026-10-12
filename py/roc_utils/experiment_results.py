"""Shared raw measurement records for the scientific experiment runners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

if not __debug__:
    raise RuntimeError(
        "experiment_results relies on runtime assertions; do not use python -O."
    )

from .evaluation_criteria import (
    is_external_auc_covered,
    is_external_roc_covered,
)
from .utilities_experiments import ReplicationMeasurements, normalize_levels


_BOUNDARY_TOLERANCE = 1e-12
METHODS = ("DiP-ROC", "BootP-ROC")

MEASUREMENT_RAW_FIELDNAMES = (
    "external_max_roc_deviation",
    "external_grid_auc",
    "roc_radius",
    "auc_interval_lower",
    "auc_interval_upper",
    "auc_width",
    "roc_covered",
    "auc_covered",
)


@dataclass(frozen=True)
class RawLevelMeasurements:
    """Raw sufficient statistics for one replication and nominal level."""

    level: float
    external_max_roc_deviation: float
    external_grid_auc: float
    roc_radius: float
    auc_interval_lower: float
    auc_interval_upper: float
    auc_width: float
    roc_covered: int
    auc_covered: int

    def __post_init__(self) -> None:
        _assert_raw_level_measurements(self)


def validate_positive_integer(name: str, value: int, *, minimum: int) -> int:
    """Return one validated integer experiment size."""

    if not isinstance(value, (int, np.integer)) or isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def parse_canonical_int(row: Mapping[str, str], field: str) -> int:
    """Parse an integer CSV field and reject non-canonical text."""

    value = int(row[field])
    assert row[field] == str(value), f"{field} must use canonical integer text."
    return value


def parse_finite_float(row: Mapping[str, str], field: str) -> float:
    """Parse one finite floating-point CSV field."""

    value = float(row[field])
    assert np.isfinite(value), f"{field} must be finite."
    return value


def assert_method_metadata(row: Mapping[str, str]) -> None:
    """Validate method-specific bounds and timing metadata of a CSV text row.

    The scientific measurement record remains method-independent. This
    additional check forbids assigning a prior to BootP or accepting a
    bootstrap-sized radius as a DiP-ROC result.
    """
    method = row["method"]
    assert method in METHODS, "Unknown predictive method."
    assert parse_canonical_int(row, "computation_time_ns") >= 0
    assert parse_canonical_int(row, "execution_position") >= 1
    assert parse_canonical_int(row, "timing_threads") == 1
    radius = parse_finite_float(row, "roc_radius")
    bound = np.sqrt(2.0) if method == "DiP-ROC" else 2.0
    assert 0.0 <= radius <= bound + _BOUNDARY_TOLERANCE
    if method == "BootP-ROC":
        assert row["prior_profile"] == ""
        assert row["prior_mass"] == ""
        if "prior_expected_grid_auc" in row:
            assert row["prior_expected_grid_auc"] == ""
    else:
        assert row["prior_profile"]
        assert parse_finite_float(row, "prior_mass") > 0.0
        if "prior_expected_grid_auc" in row:
            assert 0.0 <= parse_finite_float(row, "prior_expected_grid_auc") <= 1.0


def assert_paired_method_rows(
    dip_row: Mapping[str, str],
    boot_row: Mapping[str, str],
) -> None:
    """Verify that two method rows evaluate exactly the same external target.

    Condition identifiers may differ in the prior-sensitivity study. The
    maximum external ROC deviation is deliberately not compared: its center
    is specific to each method.
    """
    assert dip_row["method"] == "DiP-ROC"
    assert boot_row["method"] == "BootP-ROC"
    for field in ("experiment", "data_stream_id", "rng_algorithm", "master_seed"):
        assert dip_row[field] == boot_row[field], f"Unpaired {field}."
    for field in ("replication", "K", "S", "B", "n_positive", "n_negative"):
        assert (
            parse_canonical_int(dip_row, field)
            == parse_canonical_int(boot_row, field)
        ), f"Unpaired {field}."
    for field in ("level", "external_grid_auc"):
        assert (
            parse_finite_float(dip_row, field)
            == parse_finite_float(boot_row, field)
        ), f"Unpaired {field}."
    assert dip_row["simulation_stream_id"] != boot_row["simulation_stream_id"]
    assert dip_row["simulation_stream_id"] != dip_row["data_stream_id"]
    assert boot_row["simulation_stream_id"] != boot_row["data_stream_id"]


def _assert_raw_level_measurements(record: RawLevelMeasurements) -> None:
    """Assert all scalar identities of one raw level record."""

    assert np.isfinite(record.level)
    assert 0.0 < record.level < 1.0
    assert record.auc_width == (
        record.auc_interval_upper - record.auc_interval_lower
    )
    assert record.roc_covered in (0, 1)
    assert record.auc_covered in (0, 1)
    assert record.roc_covered == int(
        is_external_roc_covered(
            record.external_max_roc_deviation,
            record.roc_radius,
        )
    )
    assert record.auc_covered == int(
        is_external_auc_covered(
            record.external_grid_auc,
            (record.auc_interval_lower, record.auc_interval_upper),
        )
    )


def raw_level_measurements_from_replication(
    measurements: ReplicationMeasurements,
    levels: Iterable[float],
) -> tuple[RawLevelMeasurements, ...]:
    """Flatten one replication into validated records for all levels."""

    assert isinstance(measurements, ReplicationMeasurements)
    levels = normalize_levels(levels)
    assert set(measurements.roc_radii) == set(levels)
    assert set(measurements.auc_intervals) == set(levels)

    records = []
    for level in levels:
        roc_radius = float(measurements.roc_radii[level])
        auc_lower, auc_upper = measurements.auc_intervals[level]
        auc_lower = float(auc_lower)
        auc_upper = float(auc_upper)
        records.append(
            RawLevelMeasurements(
                level=level,
                external_max_roc_deviation=float(
                    measurements.external_max_roc_deviation
                ),
                external_grid_auc=float(measurements.external_grid_auc),
                roc_radius=roc_radius,
                auc_interval_lower=auc_lower,
                auc_interval_upper=auc_upper,
                auc_width=auc_upper - auc_lower,
                roc_covered=int(
                    is_external_roc_covered(
                        measurements.external_max_roc_deviation,
                        roc_radius,
                    )
                ),
                auc_covered=int(
                    is_external_auc_covered(
                        measurements.external_grid_auc,
                        (auc_lower, auc_upper),
                    )
                ),
            )
        )

    assert_complete_level_bundle(records, levels)
    return tuple(records)


def raw_level_measurement_fields(
    record: RawLevelMeasurements,
) -> dict[str, object]:
    """Return the ordered common CSV fields of one raw record."""

    assert isinstance(record, RawLevelMeasurements)
    fields: dict[str, object] = {
        "external_max_roc_deviation": record.external_max_roc_deviation,
        "external_grid_auc": record.external_grid_auc,
        "roc_radius": record.roc_radius,
        "auc_interval_lower": record.auc_interval_lower,
        "auc_interval_upper": record.auc_interval_upper,
        "auc_width": record.auc_width,
        "roc_covered": record.roc_covered,
        "auc_covered": record.auc_covered,
    }
    assert tuple(fields) == MEASUREMENT_RAW_FIELDNAMES
    return fields


def parse_raw_level_measurements(
    row: Mapping[str, str],
) -> RawLevelMeasurements:
    """Parse and validate the common measurement fields from one CSV row."""

    return RawLevelMeasurements(
        level=parse_finite_float(row, "level"),
        external_max_roc_deviation=parse_finite_float(
            row,
            "external_max_roc_deviation",
        ),
        external_grid_auc=parse_finite_float(row, "external_grid_auc"),
        roc_radius=parse_finite_float(row, "roc_radius"),
        auc_interval_lower=parse_finite_float(row, "auc_interval_lower"),
        auc_interval_upper=parse_finite_float(row, "auc_interval_upper"),
        auc_width=parse_finite_float(row, "auc_width"),
        roc_covered=parse_canonical_int(row, "roc_covered"),
        auc_covered=parse_canonical_int(row, "auc_covered"),
    )


def assert_complete_level_bundle(
    records: Sequence[RawLevelMeasurements],
    levels: Iterable[float],
) -> dict[float, RawLevelMeasurements]:
    """Validate shared external values and nesting across nominal levels."""

    levels = normalize_levels(levels)
    records = tuple(records)
    assert len(records) == len(levels)
    records_by_level = {record.level: record for record in records}
    assert len(records_by_level) == len(records)
    assert set(records_by_level) == set(levels)

    reference = records_by_level[levels[0]]
    for record in records_by_level.values():
        assert record.external_max_roc_deviation == (
            reference.external_max_roc_deviation
        )
        assert record.external_grid_auc == reference.external_grid_auc

    ordered = [records_by_level[level] for level in sorted(levels)]
    for lower, higher in zip(ordered[:-1], ordered[1:]):
        assert lower.roc_radius <= higher.roc_radius + _BOUNDARY_TOLERANCE
        assert higher.auc_interval_lower <= (
            lower.auc_interval_lower + _BOUNDARY_TOLERANCE
        )
        assert lower.auc_interval_upper <= (
            higher.auc_interval_upper + _BOUNDARY_TOLERANCE
        )

    return records_by_level
