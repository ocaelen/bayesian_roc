"""Summarize nine DiP-ROC prior settings and their eighteen paired contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

if not __debug__:
    raise RuntimeError("This table generator relies on runtime assertions; do not use python -O.")

import run_experiment_main_evaluation_on_synthetic_data as experiment

from roc_utils.evaluation_criteria import (
    MeanEstimate, summarize_mean, summarize_paired_differences,
)
from roc_utils.experiment_results import (
    RawLevelMeasurements, assert_complete_level_bundle,
    parse_canonical_int, parse_raw_level_measurements,
)
from roc_utils.experiment_provenance import file_sha256, read_run_metadata
from roc_utils.utilities_experiments import (
    EvaluationCriteriaSummary, ReplicationMeasurements,
    summarize_replication_measurements,
)

METHODS = experiment.METHODS
SETTINGS = experiment.SETTINGS
SETTING_METHODS = experiment.SETTING_METHODS
MAIN_PROTOCOL_VERSION = experiment.MAIN_PROTOCOL_VERSION
DISPLAY_LABELS = {
    setting: f"DiP-{experiment.SETTING_FAMILIES[setting][0].upper()}{experiment.SETTING_MASSES[setting]:g}"
    for setting in SETTINGS
}
CONTRAST_GROUPS = (
    ("Prior families at the same concentration", tuple(
        (f"dip_{left}_m{mass}", f"dip_{right}_m{mass}")
        for mass in (1, 5, 20)
        for left, right in (("linear", "uniform"), ("histograms", "uniform"), ("histograms", "linear"))
    )),
    ("Concentration: $m=5$ minus $m=1$, $m=20$ minus $m=1$, and $m=20$ minus $m=5$", tuple(
        (f"dip_{family}_m{left}", f"dip_{family}_m{right}")
        for family in ("uniform", "linear", "histograms")
        for left, right in ((5, 1), (20, 1), (20, 5))
    )),
)
CONTRASTS = tuple(contrast for _, group in CONTRAST_GROUPS for contrast in group)
assert len(CONTRASTS) == len(set(CONTRASTS)) == 18
assert all(left in SETTINGS and right in SETTINGS and left != right for left, right in CONTRASTS)
_LINEAR_SETTINGS = tuple(setting for setting in SETTINGS if experiment.SETTING_FAMILIES[setting] == "linear")
NOMINAL_LEVELS = experiment.NOMINAL_LEVELS
N_REPLICATIONS = experiment.OFFICIAL_REPLICATIONS
N_SAMPLES = experiment.OFFICIAL_SAMPLES
RAW_SCHEMA_VERSION = experiment.RAW_SCHEMA_VERSION
RAW_FIELDNAMES = experiment.RAW_FIELDNAMES
EXPERIMENT_NAME = experiment.EXPERIMENT_NAME
RESULTS_DIRECTORY_NAME = experiment.RESULTS_DIRECTORY_NAME
PY_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = experiment.DEFAULT_OUTPUT_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_CSV.parent / f"latex_tables_v{MAIN_PROTOCOL_VERSION}"
_METRIC_ATTRIBUTES = {
    "roc_coverage": "roc_covered", "auc_coverage": "auc_covered",
    "roc_radius": "roc_radius", "auc_width": "auc_width",
}


@dataclass(frozen=True)
class ConditionAnalysis:
    """One DiP-ROC prior setting's validated records, in replication order.

    Times contain one value per replication, not one per nominal level.
    """

    condition: object
    setting: str
    records_by_level: dict[float, tuple[RawLevelMeasurements, ...]]
    computation_times_ms: tuple[float, ...]
    data_stream_ids: tuple[str, ...]
    n_samples: int
    historical_auc_adjusted: tuple[bool, ...] = ()
    summaries_by_level: dict[float, EvaluationCriteriaSummary] = field(init=False)

    def __post_init__(self) -> None:
        assert self.setting in SETTINGS
        assert self.condition is not None
        assert type(self.n_samples) is int and self.n_samples > 0
        assert set(self.records_by_level) == set(NOMINAL_LEVELS)
        count = len(self.computation_times_ms)
        assert count >= 2
        if self.setting in _LINEAR_SETTINGS:
            assert len(self.historical_auc_adjusted) == count
            assert all(type(adjusted) is bool for adjusted in self.historical_auc_adjusted)
        else:
            assert self.historical_auc_adjusted == ()
        assert len(self.data_stream_ids) == count
        assert len(set(self.data_stream_ids)) == count
        assert all(self.data_stream_ids)
        assert all(math.isfinite(t) and t >= 0 for t in self.computation_times_ms)
        assert all(len(records) == count for records in self.records_by_level.values())
        assert all(record.level == level for level, records in self.records_by_level.items()
                   for record in records), "Each record must match its nominal-level key."
        replications = []
        for replication in range(count):
            bundled = assert_complete_level_bundle(
                [self.records_by_level[level][replication] for level in NOMINAL_LEVELS],
                NOMINAL_LEVELS,
            )
            assert all(record.roc_radius <= math.sqrt(2.0) + 1e-12 for record in bundled.values())
            reference = bundled[NOMINAL_LEVELS[0]]
            replications.append(ReplicationMeasurements(
                external_max_roc_deviation=reference.external_max_roc_deviation,
                external_grid_auc=reference.external_grid_auc,
                roc_radii={level: bundled[level].roc_radius for level in NOMINAL_LEVELS},
                auc_intervals={level: (
                    bundled[level].auc_interval_lower, bundled[level].auc_interval_upper,
                ) for level in NOMINAL_LEVELS},
            ))
        object.__setattr__(self, "summaries_by_level", summarize_replication_measurements(
            replications, NOMINAL_LEVELS,
        ))

    @property
    def n_replications(self) -> int:
        return len(self.computation_times_ms)

    @property
    def method(self) -> str:
        return SETTING_METHODS[self.setting]


@dataclass(frozen=True)
class PairedMethodEffect:
    """One explicitly signed setting contrast for all statistical criteria."""

    condition: object
    left_setting: str
    right_setting: str
    level: float
    metrics: dict[str, MeanEstimate]


def _validate_run_provenance(input_csv, conditions, n_replications, n_samples, master_seed):
    """Check the immutable CSV companion against the design being aggregated.

    Recorded source hashes document the run; they need not match today's files.
    """
    metadata = read_run_metadata(input_csv)
    expected = {
        "experiment": EXPERIMENT_NAME, "main_protocol_version": MAIN_PROTOCOL_VERSION,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "synthetic_design_version": experiment.SYNTHETIC_DESIGN_VERSION,
        "master_seed": master_seed, "rng_algorithm": experiment.RNG_ALGORITHM,
        "B": n_replications, "S": n_samples, "K": experiment.N_BINS,
        "nominal_levels": NOMINAL_LEVELS,
        "historical_class_counts": [experiment.HISTORICAL_N_POSITIVE, experiment.HISTORICAL_N_NEGATIVE],
        "linear_auc_restriction": experiment.HISTORICAL_AUC_BOUNDS,
        "settings": [dict(setting=s, method=SETTING_METHODS[s],
                          family=experiment.SETTING_FAMILIES[s], mass=experiment.SETTING_MASSES[s])
                     for s in SETTINGS],
        "conditions": [dict(index=c.index, identifier=c.condition_id, score_scenario=c.score_scenario,
                            n_positive=c.n_positive, n_negative=c.n_negative,
                            beta_parameters=experiment.simulation_beta_parameters(c.score_scenario))
                       for c in conditions],
        "stream_key_templates": {
            "data": [1, experiment._DATA_STREAM_ROLE, "condition_index", "replication"],
            "history": [1, experiment._HISTORICAL_DATA_STREAM_ROLE, "condition_index", "replication"],
            "simulations": {s: [1, role, "condition_index", "replication"]
                            for s, role in experiment._SETTING_STREAM_ROLES.items()},
            "warmup": {s: [1, experiment._WARMUP_STREAM_ROLE, role, "condition_index", 0]
                       for s, role in experiment._SETTING_STREAM_ROLES.items()},
        },
        "replication_indexing": "zero-based",
        "data_draw_order": ["positive_test", "negative_test", "positive_external", "negative_external"],
        "historical_draw_order": ["positive", "negative"],
        "execution_order": f"reporting order rotated left by replication modulo {len(SETTINGS)}",
        "timing_scope": "prior construction and complete predictive bands for both levels",
    }
    # Normalize tuples exactly as the JSON companion does.
    expected = json.loads(json.dumps(expected))
    configuration = metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Missing run provenance configuration.")
    for name, value in expected.items():
        recorded = configuration.get(name)
        if name == "conditions" and isinstance(recorded, list):
            # The CSV reader also accepts a different caller-supplied condition order.
            recorded = sorted(recorded, key=lambda condition: condition["index"])
            value = sorted(value, key=lambda condition: condition["index"])
        if recorded != value:
            raise ValueError(f"Run provenance configuration differs from the validated CSV design: {name}.")
    environment = metadata.get("timing_environment")
    if not isinstance(environment, dict):
        raise ValueError("Missing run provenance timing environment.")
    for name, value in (("timing_threads", 1), ("clock", "time.perf_counter_ns")):
        if environment.get(name) != value:
            raise ValueError(f"Run provenance timing environment differs from the timing protocol: {name}.")
    return metadata["raw_csv_sha256"]


def read_condition_analyses(
    input_csv: str | Path, *, conditions: Sequence | None = None,
    n_replications: int = N_REPLICATIONS, n_samples: int = N_SAMPLES,
    master_seed: int | None = None,
) -> tuple[ConditionAnalysis, ...]:
    """Validate current results and retain the records needed for paired inference.

    Defaults require the complete official experiment. Reduced integrations
    must explicitly supply both their design and their B/S values.
    """

    input_csv = Path(input_csv)
    conditions = tuple(_DIP_CONDITIONS if conditions is None else conditions)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
            raise ValueError(f"Expected the v{MAIN_PROTOCOL_VERSION} comparison CSV; earlier results belong to different synthetic protocols.")
        first = next(reader, None)
        assert first is not None, "The comparison CSV is empty."
        recorded_seed = parse_canonical_int(first, "master_seed")
    if master_seed is not None:
        assert recorded_seed == master_seed
    experiment.validate_raw_results_csv(
        input_csv, conditions=conditions, n_replications=n_replications,
        n_samples=n_samples, master_seed=recorded_seed,
    )
    expected_checksum = _validate_run_provenance(input_csv, conditions, n_replications, n_samples, recorded_seed)

    expected = [(condition.index, setting, condition) for condition in conditions for setting in SETTINGS]
    # Lists indexed by replication avoid millions of individual dictionary
    # entries at the official budget. CSV order need not be canonical.
    records = {(index, setting): {level: [None] * n_replications for level in NOMINAL_LEVELS}
               for index, setting, _ in expected}
    timings = {(index, setting): [None] * n_replications for index, setting, _ in expected}
    data_ids = {(index, setting): [None] * n_replications for index, setting, _ in expected}
    adjustments = {(condition.index, setting): [None] * n_replications
                   for condition in conditions for setting in _LINEAR_SETTINGS}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (parse_canonical_int(row, "condition_index"), row["setting"])
            assert key in records
            replication = parse_canonical_int(row, "replication")
            record = parse_raw_level_measurements(row)
            assert records[key][record.level][replication] is None
            records[key][record.level][replication] = record
            time_ns = parse_canonical_int(row, "computation_time_ns")
            if row["setting"] in _LINEAR_SETTINGS:
                adjusted = parse_canonical_int(row, "historical_auc_adjusted")
                assert adjusted in (0, 1)
                if adjustments[key][replication] is not None:
                    assert adjustments[key][replication] == bool(adjusted)
                adjustments[key][replication] = bool(adjusted)
            if timings[key][replication] is not None:
                assert timings[key][replication] == time_ns
                assert data_ids[key][replication] == row["data_stream_id"]
            else:
                timings[key][replication] = time_ns
                # All settings share each data identifier.
                data_ids[key][replication] = sys.intern(row["data_stream_id"])
    analyses = []
    for index, setting, condition in expected:
        key = (index, setting)
        assert all(value is not None for value in timings[key])
        level_records = records.pop(key)
        setting_times = timings.pop(key)
        setting_data_ids = data_ids.pop(key)
        analyses.append(ConditionAnalysis(
            condition=condition, setting=setting, n_samples=n_samples,
            records_by_level={level: tuple(level_records[level])
                              for level in NOMINAL_LEVELS},
            computation_times_ms=tuple(value / 1_000_000.0 for value in setting_times),
            data_stream_ids=tuple(setting_data_ids),
            historical_auc_adjusted=(tuple(adjustments.pop(key)) if setting in _LINEAR_SETTINGS else ()),
        ))
    result = tuple(analyses)
    _setting_bundles(result)
    if file_sha256(input_csv) != expected_checksum:
        raise ValueError("Experimental CSV changed during table aggregation; regenerate tables from one completed, unchanged run.")
    return result


def _setting_bundles(analyses: Sequence[ConditionAnalysis]):
    """Return complete paired settings in canonical condition/setting order."""

    analyses = tuple(analyses)
    assert analyses
    assert len({a.n_replications for a in analyses}) == 1
    assert len({a.n_samples for a in analyses}) == 1
    indexed = {(a.condition.index, a.setting): a for a in analyses}
    assert len(indexed) == len(analyses), "Duplicate condition/setting."
    bundles = []
    for index in sorted({a.condition.index for a in analyses}):
        assert all((index, setting) in indexed for setting in SETTINGS), "Missing paired setting."
        bundle = tuple(indexed[(index, setting)] for setting in SETTINGS)
        for comparison in bundle[1:]:
            assert comparison.condition == bundle[0].condition
            assert comparison.data_stream_ids == bundle[0].data_stream_ids
            for level in NOMINAL_LEVELS:
                assert tuple(r.external_grid_auc for r in comparison.records_by_level[level]) == tuple(
                    r.external_grid_auc for r in bundle[0].records_by_level[level]
                )
        reference_flags = indexed[(index, _LINEAR_SETTINGS[0])].historical_auc_adjusted
        assert all(indexed[(index, setting)].historical_auc_adjusted == reference_flags
                   for setting in _LINEAR_SETTINGS[1:]), "All linear concentrations must use the same historical AUC adjustments."
        bundles.append(bundle)
    return tuple(bundles)


def _values(analysis: ConditionAnalysis, level: float, metric: str) -> tuple[float, ...]:
    return tuple(float(getattr(record, _METRIC_ATTRIBUTES[metric]))
                 for record in analysis.records_by_level[level])


def build_paired_method_effects(analyses: Sequence[ConditionAnalysis]) -> tuple[PairedMethodEffect, ...]:
    effects = []
    for bundle in _setting_bundles(analyses):
        settings = {analysis.setting: analysis for analysis in bundle}
        for level in NOMINAL_LEVELS:
            for left, right in CONTRASTS:
                effects.append(PairedMethodEffect(
                    condition=bundle[0].condition, left_setting=left, right_setting=right, level=level,
                    metrics={metric: summarize_paired_differences(
                        _values(settings[left], level, metric), _values(settings[right], level, metric),
                    ) for metric in _METRIC_ATTRIBUTES},
                ))
    return tuple(effects)


def _format_estimate(mean: float, standard_error: float, *, percent=False, signed=False) -> str:
    assert math.isfinite(mean) and math.isfinite(standard_error) and standard_error >= 0
    scale = 100.0 if percent else 1.0
    digits = 2 if percent else 5
    sign = "+" if signed else ""
    return f"{scale * mean:{sign}.{digits}f} ({scale * standard_error:.{digits}f})"


def _absolute_cells(analysis: ConditionAnalysis, level: float) -> list[str]:
    summary = analysis.summaries_by_level[level]
    return [
        _format_estimate(summary.roc_coverage, summary.roc_coverage_standard_error, percent=True),
        _format_estimate(summary.auc_coverage, summary.auc_coverage_standard_error, percent=True),
        _format_estimate(summary.roc_radius_mean, summary.roc_radius_standard_error),
        _format_estimate(summary.auc_interval_width_mean, summary.auc_interval_width_standard_error),
    ]


def _difference_cells(effect: PairedMethodEffect) -> list[str]:
    return [_format_estimate(effect.metrics[metric].mean, effect.metrics[metric].standard_error,
                             percent="coverage" in metric, signed=True)
            for metric in _METRIC_ATTRIBUTES]


def _contrast_label(left: str, right: str) -> str:
    assert (left, right) in CONTRASTS
    return DISPLAY_LABELS[left] + r"$-$" + DISPLAY_LABELS[right]


def _run_note(analyses: Sequence[ConditionAnalysis]) -> str:
    first = analyses[0]
    note = (
        " Nine DiP-ROC settings compare three prior families at three concentrations."
        " DiP-U, DiP-L and DiP-H denote uniform, historical-AUC linear and historical-histogram priors;"
        " the suffix 1, 5 or 20 gives the total prior concentration per class."
        " Only DiP-L and DiP-H receive historical information, respectively an AUC summary and class histograms."
        rf" $\mathcal B={first.n_replications}$, $\mathcal S={first.n_samples}$, $\mathcal K={N_BINS}$."
    )
    if (first.n_replications != N_REPLICATIONS or first.n_samples != N_SAMPLES
            or sum(a.setting == SETTINGS[0] for a in analyses) != len(_DIP_CONDITIONS)):
        note += " Reduced validation run; these are not official experimental results."
    return note


def _table(caption: str, label: str, headers: Sequence[str], rows: Sequence,
           *, continued=False, font="small") -> str:
    lines = [r"\begin{table}[p]" if continued else r"\begin{table}[t]"]
    if continued:
        lines.append(r"\ContinuedFloat")
        # caption's default hyperref suffix is alphabetic and fails after 26
        # continuations. Numeric anchors stay unique across all 30 panels.
        lines.append(r"\ifdefined\theHtable\renewcommand{\theHtable}{\arabic{table}.\arabic{continuedfloat}}\fi")
    lines.extend([r"\centering", (r"\caption[]{" if continued else r"\caption{") + caption + "}"])
    if not continued:
        lines.append(r"\label{" + label + "}")
    lines.extend([r"\begingroup", "\\" + font, r"\setlength{\tabcolsep}{3.2pt}",
                  r"\renewcommand{\arraystretch}{1.10}",
                  r"\begin{tabular}{@{}" + "l" * len(headers) + r"@{}}", r"\hline",
                  " & ".join(headers) + r" \\", r"\hline"])
    lines.extend(row if isinstance(row, str) else " & ".join(str(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    return "\n".join(lines)


def _validated_fragments(tables: dict[str, str]) -> dict[str, str]:
    assert tuple(tables) == OUTPUT_FILENAMES
    result = {}
    for index, (filename, latex) in enumerate(tables.items()):
        assert latex.count(r"\begin{table}") == latex.count(r"\end{table}")
        assert latex.count(r"\begin{tabular}") == latex.count(r"\end{tabular}")
        assert latex.count(r"\label{" + OUTPUT_LABELS[index] + "}") == 1
        assert "nan" not in latex.lower() and "infinity" not in latex.lower()
        result[filename] = _VERSION_MARKER + latex + "\n"
    return result


def _write_table(output_path: Path, latex: str) -> None:
    """Atomically publish a fragment without overwriting an old result on failure."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(latex)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    assert output_path.read_text(encoding="utf-8") == latex


def _assert_replaceable_table_directory(output_dir: Path) -> None:
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError(f"Existing output path is protected: {output_dir}")
    if output_dir.exists():
        for previous in output_dir.iterdir():
            if (previous.name not in OUTPUT_FILENAMES or previous.is_symlink()
                    or not previous.is_file()):
                raise ValueError(f"Unrecognized output content is protected: {previous}")
            if not previous.read_text(encoding="utf-8").startswith(_VERSION_MARKER):
                raise ValueError(f"Existing earlier or unversioned table is protected: {previous}")


def generate_latex_tables(input_csv: str | Path = DEFAULT_INPUT_CSV,
                          output_dir: str | Path = DEFAULT_OUTPUT_DIR, **read_options) -> tuple[Path, ...]:
    """Stage and replace all eight fragments together, under an exclusive lock.

    A caught failure during directory publication restores the previous set.
    An interrupted process can leave the lock/backup for explicit recovery;
    another writer must never guess that an existing lock is stale.
    """
    output_dir = Path(output_dir)
    protected_names = {"latex_tables", *(f"latex_tables_v{version}" for version in range(1, MAIN_PROTOCOL_VERSION))}
    if any(part.lower() in protected_names
           for path in (output_dir, output_dir.resolve()) for part in path.parts):
        raise ValueError(f"Earlier table directories are protected; choose latex_tables_v{MAIN_PROTOCOL_VERSION} or a separate directory.")
    _assert_replaceable_table_directory(output_dir)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Table output is locked by another writer: {lock_path}") from error
    staged = None
    backup = None
    published = False
    try:
        os.close(lock_fd)
        _assert_replaceable_table_directory(output_dir)
        analyses = read_condition_analyses(input_csv, **read_options)
        tables = build_latex_tables(analyses)
        staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
        for filename, latex in tables.items():
            _write_table(staged / filename, latex)
        assert set(path.name for path in staged.iterdir()) == set(OUTPUT_FILENAMES)
        staged.chmod(0o755)
        # Check again immediately before moving any existing publication.
        _assert_replaceable_table_directory(output_dir)
        if output_dir.exists():
            backup = staged.with_name(staged.name + ".previous")
            os.replace(output_dir, backup)
        try:
            os.replace(staged, output_dir)
        except BaseException:
            if backup is not None:
                os.replace(backup, output_dir)
                backup = None
            raise
        published = True
        return tuple(output_dir / filename for filename in OUTPUT_FILENAMES)
    finally:
        try:
            if staged is not None and staged.exists():
                shutil.rmtree(staged)
            if published and backup is not None:
                shutil.rmtree(backup)
        finally:
            lock_path.unlink()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    if not arguments.input.is_file():
        parser.error(
            f"Results CSV not found: {arguments.input.resolve()}. "
            "First run python run_experiment_main_evaluation_on_synthetic_data.py "
            f"to generate the v{MAIN_PROTOCOL_VERSION} results, or use --input to select a completed "
            f"v{MAIN_PROTOCOL_VERSION} CSV with its metadata companion."
        )
    print(f"Reading {arguments.input.resolve()}", flush=True)
    print(f"DiP-ROC: {len(SETTINGS)} prior settings, {len(CONTRASTS)} paired contrasts, "
          f"{len(OUTPUT_FILENAMES)} table fragments.", flush=True)
    paths = generate_latex_tables(arguments.input, arguments.output_dir)
    print(f"Saved {len(paths)} validated table fragments to {arguments.output_dir.resolve()}")
    for path, label in zip(paths, OUTPUT_LABELS, strict=True):
        print(f"  {path.name}  [LaTeX label: {label}]")
    print("Replace the corresponding old tables when including these fragments; main.tex has not been edited.")
    return paths


CLASS_COUNT_CONDITIONS = experiment.CLASS_COUNT_CONDITIONS
CLASS_COUNT_PAIRS = experiment.CLASS_COUNT_PAIRS
SIMULATION_SCENARIOS = experiment.SIMULATION_SCENARIOS
N_BINS = experiment.N_BINS
_DIP_CONDITIONS = CLASS_COUNT_CONDITIONS
OUTPUT_FILENAMES = tuple("tab_main_evaluation_on_synthetic_data_" + name + ".tex" for name in (
    "appendix_full", "calibration_by_scenario", "balanced_sharpness", "severe_imbalance_roc",
    "paired_method_differences", "computation_time", "historical_auc_adjustments",
    "coverage_by_class_counts",
))
OUTPUT_LABELS = (
    "tab:main-evaluation-on-synthetic-data-appendix-full", "tab:class-counts-calibration-by-scenario",
    "tab:class-counts-balanced-sharpness", "tab:class-counts-severe-imbalance-roc",
    "tab:class-counts-paired-method-differences", "tab:class-counts-computation-time",
    "tab:class-counts-historical-auc-adjustments",
    "tab:class-counts-coverage-summary",
)
_VERSION_MARKER = f"% Generated from validated v{MAIN_PROTOCOL_VERSION} comparison replications.\n"


def _mean_metric(analyses, level, attribute):
    assert analyses
    return fmean(getattr(a.summaries_by_level[level], attribute) for a in analyses)


def _class_count_summary(analyses, note, *, caption, label, metrics, headers, percent=False):
    """Average scenarios equally, keeping each nine-setting count pair together."""
    rows = []
    scale, digits = (100.0, 2) if percent else (1.0, 3)
    for positive, negative in CLASS_COUNT_PAIRS:
        for setting in SETTINGS:
            selected = [a for a in analyses if a.setting == setting
                        and (a.condition.n_positive, a.condition.n_negative) == (positive, negative)]
            if selected:
                rows.append([positive, negative, DISPLAY_LABELS[setting]] + [
                    f"{scale * _mean_metric(selected, level, metric):.{digits}f}"
                    for metric in metrics for level in NOMINAL_LEVELS
                ])
    assert rows
    caption += (
        " Test and external samples have the displayed class counts."
        " Entries average the included score scenarios with equal weight."
        " Detailed estimates and Monte Carlo standard errors are reported by configuration." + note
    )
    panel_size = 3 * len(SETTINGS)
    return "\n\n".join(
        _table(caption, label, ["$n_+$", "$n_-$", "Setting", *headers],
               rows[start:start + panel_size], continued=start > 0)
        for start in range(0, len(rows), panel_size)
    )


def build_latex_tables(analyses: Sequence[ConditionAnalysis]) -> dict[str, str]:
    analyses = tuple(analyses)
    bundles = _setting_bundles(analyses)
    note = _run_note(analyses)
    scenarios = tuple(s for s in SIMULATION_SCENARIOS if any(b[0].condition.score_scenario == s for b in bundles))
    absolute_parts = []
    difference_parts = []
    timing_parts = []
    effects = {(e.condition.index, e.level, e.left_setting, e.right_setting): e
               for e in build_paired_method_effects(analyses)}
    for scenario in scenarios:
        selected = [b for b in bundles if b[0].condition.score_scenario == scenario]
        # Keep each complete condition together: at most 3 x 9 rows.
        condition_panels = [selected[start:start + 3] for start in range(0, len(selected), 3)]
        for level in NOMINAL_LEVELS:
            for panel in condition_panels:
                rows = []
                for bundle in panel:
                    c = bundle[0].condition
                    rows.extend([c.n_positive, c.n_negative, f"{level:.2f}", DISPLAY_LABELS[a.setting]]
                                + _absolute_cells(a, level) for a in bundle)
                absolute_parts.append(_table(
                    f"DiP-ROC prior comparison: {scenario} score scenario, nominal level {level:.2f}. "
                    "Estimates with Monte Carlo standard errors; coverage in percent. Smaller regions require satisfactory coverage." + note,
                    OUTPUT_LABELS[0], ["$n_+$", "$n_-$", "Level", "Setting", "ROC cov. (\\%)", "AUC cov. (\\%)", "ROC radius", "AUC width"],
                    rows, continued=bool(absolute_parts), font="scriptsize",
                ))
        # All 18 signed contrasts fit in one condition/level panel.
        for bundle in selected:
            c = bundle[0].condition
            for level in NOMINAL_LEVELS:
                differences = []
                for left, right in CONTRASTS:
                    differences.append([c.n_positive, c.n_negative, f"{level:.2f}", _contrast_label(left, right)]
                                       + _difference_cells(effects[(c.index, level, left, right)]))
                difference_parts.append(_table(
                    f"Paired setting differences (left minus right): {scenario}, $(n_+,n_-)=({c.n_positive},{c.n_negative})$, nominal level {level:.2f}. "
                    "Coverage differences are percentage points; radius and width differences use their original units. "
                    "The eighteen contrasts compare families at fixed concentration and concentrations within each family. "
                    "Each entry includes its paired Monte Carlo standard error." + note,
                    OUTPUT_LABELS[4], ["$n_+$", "$n_-$", "Level", "Contrast", r"$\Delta$ ROC cov.", r"$\Delta$ AUC cov.", r"$\Delta$ radius", r"$\Delta$ width"],
                    differences, continued=bool(difference_parts), font="scriptsize",
                ))
        timing_note = (
            " Entries are milliseconds with Monte Carlo standard errors. Both nominal levels are constructed jointly, once per setting and replication."
            " Historical sampling, preparation of historical histograms and their empirical AUC, shared data preparation, external evaluation and output are excluded."
            " Each DiP-ROC construction includes its own prior preparation: uniform parameters, linear target restriction and calibration, or histogram smoothing, normalization and scaling."
            " Measurements use one computational thread after separate warm-ups and rotation through nine execution positions." + note
        )
        for panel in condition_panels:
            times = []
            for bundle in panel:
                c = bundle[0].condition
                # These summaries use B values, never the 2B level rows.
                for analysis in bundle:
                    estimate = summarize_mean(analysis.computation_times_ms)
                    times.append([c.n_positive, c.n_negative, DISPLAY_LABELS[analysis.setting],
                                  _format_estimate(estimate.mean, estimate.standard_error)])
            timing_parts.append(_table(
                f"Descriptive computation times: {scenario}." + timing_note,
                OUTPUT_LABELS[5], ["$n_+$", "$n_-$", "Setting", "Time (ms)"], times, continued=bool(timing_parts),
            ))
    calibration = []
    for scenario in scenarios:
        for setting in SETTINGS:
            selected = [a for a in analyses if a.setting == setting and a.condition.score_scenario == scenario]
            calibration.append([scenario, DISPLAY_LABELS[setting]] + [f"{100 * _mean_metric(selected, level, metric):.2f}"
                for metric in ("roc_coverage", "auc_coverage") for level in NOMINAL_LEVELS])
    sharpness = _class_count_summary(
        analyses, note, caption="Mean uncertainty size by class counts. Interpret alongside coverage for the same settings and counts.",
        label=OUTPUT_LABELS[2], metrics=("roc_radius_mean", "auc_interval_width_mean"),
        headers=["Radius 80\\%", "Radius 95\\%", "Width 80\\%", "Width 95\\%"],
    )
    coverage = _class_count_summary(
        analyses, note, caption="External ROC and AUC coverage (percent) by class counts. Column headings give nominal targets; averages are not condition-wise coverage guarantees.",
        label=OUTPUT_LABELS[7], metrics=("roc_coverage", "auc_coverage"), percent=True,
        headers=["ROC 80\\%", "ROC 95\\%", "AUC 80\\%", "AUC 95\\%"],
    )
    imbalance_parts = []
    adjustments = []
    for bundle in bundles:
        linear = next(a for a in bundle if a.setting == "dip_linear_m1")
        c = linear.condition
        adjusted_count = sum(linear.historical_auc_adjusted)
        adjustments.append([c.score_scenario, f"$({c.n_positive},{c.n_negative})$", adjusted_count,
                            linear.n_replications, f"{100 * adjusted_count / linear.n_replications:.2f}"])
    for positive, negative in ((20, 500), (500, 20)):
        imbalance = []
        for bundle in bundles:
            c = bundle[0].condition
            if (c.n_positive, c.n_negative) == (positive, negative):
                for a in bundle:
                    imbalance.append([c.score_scenario, f"$({c.n_positive},{c.n_negative})$", DISPLAY_LABELS[a.setting]] + [
                        _format_estimate(a.summaries_by_level[level].roc_coverage,
                                         a.summaries_by_level[level].roc_coverage_standard_error, percent=True)
                        for level in NOMINAL_LEVELS])
        if imbalance:
            imbalance_parts.append(_table(
                f"Simultaneous external ROC coverage under severe imbalance, $(n_+,n_-)=({positive},{negative})$, "
                "in percent with Monte Carlo standard errors." + note,
                OUTPUT_LABELS[3], ["Scenario", "$(n_+,n_-)$", "Setting", "80\\%", "95\\%"], imbalance,
                continued=bool(imbalance_parts),
            ))
    if not imbalance_parts:
        imbalance_parts.append(_table("No severe-imbalance configuration is included in this reduced run." + note,
                                      OUTPUT_LABELS[3], ["Scenario", "Setting", "80\\%", "95\\%"], []))
    return _validated_fragments(dict(zip(OUTPUT_FILENAMES, (
        "\n\n".join(absolute_parts),
        _table("External calibration by scenario. Equal-weight descriptive averages over the included class-count configurations; not condition-wise coverage guarantees." + note,
               OUTPUT_LABELS[1], ["Scenario", "Setting", "ROC 80\\%", "ROC 95\\%", "AUC 80\\%", "AUC 95\\%"], calibration),
        sharpness,
        "\n\n".join(imbalance_parts),
        "\n\n".join(difference_parts),
        "\n\n".join(timing_parts),
        _table("Historical AUC estimates adjusted to $[0.5,0.86]$ for the linear priors only. "
               "Each of the $\\mathcal B$ historical samples is counted once, independently of concentration and nominal level;"
               " values exactly at either endpoint are not adjusted. No replication is discarded." + note,
               OUTPUT_LABELS[6], ["Scenario", "$(n_+,n_-)$", "Adjusted", r"$\mathcal B$", r"Adjusted (\%)"], adjustments),
        coverage,
    ))))


if __name__ == "__main__":
    main()
