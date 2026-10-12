"""Create DiP-ROC coverage, sharpness, timing and source tables for real data."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

if not __debug__:
    raise RuntimeError("This table generator relies on runtime assertions; do not use python -O.")

import run_experiment_empirical_evaluation_on_real_world_datasets as experiment
from roc_utils.evaluation_criteria import summarize_mean
from roc_utils.experiment_provenance import file_sha256, read_run_metadata
from roc_utils.experiment_results import (
    RawLevelMeasurements, assert_complete_level_bundle,
    parse_canonical_int, parse_raw_level_measurements,
)
from roc_utils.real_datasets import REAL_DATASET_NAMES, real_dataset_metadata
from roc_utils.utilities_experiments import (
    EvaluationCriteriaSummary, ReplicationMeasurements,
    summarize_replication_measurements,
)

METHODS = experiment.METHODS
REAL_PROTOCOL_VERSION = experiment.REAL_PROTOCOL_VERSION
NOMINAL_LEVELS = experiment.NOMINAL_LEVELS
N_REPLICATIONS = experiment.OFFICIAL_REPLICATIONS
N_SAMPLES = experiment.OFFICIAL_SAMPLES
RAW_SCHEMA_VERSION = experiment.RAW_SCHEMA_VERSION
RAW_FIELDNAMES = experiment.RAW_FIELDNAMES
REAL_DATA_CONDITIONS = experiment.REAL_DATA_CONDITIONS
# Exclude these datasets from reporting because of insufficient observations.
EXCLUDED_DATASET_NAMES = frozenset({"Alzheimer", "Fertility", "Sonar"})
TABLE_DATA_CONDITIONS = tuple(
    c for c in REAL_DATA_CONDITIONS if c.dataset_name not in EXCLUDED_DATASET_NAMES
)
N_BINS_VALUES = experiment.N_BINS_VALUES
DEFAULT_INPUT_CSV = experiment.DEFAULT_OUTPUT_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_CSV.parent / f"latex_tables_v{REAL_PROTOCOL_VERSION}"
_VERSION_MARKER = "% Generated from validated v3 DiP-ROC real-data replications.\n"
OUTPUT_FILENAMES = tuple("tab_empirical_evaluation_on_real_world_datasets_" + name + ".tex" for name in (
    "appendix_full", "core_summary", "core_coverage", "core_sharpness",
    "grid_summary", "dataset_coverage", "computation_time", "sources",
))
OUTPUT_LABELS = (
    "tab:empirical-evaluation-on-real-world-datasets-appendix-full",
    "tab:real-core-summary", "tab:real-core-coverage", "tab:real-core-sharpness",
    "tab:real-grid-summary",
    "tab:empirical-evaluation-on-real-world-datasets-dataset-coverage",
    "tab:empirical-evaluation-on-real-world-datasets-computation-time",
    "tab:real-data-sources",
)
CORE_N_BINS = experiment.MAIN_N_BINS
SUMMARY_METRICS = (
    (r"ROC coverage (\%)", "roc_coverage", True),
    (r"AUC coverage (\%)", "auc_coverage", True),
    ("Mean ROC radius", "roc_radius_mean", False),
    ("Mean AUC width", "auc_interval_width_mean", False),
)


@dataclass(frozen=True)
class ConditionAnalysis:
    """Validated DiP-ROC records, with one construction time per replication."""

    condition: object
    method: str
    records_by_level: dict[float, tuple[RawLevelMeasurements, ...]]
    computation_times_ms: tuple[float, ...]
    data_stream_ids: tuple[str, ...]
    n_samples: int
    summaries_by_level: dict[float, EvaluationCriteriaSummary] = field(init=False)

    def __post_init__(self) -> None:
        assert self.method == "DiP-ROC"
        assert self.condition in REAL_DATA_CONDITIONS
        assert type(self.n_samples) is int and self.n_samples >= 2
        assert set(self.records_by_level) == set(NOMINAL_LEVELS)
        count = len(self.computation_times_ms)
        assert count >= 2
        assert len(self.data_stream_ids) == len(set(self.data_stream_ids)) == count
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
                auc_intervals={level: (bundled[level].auc_interval_lower,
                                       bundled[level].auc_interval_upper) for level in NOMINAL_LEVELS},
            ))
        object.__setattr__(self, "summaries_by_level", summarize_replication_measurements(
            replications, NOMINAL_LEVELS,
        ))

    @property
    def n_replications(self) -> int:
        return len(self.computation_times_ms)


def _validate_analyses(analyses: Sequence[ConditionAnalysis]) -> tuple[ConditionAnalysis, ...]:
    analyses = tuple(analyses)
    assert analyses
    assert len({a.condition.index for a in analyses}) == len(analyses)
    assert len({a.n_replications for a in analyses}) == 1
    assert len({a.n_samples for a in analyses}) == 1
    for name in REAL_DATASET_NAMES:
        selected = [a for a in analyses if a.condition.dataset_name == name]
        if selected:
            assert all(a.data_stream_ids == selected[0].data_stream_ids for a in selected)
    # External ROC/AUC values are grid-specific and must not be equated across grids.
    analyses = tuple(a for a in analyses if a.condition.dataset_name not in EXCLUDED_DATASET_NAMES)
    assert analyses, "No retained datasets remain after exclusions."
    return tuple(sorted(analyses, key=lambda a: a.condition.index))


def read_condition_analyses(
    input_csv: str | Path, *, conditions: Sequence | None = None,
    n_replications: int = N_REPLICATIONS, n_samples: int = N_SAMPLES,
    master_seed: int | None = None,
) -> tuple[ConditionAnalysis, ...]:
    """Validate the complete v3 CSV and companion; return only retained datasets.

    Reduced integrations must explicitly supply their conditions and B/S values.
    Earlier results retain their original meaning, but cannot be used as v3 runs.
    """
    input_csv = Path(input_csv)
    conditions = tuple(REAL_DATA_CONDITIONS if conditions is None else conditions)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
            raise ValueError("Expected a v3 DiP-ROC real-data CSV; v1 has no timings and v2 describes a different execution protocol.")
        first = next(reader, None)
        if first is None or first["schema_version"] != RAW_SCHEMA_VERSION:
            raise ValueError("Expected nonempty v3 DiP-ROC results; earlier or mixed protocols are not accepted.")
        recorded_seed = parse_canonical_int(first, "master_seed")
    if master_seed is not None:
        assert recorded_seed == master_seed
    metadata = read_run_metadata(input_csv, metadata_schema="real_run_metadata_v1")
    experiment.validate_run_metadata(metadata, conditions=conditions,
                                     n_replications=n_replications, n_samples=n_samples,
                                     master_seed=recorded_seed)
    expected_checksum = metadata["raw_csv_sha256"]
    experiment.validate_raw_results_csv(
        input_csv, conditions=conditions, n_replications=n_replications,
        n_samples=n_samples, master_seed=recorded_seed, metadata=metadata,
    )
    records = {c.index: {level: {} for level in NOMINAL_LEVELS} for c in conditions}
    timings = {c.index: {} for c in conditions}
    data_ids = {c.index: {} for c in conditions}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == RAW_FIELDNAMES
        for row in reader:
            assert row["method"] == "DiP-ROC"
            index = parse_canonical_int(row, "condition_index")
            assert index in records
            replication = parse_canonical_int(row, "replication")
            assert 0 <= replication < n_replications
            record = parse_raw_level_measurements(row)
            assert replication not in records[index][record.level]
            records[index][record.level][replication] = record
            time_ns = parse_canonical_int(row, "computation_time_ns")
            if replication in timings[index]:
                assert timings[index][replication] == time_ns
                assert data_ids[index][replication] == row["data_stream_id"]
            else:
                timings[index][replication] = time_ns
                data_ids[index][replication] = row["data_stream_id"]
    result = []
    for condition in conditions:
        index = condition.index
        assert set(timings[index]) == set(range(n_replications))
        assert all(set(records[index][level]) == set(range(n_replications)) for level in NOMINAL_LEVELS)
        result.append(ConditionAnalysis(
            condition=condition, method="DiP-ROC", n_samples=n_samples,
            records_by_level={level: tuple(records[index][level][b] for b in range(n_replications))
                              for level in NOMINAL_LEVELS},
            computation_times_ms=tuple(timings[index][b] / 1_000_000.0 for b in range(n_replications)),
            data_stream_ids=tuple(data_ids[index][b] for b in range(n_replications)),
        ))
    if file_sha256(input_csv) != expected_checksum:
        raise RuntimeError("The CSV changed while its tables were being prepared; no tables were published.")
    return _validate_analyses(result)


def _format_estimate(mean: float, standard_error: float, *, percent=False, time=False) -> str:
    assert math.isfinite(mean) and math.isfinite(standard_error) and standard_error >= 0
    scale = 100.0 if percent else 1.0
    digits, se_digits = (2, 2) if percent else ((5, 5) if time else (5, 6))
    return f"{scale * mean:.{digits}f} ({scale * standard_error:.{se_digits}f})"


def _absolute_cells(analysis: ConditionAnalysis, level: float) -> list[str]:
    summary = analysis.summaries_by_level[level]
    return [
        _format_estimate(summary.roc_coverage, summary.roc_coverage_standard_error, percent=True),
        _format_estimate(summary.auc_coverage, summary.auc_coverage_standard_error, percent=True),
        _format_estimate(summary.roc_radius_mean, summary.roc_radius_standard_error),
        _format_estimate(summary.auc_interval_width_mean, summary.auc_interval_width_standard_error),
    ]


def _run_note(analyses: Sequence[ConditionAnalysis]) -> str:
    first = analyses[0]
    note = rf" $\mathcal B={first.n_replications}$, $\mathcal S={first.n_samples}$, uniform prior $m=1$."
    if (first.n_replications != N_REPLICATIONS or first.n_samples != N_SAMPLES
            or len(analyses) != len(TABLE_DATA_CONDITIONS)):
        note += " Reduced validation run; these are not official experimental results."
    return note


def _table(caption: str, label: str, headers: Sequence[str], rows: Sequence,
           *, continued=False, font="small", column_spec=None, header_groups=None) -> str:
    lines = [r"\begin{table}[p]" if continued else r"\begin{table}[t]"]
    if continued:
        lines.append(r"\ContinuedFloat")
    lines.extend([r"\centering", (r"\caption[]{" if continued else r"\caption{") + caption + "}"])
    if not continued:
        lines.append(r"\label{" + label + "}")
    column_spec = column_spec or "l" * len(headers)
    lines.extend([r"\begingroup", "\\" + font, r"\setlength{\tabcolsep}{4pt}",
                  r"\renewcommand{\arraystretch}{1.10}",
                  r"\begin{tabular}{@{}" + column_spec + r"@{}}", r"\hline"])
    if header_groups is not None:
        lines.append(header_groups + r" \\")
    lines.extend([" & ".join(headers) + r" \\", r"\hline"])
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
        assert "BootP" not in latex and "paired method" not in latex.lower()
        result[filename] = _VERSION_MARKER + latex + "\n"
    return result


def _split_counts(dataset_name: str) -> tuple[int, int, int, int]:
    metadata = real_dataset_metadata(dataset_name)
    hp, hn = metadata.n_positive // 5, metadata.n_negative // 5
    return metadata.n_positive - 2 * hp, metadata.n_negative - 2 * hn, hp, hn


def _mean_metric(analyses, level, attribute):
    assert analyses
    return fmean(getattr(a.summaries_by_level[level], attribute) for a in analyses)


def _format_summary_value(value: float, *, percent: bool) -> str:
    return f"{100 * value:.2f}" if percent else f"{value:.3f}"


def _mean_and_range(analyses: Sequence[ConditionAnalysis], level: float,
                    attribute: str, *, percent: bool) -> list[str]:
    """Equal-dataset mean and observed extrema of the dataset-specific means."""
    if not analyses:
        return ["---", "---"]
    values = [getattr(a.summaries_by_level[level], attribute) for a in analyses]
    return [
        _format_summary_value(fmean(values), percent=percent),
        _format_summary_value(min(values), percent=percent) + "--"
        + _format_summary_value(max(values), percent=percent),
    ]


def _latex_text(text: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
               "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(character, character) for character in text)


def _positive_class_rule(metadata) -> str:
    if metadata.positive_threshold is not None:
        return rf"$\ge {metadata.positive_threshold:g}$"
    assert metadata.positive_values
    return " or ".join(r"\texttt{" + _latex_text(value) + "}" for value in metadata.positive_values)


def build_latex_tables(analyses: Sequence[ConditionAnalysis]) -> dict[str, str]:
    analyses = _validate_analyses(analyses)
    note = _run_note(analyses)
    datasets = tuple(name for name in REAL_DATASET_NAMES if any(a.condition.dataset_name == name for a in analyses))
    appendix_parts = []
    for part, start in enumerate(range(0, len(datasets), 4)):
        rows = []
        for name in datasets[start:start + 4]:
            train_p, train_n, hp, hn = _split_counts(name)
            rows.append(r"\multicolumn{6}{@{}l}{\textbf{" + _latex_text(name) + "} "
                        + rf"($n_{{\mathrm{{tr}}}}={train_p + train_n}$, $n_+^{{\mathrm{{ts}}}}=n_+^{{\mathrm{{ext}}}}={hp}$, "
                        + rf"$n_-^{{\mathrm{{ts}}}}=n_-^{{\mathrm{{ext}}}}={hn}$)" + r"} \\")
            for analysis in analyses:
                if analysis.condition.dataset_name == name:
                    for level in NOMINAL_LEVELS:
                        rows.append([analysis.condition.n_bins, f"{level:.2f}"] + _absolute_cells(analysis, level))
            rows.append(r"\hline")
        appendix_parts.append(_table(
            "DiP-ROC: dataset- and grid-level coverage and sharpness. Each estimate is followed by its Monte Carlo standard error. "
            "Coverage is in percent; uncertainty describes randomized splits conditional on the available dataset. "
            "Test and external class counts are identical. The classifier is an $L_2$-regularized logistic regression with $C=1$." + note,
            OUTPUT_LABELS[0], ["$\\mathcal K$", "Level", r"\shortstack{ROC coverage\\(\%)}", r"\shortstack{AUC coverage\\(\%)}",
                               r"\shortstack{Mean ROC\\radius}", r"\shortstack{Mean AUC\\width}"],
            rows, continued=part > 0, font="scriptsize", column_spec="rrrrrr",
        ))
    by_dataset_grid = {(a.condition.dataset_name, a.condition.n_bins): a for a in analyses}
    core = tuple(a for a in analyses if a.condition.n_bins == CORE_N_BINS)
    core_note = "" if core else rf" No $\mathcal K={CORE_N_BINS}$ results were supplied; dashes denote unavailable estimates."
    if core and len(core) != len(datasets):
        core_note += rf" Dashes identify datasets without supplied $\mathcal K={CORE_N_BINS}$ results; these datasets are excluded from the summary."
    core_summary_note = core_note + (
        f" Each mean gives equal weight to the {len(core)} included dataset-specific estimates." if core else ""
    )
    summary_rows = [
        [title] + [cell for level in NOMINAL_LEVELS
                   for cell in _mean_and_range(core, level, metric, percent=percent)]
        for title, metric, percent in SUMMARY_METRICS
    ]
    core_coverage_rows, core_sharpness_rows, dataset_rows = [], [], []
    for name in datasets:
        analysis = by_dataset_grid.get((name, CORE_N_BINS))
        _, _, hp, hn = _split_counts(name)
        coverage_cells, sharpness_cells = [], []
        for _, metric, percent in SUMMARY_METRICS:
            cells = [
                _format_summary_value(getattr(analysis.summaries_by_level[level], metric), percent=percent)
                if analysis else "---" for level in NOMINAL_LEVELS
            ]
            (coverage_cells if percent else sharpness_cells).extend(cells)
        core_coverage_rows.append([_latex_text(name), hp, hn] + coverage_cells)
        core_sharpness_rows.append([_latex_text(name)] + sharpness_cells)
        for level in NOMINAL_LEVELS:
            values = []
            for k in N_BINS_VALUES:
                analysis = by_dataset_grid.get((name, k))
                values += ([f"{100 * getattr(analysis.summaries_by_level[level], metric):.2f}"
                            for metric in ("roc_coverage", "auc_coverage")] if analysis else ["---", "---"])
            dataset_rows.append([_latex_text(name) if level == NOMINAL_LEVELS[0] else "",
                                 int(round(100 * level))] + values)
    grid_rows, grid_counts = [], []
    for k in N_BINS_VALUES:
        selected = [a for a in analyses if a.condition.n_bins == k]
        grid_counts.append(rf"$\mathcal K={k}$: {len(selected)} datasets")
        for level in NOMINAL_LEVELS:
            grid_rows.append([k, int(round(100 * level))] + [
                _format_summary_value(_mean_metric(selected, level, metric), percent=percent)
                if selected else "---" for _, metric, percent in SUMMARY_METRICS
            ])
    grid_note = " Included datasets by grid: " + "; ".join(grid_counts) + "."
    if any((name, k) not in by_dataset_grid for name in datasets for k in N_BINS_VALUES):
        grid_note += " Only supplied dataset/grid estimates are averaged; dashes denote unavailable estimates."
    timing_parts = []
    for part, start in enumerate(range(0, len(datasets), 6)):
        rows = []
        for analysis in analyses:
            if analysis.condition.dataset_name in datasets[start:start + 6]:
                timing = summarize_mean(analysis.computation_times_ms)
                rows.append([_latex_text(analysis.condition.dataset_name), analysis.condition.n_bins,
                             _format_estimate(timing.mean, timing.standard_error, time=True)])
        timing_parts.append(_table(
            "DiP-ROC construction time in milliseconds, with Monte Carlo standard errors. Each time constructs both nominal levels jointly "
            "and is counted once per replication and grid. One computation thread is used after a separate warm-up. "
            "Training, preprocessing, scores, histogram preparation, the precomputed uniform prior, external evaluation and output are excluded." + note,
            OUTPUT_LABELS[6], ["Dataset", "$\\mathcal K$", "Mean time (SE), ms"],
            rows, continued=part > 0, font="small", column_spec="lrr",
        ))
    source_rows = []
    for name in datasets:
        metadata = real_dataset_metadata(name)
        _, _, hp, hn = _split_counts(name)
        source_rows.append([_latex_text(name), f"{metadata.openml_data_id} ({metadata.openml_version})",
                            _positive_class_rule(metadata), metadata.n_complete_rows, hp, hn])
    return _validated_fragments(dict(zip(OUTPUT_FILENAMES, (
        "\n\n".join(appendix_parts),
        _table(rf"Real data, $\mathcal K={CORE_N_BINS}$: coverage and uncertainty size across datasets. "
               "The minimum--maximum columns describe variation between datasets, not confidence intervals. "
               "Dataset-level estimates and Monte Carlo standard errors appear in the detailed appendix table." + core_summary_note + note,
               OUTPUT_LABELS[1], ["Measure", "Mean", "Minimum--maximum", "Mean", "Minimum--maximum"],
               summary_rows, column_spec="lrrrr",
               header_groups=r" & \multicolumn{2}{c}{$80\%$ nominal level} & \multicolumn{2}{c}{$95\%$ nominal level}"),
        _table(rf"Real data, $\mathcal K={CORE_N_BINS}$: class counts and coverage (\%) for uniform DiP-ROC with $m=1$. "
               "The counts apply to each of the test and external samples. Coverage headings give nominal targets. "
               "Monte Carlo standard errors appear in the detailed appendix table." + core_note + note,
               OUTPUT_LABELS[2], ["Dataset", r"$n_+^{\mathrm{ts}}$", r"$n_-^{\mathrm{ts}}$", r"$80\%$", r"$95\%$", r"$80\%$", r"$95\%$"],
               core_coverage_rows, column_spec="lrrrrrr",
               header_groups=r" & \multicolumn{2}{c}{Each held-out sample} & \multicolumn{2}{c}{ROC coverage (\%)} & \multicolumn{2}{c}{AUC coverage (\%)}"),
        _table(rf"Real data, $\mathcal K={CORE_N_BINS}$: mean uncertainty size for the same predictions as Table~\ref{{tab:real-core-coverage}}. "
               "Smaller values indicate greater precision only when coverage is adequate. "
               "Monte Carlo standard errors appear in the detailed appendix table." + core_note + note,
               OUTPUT_LABELS[3], ["Dataset", r"$80\%$", r"$95\%$", r"$80\%$", r"$95\%$"],
               core_sharpness_rows, column_spec="lrrrr",
               header_groups=r" & \multicolumn{2}{c}{Mean ROC radius} & \multicolumn{2}{c}{Mean AUC width}"),
        _table("Supplementary real-data check: coverage and uncertainty size by grid resolution. "
               "Entries average dataset-specific replication means with equal weight. "
               "These descriptive averages are not guarantees for individual datasets; detailed estimates and Monte Carlo standard errors appear in the appendix."
               + grid_note + note,
               OUTPUT_LABELS[4], [r"$\mathcal K$", r"\shortstack{Nominal\\level (\%)}", "ROC", "AUC", "ROC radius", "AUC width"],
               grid_rows, column_spec="rrrrrr",
               header_groups=r" & & \multicolumn{2}{c}{Coverage (\%)} & \multicolumn{2}{c}{Mean uncertainty size}"),
        _table("DiP-ROC dataset-level external coverage in percent at the two nominal credibility levels. The same split and fitted classifier are used "
               "across grids. Monte Carlo standard errors and sharpness appear in the detailed appendix table. Dashes denote unavailable estimates." + note,
               OUTPUT_LABELS[5], ["Dataset", r"\shortstack{Level\\(\%)}", "ROC 5", "AUC 5", "ROC 10", "AUC 10", "ROC 20", "AUC 20"],
               dataset_rows, font="small", column_spec="lrrrrrrr"),
        "\n\n".join(timing_parts),
        _table("Real-data sources and fixed binary encodings. Rules are original target labels, except Abalone, where the target is the number of rings. "
               "Complete cases exclude rows missing any source value, before predictor exclusions. The last two columns apply to each of the test and external samples; "
               "all remaining cases form training data." + note,
               OUTPUT_LABELS[7], ["Dataset", r"\shortstack{OpenML ID\\(version)}", r"\shortstack{Source rule\\for $Y=1$}",
                                  r"\shortstack{Complete\\cases}", r"$n_+^{\mathrm{ts}}$", r"$n_-^{\mathrm{ts}}$"],
               source_rows, font="small", column_spec="lclrrr"),
    ))))


def _write_table(output_path: Path, latex: str) -> None:
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
            if (previous.name not in OUTPUT_FILENAMES or previous.is_symlink() or not previous.is_file()):
                raise ValueError(f"Unrecognized output content is protected: {previous}")
            if not previous.read_text(encoding="utf-8").startswith(_VERSION_MARKER):
                raise ValueError(f"Existing earlier or unversioned table is protected: {previous}")


def generate_latex_tables(input_csv: str | Path = DEFAULT_INPUT_CSV,
                          output_dir: str | Path = DEFAULT_OUTPUT_DIR, **read_options) -> tuple[Path, ...]:
    """Validate then stage all eight fragments; restore the old set on failure."""
    input_csv, output_dir = Path(input_csv), Path(output_dir)
    protected_names = {"latex_tables", "latex_tables_v1", "latex_tables_v2",
                       "archived_experiments", "back_up_legacy_pipeline"}
    if any(part.lower() in protected_names for path in (output_dir, output_dir.resolve()) for part in path.parts):
        raise ValueError("Earlier table directories are protected; use latex_tables_v3 or a separate directory.")
    _assert_replaceable_table_directory(output_dir)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Table output is locked by another writer: {lock_path}") from error
    staged, backup, published = None, None, False
    try:
        os.close(lock_fd)
        _assert_replaceable_table_directory(output_dir)
        expected_checksum = file_sha256(input_csv)
        analyses = read_condition_analyses(input_csv, **read_options)
        tables = build_latex_tables(analyses)
        staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
        for filename, latex in tables.items():
            _write_table(staged / filename, latex)
        assert set(path.name for path in staged.iterdir()) == set(OUTPUT_FILENAMES)
        staged.chmod(0o755)
        if file_sha256(input_csv) != expected_checksum:
            raise RuntimeError("The CSV changed during table generation; no tables were published.")
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
            "First run python run_experiment_empirical_evaluation_on_real_world_datasets.py "
            f"to generate the v{REAL_PROTOCOL_VERSION} results, or use --input to select a completed "
            f"v{REAL_PROTOCOL_VERSION} CSV with its metadata companion."
        )
    print(f"Reading {arguments.input.resolve()}", flush=True)
    print(f"DiP-ROC v{REAL_PROTOCOL_VERSION}: main K={CORE_N_BINS} presentation and grids "
          f"{N_BINS_VALUES}; {len(OUTPUT_FILENAMES)} table fragments.", flush=True)
    paths = generate_latex_tables(arguments.input, arguments.output_dir)
    print(f"Saved {len(paths)} validated table fragments to {arguments.output_dir.resolve()}")
    for path, label in zip(paths, OUTPUT_LABELS, strict=True):
        print(f"  {path.name}  [LaTeX label: {label}]")
    print("Replace the corresponding old tables when including these fragments; main.tex has not been edited.")
    return paths


if __name__ == "__main__":
    main()
