"""Create paired comparison tables and within-DiP prior-sensitivity contrasts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

if not __debug__:
    raise RuntimeError("This table generator relies on runtime assertions; do not use python -O.")

from roc_utils.simulation_datasets import SYNTHETIC_DESIGN_VERSION

import run_experiment_sensitivity_to_prior_specification as experiment

from roc_utils.evaluation_criteria import (
    MeanEstimate, summarize_mean, summarize_paired_differences,
)
from roc_utils.experiment_results import (
    RawLevelMeasurements, assert_complete_level_bundle,
    parse_canonical_int, parse_raw_level_measurements,
)
from roc_utils.utilities_experiments import (
    EvaluationCriteriaSummary, ReplicationMeasurements,
    summarize_replication_measurements,
)

METHODS = experiment.METHODS
NOMINAL_LEVELS = experiment.NOMINAL_LEVELS
N_REPLICATIONS = experiment.OFFICIAL_REPLICATIONS
N_SAMPLES = experiment.OFFICIAL_SAMPLES
RAW_SCHEMA_VERSION = experiment.RAW_SCHEMA_VERSION
RAW_FIELDNAMES = experiment.RAW_FIELDNAMES
EXPERIMENT_NAME = experiment.EXPERIMENT_NAME
RESULTS_DIRECTORY_NAME = experiment.RESULTS_DIRECTORY_NAME
PY_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = experiment.DEFAULT_OUTPUT_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_CSV.parent / f"latex_tables_v{SYNTHETIC_DESIGN_VERSION}"
_METRIC_ATTRIBUTES = {
    "roc_coverage": "roc_covered", "auc_coverage": "auc_covered",
    "roc_radius": "roc_radius", "auc_width": "auc_width",
}


@dataclass(frozen=True)
class ConditionAnalysis:
    """One method's validated records, in replication order, for one setting.

    The shared sensitivity bootstrap has condition=None: it is not a prior.
    Times contain one value per replication, not one per nominal level.
    """

    condition: object | None
    method: str
    records_by_level: dict[float, tuple[RawLevelMeasurements, ...]]
    computation_times_ms: tuple[float, ...]
    data_stream_ids: tuple[str, ...]
    n_samples: int
    summaries_by_level: dict[float, EvaluationCriteriaSummary] = field(init=False)

    def __post_init__(self) -> None:
        assert self.method in METHODS
        assert self.condition is not None or (self.method == "BootP-ROC")
        assert type(self.n_samples) is int and self.n_samples > 0
        assert set(self.records_by_level) == set(NOMINAL_LEVELS)
        count = len(self.computation_times_ms)
        assert count >= 2
        assert len(self.data_stream_ids) == count
        assert len(set(self.data_stream_ids)) == count
        assert all(self.data_stream_ids)
        assert all(math.isfinite(t) and t >= 0 for t in self.computation_times_ms)
        assert all(len(records) == count for records in self.records_by_level.values())
        replications = []
        for replication in range(count):
            bundled = assert_complete_level_bundle(
                [self.records_by_level[level][replication] for level in NOMINAL_LEVELS],
                NOMINAL_LEVELS,
            )
            limit = math.sqrt(2.0) if self.method == "DiP-ROC" else 2.0
            assert all(record.roc_radius <= limit + 1e-12 for record in bundled.values())
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


@dataclass(frozen=True)
class PairedMethodEffect:
    """DiP-ROC minus BootP-ROC for all four statistical criteria."""

    condition: object
    level: float
    metrics: dict[str, MeanEstimate]


def read_condition_analyses(
    input_csv: str | Path, *, conditions: Sequence | None = None,
    n_replications: int = N_REPLICATIONS, n_samples: int = N_SAMPLES,
    master_seed: int | None = None,
) -> tuple[ConditionAnalysis, ...]:
    """Validate v3 results and retain the records needed for paired inference.

    Defaults require the complete official experiment. Reduced integrations
    must explicitly supply both their design and their B/S values.
    """

    input_csv = Path(input_csv)
    conditions = tuple(_DIP_CONDITIONS if conditions is None else conditions)
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDNAMES:
            raise ValueError("Expected the v3 comparison CSV; v1/v2 results belong to earlier synthetic protocols.")
        first = next(reader, None)
        assert first is not None, "The comparison CSV is empty."
        recorded_seed = parse_canonical_int(first, "master_seed")
    if master_seed is not None:
        assert recorded_seed == master_seed
    experiment.validate_raw_results_csv(
        input_csv, conditions=conditions, n_replications=n_replications,
        n_samples=n_samples, master_seed=recorded_seed,
    )

    expected = [(condition.index, "DiP-ROC", condition) for condition in conditions]
    expected.append((9, "BootP-ROC", None))
    records = {(index, method): {level: {} for level in NOMINAL_LEVELS}
               for index, method, _ in expected}
    timings = {(index, method): {} for index, method, _ in expected}
    data_ids = {(index, method): {} for index, method, _ in expected}
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (parse_canonical_int(row, "condition_index"), row["method"])
            assert key in records
            replication = parse_canonical_int(row, "replication")
            record = parse_raw_level_measurements(row)
            assert replication not in records[key][record.level]
            records[key][record.level][replication] = record
            time_ns = parse_canonical_int(row, "computation_time_ns")
            if replication in timings[key]:
                assert timings[key][replication] == time_ns
                assert data_ids[key][replication] == row["data_stream_id"]
            else:
                timings[key][replication] = time_ns
                data_ids[key][replication] = row["data_stream_id"]
    analyses = []
    for index, method, condition in expected:
        key = (index, method)
        assert set(timings[key]) == set(range(n_replications))
        analyses.append(ConditionAnalysis(
            condition=condition, method=method, n_samples=n_samples,
            records_by_level={level: tuple(records[key][level][b] for b in range(n_replications))
                              for level in NOMINAL_LEVELS},
            computation_times_ms=tuple(timings[key][b] / 1_000_000.0 for b in range(n_replications)),
            data_stream_ids=tuple(data_ids[key][b] for b in range(n_replications)),
        ))
    result = tuple(analyses)
    _method_pairs(result)
    return result


def _method_pairs(analyses: Sequence[ConditionAnalysis]):
    """Return paired method analyses; a shared bootstrap is never duplicated."""

    analyses = tuple(analyses)
    assert analyses
    dip = tuple(a for a in analyses if a.method == "DiP-ROC")
    boot = tuple(a for a in analyses if a.method == "BootP-ROC")
    assert dip and len(dip) == len({a.condition.index for a in dip})
    assert len({a.n_replications for a in analyses}) == 1
    assert len({a.n_samples for a in analyses}) == 1
    assert len(boot) == 1 and boot[0].condition is None
    pairs = []
    for observed in dip:
        reference = boot[0]
        assert observed.data_stream_ids == reference.data_stream_ids
        for level in NOMINAL_LEVELS:
            assert tuple(r.external_grid_auc for r in observed.records_by_level[level]) == tuple(
                r.external_grid_auc for r in reference.records_by_level[level]
            )
        pairs.append((observed, reference))
    return tuple(pairs)


def _values(analysis: ConditionAnalysis, level: float, metric: str) -> tuple[float, ...]:
    return tuple(float(getattr(record, _METRIC_ATTRIBUTES[metric]))
                 for record in analysis.records_by_level[level])


def build_paired_method_effects(analyses: Sequence[ConditionAnalysis]) -> tuple[PairedMethodEffect, ...]:
    return tuple(PairedMethodEffect(
        condition=dip.condition, level=level,
        metrics={metric: summarize_paired_differences(_values(dip, level, metric), _values(boot, level, metric))
                 for metric in _METRIC_ATTRIBUTES},
    ) for dip, boot in _method_pairs(analyses) for level in NOMINAL_LEVELS)


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


def _time_cells(dip: ConditionAnalysis, boot: ConditionAnalysis) -> list[str]:
    estimates = (summarize_mean(dip.computation_times_ms), summarize_mean(boot.computation_times_ms),
                 summarize_paired_differences(dip.computation_times_ms, boot.computation_times_ms))
    return [_format_estimate(value.mean, value.standard_error, signed=index == 2)
            for index, value in enumerate(estimates)]


def _run_note(analyses: Sequence[ConditionAnalysis]) -> str:
    first = analyses[0]
    note = rf" $\mathcal B={first.n_replications}$, $\mathcal S={first.n_samples}$."
    if (first.n_replications != N_REPLICATIONS or first.n_samples != N_SAMPLES
            or sum(a.method == "DiP-ROC" for a in analyses) != len(_DIP_CONDITIONS)):
        note += " Reduced validation run; these are not official experimental results."
    return note


def _table(caption: str, label: str, headers: Sequence[str], rows: Sequence,
           *, continued=False, font="small") -> str:
    lines = [r"\begin{table}[p]" if continued else r"\begin{table}[t]"]
    if continued:
        lines.append(r"\ContinuedFloat")
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
        result[filename] = "% Generated from validated v3 comparison replications.\n" + latex + "\n"
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


def generate_latex_tables(input_csv: str | Path = DEFAULT_INPUT_CSV,
                          output_dir: str | Path = DEFAULT_OUTPUT_DIR, **read_options) -> tuple[Path, ...]:
    output_dir = Path(output_dir)
    if any(part.lower() in {"latex_tables", "latex_tables_v1", "latex_tables_v2"}
           for path in (output_dir, output_dir.resolve()) for part in path.parts):
        raise ValueError("Previous v1/v2 table directories are protected; choose latex_tables_v3 or a separate directory.")
    for filename in OUTPUT_FILENAMES:
        previous = output_dir / filename
        if previous.is_file() and not previous.read_text(encoding="utf-8").startswith(
            "% Generated from validated v3 comparison replications.\n"
        ):
            raise ValueError(f"Existing earlier or unversioned table is protected: {previous}")
    analyses = read_condition_analyses(input_csv, **read_options)
    tables = build_latex_tables(analyses)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, latex in tables.items():
        path = output_dir / filename
        _write_table(path, latex)
        paths.append(path)
    return tuple(paths)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    arguments = _build_argument_parser().parse_args(argv)
    return generate_latex_tables(arguments.input, arguments.output_dir)


PRIOR_CONDITIONS = experiment.PRIOR_CONDITIONS
PRIOR_PROFILES = experiment.PRIOR_PROFILES
PRIOR_MASSES = experiment.PRIOR_MASSES
N_BINS = experiment.N_BINS
_DIP_CONDITIONS = PRIOR_CONDITIONS
CONCENTRATION_CONTRASTS = ((1, 20), (1, 50))
PROFILE_LABELS = {"oracle": "Oracle", "auc_compatible": "AUC-compatible", "symmetric": "Symmetric"}
OUTPUT_FILENAMES = tuple("tab_sensitivity_to_prior_specification_" + name + ".tex" for name in (
    "appendix_full", "external_coverage", "paired_concentration_effects", "sharpness", "paired_method_differences", "computation_time",
))
OUTPUT_LABELS = tuple("tab:sensitivity-to-prior-specification-" + name for name in (
    "appendix-full", "external-coverage", "paired-concentration-effects", "sharpness", "paired-method-differences", "computation-time",
))


@dataclass(frozen=True)
class PairedConcentrationEffect:
    profile: str
    baseline_mass: int
    target_mass: int
    level: float
    metrics: dict[str, MeanEstimate]

    @property
    def roc_radius_reduction(self):
        return self.metrics["roc_radius"].mean

    @property
    def auc_width_reduction(self):
        return self.metrics["auc_width"].mean


def build_paired_concentration_effects(analyses: Sequence[ConditionAnalysis]) -> tuple[PairedConcentrationEffect, ...]:
    """Retain the original within-DiP concentration contrasts and their signs."""
    indexed = {(a.condition.prior_profile, a.condition.prior_mass): a
               for a in analyses if a.method == "DiP-ROC"}
    effects = []
    for profile in PRIOR_PROFILES:
        for baseline_mass, target_mass in CONCENTRATION_CONTRASTS:
            if (profile, baseline_mass) not in indexed or (profile, target_mass) not in indexed:
                continue
            baseline, target = indexed[(profile, baseline_mass)], indexed[(profile, target_mass)]
            assert baseline.data_stream_ids == target.data_stream_ids
            for level in NOMINAL_LEVELS:
                metrics = {}
                for metric in _METRIC_ATTRIBUTES:
                    left, right = (target, baseline) if "coverage" in metric else (baseline, target)
                    metrics[metric] = summarize_paired_differences(_values(left, level, metric), _values(right, level, metric))
                effects.append(PairedConcentrationEffect(profile, baseline_mass, target_mass, level, metrics))
    return tuple(effects)


def _prior_cells(analysis):
    if analysis.method == "BootP-ROC":
        return ["BootP-ROC", "---", "---"]
    return ["DiP-ROC", PROFILE_LABELS[analysis.condition.prior_profile], analysis.condition.prior_mass]


def build_latex_tables(analyses: Sequence[ConditionAnalysis]) -> dict[str, str]:
    analyses = tuple(analyses)
    pairs = _method_pairs(analyses)
    note = _run_note(analyses)
    display = sorted((a for a in analyses if a.method == "DiP-ROC"),
                     key=lambda a: (PRIOR_PROFILES.index(a.condition.prior_profile), a.condition.prior_mass))
    display += [a for a in analyses if a.method == "BootP-ROC"]
    absolute, coverage, sharpness = [], [], []
    for analysis in display:
        for level in NOMINAL_LEVELS:
            absolute.append(_prior_cells(analysis) + [f"{level:.2f}"] + _absolute_cells(analysis, level))
        coverage.append(_prior_cells(analysis) + [
            _format_estimate(getattr(analysis.summaries_by_level[level], metric),
                             getattr(analysis.summaries_by_level[level], metric + "_standard_error"), percent=True)
            for metric in ("roc_coverage", "auc_coverage") for level in NOMINAL_LEVELS])
        sharpness.append(_prior_cells(analysis) + [
            f"{getattr(analysis.summaries_by_level[level], metric):.3f}"
            for metric in ("roc_radius_mean", "auc_interval_width_mean") for level in NOMINAL_LEVELS])
    paired = []
    for effect in build_paired_method_effects(analyses):
        c = effect.condition
        paired.append([PROFILE_LABELS[c.prior_profile], c.prior_mass, f"{effect.level:.2f}"] + _difference_cells(effect))
    concentration = []
    for effect in build_paired_concentration_effects(analyses):
        concentration.append([PROFILE_LABELS[effect.profile], f"${effect.baseline_mass}\\to{effect.target_mass}$", f"{effect.level:.2f}"] + [
            _format_estimate(effect.metrics[metric].mean, effect.metrics[metric].standard_error,
                             percent="coverage" in metric, signed=True) for metric in _METRIC_ATTRIBUTES])
    timing = [[PROFILE_LABELS[d.condition.prior_profile], d.condition.prior_mass] + _time_cells(d, b) for d, b in pairs]
    information = (" The oracle and AUC-compatible profiles receive generating-distribution information; these are controlled prior-information comparisons. "
                   "BootP-ROC has no prior and is a single shared reference across all nine settings, based on B replications, not nine times B.")
    return _validated_fragments(dict(zip(OUTPUT_FILENAMES, (
        _table("Prior sensitivity: both methods' coverage and sharpness, with Monte Carlo standard errors; coverage in percent." + information + note,
               OUTPUT_LABELS[0], ["Method", "Prior profile", "$m$", "Level", "ROC cov. (\\%)", "AUC cov. (\\%)", "ROC radius", "AUC width"], absolute, font="scriptsize"),
        _table("External coverage in percent with Monte Carlo standard errors." + information + note,
               OUTPUT_LABELS[1], ["Method", "Prior profile", "$m$", "ROC 80\\%", "ROC 95\\%", "AUC 80\\%", "AUC 95\\%"], coverage, font="scriptsize"),
        _table("Paired DiP-ROC concentration effects: coverage is target minus baseline in percentage points; "
               "sharpness is baseline minus target, so positive values indicate smaller regions. Entries include paired Monte Carlo standard errors. "
               "Interpret jointly with external coverage." + note,
               OUTPUT_LABELS[2], ["Prior profile", "Increase", "Level", r"$\Delta$ ROC cov.", r"$\Delta$ AUC cov.", "Radius reduction", "Width reduction"], concentration),
        _table("Mean sharpness under prior specification. Smaller regions are preferable only with satisfactory external coverage. "
               "Monte Carlo standard errors are in the appendix." + information + note,
               OUTPUT_LABELS[3], ["Method", "Prior profile", "$m$", "Radius 80\\%", "Radius 95\\%", "Width 80\\%", "Width 95\\%"], sharpness),
        _table("Paired DiP-ROC minus the same shared BootP-ROC reference for every criterion. Coverage differences are percentage points; "
               "radius and width differences use their original units. Each entry includes its paired Monte Carlo standard error." + information + note,
               OUTPUT_LABELS[4], ["DiP-ROC prior", "$m$", "Level", r"$\Delta$ ROC cov.", r"$\Delta$ AUC cov.", r"$\Delta$ radius", r"$\Delta$ width"], paired),
        _table("Computation time in milliseconds with Monte Carlo standard errors, jointly for both nominal levels. "
               "Each row compares one DiP-ROC setting to the same bootstrap times, measured once per replication and not pooled across priors. "
               "The difference is DiP-ROC minus BootP-ROC." + note,
               OUTPUT_LABELS[5], ["DiP-ROC prior", "$m$", "DiP-ROC (ms)", "BootP-ROC (ms)", r"$\Delta$ (ms)"], timing),
    ))))


if __name__ == "__main__":
    main()
