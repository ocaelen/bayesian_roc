import contextlib
import csv
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_tab_experiment_empirical_evaluation_on_real_world_datasets as table
from roc_utils.experiment_provenance import file_sha256, metadata_path_for_csv
from roc_utils.experiment_results import raw_level_measurements_from_replication
from roc_utils.utilities_experiments import ReplicationMeasurements


def _analysis(condition):
    """Four transparent replications; no simulation is needed for aggregation tests."""
    low_support = condition.dataset_name in ("Alzheimer", "Fertility", "Haberman", "Sonar")
    increment = condition.n_bins / 1000.0
    deviations = (0.10, 0.20, 0.30, 0.40) if low_support else (0.05, 0.10, 0.15, 0.20)
    radii = {0.80: (0.25 if low_support else 0.155) + increment,
             0.95: (0.35 if low_support else 0.205) + increment}
    intervals = ({0.80: (0.25 - increment, 0.75 + increment),
                  0.95: (0.15 - increment, 0.85 + increment)} if low_support else
                 {0.80: (0.19 - increment, 0.61 + increment),
                  0.95: (0.19 - increment, 0.81 + increment)})
    records = [raw_level_measurements_from_replication(ReplicationMeasurements(
        external_max_roc_deviation=deviation, external_grid_auc=auc,
        roc_radii=radii, auc_intervals=intervals,
    ), table.NOMINAL_LEVELS) for deviation, auc in zip(deviations, (0.20, 0.40, 0.60, 0.80))]
    return table.ConditionAnalysis(
        condition, "DiP-ROC",
        {level: tuple(r[index] for r in records) for index, level in enumerate(table.NOMINAL_LEVELS)},
        (4.0, 6.0, 8.0, 10.0),
        tuple(f"data-{condition.dataset_index}-{b}" for b in range(4)), 40,
    )


def _analyses():
    return tuple(_analysis(condition) for condition in table.REAL_DATA_CONDITIONS)


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


class RealDataTableTests(unittest.TestCase):
    def test_design_pagination_and_independent_descriptive_averages(self):
        self.assertEqual(table.N_BINS_VALUES, (5, 10, 20))
        self.assertEqual(table.NOMINAL_LEVELS, (0.80, 0.95))
        self.assertEqual(table.N_REPLICATIONS, 300)
        self.assertEqual(table.N_SAMPLES, 4000)
        self.assertEqual(table.METHODS, ("DiP-ROC",))
        self.assertEqual(table.REAL_PROTOCOL_VERSION, 3)
        self.assertEqual(len(table.REAL_DATA_CONDITIONS), 36)
        self.assertEqual(table.EXCLUDED_DATASET_NAMES, frozenset(("Alzheimer", "Fertility", "Sonar")))
        self.assertEqual(len(table.TABLE_DATA_CONDITIONS), 27)
        self.assertEqual(table.CORE_N_BINS, 10)
        self.assertEqual(table.DEFAULT_INPUT_CSV.name, "raw_replications_v3.csv")
        self.assertEqual(table.DEFAULT_OUTPUT_DIR.name, "latex_tables_v3")
        fragments = table.build_latex_tables(_analyses())
        expected_names = ("appendix_full", "core_summary", "core_coverage", "core_sharpness",
                          "grid_summary", "dataset_coverage", "computation_time", "sources")
        expected_labels = (
            "tab:empirical-evaluation-on-real-world-datasets-appendix-full",
            "tab:real-core-summary", "tab:real-core-coverage", "tab:real-core-sharpness",
            "tab:real-grid-summary", "tab:empirical-evaluation-on-real-world-datasets-dataset-coverage",
            "tab:empirical-evaluation-on-real-world-datasets-computation-time", "tab:real-data-sources",
        )
        self.assertEqual(tuple(fragments), tuple(
            f"tab_empirical_evaluation_on_real_world_datasets_{name}.tex" for name in expected_names))
        self.assertEqual(table.OUTPUT_LABELS, expected_labels)
        appendix, summary, coverage, sharpness, grid, datasets, timing, sources = fragments.values()
        self.assertEqual(appendix.count(r"\begin{table}"), 3)
        self.assertEqual(appendix.count(r"\ContinuedFloat"), 2)
        self.assertEqual(appendix.count(" & 0.80 & "), 27)
        self.assertEqual(appendix.count(" & 0.95 & "), 27)
        self.assertEqual(appendix.count(r"\multicolumn{6}"), 9)
        self.assertIn("5 & 0.80 & 50.00 (25.00) & 50.00 (25.00)", appendix)
        self.assertIn("0.25500 (0.000000)", appendix)
        self.assertIn(r"ROC coverage (\%) & 72.22 & 50.00--75.00 & 97.22 & 75.00--100.00", summary)
        self.assertIn("Mean ROC radius & 0.176 & 0.165--0.260 & 0.231 & 0.215--0.360", summary)
        self.assertIn("equal weight to the 9 included dataset-specific estimates", summary)
        self.assertIn("Haberman & 16 & 45 & 50.00 & 75.00 & 50.00 & 100.00", coverage)
        self.assertIn("Haberman & 0.260 & 0.360 & 0.520 & 0.720", sharpness)
        self.assertIn("Adult & 0.165 & 0.215 & 0.440 & 0.640", sharpness)
        for k, radius80, radius95, width80, width95 in (
            (5, "0.171", "0.226", "0.439", "0.639"),
            (10, "0.176", "0.231", "0.449", "0.649"),
            (20, "0.186", "0.241", "0.469", "0.669"),
        ):
            self.assertIn(f"{k} & 80 & 72.22 & 72.22 & {radius80} & {width80}", grid)
            self.assertIn(f"{k} & 95 & 97.22 & 100.00 & {radius95} & {width95}", grid)
            self.assertIn(rf"$\mathcal K={k}$: 9 datasets", grid)
        for fragment, spec, rows in ((summary, "lrrrr", 4), (coverage, "lrrrrrr", 9),
                                     (sharpness, "lrrrr", 9), (grid, "rrrrrr", 6)):
            self.assertIn(r"\begin{tabular}{@{}" + spec + "@{}}", fragment)
            self.assertIn(r"\multicolumn{2}{c}", fragment)
            # Grouped heading, column heading, then the manuscript's exact data row count.
            self.assertEqual(sum(line.endswith(r" \\") for line in fragment.splitlines()), rows + 2)
        self.assertEqual(datasets.count(r"\begin{table}"), 1)
        self.assertEqual(timing.count(r"\begin{table}"), 2)
        # Mean=7; sample variance=20/3 and SE=sqrt(20/3)/2, counted once per split.
        self.assertEqual(timing.count("7.00000 (1.29099)"), 27)
        self.assertNotIn("0.80 &", timing)
        self.assertIn("precomputed uniform prior", timing)
        self.assertIn("test and external samples", sources)
        for (filename, latex), expected_label in zip(fragments.items(), expected_labels, strict=True):
            for excluded in ("Alzheimer", "Fertility", "Sonar"):
                self.assertNotIn(excluded, latex)
            self.assertIn("Reduced validation run", latex)
            self.assertNotIn("BootP", latex)
            self.assertNotIn("paired", latex.lower())
            self.assertNotIn("Low support", latex)
            self.assertNotIn("Higher support", latex)
            self.assertEqual(latex.count(r"\label{"), 1)
            self.assertIn(r"\label{" + expected_label + "}", latex)
            self.assertTrue(latex.startswith(table._VERSION_MARKER))

    def test_source_metadata_has_independent_expected_encodings_and_counts(self):
        fragments = table.build_latex_tables(_analyses())
        sources = fragments[table.OUTPUT_FILENAMES[-1]]
        coverage = fragments[table.OUTPUT_FILENAMES[2]]
        expected = (
            ("Abalone", "183 (1)", r"$\ge 10$", 4177, 416, 419),
            ("Adult", "1590 (2)", r"\texttt{>50K}", 45222, 2241, 6802),
            ("Banknote Authentication", "1462 (1)", r"\texttt{2}", 1372, 122, 152),
            ("Diabetes", "37 (1)", r"\texttt{tested\_positive}", 768, 53, 100),
            ("Haberman", "43 (1)", r"\texttt{2}", 306, 16, 45),
            ("Mammographic", "45557 (2)", r"\texttt{1}", 830, 80, 85),
            ("Titanic", "40704 (2)", r"\texttt{1}", 2201, 142, 298),
            ("Transfusion", "1464 (1)", r"\texttt{2}", 748, 35, 114),
            ("Contraceptive Method Choice", "23 (1)", r"\texttt{2} or \texttt{3}", 1473, 168, 125),
        )
        for row in expected:
            self.assertIn(" & ".join(map(str, row)) + r" \\", sources)
            self.assertIn(f"{row[0]} & {row[-2]} & {row[-1]} & ", coverage)
        self.assertIn(r"\label{tab:real-data-sources}", sources)
        self.assertIn("before predictor exclusions", sources)
        metadata = table.real_dataset_metadata("Adult")
        with patch.object(table, "real_dataset_metadata", return_value=replace(metadata, positive_values=("a_b&c%",))):
            rule = table._positive_class_rule(table.real_dataset_metadata("Adult"))
        self.assertEqual(rule, r"\texttt{a\_b\&c\%}")

    def test_excluded_measurements_and_times_cannot_change_any_fragment(self):
        analyses = _analyses()
        records = raw_level_measurements_from_replication(ReplicationMeasurements(
            external_max_roc_deviation=0.01, external_grid_auc=0.3,
            roc_radii={0.8: 0.5, 0.95: 1.0},
            auc_intervals={0.8: (0.0, 0.8), 0.95: (0.0, 1.0)},
        ), table.NOMINAL_LEVELS)
        changed = tuple(
            replace(a, records_by_level={level: (records[index],) * 4
                                         for index, level in enumerate(table.NOMINAL_LEVELS)},
                    computation_times_ms=(1000.0, 2000.0, 3000.0, 4000.0))
            if a.condition.dataset_name in ("Alzheimer", "Fertility", "Sonar") else a
            for a in analyses
        )
        self.assertEqual(table.build_latex_tables(changed), table.build_latex_tables(analyses))
        self.assertEqual(tuple(a.condition for a in table._validate_analyses(analyses)),
                         table.TABLE_DATA_CONDITIONS)

    def test_excluded_inputs_are_validated_before_filtering(self):
        analyses = _analyses()
        excluded = tuple(a for a in analyses if a.condition.dataset_name == "Fertility")
        with self.assertRaisesRegex(AssertionError, "No retained datasets remain after exclusions"):
            table.build_latex_tables(excluded)
        for invalid in (
            (analyses[0], excluded[0], excluded[0]),
            (analyses[0], replace(excluded[0], n_samples=41)),
            (analyses[0], excluded[0], replace(excluded[1], data_stream_ids=("a", "b", "c", "d"))),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                table.build_latex_tables(invalid)

    def test_full_budget_retained_conditions_are_not_labelled_reduced(self):
        analyses = tuple(replace(
            a, n_samples=4000,
            records_by_level={level: records * 75 for level, records in a.records_by_level.items()},
            computation_times_ms=a.computation_times_ms * 75,
            data_stream_ids=tuple(f"data-{a.condition.dataset_index}-{b}" for b in range(300)),
        ) for a in _analyses())
        retained = tuple(a for a in analyses if a.condition.dataset_name not in
                         ("Alzheimer", "Fertility", "Sonar"))
        fragments = table.build_latex_tables(analyses)
        self.assertEqual(fragments, table.build_latex_tables(retained))
        self.assertTrue(all("Reduced validation run" not in value for value in fragments.values()))
        self.assertTrue(all("Reduced validation run" in value
                            for value in table.build_latex_tables(retained[:-1]).values()))

    def test_sample_standard_error_and_coverage_standard_error_differ_correctly(self):
        analysis = _analysis(table.REAL_DATA_CONDITIONS[0])
        samples = []
        for b, deviation in enumerate((0.1, 0.2, 0.3, 0.4)):
            samples.append(raw_level_measurements_from_replication(ReplicationMeasurements(
                external_max_roc_deviation=deviation, external_grid_auc=0.5,
                roc_radii={0.8: 0.15 + b * 0.05, 0.95: 0.4 + b * 0.05},
                auc_intervals={0.8: (0.1, 0.6 + b * 0.05), 0.95: (0.0, 0.8 + b * 0.05)},
            ), table.NOMINAL_LEVELS))
        analysis = replace(analysis, records_by_level={level: tuple(s[i] for s in samples)
                                                       for i, level in enumerate(table.NOMINAL_LEVELS)})
        summary = analysis.summaries_by_level[0.8]
        radii = (0.15, 0.20, 0.25, 0.30)
        self.assertAlmostEqual(summary.roc_radius_mean, statistics.mean(radii))
        self.assertAlmostEqual(summary.roc_radius_standard_error, statistics.stdev(radii) / 2)
        self.assertAlmostEqual(summary.roc_coverage, 0.5)
        self.assertAlmostEqual(summary.roc_coverage_standard_error, math.sqrt(0.5 * 0.5 / 4))
        self.assertAlmostEqual(summary.auc_interval_width_standard_error, statistics.stdev((0.5, 0.55, 0.6, 0.65)) / 2)
        appendix = table.build_latex_tables((analysis,))[table.OUTPUT_FILENAMES[0]]
        self.assertIn("5 & 0.80 & 50.00 (25.00) & 100.00 (0.00) & 0.22500 (0.032275) & 0.57500 (0.032275)", appendix)
        self.assertIn("5 & 0.95 & 100.00 (0.00) & 100.00 (0.00) & 0.47500 (0.032275) & 0.87500 (0.032275)", appendix)

    def test_core_means_ranges_and_grid_means_use_equal_dataset_weight_and_separate_levels(self):
        analyses = []
        for condition in table.REAL_DATA_CONDITIONS:
            if condition.dataset_name not in ("Adult", "Haberman"):
                continue
            if condition.n_bins == 10:
                if condition.dataset_name == "Adult":
                    radii = {0.8: 0.35, 0.95: 0.5}
                    intervals = {0.8: (0.0, 0.8), 0.95: (0.0, 1.0)}
                else:
                    radii = {0.8: 0.2, 0.95: 0.3}
                    intervals = {0.8: (0.2, 0.8), 0.95: (0.1, 0.9)}
            else:
                radii = ({0.8: 0.8, 0.95: 0.9} if condition.n_bins == 5 else {0.8: 0.9, 0.95: 1.0})
                intervals = {0.8: (0.45, 0.55), 0.95: (0.4, 0.6)}
            records = [raw_level_measurements_from_replication(ReplicationMeasurements(
                external_max_roc_deviation=deviation, external_grid_auc=auc,
                roc_radii=radii, auc_intervals=intervals,
            ), (0.8, 0.95)) for deviation, auc in zip((0.1, 0.2, 0.3, 0.4), (0.1, 0.3, 0.7, 0.9))]
            analyses.append(replace(_analysis(condition), records_by_level={
                level: tuple(row[index] for row in records) for index, level in enumerate((0.8, 0.95))
            }))
        fragments = table.build_latex_tables(analyses)
        summary, coverage, sharpness, grid = [fragments[name] for name in table.OUTPUT_FILENAMES[1:5]]
        # Adult has 9,043 observations in each held-out sample; Haberman has 61.
        # The mean nevertheless gives these two dataset estimates equal weight.
        for row in (
            r"ROC coverage (\%) & 62.50 & 50.00--75.00 & 87.50 & 75.00--100.00",
            r"AUC coverage (\%) & 62.50 & 50.00--75.00 & 100.00 & 100.00--100.00",
            "Mean ROC radius & 0.275 & 0.200--0.350 & 0.400 & 0.300--0.500",
            "Mean AUC width & 0.700 & 0.600--0.800 & 0.900 & 0.800--1.000",
        ):
            self.assertIn(row, summary)
        self.assertIn("not confidence intervals", summary)
        self.assertIn("equal weight to the 2 included dataset-specific estimates", summary)
        self.assertIn("Adult & 2241 & 6802 & 75.00 & 100.00 & 75.00 & 100.00", coverage)
        self.assertIn("Haberman & 16 & 45 & 50.00 & 75.00 & 50.00 & 100.00", coverage)
        self.assertIn("Adult & 0.350 & 0.500 & 0.800 & 1.000", sharpness)
        self.assertIn("Haberman & 0.200 & 0.300 & 0.600 & 0.800", sharpness)
        for row in (
            "5 & 80 & 100.00 & 0.00 & 0.800 & 0.100", "5 & 95 & 100.00 & 0.00 & 0.900 & 0.200",
            "10 & 80 & 62.50 & 62.50 & 0.275 & 0.700", "10 & 95 & 87.50 & 100.00 & 0.400 & 0.900",
            "20 & 80 & 100.00 & 0.00 & 0.900 & 0.100", "20 & 95 & 100.00 & 0.00 & 1.000 & 0.200",
        ):
            self.assertIn(row, grid)

    def test_assertions_reject_invalid_analyses_and_preserve_grid_specific_auc(self):
        analyses = _analyses()
        for changes in (
            {"method": "BootP-ROC"}, {"computation_times_ms": (1.0, 2.0)},
            {"computation_times_ms": (1.0, 2.0, float("nan"), 4.0)},
            {"data_stream_ids": ("same",) * 4},
            {"records_by_level": dict(reversed([(0.95, analyses[0].records_by_level[0.8]),
                                                (0.8, analyses[0].records_by_level[0.95])]))},
        ):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                replace(analyses[0], **changes)
        with self.assertRaises(AssertionError):
            table.build_latex_tables((analyses[0], analyses[0]))
        with self.assertRaises(AssertionError):
            table.build_latex_tables((analyses[0], replace(analyses[1], data_stream_ids=("a", "b", "c", "d"))))
        changed_levels = {}
        for level, records in analyses[1].records_by_level.items():
            # AUC stays inside the intervals so the existing coverage flags remain correct.
            changed_levels[level] = tuple(replace(r, external_grid_auc=r.external_grid_auc + 0.001) for r in records)
        changed = replace(analyses[1], records_by_level=changed_levels)
        self.assertEqual(table.build_latex_tables((analyses[0], changed)),
                         table.build_latex_tables((changed, analyses[0])))

    def test_subset_tables_are_explicitly_reduced_and_keep_unknown_grids_blank(self):
        haberman = next(a for a in _analyses() if a.condition.dataset_name == "Haberman")
        fragments = table.build_latex_tables((haberman,))
        self.assertTrue(all("Reduced validation run" in value for value in fragments.values()))
        self.assertIn("--- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[5]])
        self.assertEqual(fragments[table.OUTPUT_FILENAMES[0]].count(r"\begin{table}"), 1)
        self.assertNotIn("Adult &", fragments[table.OUTPUT_FILENAMES[-1]])
        for name in table.OUTPUT_FILENAMES[1:4]:
            self.assertIn(r"No $\mathcal K=10$ results were supplied", fragments[name])
            self.assertIn("--- & --- & --- & ---", fragments[name])
        self.assertIn("Haberman & 16 & 45 & --- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[2]])
        self.assertIn("10 & 80 & --- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[4]])
        self.assertIn("20 & 95 & --- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[4]])

    def test_partial_core_omits_missing_dataset_from_summary_without_substitution(self):
        available = tuple(a for a in _analyses() if
                          (a.condition.dataset_name, a.condition.n_bins) in (("Adult", 10), ("Haberman", 20)))
        fragments = table.build_latex_tables(available)
        summary = fragments[table.OUTPUT_FILENAMES[1]]
        self.assertIn("equal weight to the 1 included dataset-specific estimates", summary)
        self.assertIn("these datasets are excluded from the summary", summary)
        self.assertIn("Mean ROC radius & 0.165 & 0.165--0.165 & 0.215 & 0.215--0.215", summary)
        self.assertIn("Haberman & 16 & 45 & --- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[2]])
        self.assertIn("Haberman & --- & --- & --- & ---", fragments[table.OUTPUT_FILENAMES[3]])
        self.assertIn("20 & 80 & 50.00 & 50.00 & 0.270 & 0.540", fragments[table.OUTPUT_FILENAMES[4]])

    def test_publication_is_all_or_nothing_and_old_directories_are_protected(self):
        analyses = _analyses()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dummy.csv"
            source.write_text("an immutable dummy for mocked reader\n")
            destination = root / "published"
            with patch.object(table, "read_condition_analyses", return_value=analyses):
                paths = table.generate_latex_tables(source, destination)
                self.assertEqual(tuple(p.name for p in paths), table.OUTPUT_FILENAMES)
                before = _snapshot(destination)
                real_write = table._write_table
                calls = 0
                def fail_mid_staging(path, latex):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise OSError("injected stage failure")
                    return real_write(path, latex)
                with patch.object(table, "_write_table", side_effect=fail_mid_staging):
                    with self.assertRaisesRegex(OSError, "stage failure"):
                        table.generate_latex_tables(source, destination)
                self.assertEqual(_snapshot(destination), before)
                real_replace = table.os.replace
                failed = False
                def fail_directory_publication(src, dst):
                    nonlocal failed
                    if Path(src).is_dir() and Path(dst).resolve() == destination.resolve() and not failed:
                        failed = True
                        raise OSError("injected final publication failure")
                    return real_replace(src, dst)
                with patch.object(table.os, "replace", side_effect=fail_directory_publication):
                    with self.assertRaisesRegex(OSError, "final publication"):
                        table.generate_latex_tables(source, destination)
                self.assertEqual(_snapshot(destination), before)
                self.assertFalse(list(root.glob(".published.*")))
                self.assertEqual(table.generate_latex_tables(source, destination), paths)
            for old_name in ("latex_tables", "latex_tables_v1", "latex_tables_v2",
                             "archived_experiments", "back_up_legacy_pipeline"):
                with self.assertRaisesRegex(ValueError, "protected"):
                    table.generate_latex_tables(source, root / old_name)
            renamed = root / "renamed_old"
            renamed.mkdir()
            (renamed / table.OUTPUT_FILENAMES[0]).write_text("% Generated from validated v2 comparison replications.\n")
            with self.assertRaisesRegex(ValueError, "protected"):
                table.generate_latex_tables(source, renamed)
            (renamed / table.OUTPUT_FILENAMES[0]).unlink()
            (renamed / "unexpected.txt").write_text("retain me")
            with self.assertRaisesRegex(ValueError, "protected"):
                table.generate_latex_tables(source, renamed)
            (renamed / "unexpected.txt").unlink()
            obsolete = renamed / "tab_empirical_evaluation_on_real_world_datasets_fine_grid_support.tex"
            obsolete.write_text(table._VERSION_MARKER + "old seven-family output must be retained\n")
            with self.assertRaisesRegex(ValueError, "Unrecognized output content is protected"):
                table.generate_latex_tables(source, renamed)
            self.assertTrue(obsolete.is_file())
            linked = root / "linked"
            linked.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "protected"):
                table.generate_latex_tables(source, linked)

    def test_lock_and_source_change_leave_previous_fragments_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dummy.csv"
            source.write_text("dummy original\n")
            destination = root / "published"
            with patch.object(table, "read_condition_analyses", return_value=_analyses()):
                table.generate_latex_tables(source, destination)
                before = _snapshot(destination)
                lock = root / ".published.lock"
                lock.write_text("another writer")
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    table.generate_latex_tables(source, destination)
                self.assertEqual(lock.read_text(), "another writer")
                lock.unlink()
                real_build = table.build_latex_tables
                def mutate_source(analyses):
                    result = real_build(analyses)
                    source.write_text("changed after reading\n")
                    return result
                with patch.object(table, "build_latex_tables", side_effect=mutate_source):
                    with self.assertRaisesRegex(RuntimeError, "CSV changed"):
                        table.generate_latex_tables(source, destination)
                self.assertEqual(_snapshot(destination), before)
                self.assertFalse(list(root.glob(".published.*")))

    def test_atomic_fragment_failure_and_plain_cli_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / table.OUTPUT_FILENAMES[0]
            path.write_text("existing validated fragment\n")
            with patch.object(table.os, "replace", side_effect=OSError("injected replacement failure")):
                with self.assertRaises(OSError):
                    table._write_table(path, "new fragment\n")
            self.assertEqual(path.read_text(), "existing validated fragment\n")
            self.assertFalse(list(path.parent.glob(".*.tmp")))
        expected = tuple(table.DEFAULT_OUTPUT_DIR / name for name in table.OUTPUT_FILENAMES)
        with (patch.object(table, "generate_latex_tables", return_value=expected) as generate,
              patch.object(Path, "is_file", return_value=True)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(table.main([]), expected)
            generate.assert_called_once_with(table.DEFAULT_INPUT_CSV, table.DEFAULT_OUTPUT_DIR)
            self.assertIn(str(table.DEFAULT_OUTPUT_DIR.resolve()), output.getvalue())
            self.assertIn("8 table fragments", output.getvalue())
            self.assertIn("main K=10", output.getvalue())
            self.assertIn(str(table.DEFAULT_INPUT_CSV.resolve()), output.getvalue())
            for path, label in zip(expected, table.OUTPUT_LABELS, strict=True):
                self.assertIn(f"{path.name}  [LaTeX label: {label}]", output.getvalue())

    def test_cli_missing_csv_explains_required_run_and_leaves_output_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing.csv"
            destination = Path(directory) / "output"
            with (patch.object(table, "generate_latex_tables") as generate,
                  contextlib.redirect_stderr(io.StringIO()) as error):
                with self.assertRaises(SystemExit) as raised:
                    table.main(["--input", str(source), "--output-dir", str(destination)])
            self.assertEqual(raised.exception.code, 2)
            generate.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertIn(str(source.resolve()), error.getvalue())
            self.assertIn("run_experiment_empirical_evaluation_on_real_world_datasets.py", error.getvalue())
            self.assertIn("v3 CSV with its metadata companion", error.getvalue())

    def test_cli_accepts_explicit_completed_input_and_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "complete.csv"
            source.write_text("mocked completed run\n")
            destination = Path(directory) / "tables"
            expected = tuple(destination / name for name in table.OUTPUT_FILENAMES)
            with (patch.object(table, "generate_latex_tables", return_value=expected) as generate,
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(table.main(["--input", str(source), "--output-dir", str(destination)]), expected)
            generate.assert_called_once_with(source, destination)
            self.assertIn(str(destination), output.getvalue())

    def test_optimized_mode_is_rejected(self):
        completed = subprocess.run([sys.executable, "-O", str(table.__file__), "--help"],
                                   capture_output=True, text=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("do not use python -O", completed.stderr)


@unittest.skipUnless(all((PY_DIR / "data" / name).is_file() for name in ("fertility.arff", "haberman.arff")),
                     "Local Fertility or Haberman data unavailable; no network access in tests.")
class RealDataTableIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source.csv"
        cls.conditions = tuple(c for c in table.REAL_DATA_CONDITIONS if c.dataset_name in ("Fertility", "Haberman"))
        cls.options = dict(conditions=cls.conditions, n_replications=2, n_samples=40, master_seed=41)
        table.experiment.run_experiment(cls.source, **cls.options, data_dir=PY_DIR / "data", show_progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _copy_source(self, root):
        path = root / "copy.csv"
        shutil.copyfile(self.source, path)
        shutil.copyfile(metadata_path_for_csv(self.source), metadata_path_for_csv(path))
        return path

    def _write_changed_rows(self, path, rows):
        metadata = json.loads(metadata_path_for_csv(self.source).read_text())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=table.RAW_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        metadata["raw_csv_sha256"] = file_sha256(path)
        metadata_path_for_csv(path).write_text(json.dumps(metadata))

    def test_complete_reduced_csv_reads_timings_once_and_generates_eight_fragments(self):
        with self.assertRaises((AssertionError, ValueError)):
            table.read_condition_analyses(self.source)
        analyses = table.read_condition_analyses(self.source, **self.options)
        self.assertEqual(len(analyses), 3)
        self.assertEqual({a.condition.dataset_name for a in analyses}, {"Haberman"})
        with self.source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["dataset_name"] for row in rows}, {"Fertility", "Haberman"})
        for analysis in analyses:
            times = [int(row["computation_time_ns"]) / 1_000_000 for row in rows
                     if row["condition_index"] == str(analysis.condition.index) and float(row["level"]) == 0.80]
            self.assertEqual(analysis.computation_times_ms, tuple(times))
            self.assertEqual(len(times), 2)
        with tempfile.TemporaryDirectory() as directory:
            paths = table.generate_latex_tables(self.source, Path(directory) / "output", **self.options)
            self.assertEqual(tuple(path.name for path in paths), table.OUTPUT_FILENAMES)
            self.assertTrue(all("BootP" not in path.read_text() for path in paths))
            self.assertTrue(all("Reduced validation run" in path.read_text() for path in paths))
            self.assertTrue(all("Fertility" not in path.read_text() for path in paths))
            self.assertIn("Haberman", paths[-1].read_text())

    def test_reader_still_rejects_corrupted_excluded_dataset_rows(self):
        with self.source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        next(row for row in rows if row["dataset_name"] == "Fertility")["prior_mass"] = "5"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupted_excluded.csv"
            self._write_changed_rows(path, rows)
            with self.assertRaises((AssertionError, ValueError)):
                table.read_condition_analyses(path, **self.options)

    def test_reader_rejects_missing_companion_wrong_metadata_and_mid_read_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._copy_source(Path(directory))
            metadata_file = metadata_path_for_csv(source)
            original = metadata_file.read_text()
            metadata_file.unlink()
            with self.assertRaises(FileNotFoundError):
                table.read_condition_analyses(source, **self.options)
            metadata_file.write_text(original)
            payload = json.loads(original)
            payload["configuration"]["B"] = 999
            metadata_file.write_text(json.dumps(payload))
            with self.assertRaises((AssertionError, ValueError)):
                table.read_condition_analyses(source, **self.options)
            metadata_file.write_text(original)
            with patch.object(table, "file_sha256", return_value="changed"):
                with self.assertRaisesRegex(RuntimeError, "CSV changed"):
                    table.read_condition_analyses(source, **self.options)

    def test_reader_rejects_missing_duplicate_mixed_and_corrupted_records(self):
        with self.source.open(newline="") as handle:
            original = list(csv.DictReader(handle))
        corruptions = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.append(dict(rows[0])),
            "mixed_schema": lambda rows: rows[-1].update(schema_version="real_data_raw_v2"),
            "wrong_method": lambda rows: rows[-1].update(method="BootP-ROC"),
            "wrong_count": lambda rows: rows[-1].update(n_positive="3"),
            "wrong_time": lambda rows: rows[1].update(computation_time_ns=str(int(rows[0]["computation_time_ns"]) + 1)),
            "wrong_prior": lambda rows: rows[-1].update(prior_mass="5"),
            "wrong_split": lambda rows: rows[-1].update(data_stream_id="unshared"),
            "wrong_level": lambda rows: rows[-1].update(level="0.9"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in corruptions.items():
                with self.subTest(name=name):
                    rows = [dict(row) for row in original]
                    mutate(rows)
                    path = Path(directory) / f"{name}.csv"
                    self._write_changed_rows(path, rows)
                    with self.assertRaises((AssertionError, ValueError)):
                        table.read_condition_analyses(path, **self.options)
            for version in (1, 2):
                rows = [dict(row, schema_version=f"real_data_raw_v{version}") for row in original]
                path = Path(directory) / f"old_v{version}.csv"
                self._write_changed_rows(path, rows)
                with self.assertRaisesRegex(ValueError, "v3"):
                    table.read_condition_analyses(path, **self.options)


if __name__ == "__main__":
    unittest.main()
