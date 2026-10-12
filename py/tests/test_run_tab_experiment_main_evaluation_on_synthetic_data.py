import csv
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_tab_experiment_main_evaluation_on_synthetic_data as table
from roc_utils.experiment_results import raw_level_measurements_from_replication
from roc_utils import experiment_provenance as provenance
from roc_utils.utilities_experiments import ReplicationMeasurements


SETTINGS = ("dip_uniform_m1", "dip_uniform_m5", "dip_uniform_m20",
            "dip_linear_m1", "dip_linear_m5", "dip_linear_m20",
            "dip_histograms_m1", "dip_histograms_m5", "dip_histograms_m20")
DEVIATIONS = dict(zip(SETTINGS, (
    (.10, .30, .20, .40), (.40, .20, .30, .10), (.20, .35, .10, .40),
    (.10, .20, .30, .40), (.30, .10, .40, .20), (.35, .20, .40, .10),
    (.20, .10, .40, .30), (.40, .30, .10, .20), (.10, .40, .20, .35),
)))
RADII = dict(zip(SETTINGS, (.25, .20, .18, .35, .30, .24, .28, .22, .19)))
INTERVALS = dict(zip(SETTINGS, ((.10, .75), (.15, .70), (.20, .65),
                              (.05, .80), (.18, .62), (.16, .70),
                              (.08, .77), (.22, .81), (.23, .71))))
RADIUS_OFFSETS = {
    "dip_uniform_m20": (.00, .02, -.01, .04),
    "dip_linear_m20": (.04, .00, .03, -.02),
    "dip_histograms_m20": (-.02, .05, .00, .01),
}
WIDTH_OFFSETS = {
    "dip_uniform_m20": (.03, -.02, .00, .01),
    "dip_linear_m20": (.00, .04, -.01, .02),
    "dip_histograms_m20": (.02, .00, .03, -.02),
}
TIMES = dict(zip(SETTINGS, ((1., 2., 3., 4.), (2., 3., 5., 7.), (4., 3., 8., 10.),
                          (4., 6., 8., 10.), (3., 7., 6., 9.), (9., 4., 12., 7.),
                          (2., 5., 8., 9.), (3., 8., 11., 14.), (8., 3., 10., 15.))))
EXTERNAL_AUCS = (.20, .40, .60, .80)
EXPECTED_CONTRASTS = (
    ("dip_linear_m1", "dip_uniform_m1"), ("dip_histograms_m1", "dip_uniform_m1"),
    ("dip_histograms_m1", "dip_linear_m1"),
    ("dip_linear_m5", "dip_uniform_m5"), ("dip_histograms_m5", "dip_uniform_m5"),
    ("dip_histograms_m5", "dip_linear_m5"),
    ("dip_linear_m20", "dip_uniform_m20"), ("dip_histograms_m20", "dip_uniform_m20"),
    ("dip_histograms_m20", "dip_linear_m20"),
    ("dip_uniform_m5", "dip_uniform_m1"), ("dip_uniform_m20", "dip_uniform_m1"),
    ("dip_uniform_m20", "dip_uniform_m5"),
    ("dip_linear_m5", "dip_linear_m1"), ("dip_linear_m20", "dip_linear_m1"),
    ("dip_linear_m20", "dip_linear_m5"),
    ("dip_histograms_m5", "dip_histograms_m1"), ("dip_histograms_m20", "dip_histograms_m1"),
    ("dip_histograms_m20", "dip_histograms_m5"),
)


def _analysis(condition, setting, *, large_radius=False, radius_shift=0.0):
    records = []
    for index, (deviation, auc) in enumerate(zip(DEVIATIONS[setting], EXTERNAL_AUCS)):
        offset = RADIUS_OFFSETS.get(setting, (0.,) * 4)[index]
        lower, upper = INTERVALS[setting]
        upper += WIDTH_OFFSETS.get(setting, (0.,) * 4)[index]
        measurements = ReplicationMeasurements(
            external_max_roc_deviation=deviation, external_grid_auc=auc,
            roc_radii={.80: 1.6 if large_radius else RADII[setting] + radius_shift + offset,
                       .95: 1.8 if large_radius else .60 + radius_shift + offset},
            auc_intervals={.80: (lower, upper), .95: (0., 1.)},
        )
        records.append(raw_level_measurements_from_replication(measurements, table.NOMINAL_LEVELS))
    return table.ConditionAnalysis(
        condition, setting,
        {level: tuple(r[index] for r in records) for index, level in enumerate(table.NOMINAL_LEVELS)},
        TIMES[setting], tuple(f"data-{condition.index}-{b}" for b in range(4)), 40,
        (True, False, True, False) if setting.startswith("dip_linear_") else (),
    )


def _write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _metric_values(setting, level, metric):
    """Reference values from the literal synthetic example, not the reader."""
    radii = [(RADII[setting] if level == .80 else .60) + offset
             for offset in RADIUS_OFFSETS.get(setting, (0.,) * 4)]
    lower, upper = INTERVALS[setting] if level == .80 else (0., 1.)
    uppers = [upper + offset for offset in
              (WIDTH_OFFSETS.get(setting, (0.,) * 4) if level == .80 else (0.,) * 4)]
    if metric == "roc_coverage":
        return [int(value <= radius) for value, radius in zip(DEVIATIONS[setting], radii)]
    if metric == "auc_coverage":
        return [int(lower <= value <= upper) for value, upper in zip(EXTERNAL_AUCS, uppers)]
    return radii if metric == "roc_radius" else [upper - lower for upper in uppers]


class SyntheticComparisonTableTests(unittest.TestCase):
    def test_plain_command_reports_all_generated_paths_and_manuscript_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw_replications_v8.csv"
            source.touch()
            output = root / "latex_tables_v8"
            paths = tuple(output / name for name in table.OUTPUT_FILENAMES)
            stdout = io.StringIO()
            with patch.object(table, "DEFAULT_INPUT_CSV", source), \
                    patch.object(table, "DEFAULT_OUTPUT_DIR", output), \
                    patch.object(table, "generate_latex_tables", return_value=paths) as generate, \
                    redirect_stdout(stdout):
                self.assertEqual(table.main([]), paths)
            generate.assert_called_once_with(source, output)
            text = stdout.getvalue()
            self.assertIn(str(source.resolve()), text)
            self.assertIn(str(output.resolve()), text)
            self.assertIn("9 prior settings, 18 paired contrasts, 8 table fragments", text)
            for filename, label in zip(table.OUTPUT_FILENAMES, table.OUTPUT_LABELS):
                self.assertIn(f"{filename}  [LaTeX label: {label}]", text)
            self.assertNotIn("BootP", text)
            self.assertIn("main.tex has not been edited", text)

    def test_missing_csv_command_explains_required_run_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing.csv"
            output = Path(directory) / "tables"
            stderr = io.StringIO()
            with patch.object(table, "generate_latex_tables") as generate, \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                table.main(["--input", str(source), "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            generate.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn(str(source), stderr.getvalue())
            self.assertIn("python run_experiment_main_evaluation_on_synthetic_data.py", stderr.getvalue())
            self.assertIn("completed v8 CSV with its metadata companion", stderr.getvalue())

    def test_failed_command_does_not_report_tables_as_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.csv"
            source.touch()
            stdout = io.StringIO()
            with patch.object(table, "generate_latex_tables", side_effect=OSError("publication failed")), \
                    redirect_stdout(stdout), self.assertRaisesRegex(OSError, "publication failed"):
                table.main(["--input", str(source), "--output-dir", directory])
            self.assertNotIn("Saved", stdout.getvalue())
            self.assertNotIn("LaTeX label:", stdout.getvalue())

    def test_four_absolute_estimates_have_independent_means_and_standard_errors(self):
        condition = table.CLASS_COUNT_CONDITIONS[0]
        for setting in SETTINGS:
            analysis = _analysis(condition, setting)
            for level in table.NOMINAL_LEVELS:
                expected_cells = []
                for metric in ("roc_coverage", "auc_coverage", "roc_radius", "auc_width"):
                    values = _metric_values(setting, level, metric)
                    mean = sum(values) / 4
                    if "coverage" in metric:
                        se = math.sqrt(mean * (1 - mean) / 4)
                        expected_cells.append(f"{100 * mean:.2f} ({100 * se:.2f})")
                    else:
                        se = math.sqrt(sum((value - mean) ** 2 for value in values) / 3) / 2
                        expected_cells.append(f"{mean:.5f} ({se:.5f})")
                with self.subTest(setting=setting, level=level):
                    self.assertEqual(table._absolute_cells(analysis, level), expected_cells)

    def test_eighteen_contrasts_have_independent_sign_and_standard_error_references(self):
        self.assertEqual(table.SETTINGS, SETTINGS)
        self.assertEqual(table.CONTRASTS, EXPECTED_CONTRASTS)
        condition = table.CLASS_COUNT_CONDITIONS[0]
        effects = table.build_paired_method_effects(tuple(_analysis(condition, s) for s in SETTINGS))
        self.assertEqual(len(effects), 36)
        self.assertEqual([(e.left_setting, e.right_setting) for e in effects], list(EXPECTED_CONTRASTS) * 2)
        for effect in effects:
            self.assertEqual(set(effect.metrics), {"roc_coverage", "auc_coverage", "roc_radius", "auc_width"})
            for metric, estimate in effect.metrics.items():
                left = _metric_values(effect.left_setting, effect.level, metric)
                right = _metric_values(effect.right_setting, effect.level, metric)
                differences = [x - y for x, y in zip(left, right)]
                mean = sum(differences) / 4
                # Four observations: sample variance has denominator 3 and SE denominator sqrt(4).
                se = math.sqrt(sum((value - mean) ** 2 for value in differences) / 3) / 2
                with self.subTest(contrast=(effect.left_setting, effect.right_setting), level=effect.level, metric=metric):
                    self.assertAlmostEqual(estimate.mean, mean)
                    self.assertAlmostEqual(estimate.standard_error, se)
                    self.assertEqual(estimate.n_replications, 4)
        linear_minus_uniform = next(e.metrics["roc_coverage"] for e in effects
                                    if (e.left_setting, e.right_setting, e.level) ==
                                    ("dip_linear_m1", "dip_uniform_m1", .80))
        # The four paired indicators are exactly (0, 1, 0, 0).
        self.assertEqual(linear_minus_uniform.mean, .25)
        self.assertEqual(linear_minus_uniform.standard_error, .25)
        varying = next(e for e in effects if (e.left_setting, e.right_setting, e.level) ==
                       ("dip_uniform_m20", "dip_uniform_m1", .80))
        self.assertAlmostEqual(varying.metrics["roc_radius"].mean, -.0575)
        self.assertGreater(varying.metrics["roc_radius"].standard_error, 0)
        self.assertGreater(varying.metrics["auc_width"].standard_error, 0)

    def test_pairing_radius_and_shared_adjustment_invariants(self):
        c = table.CLASS_COUNT_CONDITIONS[0]
        analyses = tuple(_analysis(c, s) for s in SETTINGS)
        for setting in SETTINGS:
            with self.subTest(setting=setting), self.assertRaises(AssertionError):
                _analysis(c, setting, large_radius=True)
        with self.assertRaises(AssertionError):
            replace(analyses[0], setting="bootp")
        malformed = (analyses[:-1], analyses + analyses[-1:],
                     analyses[:-1] + (replace(analyses[-1], data_stream_ids=("a", "b", "c", "d")),),
                     analyses[:3] + (replace(analyses[3], historical_auc_adjusted=(False,) * 4),) + analyses[4:],
                     tuple(replace(a, historical_auc_adjusted=(False,) * 4)
                           if a.setting == "dip_linear_m20" else a for a in analyses))
        for case in malformed:
            with self.subTest(settings=[a.setting for a in case]):
                with self.assertRaises(AssertionError):
                    table.build_paired_method_effects(case)
        for setting, flags in (("dip_linear_m1", (True, False)), ("dip_linear_m5", (1, 0, 1, 0)),
                               ("dip_linear_m20", (True, False)),
                               ("dip_uniform_m1", (False,) * 4), ("dip_histograms_m1", (False,) * 4)):
            with self.subTest(setting=setting):
                with self.assertRaises(AssertionError):
                    replace(_analysis(c, setting), historical_auc_adjusted=flags)

    def test_analysis_rejects_records_under_the_wrong_nominal_level(self):
        analysis = _analysis(table.CLASS_COUNT_CONDITIONS[0], "dip_linear_m1")
        swapped = {.80: analysis.records_by_level[.95], .95: analysis.records_by_level[.80]}
        with self.assertRaisesRegex(AssertionError, "nominal-level key"):
            replace(analysis, records_by_level=swapped)

    def test_reduced_panels_include_all_nine_settings(self):
        condition = table.CLASS_COUNT_CONDITIONS[0]
        analyses = tuple(_analysis(condition, setting) for setting in SETTINGS)
        fragments = table.build_latex_tables(analyses)
        self.assertEqual(len(fragments), 8)
        for label in table.DISPLAY_LABELS.values():
            self.assertEqual(fragments[table.OUTPUT_FILENAMES[0]].count(f" & {label} & "), 2)
        self.assertNotIn("BootP", "".join(fragments.values()))
        self.assertIn("No severe-imbalance configuration", fragments[table.OUTPUT_FILENAMES[3]])
        self.assertEqual(fragments[table.OUTPUT_FILENAMES[3]].count(r"\label{"), 1)

    def test_eight_table_families_panels_and_single_counting(self):
        analyses = tuple(_analysis(c, s) for c in table.CLASS_COUNT_CONDITIONS for s in SETTINGS)
        fragments = table.build_latex_tables(analyses)
        self.assertEqual(len(fragments), 8)
        appendix = fragments[table.OUTPUT_FILENAMES[0]]
        self.assertEqual(appendix.count(r"\begin{table}"), 12)
        self.assertEqual(appendix.count(r"\ContinuedFloat"), 11)
        for level in ("0.80", "0.95"):
            self.assertEqual(appendix.count(f" & {level} & "), 135)
        paired = fragments[table.OUTPUT_FILENAMES[4]]
        self.assertEqual(paired.count(r"\begin{table}"), 30)
        numeric_anchor = r"\ifdefined\theHtable\renewcommand{\theHtable}{\arabic{table}.\arabic{continuedfloat}}\fi"
        self.assertEqual(paired.count("\\ContinuedFloat\n" + numeric_anchor + "\n\\centering\n\\caption[]"), 29)
        for level in ("0.80", "0.95"):
            self.assertEqual(paired.count(f" & {level} & "), 270)
        timing = fragments[table.OUTPUT_FILENAMES[5]]
        self.assertEqual(timing.count(r"\begin{table}"), 6)
        absolute_rows = [line for line in timing.splitlines() if any(f" & {label} & " in line for label in table.DISPLAY_LABELS.values())]
        paired_rows = [line for line in timing.splitlines() if "$-$" in line and " & " in line]
        self.assertEqual(len(absolute_rows), 135)
        self.assertEqual(paired_rows, [])
        for setting in SETTINGS:
            values = TIMES[setting]
            expected = f"{np.mean(values):.5f} ({np.std(values, ddof=1) / 2:.5f})"
            self.assertEqual(timing.count(f" & {table.DISPLAY_LABELS[setting]} & {expected}"), 15)
        self.assertNotIn("0.80 &", timing)
        self.assertNotIn("Paired computation", timing)
        self.assertIn("nine execution positions", timing)
        self.assertIn("its own prior preparation", timing)
        self.assertIn("histogram smoothing, normalization and scaling", timing)
        adjustments = fragments[table.OUTPUT_FILENAMES[6]]
        self.assertEqual(adjustments.count(" & 2 & 4 & 50.00"), 15)
        self.assertIn("for the linear priors only", adjustments)
        self.assertIn("values exactly at either endpoint are not adjusted", adjustments)
        self.assertEqual(fragments[table.OUTPUT_FILENAMES[3]].count(r"\begin{table}"), 2)
        for index in (2, 7):
            summary = fragments[table.OUTPUT_FILENAMES[index]]
            self.assertEqual(summary.count(r"\begin{table}"), 2)
            for positive, negative in table.CLASS_COUNT_PAIRS:
                self.assertEqual(sum(line.startswith(f"{positive} & {negative} & ")
                                     for line in summary.splitlines()), 9)
            self.assertIn("Test and external samples have the displayed class counts", summary)
        self.assertIn(r"\label{tab:class-counts-coverage-summary}", fragments[table.OUTPUT_FILENAMES[7]])
        for latex in fragments.values():
            self.assertEqual(latex.count(numeric_anchor), latex.count(r"\ContinuedFloat"))
            for panel in latex.split(r"\begin{tabular}")[1:]:
                body = panel.split(r"\end{tabular}")[0]
                self.assertLessEqual(sum(" & " in line for line in body.splitlines()), 31)

    def test_descriptive_tables_keep_equal_configuration_and_scenario_weights(self):
        analyses = tuple(_analysis(c, s, radius_shift=.011 * c.index)
                         for c in table.CLASS_COUNT_CONDITIONS for s in SETTINGS)
        fragments = table.build_latex_tables(analyses)
        calibration = fragments[table.OUTPUT_FILENAMES[1]]
        for scenario in table.SIMULATION_SCENARIOS:
            conditions = [c for c in table.CLASS_COUNT_CONDITIONS if c.score_scenario == scenario]
            for setting in SETTINGS:
                coverage = np.mean([np.mean(np.asarray(DEVIATIONS[setting]) <=
                                            np.asarray(_metric_values(setting, .80, "roc_radius")) + .011 * c.index)
                                    for c in conditions])
                self.assertIn(f"{scenario} & {table.DISPLAY_LABELS[setting]} & {100 * coverage:.2f} & 100.00", calibration)
        sharpness = fragments[table.OUTPUT_FILENAMES[2]]
        coverage_by_counts = fragments[table.OUTPUT_FILENAMES[7]]
        for positive, negative in table.CLASS_COUNT_PAIRS:
            conditions = [c for c in table.CLASS_COUNT_CONDITIONS
                          if (c.n_positive, c.n_negative) == (positive, negative)]
            for setting in SETTINGS:
                prefix = f"{positive} & {negative} & {table.DISPLAY_LABELS[setting]} & "
                size_values = [sum(np.mean(_metric_values(setting, level, metric)) +
                                   (.011 * c.index if metric == "roc_radius" else 0)
                                   for c in conditions) / len(conditions)
                               for metric in ("roc_radius", "auc_width") for level in table.NOMINAL_LEVELS]
                row = next(line for line in sharpness.splitlines() if line.startswith(prefix))
                displayed = [float(cell.removesuffix(r" \\")) for cell in row.split(" & ")[3:]]
                for value, expected in zip(displayed, size_values):
                    self.assertAlmostEqual(value, expected, delta=.000500001)
                coverage_values = []
                for metric in ("roc_coverage", "auc_coverage"):
                    for level in table.NOMINAL_LEVELS:
                        if metric == "roc_coverage":
                            mean = np.mean([np.mean(np.asarray(DEVIATIONS[setting]) <=
                                                   np.asarray(_metric_values(setting, level, "roc_radius")) + .011 * c.index)
                                            for c in conditions])
                        else:
                            mean = np.mean(_metric_values(setting, level, metric))
                        coverage_values.append(100 * mean)
                self.assertIn(prefix + " & ".join(f"{value:.2f}" for value in coverage_values), coverage_by_counts)

    def test_canonical_setting_order_and_information_roles_in_captions(self):
        analyses = tuple(_analysis(c, s) for c in reversed(table.CLASS_COUNT_CONDITIONS) for s in reversed(SETTINGS))
        fragments = table.build_latex_tables(analyses)
        for filename, latex in fragments.items():
            with self.subTest(filename=filename):
                for caption in (line for line in latex.splitlines() if line.startswith(r"\caption")):
                    self.assertIn("Nine DiP-ROC settings", caption)
                    self.assertNotIn("BootP", caption)
                    self.assertIn("uniform, historical-AUC linear and historical-histogram priors", caption)
                    self.assertIn("suffix 1, 5 or 20", caption)
                    self.assertIn("Only DiP-L and DiP-H receive historical information", caption)
                    self.assertIn(r"$\mathcal B=4$, $\mathcal S=40$", caption)
                    self.assertIn("Reduced validation run", caption)
        for filename in (*table.OUTPUT_FILENAMES[:4], table.OUTPUT_FILENAMES[7]):
            labels = [cell for line in fragments[filename].splitlines() for cell in line.split(" & ")
                      if cell in table.DISPLAY_LABELS.values()]
            self.assertEqual(labels, list(table.DISPLAY_LABELS.values()) * (len(labels) // 9))
        self.assertIn("left minus right", fragments[table.OUTPUT_FILENAMES[4]])

    def test_real_full_design_pilot_has_540_rows_and_eight_fragments(self):
        options = dict(conditions=table.CLASS_COUNT_CONDITIONS, n_replications=2, n_samples=40)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "comparison.csv"
            table.experiment.run_experiment(source, master_seed=31, show_progress=False, **options)
            with self.assertRaises(AssertionError):
                table.read_condition_analyses(source)
            analyses = table.read_condition_analyses(source, **options)
            self.assertEqual(len(analyses), 135)
            with source.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 540)
            for analysis in analyses:
                selected = [row for row in rows if row["setting"] == analysis.setting
                            and row["condition_index"] == str(analysis.condition.index) and float(row["level"]) == .80]
                self.assertEqual(analysis.computation_times_ms, tuple(int(row["computation_time_ns"]) / 1_000_000 for row in selected))
                if analysis.setting.startswith("dip_linear_"):
                    self.assertEqual(analysis.historical_auc_adjusted, tuple(bool(int(row["historical_auc_adjusted"])) for row in selected))
            shuffled = root / "shuffled.csv"
            _write_rows(shuffled, list(reversed(rows)))
            provenance.publish_run_metadata(shuffled, shuffled, provenance.read_run_metadata(source))
            self.assertEqual(analyses, table.read_condition_analyses(shuffled, **options))
            paths = table.generate_latex_tables(source, root / "latex_tables_v8", **options)
            self.assertEqual(tuple(path.name for path in paths), table.OUTPUT_FILENAMES)
            self.assertTrue(all(path.read_text().startswith("% Generated from validated v8 comparison replications.\n") for path in paths))

    def test_reader_rejects_corrupted_pairings_and_historical_metadata(self):
        options = dict(conditions=table.CLASS_COUNT_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            table.experiment.run_experiment(source, show_progress=False, **options)
            with source.open(newline="") as handle:
                original = list(csv.DictReader(handle))
            cases = ("missing setting", "duplicate setting", "timing", "adjustment", "historical count",
                     "historical auc levels", "false historical baseline", "external auc", "simulation stream",
                     "incorrect concentration", "histogram adjustment", "m20 timing", "m20 adjustment", "m20 stream",
                     "bootstrap setting", "bootstrap method")
            for case in cases:
                rows = [dict(row) for row in original]
                linear = [row for row in rows if row["setting"] == "dip_linear_m1"]
                baseline = next(row for row in rows if row["setting"] == "dip_uniform_m1")
                if case == "missing setting":
                    rows = [row for row in rows if row["setting"] != "dip_histograms_m20"]
                elif case == "duplicate setting":
                    rows.append(dict(rows[0]))
                elif case == "timing":
                    linear[1]["computation_time_ns"] = str(int(linear[1]["computation_time_ns"]) + 1)
                elif case == "adjustment":
                    linear[0]["historical_auc_adjusted"] = str(1 - int(linear[0]["historical_auc_adjusted"]))
                elif case == "historical count":
                    linear[0]["historical_n_positive"] = "499"
                elif case == "historical auc levels":
                    linear[1]["historical_grid_auc"] = "0.51"
                elif case == "false historical baseline":
                    baseline["historical_grid_auc"] = "0.7"
                elif case == "external auc":
                    baseline["external_grid_auc"] = "0.123456"
                elif case == "incorrect concentration":
                    for row in rows:
                        if row["setting"] == "dip_linear_m20":
                            row["prior_mass"] = "5.0"
                elif case == "histogram adjustment":
                    next(row for row in rows if row["setting"] == "dip_histograms_m1")["historical_auc_adjusted"] = "0"
                elif case == "m20 timing":
                    row = next(row for row in rows if row["setting"] == "dip_uniform_m20")
                    row["computation_time_ns"] = str(int(row["computation_time_ns"]) + 1)
                elif case == "m20 adjustment":
                    row = next(row for row in rows if row["setting"] == "dip_linear_m20")
                    row["historical_auc_adjusted"] = str(1 - int(row["historical_auc_adjusted"]))
                elif case == "m20 stream":
                    next(row for row in rows if row["setting"] == "dip_histograms_m20")["simulation_stream_id"] = linear[0]["simulation_stream_id"]
                elif case == "bootstrap setting":
                    baseline["setting"] = "bootp"
                elif case == "bootstrap method":
                    baseline["method"] = "BootP-ROC"
                else:
                    baseline["simulation_stream_id"] = linear[0]["simulation_stream_id"]
                with self.subTest(case=case):
                    bad = Path(directory) / "bad.csv"
                    _write_rows(bad, rows)
                    with self.assertRaises((AssertionError, ValueError)):
                        table.read_condition_analyses(bad, **options)

    def test_v8_reader_rejects_old_mixed_and_relabelled_csvs(self):
        self.assertEqual(table.MAIN_PROTOCOL_VERSION, 8)
        self.assertEqual(table.RAW_SCHEMA_VERSION, "class_counts_raw_v8")
        self.assertEqual(table.DEFAULT_INPUT_CSV.name, "raw_replications_v8.csv")
        self.assertEqual(table.DEFAULT_OUTPUT_DIR.name, "latex_tables_v8")
        options = dict(conditions=table.CLASS_COUNT_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            table.experiment.run_experiment(source, show_progress=False, **options)
            with source.open(newline="") as handle:
                original = list(csv.DictReader(handle))
            for version in range(1, 8):
                for mixed in (False, True):
                    rows = [dict(row) for row in original]
                    for row in (rows[-1:] if mixed else rows):
                        row["schema_version"] = f"class_counts_raw_v{version}"
                    old = Path(directory) / "old.csv"
                    _write_rows(old, rows)
                    with self.subTest(version=version, mixed=mixed):
                        with self.assertRaisesRegex(ValueError, "old or mixed"):
                            table.read_condition_analyses(old, **options)
            rows = [dict(row) for row in original]
            rows[-1]["positive_beta_a"] = "5.0"
            _write_rows(source, rows)
            with self.assertRaisesRegex(ValueError, "Beta metadata"):
                table.read_condition_analyses(source, **options)
            source.write_text("schema_version,experiment\nclass_counts_raw_v5,class_counts_and_imbalance\n")
            with self.assertRaisesRegex(ValueError, "v8 comparison CSV"):
                table.read_condition_analyses(source)

    def test_old_directories_renamed_fragments_and_failed_writes_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("latex_tables", *(f"latex_tables_v{version}" for version in range(1, 8))):
                for output in (root / name, root / name / "nested"):
                    with self.subTest(output=output), patch.object(table, "read_condition_analyses") as read:
                        with self.assertRaisesRegex(ValueError, "protected"):
                            table.generate_latex_tables(root / "missing.csv", output)
                        read.assert_not_called()
                        self.assertFalse(output.exists())
            custom = root / "custom"
            custom.mkdir()
            previous = custom / table.OUTPUT_FILENAMES[-1]
            for old_text in ("% Old unversioned table\n", *(f"% Generated from validated v{version} comparison replications.\n" for version in range(1, 8))):
                previous.write_text(old_text)
                with patch.object(table, "read_condition_analyses") as read:
                    with self.assertRaisesRegex(ValueError, "protected"):
                        table.generate_latex_tables(root / "missing.csv", custom)
                    read.assert_not_called()
                self.assertEqual(previous.read_text(), old_text)
            original_text = previous.read_text()
            with patch.object(table.os, "replace", side_effect=OSError("publication failed")):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    table._write_table(previous, "replacement")
            self.assertEqual(previous.read_text(), original_text)
            self.assertEqual(list(custom.iterdir()), [previous])

    def test_reader_requires_matching_checksum_and_design_provenance(self):
        options = dict(conditions=table.CLASS_COUNT_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            table.experiment.run_experiment(source, show_progress=False, **options)
            companion = provenance.metadata_path_for_csv(source)
            original = json.loads(companion.read_text())
            companion.unlink()
            with self.assertRaises(FileNotFoundError):
                table.read_condition_analyses(source, **options)
            cases = (
                ("metadata_schema", "unknown"), ("raw_csv_sha256", "0" * 64),
                ("B", 3), ("S", 40), ("main_protocol_version", 6),
                ("raw_schema_version", "class_counts_raw_v6"), ("settings", []),
                ("stream_key_templates", {}), ("conditions", []),
                ("execution_order", "fixed order"), ("timing_scope", "sampling only"),
                ("replication_indexing", "one-based"), ("data_draw_order", []),
                ("historical_draw_order", ["negative", "positive"]),
                ("timing_threads", 2), ("clock", "time.process_time_ns"),
            )
            for name, value in cases:
                payload = json.loads(json.dumps(original))
                if name in ("metadata_schema", "raw_csv_sha256"):
                    target = payload
                elif name in ("timing_threads", "clock"):
                    target = payload["timing_environment"]
                else:
                    target = payload["configuration"]
                target[name] = value
                companion.write_text(json.dumps(payload))
                with self.subTest(field=name), self.assertRaises(ValueError):
                    table.read_condition_analyses(source, **options)
            payload = json.loads(json.dumps(original))
            payload["source_sha256"] = {"py/earlier_source.py": "a" * 64}
            companion.write_text(json.dumps(payload))
            self.assertEqual(len(table.read_condition_analyses(source, **options)), 9)
            # Even a semantically identical rewrite has distinct bytes and needs its own companion.
            source.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
            with self.assertRaises(FileNotFoundError):
                table.read_condition_analyses(source, **options)

    def test_reader_rejects_concurrent_csv_replacement_after_provenance_validation(self):
        options = dict(conditions=table.CLASS_COUNT_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "requested.csv"
            replacement = Path(directory) / "concurrent.csv"
            table.experiment.run_experiment(source, master_seed=71, show_progress=False, **options)
            table.experiment.run_experiment(replacement, master_seed=72, show_progress=False, **options)
            original_checksum = provenance.file_sha256(source)
            validate = table._validate_run_provenance

            def replace_after_validation(*args):
                checksum = validate(*args)
                self.assertEqual(checksum, original_checksum)
                table.os.replace(replacement, source)
                return checksum

            with patch.object(table, "_validate_run_provenance", side_effect=replace_after_validation):
                with self.assertRaisesRegex(ValueError, "CSV changed during table aggregation"):
                    table.read_condition_analyses(source, master_seed=71, **options)
            self.assertNotEqual(provenance.file_sha256(source), original_checksum)

    @staticmethod
    def _previous_table_set(output):
        output.mkdir()
        for filename in table.OUTPUT_FILENAMES:
            (output / filename).write_text(table._VERSION_MARKER + "previous " + filename + "\n")
        return {path.name: path.read_bytes() for path in output.iterdir()}

    def test_second_staging_write_failure_preserves_the_entire_previous_set(self):
        analyses = tuple(_analysis(table.CLASS_COUNT_CONDITIONS[0], s) for s in SETTINGS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latex_tables_v8"
            previous = self._previous_table_set(output)
            write = table._write_table
            calls = []

            def fail_second(path, latex):
                calls.append(path)
                if len(calls) == 2:
                    raise OSError("second staging write failed")
                write(path, latex)

            with patch.object(table, "read_condition_analyses", return_value=analyses), \
                    patch.object(table, "_write_table", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "second staging"):
                    table.generate_latex_tables(root / "input.csv", output)
            self.assertEqual(len(calls), 2)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, previous)
            self.assertEqual(list(root.iterdir()), [output])

    def test_directory_publication_failure_restores_previous_set_and_success_replaces_all(self):
        analyses = tuple(_analysis(table.CLASS_COUNT_CONDITIONS[0], s) for s in SETTINGS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latex_tables_v8"
            previous = self._previous_table_set(output)
            replace_file = table.os.replace
            failed = False

            def fail_publication(source, destination):
                nonlocal failed
                if Path(destination).resolve() == output.resolve() and not failed:
                    failed = True
                    self.assertFalse(output.exists())
                    self.assertEqual({p.name for p in Path(source).iterdir()}, set(table.OUTPUT_FILENAMES))
                    raise OSError("directory publication failed")
                return replace_file(source, destination)

            with patch.object(table, "read_condition_analyses", return_value=analyses), \
                    patch.object(table.os, "replace", side_effect=fail_publication):
                with self.assertRaisesRegex(OSError, "directory publication"):
                    table.generate_latex_tables(root / "input.csv", output)
            self.assertTrue(failed)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, previous)
            self.assertEqual(list(root.iterdir()), [output])
            with patch.object(table, "read_condition_analyses", return_value=analyses):
                paths = table.generate_latex_tables(root / "input.csv", output)
            expected = table.build_latex_tables(analyses)
            self.assertEqual({p.name: p.read_text() for p in paths}, expected)
            self.assertEqual(output.stat().st_mode & 0o777, 0o755)
            self.assertEqual(list(root.iterdir()), [output])

    def test_existing_lock_and_unrelated_output_content_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "latex_tables_v8"
            previous = self._previous_table_set(output)
            lock = root / ".latex_tables_v8.lock"
            lock.write_text("another writer")
            with patch.object(table, "read_condition_analyses") as read:
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    table.generate_latex_tables(root / "input.csv", output)
                read.assert_not_called()
            self.assertEqual(lock.read_text(), "another writer")
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, previous)
            lock.unlink()
            unrelated = output / "notes.txt"
            unrelated.write_text("retain this file")
            with patch.object(table, "read_condition_analyses") as read:
                with self.assertRaisesRegex(ValueError, "Unrecognized output content"):
                    table.generate_latex_tables(root / "input.csv", output)
                read.assert_not_called()
            self.assertEqual(unrelated.read_text(), "retain this file")
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir() if p != unrelated}, previous)
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
