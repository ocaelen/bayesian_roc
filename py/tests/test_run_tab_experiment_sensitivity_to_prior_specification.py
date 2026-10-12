import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_tab_experiment_sensitivity_to_prior_specification as table
from roc_utils.experiment_results import raw_level_measurements_from_replication
from roc_utils.utilities_experiments import ReplicationMeasurements


def _analysis(condition, method):
    mass = condition.prior_mass if condition is not None else 1
    shrinkage = {1: 0.0, 20: 0.05, 50: 0.10}[mass]
    records = []
    deviations = (0.16, 0.24, 0.29, 0.34) if method == "DiP-ROC" else (0.16, 0.34, 0.29, 0.24)
    for deviation, auc in zip(deviations, (0.34, 0.43, 0.57, 0.66)):
        half = 0.17 - shrinkage / 2
        measurements = ReplicationMeasurements(
            external_max_roc_deviation=deviation, external_grid_auc=auc,
            roc_radii={0.80: 0.30-shrinkage, 0.95: 0.38-shrinkage},
            auc_intervals={0.80: (0.50-half, 0.50+half), 0.95: (0.25, 0.75)},
        )
        records.append(raw_level_measurements_from_replication(measurements, table.NOMINAL_LEVELS))
    return table.ConditionAnalysis(
        condition, method,
        {level: tuple(r[index] for r in records) for index, level in enumerate(table.NOMINAL_LEVELS)},
        (4.0, 6.0, 8.0, 10.0) if method == "DiP-ROC" else (1.0, 2.0, 3.0, 4.0),
        tuple(f"data-{b}" for b in range(4)), 40,
    )


def _analyses():
    return tuple(_analysis(c, "DiP-ROC") for c in table.PRIOR_CONDITIONS) + (_analysis(None, "BootP-ROC"),)


class PriorSensitivityTableTests(unittest.TestCase):
    def test_nine_prior_settings_and_one_prior_free_reference(self):
        self.assertEqual(table.PRIOR_MASSES, (1,20,50))
        self.assertEqual(tuple(c.condition_id for c in table.PRIOR_CONDITIONS), (
            "oracle_m_1", "oracle_m_20", "auc_compatible_m_1", "auc_compatible_m_20",
            "symmetric_m_1", "symmetric_m_20", "oracle_m_50", "auc_compatible_m_50", "symmetric_m_50",
        ))
        analyses = _analyses()
        pairs = table._method_pairs(analyses)
        self.assertEqual(len(analyses), 10)
        self.assertEqual(len(pairs), 9)
        self.assertTrue(all(reference is analyses[-1] for _, reference in pairs))
        effects = table.build_paired_method_effects(analyses)
        self.assertEqual(len(effects), 18)
        self.assertTrue(all(value.n_replications == 4 for effect in effects for value in effect.metrics.values()))
        reference = analyses[-1].summaries_by_level[0.80]
        self.assertAlmostEqual(reference.roc_coverage_standard_error,
                               math.sqrt(reference.roc_coverage*(1-reference.roc_coverage)/4))
        self.assertEqual(len(table.OUTPUT_FILENAMES), 6)

    def test_concentration_signs_remain_distinct_from_uniform_method_differences(self):
        analyses = _analyses()
        effects = table.build_paired_concentration_effects(analyses)
        self.assertEqual(len(effects), 12)
        self.assertEqual({(e.baseline_mass,e.target_mass) for e in effects}, {(1,20),(1,50)})
        self.assertTrue(all(e.roc_radius_reduction > 0 for e in effects))
        self.assertTrue(all(e.metrics["roc_coverage"].mean <= 0 for e in effects))
        comparisons = table.build_paired_method_effects(analyses)
        target = next(e for e in comparisons if e.condition.prior_profile == "oracle" and e.condition.prior_mass == 50 and e.level == 0.80)
        self.assertAlmostEqual(target.metrics["roc_radius"].mean, -0.10)
        dip = next(a for a in analyses if a.condition == target.condition)
        expected = np.array([r.roc_covered for r in dip.records_by_level[0.80]]) - np.array([r.roc_covered for r in analyses[-1].records_by_level[0.80]])
        self.assertAlmostEqual(target.metrics["roc_coverage"].standard_error, expected.std(ddof=1)/2)
        fragments = table.build_latex_tables(analyses)
        appendix = fragments[table.OUTPUT_FILENAMES[0]]
        self.assertEqual(appendix.count("BootP-ROC & --- & --- &"), 2)
        self.assertEqual(appendix.count(" & 0.80 &"), 10)
        self.assertIn("target minus baseline", fragments[table.OUTPUT_FILENAMES[2]])
        self.assertIn("baseline minus target", fragments[table.OUTPUT_FILENAMES[2]])
        self.assertIn("DiP-ROC minus", fragments[table.OUTPUT_FILENAMES[4]])
        self.assertIn("+4.50000 (", fragments[table.OUTPUT_FILENAMES[5]])
        self.assertIn("not pooled across priors", fragments[table.OUTPUT_FILENAMES[5]])

    def test_real_csv_shared_bootstrap_is_read_once_and_prior_fields_remain_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "comparison.csv"
            table.experiment.run_experiment(source, n_replications=2, n_samples=40,
                                            master_seed=37, show_progress=False)
            analyses = table.read_condition_analyses(source, n_replications=2, n_samples=40)
            self.assertEqual(len(analyses), 10)
            with source.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 40)
            boot = [row for row in rows if row["method"] == "BootP-ROC"]
            self.assertEqual(len(boot), 4)
            self.assertTrue(all(row[field] == "" for row in boot for field in ("prior_profile", "prior_mass", "prior_expected_grid_auc")))
            reference = next(a for a in analyses if a.method == "BootP-ROC")
            self.assertIsNone(reference.condition)
            self.assertEqual(reference.n_replications, 2)
            self.assertEqual(reference.computation_times_ms, tuple(int(row["computation_time_ns"])/1_000_000 for row in boot if float(row["level"]) == 0.80))
            paths = table.generate_latex_tables(source, root/"latex_tables_v3", n_replications=2, n_samples=40)
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.is_file() for path in paths))
            with self.assertRaises(AssertionError):
                table.read_condition_analyses(source)
            boot[0]["prior_mass"] = "0"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=table.RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(AssertionError):
                table.read_condition_analyses(source, n_replications=2, n_samples=40)


    def test_v3_table_reader_rejects_old_mixed_and_relabelled_csvs(self):
        self.assertEqual(table.SYNTHETIC_DESIGN_VERSION, 3)
        self.assertEqual(table.RAW_SCHEMA_VERSION, "prior_shape_raw_v3")
        self.assertEqual(table.DEFAULT_INPUT_CSV.name, "raw_replications_v3.csv")
        self.assertEqual(table.DEFAULT_OUTPUT_DIR.name, "latex_tables_v3")
        arguments = dict(conditions=table.PRIOR_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            table.experiment.run_experiment(source, show_progress=False, **arguments)
            with source.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for case in ("old v2", "mixed versions", "relabelled old beta"):
                changed = [dict(row) for row in rows]
                if case == "relabelled old beta":
                    changed[-1].update(dict(zip(
                        ("positive_beta_a", "positive_beta_b", "negative_beta_a", "negative_beta_b"),
                        map(str, (5.0, 3.0, 3.0, 5.0)),
                    )))
                    message = "Beta metadata"
                else:
                    for row in (changed if case == "old v2" else changed[-1:]):
                        row["schema_version"] = "prior_shape_raw_v2"
                    message = "old or mixed synthetic protocol"
                with self.subTest(case=case):
                    bad = Path(directory) / "old.csv"
                    with bad.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=table.RAW_FIELDNAMES)
                        writer.writeheader()
                        writer.writerows(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        table.read_condition_analyses(bad, **arguments)

    def test_previous_table_directories_and_renamed_fragments_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("latex_tables", "latex_tables_v1", "latex_tables_v2"):
                for output in (root / name, root / name / "nested"):
                    with self.subTest(output=output):
                        with patch.object(table, "read_condition_analyses") as read:
                            with self.assertRaisesRegex(ValueError, "protected"):
                                table.generate_latex_tables(root / "missing.csv", output)
                            read.assert_not_called()
                        self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "protected"):
                table.main(["--input", str(root / "missing.csv"),
                            "--output-dir", str(root / "latex_tables_v2")])
            custom = root / "custom"
            custom.mkdir()
            previous = custom / table.OUTPUT_FILENAMES[-1]
            for old_text in ("% Old unversioned table\n", "% Generated from validated v2 comparison replications.\n"):
                previous.write_text(old_text)
                with patch.object(table, "read_condition_analyses") as read:
                    with self.assertRaisesRegex(ValueError, "protected"):
                        table.generate_latex_tables(root / "missing.csv", custom)
                    read.assert_not_called()
                self.assertEqual(previous.read_text(), old_text)
                self.assertEqual(list(custom.iterdir()), [previous])

if __name__ == "__main__":
    unittest.main()
