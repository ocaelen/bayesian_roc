import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils.experiment_results import (  # noqa: E402
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
from roc_utils.utilities_experiments import ReplicationMeasurements  # noqa: E402


class ExperimentResultTests(unittest.TestCase):
    def setUp(self):
        self.measurements = ReplicationMeasurements(
            external_max_roc_deviation=0.2 + 0.5e-12,
            external_grid_auc=0.4 - 0.5e-12,
            roc_radii={0.80: 0.2, 0.95: 0.3},
            auc_intervals={0.80: (0.4, 0.7), 0.95: (0.3, 0.8)},
        )

    def _paired_rows(self):
        record = raw_level_measurements_from_replication(
            self.measurements, (0.80, 0.95),
        )[0]
        dip = {
            "method": "DiP-ROC",
            "experiment": "paired_fixture",
            "condition_id": "symmetric_m_1",
            "replication": "0",
            "K": "2",
            "S": "40",
            "B": "4",
            "n_positive": "4",
            "n_negative": "4",
            "level": str(record.level),
            "rng_algorithm": "PCG64",
            "master_seed": "17",
            "data_stream_id": "PCG64:17:1:1:0",
            "simulation_stream_id": "PCG64:17:1:2:0",
            "prior_profile": "symmetric",
            "prior_mass": "1.0",
            "prior_expected_grid_auc": "0.5",
            "true_grid_auc": "0.85",
            "computation_time_ns": "1000",
            "execution_position": "1",
            "timing_threads": "1",
            **{
                field: str(value)
                for field, value in raw_level_measurement_fields(record).items()
            },
        }
        boot = {
            **dip,
            "method": "BootP-ROC",
            "condition_id": "bootstrap",
            "simulation_stream_id": "PCG64:17:1:3:0",
            "prior_profile": "",
            "prior_mass": "",
            "prior_expected_grid_auc": "",
            "computation_time_ns": "0",
            "execution_position": "2",
            "external_max_roc_deviation": "0.5",
            "roc_radius": "1.8",
        }
        return dip, boot

    def test_exact_flattening_coverage_and_nesting(self):
        records = raw_level_measurements_from_replication(
            self.measurements,
            (0.80, 0.95),
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].level, 0.80)
        self.assertEqual(records[1].level, 0.95)
        self.assertEqual(records[0].roc_covered, 1)
        self.assertEqual(records[0].auc_covered, 1)
        self.assertEqual(records[0].auc_width, 0.7 - 0.4)
        self.assertEqual(records[1].auc_width, 0.8 - 0.3)
        self.assertEqual(
            tuple(raw_level_measurement_fields(records[0])),
            MEASUREMENT_RAW_FIELDNAMES,
        )
        records_by_level = assert_complete_level_bundle(
            records,
            (0.80, 0.95),
        )
        self.assertEqual(set(records_by_level), {0.80, 0.95})

    def test_csv_round_trip_is_exact(self):
        record = raw_level_measurements_from_replication(
            self.measurements,
            (0.80, 0.95),
        )[0]
        row = {
            "level": str(record.level),
            **{
                field: str(value)
                for field, value in raw_level_measurement_fields(record).items()
            },
        }

        self.assertEqual(parse_raw_level_measurements(row), record)
        self.assertEqual(parse_canonical_int({"x": "12"}, "x"), 12)
        self.assertEqual(
            parse_finite_float({"x": "0.12345678901234568"}, "x"),
            0.12345678901234568,
        )

    def test_invalid_raw_identites_are_rejected(self):
        valid = raw_level_measurements_from_replication(
            self.measurements,
            (0.80, 0.95),
        )[0]
        common = {
            "level": valid.level,
            "external_max_roc_deviation": valid.external_max_roc_deviation,
            "external_grid_auc": valid.external_grid_auc,
            "roc_radius": valid.roc_radius,
            "auc_interval_lower": valid.auc_interval_lower,
            "auc_interval_upper": valid.auc_interval_upper,
            "auc_width": valid.auc_width,
            "roc_covered": valid.roc_covered,
            "auc_covered": valid.auc_covered,
        }

        with self.assertRaises(AssertionError):
            RawLevelMeasurements(**{**common, "auc_width": 0.123})
        with self.assertRaises(AssertionError):
            RawLevelMeasurements(**{**common, "roc_covered": 0})
        with self.assertRaises(AssertionError):
            RawLevelMeasurements(**{**common, "auc_covered": 0})
        with self.assertRaises(AssertionError):
            RawLevelMeasurements(**{**common, "external_grid_auc": np.nan})

    def test_invalid_bundles_and_parsers_are_rejected(self):
        records = raw_level_measurements_from_replication(
            self.measurements,
            (0.80, 0.95),
        )
        with self.assertRaises(AssertionError):
            assert_complete_level_bundle(records[:1], (0.80, 0.95))

        mismatched_external = RawLevelMeasurements(
            level=0.95,
            external_max_roc_deviation=0.1,
            external_grid_auc=records[1].external_grid_auc,
            roc_radius=records[1].roc_radius,
            auc_interval_lower=records[1].auc_interval_lower,
            auc_interval_upper=records[1].auc_interval_upper,
            auc_width=records[1].auc_width,
            roc_covered=1,
            auc_covered=records[1].auc_covered,
        )
        with self.assertRaises(AssertionError):
            assert_complete_level_bundle(
                (records[0], mismatched_external),
                (0.80, 0.95),
            )

        nonnested = RawLevelMeasurements(
            level=0.95,
            external_max_roc_deviation=records[1].external_max_roc_deviation,
            external_grid_auc=records[1].external_grid_auc,
            roc_radius=0.19,
            auc_interval_lower=0.45,
            auc_interval_upper=0.65,
            auc_width=0.65 - 0.45,
            roc_covered=0,
            auc_covered=0,
        )
        with self.assertRaises(AssertionError):
            assert_complete_level_bundle(
                (records[0], nonnested),
                (0.80, 0.95),
            )

        with self.assertRaises(AssertionError):
            parse_canonical_int({"x": "01"}, "x")
        with self.assertRaises(AssertionError):
            parse_finite_float({"x": "nan"}, "x")
        with self.assertRaises(TypeError):
            validate_positive_integer("B", True, minimum=1)
        with self.assertRaises(ValueError):
            validate_positive_integer("B", 0, minimum=1)

    def test_method_metadata_accepts_distinct_radius_bounds_and_no_bootstrap_prior(self):
        dip, boot = self._paired_rows()
        assert_method_metadata(dip)
        assert_method_metadata(boot)
        record = parse_raw_level_measurements(boot)
        self.assertEqual(record.roc_radius, 1.8)
        self.assertEqual(record.roc_covered, 1)
        self.assertEqual(boot["true_grid_auc"], "0.85")
        without_prior_auc = dict(boot)
        del without_prior_auc["prior_expected_grid_auc"]
        assert_method_metadata(without_prior_auc)
        assert_method_metadata({**boot, "roc_radius": "2.0"})
        assert_method_metadata({**dip, "roc_radius": str(np.sqrt(2.0))})
        with self.assertRaises(AssertionError):
            assert_method_metadata({**dip, "roc_radius": "1.8"})
        with self.assertRaises(AssertionError):
            assert_method_metadata({**boot, "roc_radius": "2.000001"})
        with self.assertRaises(AssertionError):
            parse_raw_level_measurements({**boot, "roc_radius": "2.000001"})

    def test_method_metadata_rejects_invalid_timing_methods_and_prior_fields(self):
        dip, boot = self._paired_rows()
        for field, value in (
            ("method", "bootstrap"),
            ("computation_time_ns", "-1"),
            ("computation_time_ns", "01"),
            ("execution_position", "0"),
            ("execution_position", "-1"),
            ("execution_position", "01"),
            ("timing_threads", "0"),
            ("timing_threads", "2"),
            ("roc_radius", "nan"),
            ("roc_radius", "inf"),
            ("roc_radius", "-0.1"),
            ("prior_profile", "symmetric"),
            ("prior_mass", "0"),
            ("prior_expected_grid_auc", "0.5"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(AssertionError):
                    assert_method_metadata({**boot, field: value})
        for invalid_time in ("nan", "inf", "0.5", "True", ""):
            with self.subTest(time=invalid_time):
                with self.assertRaises(ValueError):
                    assert_method_metadata({**boot, "computation_time_ns": invalid_time})
        for field, value in (
            ("prior_profile", ""),
            ("prior_mass", "0"),
            ("prior_mass", "-1"),
            ("prior_mass", "nan"),
            ("prior_expected_grid_auc", "1.1"),
        ):
            with self.subTest(dip_field=field, value=value):
                with self.assertRaises(AssertionError):
                    assert_method_metadata({**dip, field: value})

    def test_pairing_allows_distinct_centers_conditions_priors_and_times(self):
        dip, boot = self._paired_rows()
        self.assertNotEqual(dip["condition_id"], boot["condition_id"])
        self.assertNotEqual(dip["external_max_roc_deviation"], boot["external_max_roc_deviation"])
        self.assertNotEqual(dip["computation_time_ns"], boot["computation_time_ns"])
        assert_paired_method_rows(dip, boot)
        # Text representations may differ when their exact numeric values agree.
        assert_paired_method_rows(dip, {**boot, "level": "0.8000000000000000"})
        with self.assertRaises(AssertionError):
            assert_paired_method_rows(boot, dip)
        with self.assertRaises(AssertionError):
            assert_paired_method_rows(dip, dip)

    def test_pairing_rejects_mismatched_targets_design_and_random_streams(self):
        dip, boot = self._paired_rows()
        corruptions = {
            "experiment": "different_experiment",
            "data_stream_id": "different_data",
            "rng_algorithm": "different_rng",
            "master_seed": "18",
            "replication": "1",
            "K": "3",
            "S": "41",
            "B": "5",
            "n_positive": "5",
            "n_negative": "5",
            "level": "0.95",
            "external_grid_auc": str(np.nextafter(float(dip["external_grid_auc"]), 1.0)),
            "simulation_stream_id": dip["simulation_stream_id"],
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                with self.assertRaises(AssertionError):
                    assert_paired_method_rows(dip, {**boot, field: value})
        with self.assertRaises(AssertionError):
            assert_paired_method_rows(dip, {**boot, "simulation_stream_id": boot["data_stream_id"]})
        with self.assertRaises(AssertionError):
            assert_paired_method_rows({**dip, "simulation_stream_id": dip["data_stream_id"]}, boot)

    def test_wide_bootstrap_records_keep_all_measurement_identities_and_nesting(self):
        measurements = ReplicationMeasurements(
            external_max_roc_deviation=np.sqrt(2.0),
            external_grid_auc=0.0,
            roc_radii={0.80: 1.8, 0.95: 2.0},
            auc_intervals={0.80: (0.0, 0.5), 0.95: (0.0, 1.0)},
        )
        records = raw_level_measurements_from_replication(measurements, (0.80, 0.95))
        self.assertEqual([record.roc_radius for record in records], [1.8, 2.0])
        self.assertEqual([record.auc_width for record in records], [0.5, 1.0])
        self.assertTrue(all(record.roc_covered == record.auc_covered == 1 for record in records))
        bundled = assert_complete_level_bundle(tuple(reversed(records)), (0.95, 0.80))
        self.assertEqual(set(bundled), {0.80, 0.95})

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.experiment_results"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "experiment_results relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
