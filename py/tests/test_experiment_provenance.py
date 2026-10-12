"""Reproducibility records must identify the exact data they accompany."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils import experiment_provenance as provenance


class ExperimentProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "experiment.py"
        self.source.write_text("# Small reproducibility fixture.\n", encoding="utf-8")
        self.configuration = {"protocol": 6, "seed": 20250731, "settings": ["example"]}

    def metadata(self):
        return provenance.collect_run_metadata(
            source_root=self.root, source_paths=[self.source],
            configuration=self.configuration,
        )

    def test_records_sources_configuration_versions_and_timing_environment(self):
        metadata = self.metadata()
        self.assertEqual(metadata["configuration"], self.configuration)
        self.assertEqual(metadata["source_sha256"], {
            "experiment.py": hashlib.sha256(self.source.read_bytes()).hexdigest(),
        })
        self.assertIsNone(metadata["git_revision"])
        self.assertIsNone(metadata["git_worktree_modified"])
        for package in ("python", "numpy", "scipy", "threadpoolctl"):
            self.assertTrue(metadata["software"][package])
        environment = metadata["timing_environment"]
        self.assertEqual(environment["timing_threads"], 1)
        self.assertEqual(environment["clock"], "time.perf_counter_ns")
        self.assertTrue(environment["operating_system"])
        self.assertTrue(environment["processor"])
        self.assertTrue(all(info["num_threads"] == 1 for info in
                            environment["native_libraries_under_timing_limit"]))

    def test_publish_and_read_exact_csv_checksum(self):
        csv_path = self.root / "raw.csv"
        csv_path.write_bytes(b"value\n42\n")
        metadata = self.metadata()
        path = provenance.publish_run_metadata(csv_path, csv_path, metadata)
        expected_hash = hashlib.sha256(b"value\n42\n").hexdigest()
        self.assertEqual(path.name, f"raw.{expected_hash}.metadata.json")
        self.assertEqual(provenance.read_run_metadata(csv_path),
                         dict(metadata, raw_csv_sha256=expected_hash))
        self.assertEqual(provenance.publish_run_metadata(csv_path, csv_path, metadata), path)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_source_changes_are_detected_before_publication(self):
        metadata = self.metadata()
        provenance.assert_sources_unchanged(metadata, source_root=self.root)
        self.source.write_text("# Edited during the experiment.\n")
        with self.assertRaisesRegex(RuntimeError, "source changed.*experiment.py"):
            provenance.assert_sources_unchanged(metadata, source_root=self.root)
        self.source.unlink()
        with self.assertRaisesRegex(RuntimeError, "source changed.*experiment.py"):
            provenance.assert_sources_unchanged(metadata, source_root=self.root)

    def test_modified_csv_cannot_silently_reuse_old_provenance(self):
        csv_path = self.root / "raw.csv"
        csv_path.write_text("old data\n")
        path = provenance.publish_run_metadata(csv_path, csv_path, self.metadata())
        original = path.read_bytes()
        csv_path.write_text("new data\n")
        with self.assertRaises(FileNotFoundError):
            provenance.read_run_metadata(csv_path)
        self.assertEqual(path.read_bytes(), original)

    def test_rejects_corrupt_schema_checksum_and_conflicting_provenance(self):
        csv_path = self.root / "raw.csv"
        csv_path.write_text("data\n")
        metadata = self.metadata()
        path = provenance.publish_run_metadata(csv_path, csv_path, metadata)
        valid = json.loads(path.read_text())
        for field, bad in (("metadata_schema", "unknown"), ("raw_csv_sha256", "0" * 64)):
            path.write_text(json.dumps(dict(valid, **{field: bad})))
            with self.subTest(field=field), self.assertRaises(ValueError):
                provenance.read_run_metadata(csv_path)
        with self.assertRaises(ValueError):
            provenance.publish_run_metadata(csv_path, csv_path, metadata)

    def test_failed_new_publication_preserves_previous_data_and_metadata(self):
        output = self.root / "raw.csv"
        output.write_text("old data\n")
        metadata = self.metadata()
        old_record = provenance.publish_run_metadata(output, output, metadata)
        old_bytes = old_record.read_bytes()
        temporary_csv = self.root / "new.csv.tmp"
        temporary_csv.write_text("new data\n")
        with patch.object(provenance.os, "replace", side_effect=OSError("injected failure")):
            with self.assertRaisesRegex(OSError, "injected failure"):
                provenance.publish_run_metadata(temporary_csv, output, metadata)
        self.assertEqual(output.read_text(), "old data\n")
        self.assertEqual(old_record.read_bytes(), old_bytes)
        self.assertFalse(provenance.metadata_path_for_csv(
            output, checksum=provenance.file_sha256(temporary_csv)).exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [temporary_csv])

    def test_runner_records_actual_settings_streams_and_budgets(self):
        import run_experiment_main_evaluation_on_synthetic_data as runner
        output = self.root / "raw_v8.csv"
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        runner.run_experiment(output, conditions=(condition,), n_replications=2,
                              n_samples=40, master_seed=71, show_progress=False)
        metadata = provenance.read_run_metadata(output)
        configuration = metadata["configuration"]
        self.assertEqual(configuration["main_protocol_version"], 8)
        self.assertEqual(configuration["raw_schema_version"], "class_counts_raw_v8")
        self.assertEqual(configuration["synthetic_design_version"], 3)
        self.assertEqual(configuration["B"], 2)
        self.assertEqual(configuration["S"], 40)
        self.assertEqual(configuration["master_seed"], 71)
        self.assertEqual(configuration["historical_class_counts"], [500, 500])
        expected_settings = [
            dict(setting="dip_uniform_m1", method="DiP-ROC", family="uniform", mass=1.0),
            dict(setting="dip_uniform_m5", method="DiP-ROC", family="uniform", mass=5.0),
            dict(setting="dip_uniform_m20", method="DiP-ROC", family="uniform", mass=20.0),
            dict(setting="dip_linear_m1", method="DiP-ROC", family="linear", mass=1.0),
            dict(setting="dip_linear_m5", method="DiP-ROC", family="linear", mass=5.0),
            dict(setting="dip_linear_m20", method="DiP-ROC", family="linear", mass=20.0),
            dict(setting="dip_histograms_m1", method="DiP-ROC", family="histograms", mass=1.0),
            dict(setting="dip_histograms_m5", method="DiP-ROC", family="histograms", mass=5.0),
            dict(setting="dip_histograms_m20", method="DiP-ROC", family="histograms", mass=20.0),
        ]
        self.assertEqual(configuration["settings"], expected_settings)
        reporting_order = tuple(entry["setting"] for entry in expected_settings)
        self.assertEqual(runner.SETTINGS, reporting_order)
        templates = configuration["stream_key_templates"]
        expected_roles = {
            "dip_uniform_m1": 2, "dip_uniform_m5": 7, "dip_uniform_m20": 11,
            "dip_linear_m1": 6, "dip_linear_m5": 8, "dip_linear_m20": 12,
            "dip_histograms_m1": 9, "dip_histograms_m5": 10, "dip_histograms_m20": 13,
        }
        self.assertEqual(templates, {
            "data": [1, 1, "condition_index", "replication"],
            "history": [1, 5, "condition_index", "replication"],
            "simulations": {
                setting: [1, role, "condition_index", "replication"]
                for setting, role in expected_roles.items()
            },
            "warmup": {
                setting: [1, 4, role, "condition_index", 0]
                for setting, role in expected_roles.items()
            },
        })
        for setting, role in expected_roles.items():
            with self.subTest(setting=setting):
                self.assertEqual(runner._simulation_stream_key(condition, 3, setting),
                                 (1, role, condition.index, 3))
                self.assertEqual(runner._warmup_stream_key(condition.index, setting),
                                 (1, 4, role, condition.index, 0))
        self.assertEqual(configuration["execution_order"],
                         "reporting order rotated left by replication modulo 9")
        for replication in range(18):
            with self.subTest(replication=replication):
                offset = replication % 9
                self.assertEqual(runner._execution_order(replication),
                                 reporting_order[offset:] + reporting_order[:offset])
        self.assertIn("py/run_experiment_main_evaluation_on_synthetic_data.py", metadata["source_sha256"])
        self.assertIn("py/roc_utils/prior_profiles.py", metadata["source_sha256"])

    def test_final_csv_failure_after_metadata_publish_preserves_old_run(self):
        import run_experiment_main_evaluation_on_synthetic_data as runner
        output = self.root / "raw_v8.csv"
        options = dict(conditions=runner.CLASS_COUNT_CONDITIONS[:1], n_replications=2,
                       n_samples=40, show_progress=False)
        runner.run_experiment(output, master_seed=71, **options)
        old_csv = output.read_bytes()
        old_metadata_path = provenance.metadata_path_for_csv(output)
        old_metadata = old_metadata_path.read_bytes()
        initial_records = set(self.root.glob("*.metadata.json"))
        original_replace = os.replace

        def fail_csv_only(source, destination):
            if Path(destination) == output:
                raise OSError("injected final CSV failure")
            return original_replace(source, destination)

        with patch.object(runner.os, "replace", side_effect=fail_csv_only):
            with self.assertRaisesRegex(OSError, "injected final CSV failure"):
                runner.run_experiment(output, master_seed=72, overwrite=True, **options)
        self.assertEqual(output.read_bytes(), old_csv)
        self.assertEqual(old_metadata_path.read_bytes(), old_metadata)
        self.assertEqual(provenance.read_run_metadata(output)["configuration"]["master_seed"], 71)
        self.assertEqual(len(set(self.root.glob("*.metadata.json")) - initial_records), 1)
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
