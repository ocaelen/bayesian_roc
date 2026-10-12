"""Offline provenance, publication, and diagnostic contracts for real-data v3."""

import ast
import copy
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sklearn import __version__ as SKLEARN_VERSION

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_experiment_empirical_evaluation_on_real_world_datasets as runner  # noqa: E402
from roc_utils import experiment_provenance as provenance  # noqa: E402
from roc_utils.semantic_rng import semantic_stream_id  # noqa: E402
from test_run_experiment_real_data import (  # noqa: E402
    FERTILITY_CONDITIONS,
    _fertility_loaded,
    _read_rows,
    _score_fixture,
    _write_rows,
)


class RealRunnerV3SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.output = self.root / "raw_replications_v3.csv"
        self.loaded = _fertility_loaded(self.root / "fertility.arff")
        self.condition = FERTILITY_CONDITIONS[0]
        self.options = dict(
            conditions=(self.condition,),
            n_replications=1,
            n_samples=20,
            master_seed=71,
            show_progress=False,
        )

    def _run(self, output=None, **overrides):
        options = dict(self.options, **overrides)
        with (
            patch.object(runner, "load_real_dataset", return_value=self.loaded),
            patch.object(runner, "fit_logistic_regression_scores", return_value=_score_fixture()),
        ):
            return runner.run_experiment(output or self.output, **options)

    def _validate(self, output=None, **overrides):
        options = {key: value for key, value in self.options.items()
                   if key != "show_progress"}
        return runner.validate_raw_results_csv(
            output or self.output, **dict(options, **overrides)
        )

    def _metadata(self):
        return runner.read_run_metadata(
            self.output, metadata_schema="real_run_metadata_v1"
        )

    def _published_snapshot(self):
        companion = provenance.metadata_path_for_csv(self.output)
        return self.output.read_bytes(), companion, companion.read_bytes()

    def _assert_published_snapshot(self, snapshot):
        csv_bytes, companion, metadata_bytes = snapshot
        self.assertEqual(self.output.read_bytes(), csv_bytes)
        self.assertEqual(companion.read_bytes(), metadata_bytes)
        self.assertEqual(
            self._metadata()["configuration"]["master_seed"],
            71,
        )
        self.assertFalse(list(self.root.glob(".*.tmp")))

    @staticmethod
    def _diagnostic_context(error):
        contexts = []
        for note in getattr(error, "__notes__", ()):
            try:
                value = ast.literal_eval(note.split(": ", 1)[1])
            except (IndexError, SyntaxError, ValueError):
                continue
            if isinstance(value, dict):
                contexts.append(value)
        merged = {}
        for context in contexts:
            merged.update(context)
        return merged

    def test_completed_run_has_checksum_bound_real_v3_provenance(self):
        self._run()
        metadata = self._metadata()
        self.assertEqual(runner.REAL_PROTOCOL_VERSION, 3)
        self.assertEqual(runner.RAW_SCHEMA_VERSION, "real_data_raw_v3")
        self.assertEqual(runner.METHODS, ("DiP-ROC",))
        self.assertEqual(metadata["metadata_schema"], "real_run_metadata_v1")
        self.assertEqual(metadata["raw_csv_sha256"], provenance.file_sha256(self.output))
        self.assertEqual(metadata["software"]["scikit-learn"], SKLEARN_VERSION)
        self.assertEqual({row["sklearn_version"] for row in _read_rows(self.output)},
                         {SKLEARN_VERSION})
        self.assertEqual(self._validate(), 2)
        self.assertIn("py/roc_utils/real_model.py", metadata["source_sha256"])
        self.assertIn("py/roc_utils/real_datasets.py", metadata["source_sha256"])
        self.assertEqual(metadata["timing_environment"]["timing_threads"], 1)

    def test_companion_is_required_unless_metadata_is_explicitly_supplied(self):
        self._run()
        metadata = self._metadata()
        provenance.metadata_path_for_csv(self.output).unlink()
        with self.assertRaises(FileNotFoundError):
            self._metadata()
        with self.assertRaises(FileNotFoundError):
            self._validate()
        self.assertEqual(self._validate(metadata=metadata), 2)

    def test_reader_rejects_foreign_metadata_schema_and_wrong_csv_digest(self):
        self._run()
        path = provenance.metadata_path_for_csv(self.output)
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, invalid in (
            ("metadata_schema", "synthetic_run_metadata_v1"),
            ("raw_csv_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                payload = dict(original, **{field: invalid})
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self._metadata()
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_changed_csv_cannot_reuse_its_previous_companion(self):
        self._run()
        self.output.write_bytes(self.output.read_bytes() + b"\n")
        with self.assertRaises(FileNotFoundError):
            self._metadata()

    def test_metadata_configuration_rejects_changed_protocol_and_design(self):
        self._run()
        metadata = self._metadata()
        configuration = metadata["configuration"]
        self.assertEqual(configuration["real_protocol_version"], 3)
        self.assertEqual(configuration["raw_schema_version"], "real_data_raw_v3")
        self.assertEqual(configuration["methods"], ["DiP-ROC"])
        options = {key: value for key, value in self.options.items()
                   if key != "show_progress"}
        runner.validate_run_metadata(metadata, **options)
        for field, invalid in (
            ("real_protocol_version", 2),
            ("raw_schema_version", "real_data_raw_v2"),
            ("methods", ["DiP-ROC", "BootP-ROC"]),
            ("master_seed", 72),
            ("rng_algorithm", "MT19937"),
            ("B", 2),
            ("S", 21),
            ("K_values", [10]),
            ("nominal_levels", [0.90, 0.95]),
            ("model", {}),
            ("prior", {}),
            ("datasets", []),
            ("conditions", []),
            ("stream_key_templates", {}),
            ("timing_scope", "training included"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(metadata)
                changed["configuration"][field] = invalid
                with self.assertRaises((AssertionError, ValueError)):
                    runner.validate_run_metadata(changed, **options)
                with self.assertRaises((AssertionError, ValueError)):
                    self._validate(metadata=changed)

    def test_row_software_version_must_match_recorded_run(self):
        self._run()
        metadata = self._metadata()
        rows = _read_rows(self.output)
        for row in rows:
            row["sklearn_version"] = "0.0.0-unrecorded"
        _write_rows(self.output, rows)
        with self.assertRaises((AssertionError, ValueError)) as raised:
            self._validate(metadata=metadata)
        details = self._diagnostic_context(raised.exception)
        self.assertEqual(details["phase"], "CSV validation")
        self.assertEqual(details["csv_line"], 2)
        self.assertEqual(details["condition_id"], "fertility_k_5")
        self.assertEqual(str(details["replication_zero_based"]), "0")
        self.assertEqual(details["data_stream_id"], semantic_stream_id(71, 3, 1, 5, 0))

        # A valid historical run must remain readable under a newer installation.
        metadata["software"]["scikit-learn"] = "0.0.0-unrecorded"
        self.assertEqual(self._validate(metadata=metadata), 2)

    def test_old_csv_schemas_are_protected_even_with_overwrite(self):
        self._run()
        original_rows = _read_rows(self.output)
        for schema in ("real_data_raw_v1", "real_data_raw_v2"):
            with self.subTest(schema=schema):
                output = self.root / f"preserved_{schema}.csv"
                rows = [dict(row, schema_version=schema) for row in original_rows]
                _write_rows(output, rows)
                old_bytes = output.read_bytes()
                with patch.object(runner, "load_real_dataset") as load:
                    with self.assertRaises((ValueError, FileExistsError)):
                        runner.run_experiment(output, overwrite=True, **self.options)
                load.assert_not_called()
                self.assertEqual(output.read_bytes(), old_bytes)

    def test_legacy_destination_paths_are_protected_even_when_absent(self):
        reserved = (
            Path("raw_replications.csv"),
            Path("raw_replications_v1.csv"),
            Path("raw_replications_v2.csv"),
            Path("latex_tables/output.csv"),
            Path("latex_tables_v1/output.csv"),
            Path("latex_tables_v2/output.csv"),
            Path("results/archived_experiments/real_data/output.csv"),
            Path("py/back_up_legacy_pipeline/output.csv"),
        )
        for relative_path in reserved:
            with self.subTest(path=str(relative_path)):
                output = self.root / relative_path
                with patch.object(runner, "load_real_dataset") as load:
                    with self.assertRaises((ValueError, FileExistsError)):
                        runner.run_experiment(output, overwrite=True, **self.options)
                load.assert_not_called()
                self.assertFalse(output.exists())

    def test_no_clobber_handles_a_competing_csv_at_final_publication(self):
        competing_bytes = b"results published by another process\n"
        original_link = os.link
        publication_attempts = []

        def competing_link(source, destination, *args, **kwargs):
            if Path(destination) == self.output:
                publication_attempts.append(Path(source))
                self.output.write_bytes(competing_bytes)
            return original_link(source, destination, *args, **kwargs)

        with patch.object(runner.os, "link", side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self._run()
        self.assertEqual(len(publication_attempts), 1)
        self.assertEqual(self.output.read_bytes(), competing_bytes)
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_failed_csv_replacement_preserves_previous_csv_and_companion(self):
        self._run()
        snapshot = self._published_snapshot()
        old_companions = set(self.root.glob("*.metadata.json"))
        original_replace = os.replace

        def fail_csv_only(source, destination):
            if Path(destination) == self.output:
                raise OSError("injected final CSV replacement failure")
            return original_replace(source, destination)

        with patch.object(runner.os, "replace", side_effect=fail_csv_only):
            with self.assertRaisesRegex(OSError, "final CSV replacement failure"):
                self._run(master_seed=72, overwrite=True)
        self._assert_published_snapshot(snapshot)
        self.assertEqual(len(set(self.root.glob("*.metadata.json")) - old_companions), 1)

    def test_failed_metadata_publication_preserves_previous_run(self):
        self._run()
        snapshot = self._published_snapshot()
        error = OSError("injected metadata publication failure")
        with patch.object(runner, "publish_run_metadata", side_effect=error):
            with self.assertRaises(OSError) as raised:
                self._run(master_seed=72, overwrite=True)
        self.assertIs(raised.exception, error)
        self._assert_published_snapshot(snapshot)

    def test_source_change_aborts_before_any_new_publication(self):
        self._run()
        snapshot = self._published_snapshot()
        companions = set(self.root.glob("*.metadata.json"))
        error = RuntimeError("Experiment source changed during this run: fixture.py")
        with (
            patch.object(runner, "assert_sources_unchanged", side_effect=error),
            patch.object(runner, "publish_run_metadata") as publish,
        ):
            with self.assertRaisesRegex(RuntimeError, "source changed") as raised:
                self._run(master_seed=72, overwrite=True)
        self.assertIs(raised.exception, error)
        publish.assert_not_called()
        self._assert_published_snapshot(snapshot)
        self.assertEqual(set(self.root.glob("*.metadata.json")), companions)

    def test_split_failure_preserves_error_and_identifies_its_random_stream(self):
        error = AssertionError("injected split failure")
        with patch.object(runner, "split_and_encode_real_dataset", side_effect=error):
            with self.assertRaises(AssertionError) as raised:
                self._run()
        self.assertIs(raised.exception, error)
        details = self._diagnostic_context(error)
        self.assertEqual(details["master_seed"], 71)
        self.assertEqual(details["replication_zero_based"], 0)
        self.assertEqual(details["data_stream_id"], semantic_stream_id(71, 3, 1, 5, 0))
        self.assertIn("Fertility", str(details))
        self.assertIn("split", str(details).lower())
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_model_failure_notes_include_model_and_split_context(self):
        error = RuntimeError("injected model convergence failure")
        with (
            patch.object(runner, "load_real_dataset", return_value=self.loaded),
            patch.object(runner, "fit_logistic_regression_scores", side_effect=error),
        ):
            with self.assertRaises(RuntimeError) as raised:
                runner.run_experiment(self.output, **self.options)
        self.assertIs(raised.exception, error)
        details = self._diagnostic_context(error)
        self.assertEqual(details["master_seed"], 71)
        self.assertEqual(details["replication_zero_based"], 0)
        self.assertEqual(details["data_stream_id"], semantic_stream_id(71, 3, 1, 5, 0))
        self.assertIn("Fertility", str(details))
        self.assertIn(runner.MODEL_SPEC_VERSION, str(details))
        self.assertIn("lbfgs", str(details))
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_external_evaluation_failure_retains_grid_and_simulation_context(self):
        error = AssertionError("injected external evaluation failure")
        with patch.object(runner, "evaluate_replication", side_effect=error):
            with self.assertRaises(AssertionError) as raised:
                self._run()
        self.assertIs(raised.exception, error)
        details = self._diagnostic_context(error)
        self.assertEqual(details["condition_id"], "fertility_k_5")
        self.assertEqual(details["replication_zero_based"], 0)
        self.assertEqual(details["master_seed"], 71)
        self.assertEqual(details["data_stream_id"], semantic_stream_id(71, 3, 1, 5, 0))
        self.assertEqual(details["simulation_stream_id"], semantic_stream_id(71, 3, 2, 15, 0))
        self.assertIn("external", str(details).lower())
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_csv_write_failure_retains_replication_context_and_previous_run(self):
        self._run()
        snapshot = self._published_snapshot()
        error = OSError("injected CSV row write failure")
        with patch.object(csv.DictWriter, "writerows", side_effect=error):
            with self.assertRaises(OSError) as raised:
                self._run(master_seed=72, overwrite=True)
        self.assertIs(raised.exception, error)
        details = self._diagnostic_context(error)
        self.assertEqual(details["condition_id"], "fertility_k_5")
        self.assertEqual(details["replication_zero_based"], 0)
        self.assertEqual(details["master_seed"], 72)
        self.assertEqual(details["data_stream_id"], semantic_stream_id(72, 3, 1, 5, 0))
        self.assertEqual(details["simulation_stream_id"], semantic_stream_id(72, 3, 2, 15, 0))
        self.assertIn("csv", str(details).lower())
        self._assert_published_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
