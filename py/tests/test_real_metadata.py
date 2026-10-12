"""Independent checks of the public real-data definitions and provenance."""

from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils import experiment_provenance as provenance
from roc_utils import real_datasets as datasets


class RealMetadataTests(unittest.TestCase):
    def test_public_binary_rules_match_the_twelve_documented_tasks(self):
        expected = {
            "Abalone": ((), (), 10.0),
            "Adult": ((">50K",), ("<=50K",), None),
            "Alzheimer": (("P",), ("H",), None),
            "Banknote Authentication": (("2",), ("1",), None),
            "Diabetes": (("tested_positive",), ("tested_negative",), None),
            "Fertility": (("2",), ("1",), None),
            "Haberman": (("2",), ("1",), None),
            "Mammographic": (("1",), ("0",), None),
            "Sonar": (("Mine",), ("Rock",), None),
            "Titanic": (("1",), ("-1",), None),
            "Transfusion": (("2",), ("1",), None),
            "Contraceptive Method Choice": (("2", "3"), ("1",), None),
        }
        self.assertEqual(tuple(expected), datasets.REAL_DATASET_NAMES)
        with patch.object(datasets, "urlopen", side_effect=AssertionError("No download needed")):
            for name, rule in expected.items():
                with self.subTest(dataset=name):
                    metadata = datasets.real_dataset_metadata(name)
                    self.assertEqual((metadata.positive_values, metadata.negative_values,
                                      metadata.positive_threshold), rule)
                    self.assertTrue(metadata.target_column)
                    self.assertEqual(json.loads(json.dumps(asdict(metadata)))["name"], name)

    def test_public_predictor_overrides_match_the_manuscript(self):
        for name in datasets.REAL_DATASET_NAMES:
            metadata = datasets.real_dataset_metadata(name)
            expected = {"Fertility": ("V1", "V3", "V4", "V5", "V6", "V7", "V8"),
                        "Titanic": ("Class", "Age", "Sex")}.get(name, ())
            self.assertEqual(metadata.categorical_overrides, expected)
            self.assertEqual(metadata.drop_features, ("BI-RADS",) if name == "Mammographic" else ())

    def test_invalid_public_rule_is_rejected(self):
        source = datasets._SPECS_BY_KEY["adult"]
        for changes in ({"positive_values": ()}, {"negative_values": (">50K",)},
                        {"target_column": ""}, {"positive_threshold": float("nan")},
                        {"positive_threshold": 10.0}):
            with self.subTest(changes=changes), patch.dict(
                datasets._SPECS_BY_KEY, {"adult": replace(source, **changes)},
            ), self.assertRaises(AssertionError):
                datasets.real_dataset_metadata("Adult")

    def test_complete_cases_check_excluded_predictor_and_target(self):
        # Only the final two rows are missing values: first in a predictor
        # excluded from fitting, then in the target. Both must be removed.
        data = "\n".join(f"{i},2,{int(i >= 10)}" for i in range(20))
        text = ("@relation complete_cases\n@attribute x numeric\n"
                "@attribute excluded numeric\n@attribute class {0,1}\n@data\n"
                + data + "\n20,?,1\n21,2,?\n")
        source = datasets._RealDatasetSpec(
            name="Toy", slug="toy", openml_data_id=1, openml_version=1,
            openml_file_id=1, md5_checksum="0" * 32, target_column="class",
            positive_values=("1",), negative_values=("0",), positive_threshold=None,
            drop_features=("excluded",), categorical_overrides=(),
            expected_raw_rows=22, expected_complete_rows=20, expected_positive=10,
            expected_feature_count=1, expected_categorical_features=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.arff"
            path.write_text(text)
            X, y, indices, features, categorical = datasets._load_complete_dataset(source, path)
        self.assertEqual(features, ("x",))
        self.assertEqual(categorical, ())
        np.testing.assert_array_equal(indices, np.arange(20))
        np.testing.assert_array_equal(X[:, 0].astype(float), np.arange(20))
        np.testing.assert_array_equal(y, [0] * 10 + [1] * 10)

    def test_real_provenance_records_sklearn_and_requires_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment.py"
            source.write_text("# fixture\n")
            kwargs = dict(source_root=root, source_paths=[source], configuration={"protocol": 3})
            old = provenance.collect_run_metadata(**kwargs)
            self.assertEqual(old["metadata_schema"], "synthetic_run_metadata_v1")
            self.assertNotIn("scikit-learn", old["software"])
            real = provenance.collect_run_metadata(**kwargs, metadata_schema="real_run_metadata_v1")
            self.assertEqual(real["software"]["scikit-learn"], version("scikit-learn"))
            csv_path = root / "raw.csv"
            csv_path.write_text("value\n1\n")
            provenance.publish_run_metadata(csv_path, csv_path, real)
            loaded = provenance.read_run_metadata(csv_path, metadata_schema="real_run_metadata_v1")
            self.assertEqual(loaded["configuration"], real["configuration"])
            with self.assertRaisesRegex(ValueError, "Expected synthetic"):
                provenance.read_run_metadata(csv_path)
            for operation in (
                lambda: provenance.collect_run_metadata(**kwargs, metadata_schema="unknown"),
                lambda: provenance.read_run_metadata(csv_path, metadata_schema="unknown"),
            ):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    operation()

    def test_corrupt_real_companion_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("# source\n")
            csv_path = root / "raw.csv"
            csv_path.write_text("data\n")
            metadata = provenance.collect_run_metadata(
                source_root=root, source_paths=[source], configuration={},
                metadata_schema="real_run_metadata_v1",
            )
            companion = provenance.publish_run_metadata(csv_path, csv_path, metadata)
            payload = json.loads(companion.read_text())
            payload["raw_csv_sha256"] = "0" * 64
            companion.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "does not match"):
                provenance.read_run_metadata(csv_path, metadata_schema="real_run_metadata_v1")


if __name__ == "__main__":
    unittest.main()
