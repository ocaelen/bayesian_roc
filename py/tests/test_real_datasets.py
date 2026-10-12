import io
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import md5
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import roc_utils.real_datasets as real_datasets  # noqa: E402
from roc_utils.real_datasets import (  # noqa: E402
    LoadedRealDataset,
    REAL_DATASET_NAMES,
    load_real_dataset,
    prepare_real_dataset,
    real_dataset_metadata,
    split_and_encode_real_dataset,
)
from roc_utils.real_model import fit_logistic_regression_scores  # noqa: E402


TOY_ARFF = b"""\
@relation toy
@attribute number numeric
@attribute color {red,blue,green}
@attribute class {no,yes}
@data
0,green,no
1,red,no
2,red,no
3,red,no
4,red,no
5,red,no
6,red,no
7,red,no
8,red,no
9,red,no
10,blue,yes
11,blue,yes
12,blue,yes
13,blue,yes
14,blue,yes
15,blue,yes
16,blue,yes
17,blue,yes
18,blue,yes
19,blue,yes
?,red,no
"""


class IdentityPermutationRng:
    def permutation(self, values):
        return np.asarray(values).copy()


def toy_spec():
    return real_datasets._RealDatasetSpec(
        name="Toy",
        slug="toy",
        openml_data_id=999,
        openml_version=1,
        openml_file_id=123,
        md5_checksum=md5(TOY_ARFF).hexdigest(),
        target_column="class",
        positive_values=("yes",),
        negative_values=("no",),
        positive_threshold=None,
        drop_features=(),
        categorical_overrides=(),
        expected_raw_rows=21,
        expected_complete_rows=20,
        expected_positive=10,
        expected_feature_count=2,
        expected_categorical_features=("color",),
    )


class RealDatasetTests(unittest.TestCase):
    def test_registry_pins_the_twelve_manuscript_datasets(self):
        expected_sources = (
            ("Abalone", 183, 1, 3620, "a9555b1cfe39eb55df2c520f09172a96"),
            ("Adult", 1590, 2, 1595261, "bb6510925e5d4b23d136715febb2cdf5"),
            ("Alzheimer", 46849, 2, 22124412, "83e3bd9a41795e8e6916b0e511416c2c"),
            (
                "Banknote Authentication",
                1462,
                1,
                1586223,
                "baa2dc5b745775a943ebeb9c276401f8",
            ),
            ("Diabetes", 37, 1, 37, "3cbaa3e54586aa88cf6aacb4033e4470"),
            ("Fertility", 1473, 1, 1587930, "6bfc3c6c5c7cd86324ce16e63a623bef"),
            ("Haberman", 43, 1, 43, "fe063732e696d8e7aa79746032f0191b"),
            (
                "Mammographic",
                45557,
                2,
                22116525,
                "16eda295e59d814dde5cadbb97330117",
            ),
            ("Sonar", 40, 1, 40, "3ab630fbbfe25ab48b9bb47ce5759203"),
            ("Titanic", 40704, 2, 4965305, "08416114dd85d0ebd932fcb1d87650c1"),
            (
                "Transfusion",
                1464,
                1,
                1586225,
                "c3242468edab8c2e7a907674122dc851",
            ),
            (
                "Contraceptive Method Choice",
                23,
                1,
                23,
                "3149646ecff276abac3e892d1556655f",
            ),
        )
        actual_sources = tuple(
            (
                spec.name,
                spec.openml_data_id,
                spec.openml_version,
                spec.openml_file_id,
                spec.md5_checksum,
            )
            for spec in real_datasets._REAL_DATASET_SPECS
        )
        self.assertEqual(actual_sources, expected_sources)
        self.assertEqual(
            REAL_DATASET_NAMES,
            tuple(source[0] for source in expected_sources),
        )
        expected_processing = (
            ("Abalone", "Class_number_of_rings", 4177, 4177, 2081, 8, ("Sex",)),
            ("Adult", "class", 48842, 45222, 11208, 14, (
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country",
            )),
            ("Alzheimer", "class", 174, 174, 89, 450, ()),
            ("Banknote Authentication", "Class", 1372, 1372, 610, 4, ()),
            ("Diabetes", "class", 768, 768, 268, 8, ()),
            ("Fertility", "Class", 100, 100, 12, 9, (
                "V1", "V3", "V4", "V5", "V6", "V7", "V8",
            )),
            ("Haberman", "Survival_status", 306, 306, 81, 3, (
                "Patients_year_of_operation",
            )),
            ("Mammographic", "Severity", 961, 830, 403, 4, (
                "Shape", "Margin",
            )),
            ("Sonar", "Class", 208, 208, 111, 60, ()),
            ("Titanic", "class", 2201, 2201, 711, 3, (
                "Class", "Age", "Sex",
            )),
            ("Transfusion", "Class", 748, 748, 178, 4, ()),
            (
                "Contraceptive Method Choice",
                "Contraceptive_method_used",
                1473,
                1473,
                844,
                9,
                (
                    "Wifes_education",
                    "Husbands_education",
                    "Wifes_religion",
                    "Wifes_now_working%3F",
                    "Husbands_occupation",
                    "Standard-of-living_index",
                    "Media_exposure",
                ),
            ),
        )
        actual_processing = tuple(
            (
                spec.name,
                spec.target_column,
                spec.expected_raw_rows,
                spec.expected_complete_rows,
                spec.expected_positive,
                spec.expected_feature_count,
                spec.expected_categorical_features,
            )
            for spec in real_datasets._REAL_DATASET_SPECS
        )
        self.assertEqual(actual_processing, expected_processing)
        for name, _, source_rows, complete_rows, positive, _, _ in (
            expected_processing
        ):
            metadata = real_dataset_metadata(name)
            self.assertEqual(metadata.name, name)
            self.assertEqual(metadata.n_source_rows, source_rows)
            self.assertEqual(metadata.n_complete_rows, complete_rows)
            self.assertEqual(metadata.n_positive, positive)
            self.assertEqual(
                metadata.n_negative,
                complete_rows - positive,
            )
            self.assertEqual(
                metadata.n_removed_rows,
                source_rows - complete_rows,
            )

    def test_all_fixed_positive_class_recodings(self):
        cases = {
            "Abalone": ([9, 10], [0, 1]),
            "Adult": (["<=50K", ">50K"], [0, 1]),
            "Alzheimer": (["H", "P"], [0, 1]),
            "Banknote Authentication": (["1", "2"], [0, 1]),
            "Diabetes": (
                ["tested_negative", "tested_positive"],
                [0, 1],
            ),
            "Fertility": (["1", "2"], [0, 1]),
            "Haberman": (["1", "2"], [0, 1]),
            "Mammographic": (["0", "1"], [0, 1]),
            "Sonar": (["Rock", "Mine"], [0, 1]),
            "Titanic": (["-1", "1"], [0, 1]),
            "Transfusion": (["1", "2"], [0, 1]),
            "Contraceptive Method Choice": (["1", "2", "3"], [0, 1, 1]),
        }

        for name, (raw_target, expected) in cases.items():
            with self.subTest(dataset=name):
                spec = next(
                    spec
                    for spec in real_datasets._REAL_DATASET_SPECS
                    if spec.name == name
                )
                encoded = real_datasets._encode_target(raw_target, spec)
                np.testing.assert_array_equal(encoded, expected)

        adult_spec = next(
            spec
            for spec in real_datasets._REAL_DATASET_SPECS
            if spec.name == "Adult"
        )
        with self.assertRaises(AssertionError):
            real_datasets._encode_target(["<=50K", "unexpected"], adult_spec)

    def test_cache_download_exact_split_and_train_only_encoding(self):
        spec = toy_spec()
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "new_data_directory"
            with (
                patch.object(
                    real_datasets,
                    "_SPECS_BY_KEY",
                    {"toy": spec},
                ),
                patch.object(
                    real_datasets,
                    "urlopen",
                    return_value=io.BytesIO(TOY_ARFF),
                ) as mocked_urlopen,
            ):
                dataset = prepare_real_dataset(
                    "Toy",
                    rng=IdentityPermutationRng(),
                    data_dir=data_dir,
                )

            mocked_urlopen.assert_called_once()
            self.assertTrue(data_dir.is_dir())
            self.assertEqual(dataset.source_path, data_dir / "toy.arff")
            self.assertEqual(dataset.source_path.read_bytes(), TOY_ARFF)
            self.assertEqual(dataset.n_source_rows, 21)
            self.assertEqual(dataset.n_complete_rows, 20)
            self.assertEqual(dataset.n_removed_rows, 1)
            self.assertEqual(dataset.openml_file_id, 123)
            self.assertEqual(dataset.md5_checksum, md5(TOY_ARFF).hexdigest())

            np.testing.assert_array_equal(dataset.test_row_indices, [0, 1, 10, 11])
            np.testing.assert_array_equal(
                dataset.external_row_indices,
                [2, 3, 12, 13],
            )
            np.testing.assert_array_equal(
                dataset.train_row_indices,
                [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19],
            )
            np.testing.assert_array_equal(dataset.y_test, [0, 0, 1, 1])
            np.testing.assert_array_equal(dataset.y_external, [0, 0, 1, 1])
            self.assertEqual(
                tuple(dataset.feature_names),
                ("number", "color_blue", "color_red"),
            )
            self.assertNotIn("color_green", dataset.feature_names)
            np.testing.assert_array_equal(dataset.X_test[0], [0.0, 0.0, 0.0])

            with (
                patch.object(
                    real_datasets,
                    "_SPECS_BY_KEY",
                    {"toy": spec},
                ),
                patch.object(
                    real_datasets,
                    "urlopen",
                    side_effect=AssertionError("cache was not reused"),
                ) as second_urlopen,
            ):
                cached_dataset = prepare_real_dataset(
                    "toy",
                    rng=IdentityPermutationRng(),
                    data_dir=data_dir,
                )

            second_urlopen.assert_not_called()
            np.testing.assert_array_equal(
                cached_dataset.X_train,
                dataset.X_train,
            )

    def test_seeded_split_is_reproducible(self):
        spec = toy_spec()
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "toy.arff").write_bytes(TOY_ARFF)
            with patch.object(
                real_datasets,
                "_SPECS_BY_KEY",
                {"toy": spec},
            ):
                first = prepare_real_dataset(
                    "Toy",
                    rng=np.random.default_rng(712),
                    data_dir=data_dir,
                )
                second = prepare_real_dataset(
                    "Toy",
                    rng=np.random.default_rng(712),
                    data_dir=data_dir,
                )

            for first_values, second_values in (
                (first.train_row_indices, second.train_row_indices),
                (first.test_row_indices, second.test_row_indices),
                (first.external_row_indices, second.external_row_indices),
                (first.X_train, second.X_train),
                (first.X_test, second.X_test),
                (first.X_external, second.X_external),
            ):
                np.testing.assert_array_equal(first_values, second_values)

    def test_loaded_source_is_immutable_and_reused_across_replications(self):
        spec = toy_spec()
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "toy.arff").write_bytes(TOY_ARFF)
            original_loader = real_datasets._load_complete_dataset

            with (
                patch.object(
                    real_datasets,
                    "_SPECS_BY_KEY",
                    {"toy": spec},
                ),
                patch.object(
                    real_datasets,
                    "_load_complete_dataset",
                    wraps=original_loader,
                ) as mocked_loader,
            ):
                loaded = load_real_dataset("Toy", data_dir=data_dir)
                self.assertIsInstance(loaded, LoadedRealDataset)
                self.assertEqual(mocked_loader.call_count, 1)

                raw_snapshot = loaded.X_raw.copy()
                target_snapshot = loaded.y.copy()
                row_snapshot = loaded.source_row_indices.copy()

                first = split_and_encode_real_dataset(
                    loaded,
                    rng=np.random.default_rng(712),
                )
                second = split_and_encode_real_dataset(
                    loaded,
                    rng=np.random.default_rng(713),
                )
                self.assertEqual(mocked_loader.call_count, 1)
                self.assertFalse(loaded.X_raw.flags.writeable)
                self.assertFalse(loaded.y.flags.writeable)
                self.assertFalse(loaded.source_row_indices.flags.writeable)
                np.testing.assert_array_equal(loaded.X_raw, raw_snapshot)
                np.testing.assert_array_equal(loaded.y, target_snapshot)
                np.testing.assert_array_equal(
                    loaded.source_row_indices,
                    row_snapshot,
                )
                self.assertFalse(
                    np.array_equal(
                        first.train_row_indices,
                        second.train_row_indices,
                    )
                )

                wrapper = prepare_real_dataset(
                    "Toy",
                    rng=np.random.default_rng(712),
                    data_dir=data_dir,
                )
                self.assertEqual(mocked_loader.call_count, 2)

            for direct_values, wrapper_values in (
                (first.train_row_indices, wrapper.train_row_indices),
                (first.test_row_indices, wrapper.test_row_indices),
                (first.external_row_indices, wrapper.external_row_indices),
                (first.X_train, wrapper.X_train),
                (first.X_test, wrapper.X_test),
                (first.X_external, wrapper.X_external),
                (first.y_train, wrapper.y_train),
                (first.y_test, wrapper.y_test),
                (first.y_external, wrapper.y_external),
            ):
                np.testing.assert_array_equal(direct_values, wrapper_values)

            with self.assertRaises(ValueError):
                loaded.y[0] = 1 - loaded.y[0]

    def test_classwise_split_uses_floors_with_imbalanced_remainders(self):
        y = np.asarray([0] * 11 + [1] * 13, dtype=np.int64)
        train, test, external = real_datasets._split_within_classes(
            y,
            np.random.default_rng(42),
        )

        for held_out in (test, external):
            self.assertEqual(np.count_nonzero(y[held_out] == 0), 2)
            self.assertEqual(np.count_nonzero(y[held_out] == 1), 2)
        self.assertEqual(np.count_nonzero(y[train] == 0), 7)
        self.assertEqual(np.count_nonzero(y[train] == 1), 9)
        combined = np.concatenate((train, test, external))
        np.testing.assert_array_equal(np.sort(combined), np.arange(y.size))

    def test_download_and_cache_failures_are_explicit(self):
        spec = toy_spec()
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (
                patch.object(
                    real_datasets,
                    "_SPECS_BY_KEY",
                    {"toy": spec},
                ),
                patch.object(
                    real_datasets,
                    "urlopen",
                    side_effect=OSError("network unavailable"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Could not download 'Toy'.*network unavailable",
                ):
                    prepare_real_dataset(
                        "Toy",
                        rng=IdentityPermutationRng(),
                        data_dir=data_dir,
                    )
            self.assertFalse((data_dir / "toy.arff").exists())
            self.assertFalse((data_dir / "toy.arff.part").exists())

            with (
                patch.object(
                    real_datasets,
                    "_SPECS_BY_KEY",
                    {"toy": spec},
                ),
                patch.object(
                    real_datasets,
                    "urlopen",
                    return_value=io.BytesIO(b"wrong download"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Downloaded MD5"):
                    prepare_real_dataset(
                        "Toy",
                        rng=IdentityPermutationRng(),
                        data_dir=data_dir,
                    )
            self.assertFalse((data_dir / "toy.arff").exists())
            self.assertFalse((data_dir / "toy.arff.part").exists())

            (data_dir / "toy.arff").write_bytes(b"corrupted")
            with patch.object(
                real_datasets,
                "_SPECS_BY_KEY",
                {"toy": spec},
            ):
                with self.assertRaisesRegex(RuntimeError, "has MD5"):
                    prepare_real_dataset(
                        "Toy",
                        rng=IdentityPermutationRng(),
                        data_dir=data_dir,
                    )

    def test_runtime_assertions_reject_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            prepare_real_dataset(
                "Unknown",
                rng=np.random.default_rng(1),
            )
        with self.assertRaises(AssertionError):
            prepare_real_dataset(
                "Adult",
                rng=None,
            )

        spec = toy_spec()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "not_a_directory"
            path.write_text("file", encoding="utf-8")
            with patch.object(
                real_datasets,
                "_SPECS_BY_KEY",
                {"toy": spec},
            ):
                with self.assertRaises(AssertionError):
                    prepare_real_dataset(
                        "Toy",
                        rng=IdentityPermutationRng(),
                        data_dir=path,
                    )

    @unittest.skipUnless(
        os.environ.get("BAYESIAN_ROC_NETWORK_TESTS") == "1",
        "set BAYESIAN_ROC_NETWORK_TESTS=1 for all twelve OpenML sources",
    )
    def test_all_pinned_openml_sources_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            for dataset_index, name in enumerate(REAL_DATASET_NAMES):
                with self.subTest(dataset=name):
                    loaded = load_real_dataset(
                        name,
                        data_dir=temporary_directory,
                    )
                    self.assertEqual(loaded.name, name)
                    self.assertEqual(
                        loaded.n_source_rows,
                        loaded.n_complete_rows + loaded.n_removed_rows,
                    )
                    for split_index in range(3):
                        with self.subTest(split=split_index):
                            dataset = split_and_encode_real_dataset(
                                loaded,
                                rng=np.random.default_rng(
                                    1000 * dataset_index + split_index
                                ),
                            )
                            scores = fit_logistic_regression_scores(dataset)
                            self.assertEqual(
                                scores.positive_test_scores.size,
                                np.count_nonzero(dataset.y_test == 1),
                            )
                            self.assertEqual(
                                scores.negative_test_scores.size,
                                np.count_nonzero(dataset.y_test == 0),
                            )
                            self.assertEqual(
                                scores.positive_external_scores.size,
                                np.count_nonzero(dataset.y_external == 1),
                            )
                            self.assertEqual(
                                scores.negative_external_scores.size,
                                np.count_nonzero(dataset.y_external == 0),
                            )

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.real_datasets"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "real_datasets relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
