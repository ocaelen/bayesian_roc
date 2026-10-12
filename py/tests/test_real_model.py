import subprocess
import sys
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import roc_utils.real_model as real_model  # noqa: E402
from roc_utils.real_datasets import PreparedRealDataset  # noqa: E402
from roc_utils.real_model import fit_logistic_regression_scores  # noqa: E402


def _prepared_dataset() -> PreparedRealDataset:
    X_train = np.asarray(
        [
            [-2.0, 0.0],
            [-1.5, 1.0],
            [-1.0, 0.0],
            [-0.5, 1.0],
            [0.5, 0.0],
            [1.0, 1.0],
            [1.5, 0.0],
            [2.0, 1.0],
        ]
    )
    y_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    X_test = np.asarray(
        [[1.2, 0.0], [-1.2, 1.0], [1.8, 1.0], [-1.8, 0.0]]
    )
    y_test = np.asarray([1, 0, 1, 0], dtype=np.int64)
    X_external = np.asarray(
        [[-0.8, 0.0], [0.8, 1.0], [-1.6, 1.0], [1.6, 0.0]]
    )
    y_external = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return PreparedRealDataset(
        name="Toy",
        slug="toy",
        source="test fixture",
        source_path=Path("/unused/toy.arff"),
        openml_data_id=1,
        openml_version=1,
        openml_file_id=1,
        md5_checksum="0" * 32,
        n_source_rows=16,
        n_complete_rows=16,
        n_removed_rows=0,
        numeric_feature_names=("x",),
        categorical_feature_names=("category",),
        feature_names=("x", "category_a"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_external=X_external,
        y_external=y_external,
        train_row_indices=np.arange(8, dtype=np.int64),
        test_row_indices=np.arange(8, 12, dtype=np.int64),
        external_row_indices=np.arange(12, 16, dtype=np.int64),
    )


def _fake_classifier(dataset: PreparedRealDataset) -> Mock:
    classifier = Mock()
    classifier.classes_ = np.asarray([1, 0], dtype=np.int64)
    classifier.coef_ = np.asarray([[0.5, -0.25]])
    classifier.intercept_ = np.asarray([0.1])
    classifier.n_iter_ = np.asarray([7], dtype=np.int64)
    classifier.fit.return_value = classifier
    test_positive = np.asarray([0.8, 0.2, 0.7, 0.1])
    external_positive = np.asarray([0.3, 0.6, 0.4, 0.9])
    classifier.predict_proba.side_effect = (
        np.column_stack((test_positive, 1.0 - test_positive)),
        np.column_stack((external_positive, 1.0 - external_positive)),
    )
    return classifier


class RealModelTests(unittest.TestCase):
    def test_exact_model_contract_and_explicit_positive_class_column(self):
        dataset = _prepared_dataset()
        classifier = _fake_classifier(dataset)
        original_train = dataset.X_train.copy()
        original_test = dataset.X_test.copy()
        original_external = dataset.X_external.copy()

        with patch.object(
            real_model,
            "LogisticRegression",
            return_value=classifier,
        ) as constructor:
            scores = fit_logistic_regression_scores(dataset)

        constructor.assert_called_once_with(
            C=1.0,
            l1_ratio=0.0,
            solver="lbfgs",
            fit_intercept=True,
            class_weight=None,
            max_iter=10_000,
            tol=1e-8,
        )
        fit_X, fit_y = classifier.fit.call_args.args
        self.assertIsNot(fit_X, dataset.X_train)
        self.assertIs(fit_y, dataset.y_train)
        expected_train_numeric = (
            dataset.X_train[:, 0] - np.mean(dataset.X_train[:, 0])
        ) / np.std(dataset.X_train[:, 0])
        np.testing.assert_allclose(fit_X[:, 0], expected_train_numeric)
        np.testing.assert_array_equal(fit_X[:, 1], dataset.X_train[:, 1])
        self.assertEqual(classifier.predict_proba.call_count, 2)
        scored_test = classifier.predict_proba.call_args_list[0].args[0]
        scored_external = classifier.predict_proba.call_args_list[1].args[0]
        np.testing.assert_allclose(
            scored_test[:, 0],
            (dataset.X_test[:, 0] - np.mean(dataset.X_train[:, 0]))
            / np.std(dataset.X_train[:, 0]),
        )
        np.testing.assert_allclose(
            scored_external[:, 0],
            (dataset.X_external[:, 0] - np.mean(dataset.X_train[:, 0]))
            / np.std(dataset.X_train[:, 0]),
        )
        np.testing.assert_array_equal(scored_test[:, 1], dataset.X_test[:, 1])
        np.testing.assert_array_equal(
            scored_external[:, 1],
            dataset.X_external[:, 1],
        )
        np.testing.assert_array_equal(dataset.X_train, original_train)
        np.testing.assert_array_equal(dataset.X_test, original_test)
        np.testing.assert_array_equal(dataset.X_external, original_external)
        np.testing.assert_array_equal(scores.positive_test_scores, [0.8, 0.7])
        np.testing.assert_array_equal(scores.negative_test_scores, [0.2, 0.1])
        np.testing.assert_array_equal(
            scores.positive_external_scores,
            [0.6, 0.9],
        )
        np.testing.assert_array_equal(
            scores.negative_external_scores,
            [0.3, 0.4],
        )
        self.assertEqual(scores.n_iterations, 7)
        for values in (
            scores.positive_test_scores,
            scores.negative_test_scores,
            scores.positive_external_scores,
            scores.negative_external_scores,
        ):
            self.assertFalse(values.flags.writeable)

    def test_true_fit_is_deterministic_and_returns_valid_probabilities(self):
        dataset = _prepared_dataset()
        first = fit_logistic_regression_scores(dataset)
        second = fit_logistic_regression_scores(dataset)

        self.assertEqual(first.n_iterations, second.n_iterations)
        for first_values, second_values in zip(
            (
                first.positive_test_scores,
                first.negative_test_scores,
                first.positive_external_scores,
                first.negative_external_scores,
            ),
            (
                second.positive_test_scores,
                second.negative_test_scores,
                second.positive_external_scores,
                second.negative_external_scores,
            ),
        ):
            np.testing.assert_array_equal(first_values, second_values)
            self.assertTrue(np.all((0.0 <= first_values) & (first_values <= 1.0)))

    def test_zero_iterations_is_a_valid_sklearn_result(self):
        dataset = _prepared_dataset()
        classifier = _fake_classifier(dataset)
        classifier.n_iter_ = np.asarray([0], dtype=np.int64)

        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            scores = fit_logistic_regression_scores(dataset)

        self.assertEqual(scores.n_iterations, 0)

    def test_numeric_scaling_uses_training_only_and_handles_constants(self):
        base = _prepared_dataset()
        dataset = replace(
            base,
            numeric_feature_names=("x", "constant"),
            feature_names=("x", "constant", "category_a"),
            X_train=np.column_stack(
                (base.X_train[:, 0], np.full(base.y_train.size, 12.0),
                 base.X_train[:, 1])
            ),
            X_test=np.column_stack(
                (base.X_test[:, 0] * 1e6, np.full(base.y_test.size, 12.0),
                 base.X_test[:, 1])
            ),
            X_external=np.column_stack(
                (base.X_external[:, 0] * -1e6,
                 np.full(base.y_external.size, 12.0),
                 base.X_external[:, 1])
            ),
        )
        originals = tuple(
            X.copy()
            for X in (dataset.X_train, dataset.X_test, dataset.X_external)
        )

        X_train, X_test, X_external = real_model._model_matrices(dataset)

        np.testing.assert_allclose(np.mean(X_train[:, :2], axis=0), 0.0)
        self.assertAlmostEqual(float(np.var(X_train[:, 0])), 1.0)
        np.testing.assert_array_equal(X_train[:, 1], 0.0)
        self.assertGreater(np.max(np.abs(X_test[:, 0])), 1e5)
        self.assertGreater(np.max(np.abs(X_external[:, 0])), 1e5)
        for source, original, transformed in zip(
            (dataset.X_train, dataset.X_test, dataset.X_external),
            originals,
            (X_train, X_test, X_external),
        ):
            np.testing.assert_array_equal(source, original)
            np.testing.assert_array_equal(transformed[:, 2:], source[:, 2:])
            self.assertFalse(np.shares_memory(source, transformed))

    def test_invalid_classifier_outputs_are_rejected_without_clipping(self):
        dataset = _prepared_dataset()

        classifier = _fake_classifier(dataset)
        invalid = np.asarray(
            [[-0.001, 1.001], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
        )
        classifier.predict_proba.side_effect = (invalid, invalid)
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaises(AssertionError):
                fit_logistic_regression_scores(dataset)

        classifier = _fake_classifier(dataset)
        invalid = np.asarray(
            [[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.4]]
        )
        classifier.predict_proba.side_effect = (invalid, invalid)
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaises(AssertionError):
                fit_logistic_regression_scores(dataset)

        classifier = _fake_classifier(dataset)
        classifier.n_iter_ = np.asarray([1.5])
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaises(AssertionError):
                fit_logistic_regression_scores(dataset)

        classifier = _fake_classifier(dataset)
        classifier.classes_ = np.asarray([0, 0], dtype=np.int64)
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaises(AssertionError):
                fit_logistic_regression_scores(dataset)

        classifier = _fake_classifier(dataset)
        classifier.predict_proba.side_effect = (
            np.full((4, 3), 1.0 / 3.0),
            np.full((4, 3), 1.0 / 3.0),
        )
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaises(AssertionError):
                fit_logistic_regression_scores(dataset)

    def test_invalid_data_and_convergence_warnings_are_rejected(self):
        dataset = _prepared_dataset()
        dataset.X_train[0, 0] = np.nan
        with self.assertRaises(AssertionError):
            fit_logistic_regression_scores(dataset)

        dataset = _prepared_dataset()
        classifier = _fake_classifier(dataset)

        def warn_on_fit(*unused_arguments, **unused_keywords):
            warnings.warn("iteration limit", ConvergenceWarning)

        classifier.fit.side_effect = warn_on_fit
        with patch.object(real_model, "LogisticRegression", return_value=classifier):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_logistic_regression_scores(dataset)

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.real_model"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "real_model relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
