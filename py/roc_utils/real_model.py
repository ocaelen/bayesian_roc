"""Fit the fixed real-data classifier and return class-separated scores."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if not __debug__:
    raise RuntimeError(
        "real_model relies on runtime assertions; do not use python -O."
    )

from .real_datasets import PreparedRealDataset


MODEL_SPEC_VERSION = "logistic_l2_c1_lbfgs_numeric_standardization_v2"
MODEL_PENALTY = "l2"
MODEL_C = 1.0
# In scikit-learn 1.9, finite C with l1_ratio=0 is the warning-free pure-L2
# parameterization; passing penalty="l2" is deprecated.
MODEL_L1_RATIO = 0.0
MODEL_SOLVER = "lbfgs"
MODEL_FIT_INTERCEPT = True
MODEL_CLASS_WEIGHT = None
MODEL_MAX_ITER = 10_000
MODEL_TOL = 1e-8
NUMERIC_SCALING = "training_only_numeric_standardization"


@dataclass(frozen=True)
class RealDatasetScores:
    """Held-out positive-class probabilities from one fitted classifier."""

    positive_test_scores: np.ndarray
    negative_test_scores: np.ndarray
    positive_external_scores: np.ndarray
    negative_external_scores: np.ndarray
    n_iterations: int

    def __post_init__(self) -> None:
        for scores in (
            self.positive_test_scores,
            self.negative_test_scores,
            self.positive_external_scores,
            self.negative_external_scores,
        ):
            _assert_score_vector(scores)
            assert not scores.flags.writeable
        assert type(self.n_iterations) is int
        # scikit-learn may report zero iterations when the initial parameters
        # already satisfy the stopping criterion.
        assert 0 <= self.n_iterations <= MODEL_MAX_ITER


def _assert_score_vector(scores: np.ndarray) -> None:
    """Assert one non-empty finite vector of probabilities."""

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert scores.size > 0
    assert np.all(np.isfinite(scores))
    assert np.all((0.0 <= scores) & (scores <= 1.0))


def _immutable_scores(values: np.ndarray) -> np.ndarray:
    """Return a read-only floating-point copy of one score vector."""

    values = np.asarray(values, dtype=float).copy()
    values.setflags(write=False)
    _assert_score_vector(values)
    return values


def _assert_prepared_dataset(dataset: PreparedRealDataset) -> None:
    """Assert the model-facing subset of the prepared-data contract."""

    assert isinstance(dataset, PreparedRealDataset)
    feature_count = dataset.X_train.shape[1]
    assert feature_count > 0
    assert feature_count == len(dataset.feature_names)

    for X, y in (
        (dataset.X_train, dataset.y_train),
        (dataset.X_test, dataset.y_test),
        (dataset.X_external, dataset.y_external),
    ):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape == (y.size, feature_count)
        assert np.all(np.isfinite(X))
        assert set(np.unique(y)) == {0, 1}

    for label in (0, 1):
        assert np.count_nonzero(dataset.y_test == label) == np.count_nonzero(
            dataset.y_external == label
        )


def _model_matrices(
    dataset: PreparedRealDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize training numeric columns and reuse that transformation."""

    n_numeric = len(dataset.numeric_feature_names)
    assert 0 <= n_numeric <= dataset.X_train.shape[1]
    assert dataset.feature_names[:n_numeric] == dataset.numeric_feature_names

    matrices = tuple(
        np.asarray(X, dtype=float).copy()
        for X in (dataset.X_train, dataset.X_test, dataset.X_external)
    )
    for source, matrix in zip(
        (dataset.X_train, dataset.X_test, dataset.X_external),
        matrices,
    ):
        assert matrix.shape == source.shape
        assert not np.shares_memory(matrix, source)

    if n_numeric:
        scaler = StandardScaler()
        matrices[0][:, :n_numeric] = scaler.fit_transform(
            dataset.X_train[:, :n_numeric]
        )
        matrices[1][:, :n_numeric] = scaler.transform(
            dataset.X_test[:, :n_numeric]
        )
        matrices[2][:, :n_numeric] = scaler.transform(
            dataset.X_external[:, :n_numeric]
        )

        assert scaler.mean_.shape == (n_numeric,)
        assert scaler.var_.shape == (n_numeric,)
        assert scaler.scale_.shape == (n_numeric,)
        assert np.all(np.isfinite(scaler.mean_))
        assert np.all(np.isfinite(scaler.var_))
        assert np.all(np.isfinite(scaler.scale_))
        assert np.all(scaler.var_ >= 0.0)
        assert np.all(scaler.scale_ > 0.0)
        np.testing.assert_allclose(
            np.mean(matrices[0][:, :n_numeric], axis=0),
            0.0,
            rtol=0.0,
            atol=1e-10,
        )
        nonconstant = scaler.var_ > 0.0
        np.testing.assert_allclose(
            np.var(matrices[0][:, :n_numeric][:, nonconstant], axis=0),
            1.0,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            matrices[0][:, :n_numeric][:, ~nonconstant],
            0.0,
            rtol=0.0,
            atol=0.0,
        )

    for source, matrix in zip(
        (dataset.X_train, dataset.X_test, dataset.X_external),
        matrices,
    ):
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(
            matrix[:, n_numeric:],
            source[:, n_numeric:],
        )
    return matrices


def _positive_probabilities(
    classifier: LogisticRegression,
    X: np.ndarray,
    positive_column: int,
) -> np.ndarray:
    """Return validated P(Y=1) values without silently clipping them."""

    probabilities = np.asarray(classifier.predict_proba(X), dtype=float)
    assert probabilities.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    np.testing.assert_allclose(
        np.sum(probabilities, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    return probabilities[:, positive_column]


def fit_logistic_regression_scores(
    dataset: PreparedRealDataset,
) -> RealDatasetScores:
    """Fit one numeric-standardized L2 model and score both held-out samples."""

    _assert_prepared_dataset(dataset)
    X_train, X_test, X_external = _model_matrices(dataset)
    classifier = LogisticRegression(
        C=MODEL_C,
        l1_ratio=MODEL_L1_RATIO,
        solver=MODEL_SOLVER,
        fit_intercept=MODEL_FIT_INTERCEPT,
        class_weight=MODEL_CLASS_WEIGHT,
        max_iter=MODEL_MAX_ITER,
        tol=MODEL_TOL,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            classifier.fit(X_train, dataset.y_train)
    except ConvergenceWarning as error:
        raise RuntimeError(
            f"Logistic regression did not converge: {error}"
        ) from error
    except Warning as error:
        warning_name = type(error).__name__
        raise RuntimeError(
            f"Logistic regression emitted {warning_name}: {error}"
        ) from error

    classes = np.asarray(classifier.classes_)
    assert classes.shape == (2,)
    assert set(classes.tolist()) == {0, 1}
    positive_columns = np.flatnonzero(classes == 1)
    assert positive_columns.shape == (1,)
    positive_column = int(positive_columns[0])

    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    assert coefficients.shape == (1, dataset.X_train.shape[1])
    assert intercept.shape == (1,)
    assert np.all(np.isfinite(coefficients))
    assert np.all(np.isfinite(intercept))

    iterations = np.asarray(classifier.n_iter_)
    assert iterations.shape == (1,)
    assert np.isfinite(iterations[0])
    n_iterations = int(iterations[0])
    assert iterations[0] == n_iterations
    assert 0 <= n_iterations <= MODEL_MAX_ITER

    test_probabilities = _positive_probabilities(
        classifier,
        X_test,
        positive_column,
    )
    external_probabilities = _positive_probabilities(
        classifier,
        X_external,
        positive_column,
    )

    scores = RealDatasetScores(
        positive_test_scores=_immutable_scores(
            test_probabilities[dataset.y_test == 1]
        ),
        negative_test_scores=_immutable_scores(
            test_probabilities[dataset.y_test == 0]
        ),
        positive_external_scores=_immutable_scores(
            external_probabilities[dataset.y_external == 1]
        ),
        negative_external_scores=_immutable_scores(
            external_probabilities[dataset.y_external == 0]
        ),
        n_iterations=n_iterations,
    )
    assert scores.positive_test_scores.size == scores.positive_external_scores.size
    assert scores.negative_test_scores.size == scores.negative_external_scores.size
    return scores
