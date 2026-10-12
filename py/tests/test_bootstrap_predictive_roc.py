"""Independent numerical contracts for the stratified predictive bootstrap."""

import itertools
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import roc_utils.utilities_experiments as experiment_utils  # noqa: E402
from roc_utils.bootstrap_predictive_roc_sampler import (  # noqa: E402
    sample_stratified_bootstrap_histograms,
)
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    grid_auc,
    roc_from_histograms,
    rocs_from_histogram_samples,
)
from roc_utils.semantic_rng import make_semantic_rng  # noqa: E402
from roc_utils.utilities_experiments import (  # noqa: E402
    build_bootstrap_predictive_band,
    build_posterior_predictive_band,
)


class RecordingMultinomialGenerator(np.random.Generator):
    """A real Generator that records the requested empirical multinomials."""

    def __init__(self, seed=17):
        super().__init__(np.random.PCG64(seed))
        self.calls = []

    def multinomial(self, n, pvals, size=None):
        self.calls.append((n, np.asarray(pvals).copy(), size))
        return super().multinomial(n, pvals, size=size)


class FixedMultinomialGenerator(np.random.Generator):
    """Inject count arrays to exercise the sampler's scientific assertions."""

    def __init__(self, output):
        super().__init__(np.random.PCG64(1))
        self.output = output

    def multinomial(self, n, pvals, size=None):
        return self.output


class BootstrapPredictiveRocTests(unittest.TestCase):
    def test_sampler_uses_exact_empirical_probabilities_and_class_sizes(self):
        positive = np.array([0, 1, 3])
        negative = np.array([2, 0, 1])
        originals = positive.copy(), negative.copy()
        rng = RecordingMultinomialGenerator()

        plus, minus = sample_stratified_bootstrap_histograms(
            positive, negative, 7, rng=rng,
        )

        self.assertEqual(len(rng.calls), 2)
        self.assertEqual((rng.calls[0][0], rng.calls[0][2]), (4, 7))
        self.assertEqual((rng.calls[1][0], rng.calls[1][2]), (3, 7))
        np.testing.assert_array_equal(rng.calls[0][1], [0.0, 0.25, 0.75])
        np.testing.assert_array_equal(rng.calls[1][1], [2 / 3, 0.0, 1 / 3])
        for draws, total in ((plus, 4), (minus, 3)):
            self.assertEqual(draws.shape, (7, 3))
            self.assertTrue(np.issubdtype(draws.dtype, np.integer))
            np.testing.assert_array_equal(draws.sum(axis=1), total)
            self.assertTrue(np.all(draws >= 0))
        np.testing.assert_array_equal(plus[:, 0], 0)
        np.testing.assert_array_equal(minus[:, 1], 0)
        np.testing.assert_array_equal(positive, originals[0])
        np.testing.assert_array_equal(negative, originals[1])

    def test_seed_reproducibility_and_roc_endpoints_under_imbalance(self):
        arguments = ([0, 1, 0, 0], [1, 2, 0, 4], 40)
        first = sample_stratified_bootstrap_histograms(
            *arguments, rng=np.random.default_rng(391),
        )
        second = sample_stratified_bootstrap_histograms(
            *arguments, rng=np.random.default_rng(391),
        )
        for first_counts, second_counts in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_counts, second_counts)
        np.testing.assert_array_equal(first[0], np.tile([0, 1, 0, 0], (40, 1)))
        np.testing.assert_array_equal(first[1][:, 2], 0)
        fpr, tpr = rocs_from_histogram_samples(*first)
        self.assertEqual(fpr.shape, (40, 5))
        self.assertEqual(tpr.shape, (40, 5))
        for coordinates in (fpr, tpr):
            np.testing.assert_array_equal(coordinates[:, 0], 0.0)
            np.testing.assert_array_equal(coordinates[:, -1], 1.0)
            self.assertTrue(np.all(np.diff(coordinates, axis=1) >= 0.0))

    def test_sixteen_index_resamples_give_the_exact_band_and_auc_distribution(self):
        # Four independent index draws, two per class: each tuple has mass 1/16.
        # This enumerates resampling individual observations, independently of
        # the multinomial implementation and the numerical quantile helper.
        plus, minus = [], []
        for indices in itertools.product((0, 1), repeat=4):
            positive_upper = indices[0] + indices[1]
            negative_upper = indices[2] + indices[3]
            plus.append([2 - positive_upper, positive_upper])
            minus.append([2 - negative_upper, negative_upper])
        with patch.object(
            experiment_utils,
            "sample_stratified_bootstrap_histograms",
            return_value=(np.asarray(plus), np.asarray(minus)),
        ) as sampler:
            band = build_bootstrap_predictive_band(
                np.array([1, 1]), np.array([1, 1]),
                S=16, levels=(0.80, 0.95), rng=np.random.default_rng(1),
            )

        sampler.assert_called_once()
        np.testing.assert_array_equal(band.fpr_center, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(band.tpr_center, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(
            np.sort(band.auc_samples),
            [0.0] + [0.25] * 4 + [0.5] * 6 + [0.75] * 4 + [1.0],
        )
        self.assertAlmostEqual(band.roc_radii[0.80], 1.0)
        self.assertAlmostEqual(band.roc_radii[0.95], 1.0)
        np.testing.assert_allclose(
            band.auc_intervals[0.80],
            [0.5 - np.sqrt(2.0) / 4.0, 0.5 + np.sqrt(2.0) / 4.0],
            rtol=0.0, atol=1e-15,
        )
        self.assertEqual(band.auc_intervals[0.95], (0.0, 1.0))

    def test_auc_expansion_is_centered_on_observed_auc_not_simulation_mean(self):
        # The observed AUC is 3/4; every supplied bootstrap AUC is 1/2.
        plus_draws = np.tile([2, 2], (20, 1))
        minus_draws = np.tile([2, 2], (20, 1))
        with patch.object(
            experiment_utils,
            "sample_stratified_bootstrap_histograms",
            return_value=(plus_draws, minus_draws),
        ):
            band = build_bootstrap_predictive_band(
                np.array([1, 3]), np.array([3, 1]),
                S=20, levels=(0.80, 0.95), rng=np.random.default_rng(2),
            )

        np.testing.assert_array_equal(band.fpr_center, [0.0, 0.25, 1.0])
        np.testing.assert_array_equal(band.tpr_center, [0.0, 0.75, 1.0])
        np.testing.assert_array_equal(band.auc_samples, 0.5)
        endpoint = 0.75 - np.sqrt(2.0) / 4.0
        for level in (0.80, 0.95):
            np.testing.assert_allclose(
                band.auc_intervals[level], [endpoint, endpoint],
                rtol=0.0, atol=1e-15,
            )
            self.assertAlmostEqual(band.roc_radii[level], 0.5)

    def test_exact_production_quantile_ranks_are_used_before_expansion(self):
        # AUC draws are exactly .4 + r/20000, r=1,...,4000. The 95% ranks
        # must be 100 and 3900, including binary floating-point level rounding.
        ranks = np.arange(1, 4001, dtype=np.int64)
        plus = np.column_stack((7000 - ranks, 3000 + ranks))
        minus = np.tile([5000, 5000], (4000, 1))
        with patch.object(
            experiment_utils,
            "sample_stratified_bootstrap_histograms",
            return_value=(plus, minus),
        ) as sampler:
            band = build_bootstrap_predictive_band(
                np.array([5000, 5000]), np.array([5000, 5000]),
                S=4000, levels=(0.95, 0.80), rng=np.random.default_rng(3),
            )

        sampler.assert_called_once()
        for level, lower, upper, radius in (
            (0.80, 0.42, 0.58, 0.16),
            (0.95, 0.405, 0.595, 0.19),
        ):
            np.testing.assert_allclose(
                band.auc_intervals[level],
                [0.5 + np.sqrt(2.0) * (lower - 0.5),
                 0.5 + np.sqrt(2.0) * (upper - 0.5)],
                rtol=0.0, atol=1e-14,
            )
            self.assertAlmostEqual(band.roc_radii[level], np.sqrt(2.0) * radius)

    def test_raw_radius_above_square_diameter_is_preserved(self):
        rare_draws = np.tile([0, 100], (20, 1))
        with patch.object(
            experiment_utils,
            "sample_stratified_bootstrap_histograms",
            return_value=(rare_draws, rare_draws.copy()),
        ):
            band = build_bootstrap_predictive_band(
                np.array([99, 1]), np.array([99, 1]),
                S=20, levels=(0.80, 0.95), rng=np.random.default_rng(4),
            )
        for radius in band.roc_radii.values():
            self.assertGreater(radius, np.sqrt(2.0))
            self.assertAlmostEqual(radius, 1.98)
        self.assertEqual(band.auc_intervals[0.80], (0.5, 0.5))

    def test_single_occupied_bins_produce_valid_degenerate_predictions(self):
        for positive, negative, expected_auc in (
            ([0, 1], [2, 0], 1.0),
            ([1, 0], [0, 2], 0.0),
            ([0, 1], [0, 2], 0.5),
        ):
            with self.subTest(positive=positive, negative=negative):
                band = build_bootstrap_predictive_band(
                    np.asarray(positive), np.asarray(negative),
                    S=40, levels=(0.80, 0.95), rng=np.random.default_rng(5),
                )
                for level in (0.80, 0.95):
                    self.assertEqual(band.roc_radii[level], 0.0)
                    self.assertEqual(
                        band.auc_intervals[level], (expected_auc, expected_auc),
                    )
                np.testing.assert_array_equal(band.auc_samples, expected_auc)

    def test_real_bootstrap_keeps_perfect_auc_predictions_inside_unit_interval(self):
        # Every bootstrap resample preserves perfect separation, hence its
        # exact AUC is 1. Seed 123 previously produced values 1+one ulp and
        # failed the constructor's strict AUC assertion at the official S.
        positive = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 20])
        negative = np.array([0, 0, 6, 5, 6, 1, 2, 0, 0, 0])
        originals = positive.copy(), negative.copy()
        positive.setflags(write=False)
        negative.setflags(write=False)

        band = build_bootstrap_predictive_band(
            positive, negative, S=4000, levels=(0.80, 0.95),
            rng=np.random.default_rng(123),
        )

        self.assertEqual(band.auc_samples.shape, (4000,))
        self.assertTrue(np.all(np.isfinite(band.auc_samples)))
        self.assertTrue(np.all((0.0 <= band.auc_samples) & (band.auc_samples <= 1.0)))
        np.testing.assert_allclose(
            band.auc_samples, 1.0, rtol=0.0, atol=2 * np.finfo(float).eps,
        )
        for interval in band.auc_intervals.values():
            self.assertTrue(0.0 <= interval[0] <= interval[1] <= 1.0)
            self.assertEqual(interval[1], 1.0)
            # In-range values 1-one ulp remain untouched by the correction.
            np.testing.assert_allclose(
                interval, (1.0, 1.0), rtol=0.0, atol=2 * np.finfo(float).eps,
            )
        np.testing.assert_array_equal(positive, originals[0])
        np.testing.assert_array_equal(negative, originals[1])

    def test_historical_replication_8421_auc_roundoff_regression_is_frozen(self):
        # Preserve the actual v2 failure independently of future scenario
        # changes. Bootstrap draw 396 has perfect separation; its uncorrected
        # floating trapezoidal AUC was 1.0000000000000002 on the failing run.
        positive = np.array([0, 0, 0, 2, 0, 0, 1, 6, 6, 5])
        negative = np.array([0, 4, 10, 3, 0, 1, 2, 0, 0, 0])
        originals = positive.copy(), negative.copy()
        for counts in (positive, negative):
            counts.setflags(write=False)
        stream_key = (20250731, 1, 3, 0, 8421)
        plus_draws, minus_draws = sample_stratified_bootstrap_histograms(
            positive, negative, 4000, rng=make_semantic_rng(*stream_key),
        )
        np.testing.assert_array_equal(plus_draws[396], [0, 0, 0, 0, 0, 0, 0, 9, 7, 4])
        np.testing.assert_array_equal(minus_draws[396], [0, 0, 11, 6, 0, 1, 2, 0, 0, 0])
        self.assertGreater(np.flatnonzero(plus_draws[396]).min(), np.flatnonzero(minus_draws[396]).max())
        fpr, tpr = rocs_from_histogram_samples(plus_draws, minus_draws)
        self.assertEqual(grid_auc(fpr, tpr)[396], 1.0)

        band = build_bootstrap_predictive_band(
            positive, negative, S=4000, levels=(0.80, 0.95),
            rng=make_semantic_rng(*stream_key),
        )
        self.assertEqual(band.auc_samples[396], 1.0)
        self.assertTrue(np.all((0.0 <= band.auc_samples) & (band.auc_samples <= 1.0)))
        for lower, upper in band.auc_intervals.values():
            self.assertTrue(0.0 <= lower <= upper <= 1.0)
        np.testing.assert_array_equal(positive, originals[0])
        np.testing.assert_array_equal(negative, originals[1])

    def test_dip_constructor_also_corrects_only_auc_integration_roundoff(self):
        positive = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 20])
        negative = np.array([0, 0, 6, 5, 6, 1, 2, 0, 0, 0])
        fpr, tpr = roc_from_histograms(positive, negative)
        samples = tuple(np.tile(values, (40, 1)) for values in (fpr, tpr))
        originals = tuple(values.copy() for values in samples)
        for values in samples:
            values.setflags(write=False)
        # These count-based ROC draws are possible under the positive
        # Dirichlet prior; fixing them isolates the downstream AUC path.
        with patch.object(
            experiment_utils, "sample_posterior_predictive_rocs", return_value=samples,
        ):
            band = build_posterior_predictive_band(
                positive, negative, np.full(10, 0.1), np.full(10, 0.1),
                S=40, levels=(0.80, 0.95), rng=np.random.default_rng(123),
            )

        np.testing.assert_array_equal(band.auc_samples, 1.0)
        for interval in band.auc_intervals.values():
            self.assertEqual(interval, (1.0, 1.0))
        for values, original in zip(samples, originals, strict=True):
            np.testing.assert_array_equal(values, original)
            self.assertFalse(values.flags.writeable)

    def test_invalid_sampler_and_band_inputs_are_rejected(self):
        invalid_histograms = (
            ([], []),
            ([1, 1], [1]),
            ([[1, 1]], [[1, 1]]),
            ([0, 0], [1, 1]),
            ([1, 1], [0, 0]),
            ([-1, 2], [1, 1]),
            ([0.5, 1.5], [1, 1]),
            ([1, np.nan], [1, 1]),
            ([1, 1], [1, np.inf]),
        )
        for positive, negative in invalid_histograms:
            with self.subTest(positive=positive, negative=negative):
                with self.assertRaises(ValueError):
                    sample_stratified_bootstrap_histograms(
                        positive, negative, 20, rng=np.random.default_rng(6),
                    )
        for invalid_s in (0, -1, 1.5, True, np.bool_(True)):
            with self.subTest(S=invalid_s):
                with self.assertRaises((TypeError, ValueError)):
                    sample_stratified_bootstrap_histograms(
                        [1, 1], [1, 1], invalid_s,
                        rng=np.random.default_rng(7),
                    )
        with self.assertRaises(AssertionError):
            sample_stratified_bootstrap_histograms([1, 1], [1, 1], 20, rng=None)
        with self.assertRaises(AssertionError):
            build_bootstrap_predictive_band(
                np.array([1, 1]), np.array([1, 1]),
                S=20, levels=(0.80, 0.95), rng=None,
            )
        for levels in ((), (0.8, 0.8), (0.0,), (1.0,), (np.nan,)):
            with self.subTest(levels=levels):
                with self.assertRaises(ValueError):
                    build_bootstrap_predictive_band(
                        np.array([1, 1]), np.array([1, 1]),
                        S=20, levels=levels, rng=np.random.default_rng(8),
                    )

    def test_sampler_checks_generated_shape_integer_counts_totals_and_empty_bins(self):
        for invalid_counts in (
            np.array([0, 2]),
            np.array([[0, 2], [0, 2], [0, 2]]),
            np.array([[0.0, 2.0], [0.0, 2.0]]),
            np.array([[0, -1], [0, 2]]),
            np.array([[0, 1], [0, 2]]),
            np.array([[1, 1], [0, 2]]),
        ):
            with self.subTest(counts=invalid_counts):
                with self.assertRaises(AssertionError):
                    sample_stratified_bootstrap_histograms(
                        [0, 2], [1, 1], 2,
                        rng=FixedMultinomialGenerator(invalid_counts),
                    )

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.bootstrap_predictive_roc_sampler"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", command],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("do not use python -O", completed.stderr)


if __name__ == "__main__":
    unittest.main()
