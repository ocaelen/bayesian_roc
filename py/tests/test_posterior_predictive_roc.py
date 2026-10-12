import itertools
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import roc_utils.posterior_predictive_roc_sampler as roc_sampler  # noqa: E402
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    curve_max_euclidean_deviation,
    equal_tailed_interval,
    fixed_uniform_grid,
    grid_auc,
    histogram_from_scores,
    histograms_from_scores,
    linear_informative_priors,
    max_euclidean_deviations,
    nearest_rank_quantile,
    posterior_predictive_mean_roc,
    right_tail_sums,
    roc_from_histograms,
    rocs_from_histogram_samples,
    sample_posterior_predictive_histograms,
    sample_posterior_predictive_rocs,
    symmetric_dirichlet_priors,
    validate_grid,
)


class ScriptedPosteriorGenerator(np.random.Generator):
    """Record distribution parameters while supplying fixed possible draws."""

    def __init__(self, probability_draws, count_draws):
        super().__init__(np.random.PCG64(19))
        self.probability_draws = probability_draws
        self.count_draws = count_draws
        self.dirichlet_calls = []
        self.binomial_calls = []

    def dirichlet(self, alpha, size=None):
        self.dirichlet_calls.append((np.asarray(alpha).copy(), size))
        return np.asarray(
            self.probability_draws[len(self.dirichlet_calls) - 1], dtype=float,
        ).copy()

    def binomial(self, n, p, size=None):
        self.binomial_calls.append((np.asarray(n).copy(), np.asarray(p).copy(), size))
        return np.asarray(
            self.count_draws[len(self.binomial_calls) - 1], dtype=np.int64,
        ).copy()


class PosteriorPredictiveRocTests(unittest.TestCase):
    def test_fixed_uniform_grid_and_bin_convention(self):
        gamma = fixed_uniform_grid(2)
        np.testing.assert_allclose(gamma, [0.0, 0.5, 1.0])

        counts = histogram_from_scores([0.0, 0.25, 0.5, 0.75, 1.0], gamma)
        np.testing.assert_array_equal(counts, [2, 3])

        y_true = np.array([1, 0, 1, 0, 1])
        d_plus, d_minus = histograms_from_scores(
            y_true,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            gamma,
        )
        np.testing.assert_array_equal(d_plus, [1, 2])
        np.testing.assert_array_equal(d_minus, [1, 1])

    def test_right_tail_sums_and_roc_ordering(self):
        np.testing.assert_allclose(right_tail_sums([1, 2, 3]), [6, 5, 3, 0])

        fpr, tpr = roc_from_histograms([1, 2, 3], [3, 2, 1])
        np.testing.assert_allclose(fpr, [0.0, 1.0 / 6.0, 3.0 / 6.0, 1.0])
        np.testing.assert_allclose(tpr, [0.0, 3.0 / 6.0, 5.0 / 6.0, 1.0])

    def test_grid_roc_matches_threshold_decisions_including_scores_equal_to_one(self):
        # The manuscript's R_K is the endpoint at a threshold ABOVE 1,
        # not the empirical ROC at gamma_K=1 when some scores equal 1.
        gamma = np.array([0.0, 0.2, 0.6, 1.0])
        positive = np.array([0.0, 0.2, 0.59, 0.6, 1.0, 1.0])
        negative = np.array([0.0, 0.19, 0.2, 0.6, 0.99, 1.0])
        thresholds = [np.nextafter(1.0, np.inf), 0.6, 0.2, 0.0]
        d_plus = histogram_from_scores(positive, gamma)
        d_minus = histogram_from_scores(negative, gamma)
        fpr, tpr = roc_from_histograms(d_plus, d_minus)

        np.testing.assert_array_equal(
            fpr, [np.mean(negative >= threshold) for threshold in thresholds],
        )
        np.testing.assert_array_equal(
            tpr, [np.mean(positive >= threshold) for threshold in thresholds],
        )
        self.assertGreater(np.mean(positive >= 1.0), tpr[0])
        self.assertGreater(np.mean(negative >= 1.0), fpr[0])
        np.testing.assert_allclose(np.diff(tpr), d_plus[::-1] / positive.size)
        np.testing.assert_allclose(np.diff(fpr), d_minus[::-1] / negative.size)

    def test_grid_auc_matches_exact_pairwise_bin_order_for_all_small_samples(self):
        # An independent integer-count reference: each positive/negative
        # pair contributes 1 for a higher positive bin, 1/2 for a shared bin.
        # Enumerating all bin multisets includes ties, empty bins, imbalance,
        # perfect and reversed separation, without relying on ROC geometry.
        for K in (2, 3, 4):
            for n_plus, n_minus in ((1, 1), (2, 3), (3, 2)):
                for positive in itertools.combinations_with_replacement(range(K), n_plus):
                    for negative in itertools.combinations_with_replacement(range(K), n_minus):
                        with self.subTest(K=K, positive=positive, negative=negative):
                            twice_wins = sum(
                                2 * (p > n) + (p == n)
                                for p in positive for n in negative
                            )
                            expected = Fraction(twice_wins, 2 * n_plus * n_minus)
                            fpr, tpr = roc_from_histograms(
                                np.bincount(positive, minlength=K),
                                np.bincount(negative, minlength=K),
                            )
                            self.assertAlmostEqual(float(grid_auc(fpr, tpr)), float(expected), places=15)

    def test_perfect_auc_roundoff_is_bounded_for_scalar_batch_and_strided_inputs(self):
        # All positive bins exceed all negative bins, so every pair wins:
        # the exact AUC is 1. The floating trapezoidal sum used to exceed it
        # by one ulp for these counts, despite completely valid ROC points.
        fpr, tpr = roc_from_histograms(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 20],
            [0, 0, 6, 5, 6, 1, 2, 0, 0, 0],
        )
        for shape in ("scalar", "batch", "strided"):
            with self.subTest(shape=shape):
                if shape == "scalar":
                    inputs = (fpr, tpr)
                elif shape == "batch":
                    inputs = tuple(np.tile(values, (3, 1)) for values in (fpr, tpr))
                else:
                    buffers = [np.full((3, 2 * fpr.size), np.nan) for _ in range(2)]
                    for buffer, values in zip(buffers, (fpr, tpr), strict=True):
                        buffer[:, ::2] = values
                    inputs = tuple(buffer[:, ::2] for buffer in buffers)
                    self.assertFalse(inputs[0].flags.c_contiguous)
                originals = tuple(values.copy() for values in inputs)
                for values in inputs:
                    values.setflags(write=False)
                actual = grid_auc(*inputs)
                self.assertEqual(np.shape(actual), () if shape == "scalar" else (3,))
                np.testing.assert_array_equal(actual, 1.0)
                for values, original in zip(inputs, originals, strict=True):
                    np.testing.assert_array_equal(values, original)
                    self.assertFalse(values.flags.writeable)

    def test_grid_auc_preserves_interior_values_close_to_both_boundaries(self):
        # A horizontal ROC segment of height a and width 1 has exact area a.
        # Legitimate values near the boundaries must not be rounded to 0/1.
        epsilon = np.finfo(float).eps
        expected = np.array([epsilon, 1e-14, 0.25, 0.5, 1 - 1e-14, 1 - epsilon])
        fpr = np.tile([0.0, 0.0, 1.0, 1.0], (expected.size, 1))
        tpr = np.column_stack((np.zeros(expected.size), expected, expected, np.ones(expected.size)))
        np.testing.assert_array_equal(grid_auc(fpr, tpr), expected)
        for actual_fpr, actual_tpr, area in zip(fpr, tpr, expected, strict=True):
            self.assertEqual(float(grid_auc(actual_fpr, actual_tpr)), area)

    def test_grid_auc_rejects_invalid_integrals_instead_of_clipping_them(self):
        # A failed numerical routine must remain detectable even for valid
        # ROC inputs; the boundary correction is only for tiny roundoff.
        for invalid in (-1e-6, 1.0 + 1e-6, np.nan, -np.inf, np.inf):
            for batch in (False, True):
                with self.subTest(invalid=invalid, batch=batch):
                    coordinates = [[0.0, 1.0], [0.0, 1.0]] if batch else [0.0, 1.0]
                    result = np.array([0.5, invalid]) if batch else invalid
                    with patch.object(roc_sampler.np, "trapezoid", return_value=result):
                        with self.assertRaises(ValueError):
                            grid_auc(coordinates, coordinates)

    def test_grid_auc_still_rejects_invalid_roc_before_integration(self):
        invalid_pairs = (
            ([-0.01, 1.0], [0.0, 1.0]),
            ([0.0, 1.01, 1.0], [0.0, 0.5, 1.0]),
            ([0.0, 0.5, 1.0], [0.0, 1.01, 1.0]),
            ([0.0, 0.6, 0.4, 1.0], [0.0, 0.5, 0.6, 1.0]),
            ([0.0, 0.5, 1.0], [0.0, 0.7, 0.6]),
            ([0.01, 1.0], [0.0, 1.0]),
            ([0.0, 0.99], [0.0, 1.0]),
            ([0.0, np.nan, 1.0], [0.0, 0.5, 1.0]),
        )
        for fpr, tpr in invalid_pairs:
            with self.subTest(fpr=fpr, tpr=tpr):
                with patch.object(roc_sampler.np, "trapezoid", return_value=0.5) as integrate:
                    with self.assertRaises(ValueError):
                        grid_auc(fpr, tpr)
                    integrate.assert_not_called()

    def test_priors_match_manuscript_formulas(self):
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(4, 2.0)
        np.testing.assert_allclose(alpha_plus, [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(alpha_minus, [0.5, 0.5, 0.5, 0.5])

        alpha_plus, alpha_minus = linear_informative_priors(4, 10.0)
        np.testing.assert_allclose(alpha_plus, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(alpha_minus, [4.0, 3.0, 2.0, 1.0])

    def test_posterior_predictive_mean_curve(self):
        d_plus = np.array([0, 10])
        d_minus = np.array([10, 0])
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(2, 2.0)

        fpr_center, tpr_center = posterior_predictive_mean_roc(
            d_plus,
            d_minus,
            alpha_plus,
            alpha_minus,
        )

        np.testing.assert_allclose(fpr_center, [0.0, 1.0 / 12.0, 1.0])
        np.testing.assert_allclose(tpr_center, [0.0, 11.0 / 12.0, 1.0])

    def test_posterior_predictive_sampler_shapes_and_endpoints(self):
        d_plus = np.array([2, 4, 6])
        d_minus = np.array([6, 4, 2])
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(3, 3.0)
        rng = np.random.default_rng(123)

        d_plus_tilde, d_minus_tilde = sample_posterior_predictive_histograms(
            d_plus,
            d_minus,
            alpha_plus,
            alpha_minus,
            S=25,
            rng=rng,
        )

        self.assertEqual(d_plus_tilde.shape, (25, 3))
        self.assertEqual(d_minus_tilde.shape, (25, 3))
        np.testing.assert_array_equal(d_plus_tilde.sum(axis=1), d_plus.sum())
        np.testing.assert_array_equal(d_minus_tilde.sum(axis=1), d_minus.sum())

        fpr_samples, tpr_samples = sample_posterior_predictive_rocs(
            d_plus,
            d_minus,
            alpha_plus,
            alpha_minus,
            S=25,
            seed=456,
        )

        self.assertEqual(fpr_samples.shape, (25, 4))
        self.assertEqual(tpr_samples.shape, (25, 4))
        np.testing.assert_allclose(fpr_samples[:, 0], 0.0)
        np.testing.assert_allclose(tpr_samples[:, 0], 0.0)
        np.testing.assert_allclose(fpr_samples[:, -1], 1.0)
        np.testing.assert_allclose(tpr_samples[:, -1], 1.0)
        self.assertTrue(np.all(np.diff(fpr_samples, axis=1) >= -1e-12))
        self.assertTrue(np.all(np.diff(tpr_samples, axis=1) >= -1e-12))

    def test_posterior_draws_feed_classwise_predictive_counts_with_unequal_sizes(self):
        # Three different latent draws must each drive their own future sample;
        # replacing them by a common posterior mean would pass shape checks.
        rng = ScriptedPosteriorGenerator(
            probability_draws=(
                [[1 / 4, 1 / 4, 1 / 2], [1 / 2, 1 / 4, 1 / 4], [1 / 8, 3 / 8, 1 / 2]],
                [[1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 2, 1 / 4], [1 / 8, 1 / 8, 3 / 4]],
            ),
            count_draws=([1, 2, 0], [1, 1, 2], [3, 1, 1], [1, 3, 1]),
        )
        plus, minus = sample_posterior_predictive_histograms(
            [1, 2, 1], [3, 1, 2], [1 / 4, 1 / 2, 3 / 4], [1 / 2, 1 / 4, 3 / 4],
            S=3, rng=rng,
        )
        self.assertEqual(len(rng.dirichlet_calls), 2)
        for (actual_alpha, size), expected_alpha in zip(
            rng.dirichlet_calls, ([5 / 4, 5 / 2, 7 / 4], [7 / 2, 5 / 4, 11 / 4]),
            strict=True,
        ):
            np.testing.assert_array_equal(actual_alpha, expected_alpha)
            self.assertEqual(size, 3)
        expected_calls = (
            ([4, 4, 4], [1 / 4, 1 / 2, 1 / 8]),
            ([3, 2, 4], [1 / 3, 1 / 2, 3 / 7]),
            ([6, 6, 6], [1 / 2, 1 / 4, 1 / 8]),
            ([3, 5, 5], [1 / 2, 2 / 3, 1 / 7]),
        )
        self.assertEqual(len(rng.binomial_calls), 4)
        for (n, p, size), (expected_n, expected_p) in zip(
            rng.binomial_calls, expected_calls, strict=True,
        ):
            np.testing.assert_array_equal(n, expected_n)
            np.testing.assert_allclose(p, expected_p, rtol=0.0, atol=1e-15)
            self.assertIsNone(size)
        np.testing.assert_array_equal(plus, [[1, 1, 2], [2, 1, 1], [0, 2, 2]])
        np.testing.assert_array_equal(minus, [[3, 1, 2], [1, 3, 2], [1, 1, 4]])

    def test_sequential_binomials_give_the_exact_multinomial_law(self):
        # Enumerate every three-bin outcome and multiply the probabilities
        # requested from the RNG. Compare with independent multinomial masses.
        n = 4
        outcomes = np.array([
            [first, second, n - first - second]
            for first in range(n + 1) for second in range(n - first + 1)
        ])
        fractions = (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2))
        probabilities = np.tile([float(p) for p in fractions], (len(outcomes), 1))
        rng = ScriptedPosteriorGenerator((), (outcomes[:, 0], outcomes[:, 1]))
        actual = roc_sampler._multinomial_rows_from_probabilities(rng, n, probabilities)
        np.testing.assert_array_equal(actual, outcomes)
        path_masses = np.ones(len(outcomes))
        for column, (remaining, conditional, _) in enumerate(rng.binomial_calls):
            for row, (count, total, probability) in enumerate(zip(
                outcomes[:, column], remaining, conditional, strict=True,
            )):
                path_masses[row] *= (
                    math.comb(int(total), int(count))
                    * probability ** count * (1 - probability) ** (total - count)
                )
        exact_masses = []
        for counts in outcomes:
            coefficient = math.factorial(n) // math.prod(math.factorial(int(x)) for x in counts)
            mass = coefficient * math.prod(p ** int(x) for p, x in zip(fractions, counts, strict=True))
            exact_masses.append(float(mass))
        np.testing.assert_allclose(path_masses, exact_masses, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(path_masses.sum()), 1.0, places=15)

    def test_predictive_beta_binomial_law_by_exact_polynomial_quadrature(self):
        # For d+=(2,3), alpha+=(1,1), the latent first-bin probability is
        # Beta(3,4), with density 60*p**2*(1-p)**3. Multiplying by a Bin(5,p)
        # mass gives a degree-10 polynomial, integrated exactly by six-point
        # Gauss-Legendre quadrature. No Monte Carlo tolerance is needed.
        nodes, weights = np.polynomial.legendre.leggauss(6)
        nodes, weights = (nodes + 1) / 2, weights / 2
        latent = np.repeat(nodes, 6)
        count_draws = np.tile(np.arange(6), 6)
        probabilities = np.column_stack((latent, 1 - latent))
        rng = ScriptedPosteriorGenerator(
            (probabilities, probabilities.copy()), (count_draws, count_draws.copy()),
        )
        plus, _ = sample_posterior_predictive_histograms(
            [2, 3], [3, 2], [1, 1], [1, 1], S=36, rng=rng,
        )
        totals, requested_p, _ = rng.binomial_calls[0]
        np.testing.assert_array_equal(totals, 5)
        np.testing.assert_array_equal(plus[:, 0], count_draws)
        conditional_masses = np.array([
            math.comb(5, int(count)) * p ** count * (1 - p) ** (5 - count)
            for count, p in zip(count_draws, requested_p, strict=True)
        ]).reshape(6, 6)
        masses = np.sum(
            (weights * 60 * nodes**2 * (1 - nodes)**3)[:, None] * conditional_masses,
            axis=0,
        )
        exact = [float(Fraction(
            60 * math.comb(5, count) * math.factorial(count + 2) * math.factorial(8 - count),
            math.factorial(11),
        )) for count in range(6)]
        np.testing.assert_allclose(masses, exact, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(masses.sum()), 1.0, places=14)
        values = np.arange(6)
        self.assertAlmostEqual(float(masses @ values), 15 / 7, places=14)
        variance = float(masses @ (values - 15 / 7) ** 2)
        self.assertAlmostEqual(variance, 90 / 49, places=14)
        # A plug-in Bin(5,3/7) would instead have variance 60/49.
        self.assertGreater(variance, 60 / 49)

    def test_sequential_sampler_handles_zero_and_exhausted_probabilities(self):
        # Finite-precision Dirichlet draws can contain zeros. In particular,
        # the remaining probability can be exhausted before the final bin.
        probabilities = np.eye(3)
        counts = roc_sampler._multinomial_rows_from_probabilities(
            np.random.default_rng(617), 7, probabilities,
        )
        np.testing.assert_array_equal(counts, 7 * np.eye(3, dtype=int))

    def test_posterior_sampler_rejects_invalid_generated_histograms(self):
        # Normalizing an incorrect total would still make a plausible ROC,
        # but would silently change the future class sizes in the model.
        valid = np.tile([1, 2], (3, 1))
        invalid_draws = (
            np.tile([1, 3], (3, 1)),
            np.tile([1, 2], (2, 1)),
            np.ones((3, 3), dtype=int),
            valid.astype(float),
            np.tile([0.5, 2.5], (3, 1)),
            np.tile([-1, 4], (3, 1)),
            np.tile([np.nan, 3.0], (3, 1)),
        )
        for bad in invalid_draws:
            for invalid_class in ("positive", "negative"):
                with self.subTest(draws=bad, invalid_class=invalid_class):
                    outputs = (bad, valid) if invalid_class == "positive" else (valid, bad)
                    with patch.object(
                        roc_sampler, "_multinomial_rows_from_probabilities", side_effect=outputs,
                    ):
                        with self.assertRaises(AssertionError):
                            sample_posterior_predictive_rocs(
                                [1, 2], [2, 1], [1.0, 1.0], [1.0, 1.0],
                                S=3, rng=np.random.default_rng(123),
                            )

    def test_auc_and_nearest_rank_intervals(self):
        fpr, tpr = roc_from_histograms([0, 10], [10, 0])
        self.assertAlmostEqual(float(grid_auc(fpr, tpr)), 1.0)

        fpr, tpr = roc_from_histograms([5, 5], [5, 5])
        self.assertAlmostEqual(float(grid_auc(fpr, tpr)), 0.5)

        values = np.arange(1, 101)
        self.assertEqual(nearest_rank_quantile(values, 0.75), 75)
        self.assertEqual(equal_tailed_interval(values, 0.90), (5, 95))

    def test_production_auc_interval_uses_exact_nearest_ranks(self):
        values = np.arange(1, 4001)
        lower_probability = (1.0 - 0.95) / 2.0

        self.assertGreater(lower_probability * values.size, 100.0)
        self.assertEqual(
            nearest_rank_quantile(values, lower_probability),
            100,
        )
        self.assertEqual(
            equal_tailed_interval(values, level=0.95),
            (100, 3900),
        )
        self.assertEqual(nearest_rank_quantile(values, 0.95), 3800)

        genuinely_above_rank_100 = (100.0 + 1e-10) / values.size
        self.assertEqual(
            nearest_rank_quantile(values, genuinely_above_rank_100),
            101,
        )

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            fixed_uniform_grid(0)
        with self.assertRaises(ValueError):
            histogram_from_scores([-0.1], fixed_uniform_grid(2))
        with self.assertRaises(ValueError):
            histogram_from_scores([np.nan], fixed_uniform_grid(2))
        with self.assertRaises(ValueError):
            symmetric_dirichlet_priors(2, 0.0)
        for invalid_mass in (np.nan, np.inf, -np.inf):
            with self.subTest(prior_mass=invalid_mass):
                with self.assertRaises(ValueError):
                    symmetric_dirichlet_priors(2, invalid_mass)
                with self.assertRaises(ValueError):
                    linear_informative_priors(2, invalid_mass)
        with self.assertRaises(ValueError):
            roc_from_histograms([0, 0], [1, 1])
        with self.assertRaises(ValueError):
            grid_auc([0.0, 0.7, 0.6], [0.0, 0.8, 1.0])
        with self.assertRaises(ValueError):
            validate_grid([-1e-9, 0.5, 1.0])

        invalid_roc_pairs = (
            ([0.0, np.nan, 1.0], [0.0, 0.5, 1.0]),
            ([0.0, 0.5, 1.0], [0.0, np.inf, 1.0]),
            ([], []),
            ([0.0], [0.0]),
            ([0.0, 0.5, 1.0], [0.0, 0.7, 0.6]),
            ([0.1, 1.0], [0.0, 1.0]),
            ([0.0, 1.0], [0.0, 0.9]),
        )
        for fpr, tpr in invalid_roc_pairs:
            with self.subTest(fpr=fpr, tpr=tpr):
                with self.assertRaises(ValueError):
                    grid_auc(fpr, tpr)

        with self.assertRaises(ValueError):
            rocs_from_histogram_samples(
                [[1.0, np.nan]],
                [[1.0, 0.0]],
            )
        with self.assertRaises(ValueError):
            max_euclidean_deviations(
                [[0.0, np.nan, 1.0]],
                [[0.0, 0.5, 1.0]],
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            )
        with self.assertRaises(ValueError):
            max_euclidean_deviations([], [], [], [])
        with self.assertRaises(ValueError):
            curve_max_euclidean_deviation([], [], [], [])


if __name__ == "__main__":
    unittest.main()
