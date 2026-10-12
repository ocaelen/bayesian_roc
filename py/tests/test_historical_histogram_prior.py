"""Exact references for the historical-histogram prior in the manuscript."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils.prior_profiles import (  # noqa: E402
    dirichlet_parameters_from_bin_probabilities,
    historical_histogram_prior_bin_probabilities,
    prior_expected_grid_auc,
)


def _exact_grid_auc(positive_counts, negative_counts):
    """Enumerate bin pairs, giving a tied pair half a success."""
    total = sum(positive_counts) * sum(negative_counts)
    return sum(
        Fraction(positive * negative, total)
        * (1 if i > j else Fraction(1, 2) if i == j else 0)
        for i, positive in enumerate(positive_counts)
        for j, negative in enumerate(negative_counts)
    )


class HistoricalHistogramPriorTests(unittest.TestCase):
    def test_empty_bins_and_unequal_classes_match_exact_fractions(self):
        k, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
            [0, 2], [3, 0],
        )
        np.testing.assert_array_equal(k, [1, 2])
        np.testing.assert_allclose(q_plus, [1 / 6, 5 / 6], rtol=0, atol=1e-15)
        np.testing.assert_allclose(q_minus, [7 / 8, 1 / 8], rtol=0, atol=1e-15)
        self.assertAlmostEqual(prior_expected_grid_auc(q_plus, q_minus), 41 / 48)
        # One smoothing pseudo-count is distributed over the two bins;
        # adding one to EACH bin would produce different fractions.
        np.testing.assert_allclose(3 * q_plus - [0, 2], [0.5, 0.5])
        np.testing.assert_allclose(4 * q_minus - [3, 0], [0.5, 0.5])

    def test_masses_one_five_and_twenty_preserve_profile_and_expected_auc(self):
        _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
            [0, 2], [3, 0],
        )
        exact_plus = (Fraction(1, 6), Fraction(5, 6))
        exact_minus = (Fraction(7, 8), Fraction(1, 8))
        for mass in (1.0, 5.0, 20.0):
            with self.subTest(mass=mass):
                alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(
                    q_plus, q_minus, mass,
                )
                for alpha, exact, profile in (
                    (alpha_plus, exact_plus, q_plus),
                    (alpha_minus, exact_minus, q_minus),
                ):
                    expected = [float(int(mass) * probability) for probability in exact]
                    np.testing.assert_allclose(alpha, expected, rtol=0, atol=1e-14)
                    self.assertTrue(np.all(alpha > 0))
                    self.assertAlmostEqual(float(alpha.sum()), mass, delta=1e-12)
                    np.testing.assert_allclose(alpha / mass, profile, rtol=0, atol=1e-15)
                self.assertAlmostEqual(
                    prior_expected_grid_auc(alpha_plus / mass, alpha_minus / mass),
                    float(Fraction(41, 48)), delta=1e-12,
                )

    def test_nonmonotone_nonmirrored_profiles_are_valid(self):
        _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
            [0, 4, 0], [1, 0, 1],
        )
        np.testing.assert_allclose(q_plus, np.array([1, 13, 1]) / 15)
        np.testing.assert_allclose(q_minus, np.array([4, 1, 4]) / 9)
        self.assertFalse(np.allclose(q_plus, q_minus[::-1]))
        self.assertGreater(q_plus[1], q_plus[2])
        self.assertGreater(q_plus[1], q_plus[0])

    def test_uniform_histograms_recover_uniform_profiles(self):
        # Different class sizes are normalized independently.
        for count_plus, count_minus in ((1, 1), (3, 7), (50, 500)):
            with self.subTest(positive=count_plus, negative=count_minus):
                _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
                    [count_plus] * 10, [count_minus] * 10,
                )
                np.testing.assert_allclose(q_plus, [0.1] * 10)
                np.testing.assert_allclose(q_minus, [0.1] * 10)

    def test_historical_size_does_not_determine_final_concentration(self):
        for count in (1, 500, 50000):
            with self.subTest(count=count):
                _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
                    [0] * 9 + [count], [count] + [0] * 9,
                )
                for mass in (1.0, 5.0, 20.0):
                    with self.subTest(mass=mass):
                        alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(
                            q_plus, q_minus, mass,
                        )
                        self.assertAlmostEqual(float(alpha_plus.sum()), mass, delta=1e-12)
                        self.assertAlmostEqual(float(alpha_minus.sum()), mass, delta=1e-12)
                        self.assertTrue(np.all(alpha_plus > 0))
                        self.assertTrue(np.all(alpha_minus > 0))
                        np.testing.assert_allclose(alpha_plus / mass, q_plus, rtol=0, atol=1e-15)
                        np.testing.assert_allclose(alpha_minus / mass, q_minus, rtol=0, atol=1e-15)

    def test_no_linear_auc_range_restriction_is_applied(self):
        _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(
            [0] * 9 + [500], [500] + [0] * 9,
        )
        # Exact smoothed counts are (1,...,1,5001), with total 5010.
        exact = _exact_grid_auc([1] * 9 + [5001], [5001] + [1] * 9)
        self.assertGreater(exact, Fraction(13, 15))
        self.assertAlmostEqual(prior_expected_grid_auc(q_plus, q_minus), float(exact))
        self.assertLess(prior_expected_grid_auc(q_minus, q_plus), 0.5)
        self.assertAlmostEqual(prior_expected_grid_auc(q_minus, q_plus), float(1 - exact))

    def test_same_empirical_auc_does_not_determine_histogram_prior(self):
        positive = [0] * 9 + [4]
        negative_a = [0] * 8 + [2, 2]
        negative_b = [2] + [0] * 8 + [2]
        self.assertEqual(_exact_grid_auc(positive, negative_a), Fraction(3, 4))
        self.assertEqual(_exact_grid_auc(positive, negative_b), Fraction(3, 4))
        _, q_plus_a, q_minus_a = historical_histogram_prior_bin_probabilities(
            positive, negative_a,
        )
        _, q_plus_b, q_minus_b = historical_histogram_prior_bin_probabilities(
            positive, negative_b,
        )
        np.testing.assert_array_equal(q_plus_a, q_plus_b)
        self.assertFalse(np.array_equal(q_minus_a, q_minus_b))
        np.testing.assert_allclose(q_plus_a, np.array([1] * 9 + [41]) / 50)
        np.testing.assert_allclose(q_minus_a, np.array([1] * 8 + [21, 21]) / 50)
        np.testing.assert_allclose(q_minus_b, np.array([21] + [1] * 8 + [21]) / 50)
        self.assertAlmostEqual(prior_expected_grid_auc(q_plus_a, q_minus_a), 167 / 250)
        self.assertAlmostEqual(prior_expected_grid_auc(q_plus_b, q_minus_b), 183 / 250)

    def test_sequential_update_special_case_for_equal_historical_sizes(self):
        for positive_counts, negative_counts, mass in (
            ([0, 4, 0], [2, 0, 2], 5.0),
            ([0, 19, 0], [7, 0, 12], 20.0),
        ):
            with self.subTest(mass=mass):
                positive = np.array(positive_counts)
                negative = np.array(negative_counts)
                _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(positive, negative)
                alpha_plus, alpha_minus = dirichlet_parameters_from_bin_probabilities(
                    q_plus, q_minus, mass,
                )
                np.testing.assert_allclose(alpha_plus, positive + 1 / 3, rtol=0, atol=1e-14)
                np.testing.assert_allclose(alpha_minus, negative + 1 / 3, rtol=0, atol=1e-14)
                observed = np.array([2, 1, 3])
                np.testing.assert_allclose(
                    observed + alpha_plus, observed + positive + 1 / 3, rtol=0, atol=1e-14,
                )

    def test_sequential_update_mass_is_class_specific_for_unequal_sizes(self):
        _, q_plus, q_minus = historical_histogram_prior_bin_probabilities([0, 2], [3, 0])
        np.testing.assert_allclose(3 * q_plus, [0.5, 2.5])
        np.testing.assert_allclose(4 * q_minus, [3.5, 0.5])
        self.assertFalse(np.allclose(3 * q_minus, [3.5, 0.5]))

    def test_readonly_inputs_and_input_arrays_are_preserved(self):
        positive = np.array([0, 4, 0], dtype=np.int64)
        negative = np.array([3, 1, 0], dtype=np.int64)
        positive.setflags(write=False)
        negative.setflags(write=False)
        _, q_plus, q_minus = historical_histogram_prior_bin_probabilities(positive, negative)
        np.testing.assert_array_equal(positive, [0, 4, 0])
        np.testing.assert_array_equal(negative, [3, 1, 0])
        self.assertFalse(np.shares_memory(q_plus, positive))
        self.assertFalse(np.shares_memory(q_minus, negative))
        q_plus[0] = 0
        np.testing.assert_array_equal(positive, [0, 4, 0])

    def test_integral_float_counts_match_integer_counts(self):
        integer = historical_histogram_prior_bin_probabilities([0, 2], [3, 0])
        floating = historical_histogram_prior_bin_probabilities([0.0, 2.0], [3.0, 0.0])
        for expected, actual in zip(integer, floating, strict=True):
            np.testing.assert_array_equal(expected, actual)

    def test_invalid_counts_are_rejected_for_either_class(self):
        invalid_histograms = (
            [], [1], [[0, 2]], [0, 0], [-1, 3], [0.5, 1.5],
            [np.nan, 2], [np.inf, 2], [-np.inf, 2], [True, False],
            ["0", "2"], [0j, 2 + 0j], np.array([0, 2], dtype=object),
            [1e308, 1e308],
        )
        for counts in invalid_histograms:
            for invalid_class in ("positive", "negative"):
                with self.subTest(counts=counts, invalid_class=invalid_class):
                    positive = counts if invalid_class == "positive" else [0, 2]
                    negative = counts if invalid_class == "negative" else [3, 0]
                    with self.assertRaises(ValueError):
                        historical_histogram_prior_bin_probabilities(positive, negative)
        with self.assertRaises(ValueError):
            historical_histogram_prior_bin_probabilities([0, 2], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
