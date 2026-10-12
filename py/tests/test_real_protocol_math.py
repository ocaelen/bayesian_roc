"""Fixed-grid mathematical checks for the real-data experimental protocol."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roc_utils import utilities_experiments as utilities
from roc_utils.evaluation_criteria import is_external_auc_covered, is_external_roc_covered
from roc_utils.posterior_predictive_roc_sampler import (
    fixed_uniform_grid, grid_auc, histogram_from_scores, roc_from_histograms,
)


class RealProtocolMathTests(unittest.TestCase):
    def test_official_quantile_ranks_use_the_same_4000_curves_at_both_levels(self):
        # Each scripted curve is realizable with 4000 positives and 4000
        # negatives: negative tail 1/2, positive tail i/4000. Its area is
        # exactly 1/4 + i/8000, independently of the number of middle points.
        for K in (5, 10, 20):
            with self.subTest(K=K):
                d = np.zeros(K, dtype=np.int64)
                d[0] = d[-1] = 2000
                alpha = np.full(K, 1 / K)
                fractions = np.arange(4000, dtype=float) / 4000
                fpr = np.column_stack((np.zeros(4000), np.full((4000, K - 1), .5), np.ones(4000)))
                tpr = np.column_stack((np.zeros(4000), np.repeat(fractions[:, None], K - 1, axis=1), np.ones(4000)))
                # Independent bin-tail arithmetic for the posterior mean.
                center = np.array([0.0] + [
                    (2000 + j / K) / 4001 for j in range(1, K)
                ] + [1.0])
                deviations = sorted(max(
                    math.hypot(.5 - center[j], fraction - center[j])
                    for j in range(1, K)
                ) for fraction in fractions)
                with patch.object(utilities, "sample_posterior_predictive_rocs",
                                  return_value=(fpr, tpr)) as sample:
                    band = utilities.build_posterior_predictive_band(
                        d, d, alpha, alpha, 4000, (.8, .95), np.random.default_rng(3),
                    )
                sample.assert_called_once()
                self.assertEqual(sample.call_args.kwargs["S"], 4000)
                np.testing.assert_allclose(band.fpr_center, center, rtol=0, atol=1e-15)
                np.testing.assert_allclose(band.auc_samples, .25 + fractions / 2, rtol=0, atol=1e-15)
                for level, radius_rank, lo_rank, hi_rank in ((.8, 3200, 400, 3600), (.95, 3800, 100, 3900)):
                    self.assertAlmostEqual(band.roc_radii[level], deviations[radius_rank - 1], places=14)
                    np.testing.assert_allclose(band.auc_intervals[level],
                        [.25 + (lo_rank - 1) / 8000, .25 + (hi_rank - 1) / 8000], rtol=0, atol=1e-15)

    def test_external_auc_matches_independent_bin_pair_counting_on_each_grid(self):
        positive = np.array([.001, .009, .1, .37, .5, .99, 1.0])
        negative = np.array([0, .005, .1, .4, .51, .9])
        for K in (5, 10, 20):
            with self.subTest(K=K):
                grid = fixed_uniform_grid(K)
                # Count comparisons at the grid resolution, with half credit
                # for a pair in the same bin, including endpoints and ties.
                pbins = [sum(score >= edge for edge in grid[1:-1]) for score in positive]
                nbins = [sum(score >= edge for edge in grid[1:-1]) for score in negative]
                expected = sum((p > n) + .5 * (p == n) for p in pbins for n in nbins) / (len(pbins) * len(nbins))
                fpr, tpr = roc_from_histograms(
                    histogram_from_scores(positive, grid), histogram_from_scores(negative, grid),
                )
                self.assertAlmostEqual(float(grid_auc(fpr, tpr)), expected, places=14)
                fpr, tpr = roc_from_histograms(
                    histogram_from_scores([.009], grid), histogram_from_scores([.001], grid),
                )
                self.assertEqual(float(grid_auc(fpr, tpr)), .5)

    def test_sparse_small_classes_keep_valid_predictive_bands_on_all_grids(self):
        for K in (5, 10, 20):
            for positive_bin, negative_bin in ((0, 0), (K - 1, 0), (0, K - 1)):
                with self.subTest(K=K, positive_bin=positive_bin, negative_bin=negative_bin):
                    positive, negative = np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.int64)
                    positive[positive_bin], negative[negative_bin] = 2, 17  # Fertility counts
                    positive.setflags(write=False)
                    negative.setflags(write=False)
                    alpha = np.full(K, 1 / K)
                    alpha.setflags(write=False)
                    band = utilities.build_posterior_predictive_band(
                        positive, negative, alpha, alpha, 40, (.8, .95), np.random.default_rng(23),
                    )
                    self.assertEqual(positive.sum(), 2)
                    self.assertEqual(negative.sum(), 17)
                    self.assertEqual(band.auc_samples.shape, (40,))
                    self.assertTrue(0 <= band.roc_radii[.8] <= band.roc_radii[.95] <= math.sqrt(2))
                    lo80, hi80 = band.auc_intervals[.8]
                    lo95, hi95 = band.auc_intervals[.95]
                    self.assertTrue(0 <= lo95 <= lo80 <= hi80 <= hi95 <= 1)

    def test_inclusive_absolute_coverage_tolerance(self):
        # Clearly separated offsets avoid ambiguity from binary rounding at
        # exactly 1e-12 while exercising each side of the allowed tolerance.
        for offset, expected in ((0, True), (5e-13, True), (2e-12, False)):
            with self.subTest(offset=offset):
                self.assertEqual(is_external_roc_covered(.2 + offset, .2), expected)
                self.assertEqual(is_external_auc_covered(.2 - offset, (.2, .8)), expected)
                self.assertEqual(is_external_auc_covered(.8 + offset, (.2, .8)), expected)


if __name__ == "__main__":
    unittest.main()
