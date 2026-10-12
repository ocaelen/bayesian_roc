import sys
import unittest
from pathlib import Path

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils.evaluation_criteria import (  # noqa: E402
    MeanEstimate,
    is_external_auc_covered,
    is_external_roc_covered,
    summarize_mean,
    summarize_paired_differences,
    summarize_external_auc_coverage,
    summarize_external_roc_coverage,
    summarize_sharpness,
)
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    curve_max_euclidean_deviation,
)


class EvaluationCriteriaTests(unittest.TestCase):
    def test_external_roc_coverage_is_simultaneous_and_inclusive(self):
        fpr_center = np.array([0.0, 0.2, 0.6, 1.0])
        tpr_center = np.array([0.0, 0.2, 0.6, 1.0])
        fpr_external = np.array([0.0, 0.3, 0.9, 1.0])
        tpr_external = np.array([0.0, 0.2, 1.0, 1.0])

        deviation = curve_max_euclidean_deviation(
            fpr_external,
            tpr_external,
            fpr_center,
            tpr_center,
        )
        self.assertAlmostEqual(deviation, 0.5)

        result = summarize_external_roc_coverage(
            external_max_deviations=[deviation, deviation],
            roc_radii=[0.5, 0.5 - 1.0 / 1024.0],
        )

        self.assertEqual(result.n_replications, 2)
        self.assertAlmostEqual(result.coverage, 0.5)
        self.assertAlmostEqual(result.standard_error, np.sqrt(0.5 * 0.5 / 2.0))

        tolerance_result = summarize_external_roc_coverage(
            external_max_deviations=[0.5 + 0.5e-12, 0.5 + 2.0e-12],
            roc_radii=[0.5, 0.5],
        )
        self.assertAlmostEqual(tolerance_result.coverage, 0.5)
        self.assertTrue(is_external_roc_covered(0.5 + 0.5e-12, 0.5))
        self.assertFalse(is_external_roc_covered(0.5 + 2.0e-12, 0.5))

    def test_external_roc_coverage_aggregates_replications(self):
        deviations = [0.1, 0.2, 0.3, 0.4]

        result_080 = summarize_external_roc_coverage(
            external_max_deviations=deviations,
            roc_radii=[0.11, 0.1, 0.31, 0.5],
        )
        result_095 = summarize_external_roc_coverage(
            external_max_deviations=deviations,
            roc_radii=[0.1, 0.25, 0.2, 0.5],
        )

        self.assertEqual(result_080.n_replications, 4)
        self.assertEqual(result_095.n_replications, 4)
        self.assertAlmostEqual(result_080.coverage, 0.75)
        self.assertAlmostEqual(result_080.standard_error, np.sqrt(3.0) / 8.0)
        self.assertAlmostEqual(result_095.coverage, 0.75)
        self.assertAlmostEqual(result_095.standard_error, np.sqrt(3.0) / 8.0)

    def test_external_auc_coverage_includes_interval_endpoints(self):
        result = summarize_external_auc_coverage(
            external_aucs=[0.375, 0.625, 47.0 / 128.0, 81.0 / 128.0],
            auc_intervals=[
                [0.375, 0.625],
                [0.375, 0.625],
                [0.375, 0.625],
                [0.375, 0.625],
            ],
        )

        self.assertEqual(result.n_replications, 4)
        self.assertAlmostEqual(result.coverage, 0.5)
        self.assertAlmostEqual(result.standard_error, 0.25)

        tolerance_result = summarize_external_auc_coverage(
            external_aucs=[
                0.4 - 0.5e-12,
                0.6 + 0.5e-12,
                0.4 - 2.0e-12,
                0.6 + 2.0e-12,
            ],
            auc_intervals=[
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
                [0.4, 0.6],
            ],
        )
        self.assertAlmostEqual(tolerance_result.coverage, 0.5)
        self.assertTrue(is_external_auc_covered(0.4 - 0.5e-12, (0.4, 0.6)))
        self.assertFalse(is_external_auc_covered(0.4 - 2.0e-12, (0.4, 0.6)))

        paired_result = summarize_external_auc_coverage(
            external_aucs=[0.1, 0.4, 0.7, 0.9],
            auc_intervals=[
                [0.0, 0.2],
                [0.5, 0.6],
                [0.65, 0.75],
                [0.0, 1.0],
            ],
        )
        self.assertAlmostEqual(paired_result.coverage, 0.75)
        self.assertAlmostEqual(paired_result.standard_error, np.sqrt(3.0) / 8.0)

    def test_sharpness_matches_manuscript_standard_errors(self):
        result = summarize_sharpness(
            roc_radii=[0.1, 0.2, 0.3, 0.4],
            auc_intervals=[
                [0.4, 0.6],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
            ],
        )

        self.assertEqual(result.n_replications, 4)
        self.assertAlmostEqual(result.roc_radius_mean, 0.25)
        self.assertAlmostEqual(result.roc_radius_standard_error, np.sqrt(1.0 / 240.0))
        self.assertAlmostEqual(result.auc_interval_width_mean, 0.5)
        self.assertAlmostEqual(
            result.auc_interval_width_standard_error,
            np.sqrt(1.0 / 60.0),
        )

    def test_degenerate_regions_are_valid(self):
        coverage = summarize_external_auc_coverage(
            external_aucs=[0.5],
            auc_intervals=[[0.5, 0.5]],
        )
        self.assertEqual(coverage.coverage, 1.0)
        self.assertEqual(coverage.standard_error, 0.0)

        result = summarize_sharpness(
            roc_radii=[0.0, 0.0],
            auc_intervals=[[0.5, 0.5], [0.5, 0.5]],
        )
        self.assertEqual(result.roc_radius_mean, 0.0)
        self.assertEqual(result.roc_radius_standard_error, 0.0)
        self.assertEqual(result.auc_interval_width_mean, 0.0)
        self.assertEqual(result.auc_interval_width_standard_error, 0.0)

    def test_runtime_assertions_reject_inconsistent_inputs(self):
        with self.assertRaises(AssertionError):
            summarize_external_roc_coverage([], [])
        with self.assertRaises(AssertionError):
            summarize_external_roc_coverage([0.1, 0.2], [0.1])
        with self.assertRaises(AssertionError):
            summarize_external_roc_coverage([-0.1], [0.1])
        with self.assertRaises(AssertionError):
            summarize_external_auc_coverage([0.5], [[0.6, 0.4]])
        with self.assertRaises(AssertionError):
            summarize_external_auc_coverage([1.1], [[0.0, 1.0]])
        with self.assertRaises(AssertionError):
            summarize_sharpness([0.1], [[0.4, 0.6]])
        with self.assertRaises(AssertionError):
            summarize_sharpness([0.1, 0.2], [[0.4, 0.6]])
        with self.assertRaises(AssertionError):
            is_external_roc_covered(np.nan, 0.1)
        with self.assertRaises(AssertionError):
            is_external_auc_covered(0.5, (0.6, 0.4))

    def test_bootstrap_radius_is_preserved_above_the_square_diameter(self):
        self.assertTrue(is_external_roc_covered(np.sqrt(2.0), 1.8))
        coverage = summarize_external_roc_coverage(
            [np.sqrt(2.0), 1.0], [1.8, 2.0],
        )
        self.assertEqual(coverage.coverage, 1.0)
        sharpness = summarize_sharpness([1.6, 2.0], [[0.0, 1.0], [0.25, 0.75]])
        self.assertAlmostEqual(sharpness.roc_radius_mean, 1.8)
        self.assertAlmostEqual(sharpness.roc_radius_standard_error, 0.2)
        self.assertEqual(sharpness.auc_interval_width_mean, 0.75)
        self.assertEqual(sharpness.auc_interval_width_standard_error, 0.25)
        with self.assertRaises(AssertionError):
            is_external_roc_covered(0.1, 2.0 + 1e-6)
        with self.assertRaises(AssertionError):
            is_external_roc_covered(np.sqrt(2.0) + 1e-6, 2.0)
        with self.assertRaises(AssertionError):
            summarize_external_roc_coverage([1.5], [2.0])
        with self.assertRaises(AssertionError):
            summarize_sharpness([1.0, 2.01], [[0.0, 1.0], [0.0, 1.0]])

    def test_generic_mean_matches_independent_sample_variance_arithmetic(self):
        result = summarize_mean([1.0, 2.0, 3.0, 4.0])
        self.assertIsInstance(result, MeanEstimate)
        self.assertEqual(result.n_replications, 4)
        self.assertEqual(result.mean, 2.5)
        self.assertAlmostEqual(result.standard_error, np.sqrt(5.0 / 12.0))
        negative = summarize_mean([-2.0, -2.0, -2.0])
        self.assertEqual(negative.mean, -2.0)
        self.assertEqual(negative.standard_error, 0.0)

    def test_paired_coverage_uses_covariance_and_dip_minus_boot_direction(self):
        identical = summarize_paired_differences([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(identical.mean, 0.0)
        self.assertEqual(identical.standard_error, 0.0)
        self.assertEqual(identical.n_replications, 4)

        # Differences (-1, 1, 0, 0) have squared deviations summing to 2.
        result = summarize_paired_differences([0, 1, 1, 0], [1, 0, 1, 0])
        self.assertEqual(result.mean, 0.0)
        self.assertAlmostEqual(result.standard_error, np.sqrt(1.0 / 6.0))
        independent_se = np.sqrt(0.5 * 0.5 / 4 + 0.5 * 0.5 / 4)
        self.assertNotAlmostEqual(result.standard_error, independent_se)

        directed = summarize_paired_differences([1, 1, 1, 0], [0, 1, 0, 0])
        reversed_pair = summarize_paired_differences([0, 1, 0, 0], [1, 1, 1, 0])
        self.assertEqual(directed.mean, 0.5)
        self.assertEqual(reversed_pair.mean, -0.5)
        self.assertAlmostEqual(directed.standard_error, np.sqrt(1.0 / 12.0))
        self.assertEqual(directed.standard_error, reversed_pair.standard_error)

    def test_paired_continuous_metrics_allow_signed_differences_and_preserve_pairs(self):
        result = summarize_paired_differences([0.5, 1.0, 1.5, 2.0], [1.0] * 4)
        self.assertEqual(result.mean, 0.25)
        self.assertAlmostEqual(result.standard_error, np.sqrt(5.0 / 48.0))
        time_result = summarize_paired_differences([10, 40, 30, 80], [20, 20, 20, 20])
        self.assertEqual(time_result.mean, 20.0)
        self.assertAlmostEqual(time_result.standard_error, np.sqrt(2600.0 / 12.0))
        permutation = [2, 0, 3, 1]
        dip = np.asarray([0, 1, 1, 0])[permutation]
        boot = np.asarray([1, 0, 1, 0])[permutation]
        reordered = summarize_paired_differences(dip, boot)
        self.assertAlmostEqual(reordered.standard_error, np.sqrt(1.0 / 6.0))

    def test_boolean_and_unsigned_inputs_are_subtracted_without_wraparound(self):
        indicators = summarize_paired_differences(
            np.array([False, True], dtype=bool),
            np.array([True, True], dtype=bool),
        )
        self.assertEqual(indicators.mean, -0.5)
        self.assertEqual(indicators.standard_error, 0.5)
        timings = summarize_paired_differences(
            np.array([1, 3], dtype=np.uint64),
            np.array([3, 1], dtype=np.uint64),
        )
        self.assertEqual(timings.mean, 0.0)
        self.assertEqual(timings.standard_error, 2.0)

    def test_mean_and_paired_summaries_reject_incomplete_or_nonfinite_data(self):
        for invalid_values in ([], [1.0], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]):
            with self.subTest(values=invalid_values):
                with self.assertRaises(AssertionError):
                    summarize_mean(invalid_values)
        for dip, boot in (
            ([], []),
            ([1.0], [0.0]),
            ([1.0, 2.0], [0.0]),
            ([[1.0, 2.0]], [[0.0, 1.0]]),
            ([1.0, np.nan], [0.0, 1.0]),
            ([1.0, 2.0], [0.0, np.inf]),
        ):
            with self.subTest(dip=dip, boot=boot):
                with self.assertRaises(AssertionError):
                    summarize_paired_differences(dip, boot)


if __name__ == "__main__":
    unittest.main()
