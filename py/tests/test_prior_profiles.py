import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils.prior_profiles import (  # noqa: E402
    auc_calibrated_prior_bin_probabilities,
    beta_bin_probabilities,
    dirichlet_parameters_from_bin_probabilities,
    expected_roc_from_bin_probabilities,
    increasing_decreasing_prior_bin_probabilities,
    prior_expected_grid_auc,
    symmetric_prior_bin_probabilities,
    validate_prior_mean_probabilities,
)

CALIBRATED_TARGET_AUC = 0.65


class PriorProfileTests(unittest.TestCase):
    def test_symmetric_prior_gives_uniform_bins_and_chance_line(self):
        k, q_plus, q_minus = symmetric_prior_bin_probabilities(K=10)
        fpr, tpr, expected_auc = expected_roc_from_bin_probabilities(
            q_plus,
            q_minus,
        )

        np.testing.assert_array_equal(k, np.arange(1, 11))
        np.testing.assert_allclose(q_plus, np.full(10, 0.1))
        np.testing.assert_allclose(q_minus, np.full(10, 0.1))
        np.testing.assert_allclose(fpr, np.linspace(0, 1, 11))
        np.testing.assert_allclose(tpr, fpr)
        self.assertAlmostEqual(expected_auc, 0.5)

    def test_increasing_decreasing_prior_matches_known_example(self):
        k, q_plus, q_minus = increasing_decreasing_prior_bin_probabilities(K=4)
        fpr, tpr, expected_auc = expected_roc_from_bin_probabilities(
            q_plus,
            q_minus,
        )

        np.testing.assert_array_equal(k, [1, 2, 3, 4])
        np.testing.assert_allclose(q_plus, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(q_minus, [0.4, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(fpr, [0.0, 0.1, 0.3, 0.6, 1.0])
        np.testing.assert_allclose(tpr, [0.0, 0.4, 0.7, 0.9, 1.0])
        self.assertAlmostEqual(expected_auc, 0.75)

    def test_calibrated_example_is_exact_midpoint(self):
        K = 10
        _, q_symmetric_plus, q_symmetric_minus = (
            symmetric_prior_bin_probabilities(K)
        )
        _, q_monotone_plus, q_monotone_minus = (
            increasing_decreasing_prior_bin_probabilities(K)
        )
        _, q_calibrated_plus, q_calibrated_minus = (
            auc_calibrated_prior_bin_probabilities(
                K,
                CALIBRATED_TARGET_AUC,
            )
        )

        np.testing.assert_allclose(
            q_calibrated_plus,
            0.5 * (q_symmetric_plus + q_monotone_plus),
        )
        np.testing.assert_allclose(
            q_calibrated_minus,
            0.5 * (q_symmetric_minus + q_monotone_minus),
        )
        self.assertAlmostEqual(q_calibrated_plus[0], 13 / 220)
        self.assertAlmostEqual(q_calibrated_plus[-1], 31 / 220)

        fpr_symmetric, tpr_symmetric, _ = expected_roc_from_bin_probabilities(
            q_symmetric_plus,
            q_symmetric_minus,
        )
        fpr_monotone, tpr_monotone, _ = expected_roc_from_bin_probabilities(
            q_monotone_plus,
            q_monotone_minus,
        )
        fpr_calibrated, tpr_calibrated, expected_auc = (
            expected_roc_from_bin_probabilities(
                q_calibrated_plus,
                q_calibrated_minus,
            )
        )

        np.testing.assert_allclose(
            fpr_calibrated,
            0.5 * (fpr_symmetric + fpr_monotone),
        )
        np.testing.assert_allclose(
            tpr_calibrated,
            0.5 * (tpr_symmetric + tpr_monotone),
        )
        self.assertAlmostEqual(expected_auc, CALIBRATED_TARGET_AUC)

    def test_rocs_match_the_manuscript_closed_forms(self):
        K = 10
        plot_index = np.arange(K + 1, dtype=float)

        symmetric_fpr = plot_index / K
        symmetric_tpr = symmetric_fpr

        increasing_fpr = plot_index * (plot_index + 1) / (K * (K + 1))
        increasing_tpr = (
            plot_index * (2 * K - plot_index + 1) / (K * (K + 1))
        )

        correction = (
            3
            * (2 * CALIBRATED_TARGET_AUC - 1)
            * plot_index
            * (K - plot_index)
            / (2 * (K**2 - 1))
        )
        calibrated_fpr = plot_index / K - correction
        calibrated_tpr = plot_index / K + correction

        profiles = (
            symmetric_prior_bin_probabilities(K),
            increasing_decreasing_prior_bin_probabilities(K),
            auc_calibrated_prior_bin_probabilities(
                K,
                CALIBRATED_TARGET_AUC,
            ),
        )
        closed_forms = (
            (symmetric_fpr, symmetric_tpr, 0.5),
            (increasing_fpr, increasing_tpr, 0.8),
            (calibrated_fpr, calibrated_tpr, CALIBRATED_TARGET_AUC),
        )
        for (_, q_plus, q_minus), (fpr, tpr, auc) in zip(
            profiles,
            closed_forms,
            strict=True,
        ):
            actual_fpr, actual_tpr, actual_auc = (
                expected_roc_from_bin_probabilities(q_plus, q_minus)
            )
            np.testing.assert_allclose(actual_fpr, fpr)
            np.testing.assert_allclose(actual_tpr, tpr)
            self.assertAlmostEqual(actual_auc, auc)

    def test_historical_beta_5_3_bins_keep_the_known_grid_auc(self):
        # Frozen numerical reference from the v1/v2 design, independent of
        # the current scenario registry and its newly specified Medium case.
        _, q_plus, q_minus = beta_bin_probabilities(
            K=10,
            positive_shape=(5.0, 3.0),
            negative_shape=(3.0, 5.0),
        )

        np.testing.assert_allclose(q_plus, q_minus[::-1])
        self.assertAlmostEqual(
            prior_expected_grid_auc(q_plus, q_minus),
            0.85039269104,
            places=12,
        )

        alpha_plus, alpha_minus = (
            dirichlet_parameters_from_bin_probabilities(
                q_plus,
                q_minus,
                prior_mass=20.0,
            )
        )
        np.testing.assert_allclose(alpha_plus, 20.0 * q_plus)
        np.testing.assert_allclose(alpha_minus, 20.0 * q_minus)
        self.assertAlmostEqual(alpha_plus.sum(), 20.0)
        self.assertAlmostEqual(alpha_minus.sum(), 20.0)

    def test_infeasible_calibrated_auc_is_rejected(self):
        upper_auc = (5 * 10 + 2) / (6 * 10)

        with self.assertRaises(ValueError):
            auc_calibrated_prior_bin_probabilities(K=10, target_auc=0.49)
        with self.assertRaises(ValueError):
            auc_calibrated_prior_bin_probabilities(
                K=10,
                target_auc=upper_auc,
            )
        with self.assertRaises(ValueError):
            symmetric_prior_bin_probabilities(K=1)
        with self.assertRaises(ValueError):
            symmetric_prior_bin_probabilities(K=10.0)

    def test_invalid_probability_profiles_and_masses_are_rejected(self):
        invalid_profiles = (
            ([0.5, 0.5], [0.2, 0.2]),
            ([0.5, np.nan], [0.5, 0.5]),
            ([0.5, 0.5], [0.0, 1.0]),
            ([1.0], [1.0]),
        )
        for q_plus, q_minus in invalid_profiles:
            with self.subTest(q_plus=q_plus, q_minus=q_minus):
                with self.assertRaises(ValueError):
                    validate_prior_mean_probabilities(q_plus, q_minus)

        _, q_plus, q_minus = symmetric_prior_bin_probabilities(10)
        for invalid_mass in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(prior_mass=invalid_mass):
                with self.assertRaises(ValueError):
                    dirichlet_parameters_from_bin_probabilities(
                        q_plus,
                        q_minus,
                        invalid_mass,
                    )

        with self.assertRaises(ValueError):
            beta_bin_probabilities(
                K=10,
                positive_shape=(4.0, -3.0),
                negative_shape=(3.0, 4.0),
            )

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.prior_profiles"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "prior_profiles relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
