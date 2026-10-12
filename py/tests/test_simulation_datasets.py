import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_function

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    fixed_uniform_grid,
    histogram_from_scores,
)
from roc_utils.simulation_datasets import (  # noqa: E402
    SIMULATION_SCENARIOS,
    SYNTHETIC_DESIGN_VERSION,
    generate_simulation_dataset,
    simulation_beta_parameters,
)
from roc_utils.prior_profiles import (  # noqa: E402
    beta_bin_probabilities,
    prior_expected_grid_auc,
)


class RecordingBetaRng:
    """Small deterministic stand-in used to inspect calls to ``rng.beta``."""

    def __init__(self):
        self.calls = []

    def beta(self, a, b, size):
        self.calls.append((a, b, size))
        return np.full(size, len(self.calls) / 10.0)


class FixedOutputRng:
    """Stand-in used to verify postcondition assertions."""

    def __init__(self, output):
        self.output = output

    def beta(self, a, b, size):
        return self.output


class SimulationDatasetTests(unittest.TestCase):
    def test_public_scenario_registry_is_the_single_beta_parameter_source(self):
        expected = {
            "Easy": ((3.75, 2.25), (2.25, 3.75)),
            "Medium": ((3.5, 2.5), (2.5, 3.5)),
            "Hard": ((3.25, 2.75), (2.75, 3.25)),
        }

        self.assertEqual(SYNTHETIC_DESIGN_VERSION, 3)
        self.assertEqual(SIMULATION_SCENARIOS, tuple(expected))
        for scenario, parameters in expected.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    simulation_beta_parameters(scenario),
                    parameters,
                )

        with self.assertRaises(AssertionError):
            simulation_beta_parameters("Unknown")

    def test_all_scenarios_use_the_exact_manuscript_beta_parameters(self):
        scenario_parameters = {
            "Easy": ((3.75, 2.25), (2.25, 3.75)),
            "Medium": ((3.5, 2.5), (2.5, 3.5)),
            "Hard": ((3.25, 2.75), (2.75, 3.25)),
        }

        for scenario, (positive_parameters, negative_parameters) in (
            scenario_parameters.items()
        ):
            with self.subTest(scenario=scenario):
                rng = RecordingBetaRng()
                dataset = generate_simulation_dataset(
                    scenario,
                    n_positive=3,
                    n_negative=5,
                    rng=rng,
                )

                self.assertEqual(
                    rng.calls,
                    [
                        (*positive_parameters, 3),
                        (*negative_parameters, 5),
                        (*positive_parameters, 3),
                        (*negative_parameters, 5),
                    ],
                )
                np.testing.assert_array_equal(
                    dataset.positive_test_scores,
                    np.full(3, 0.1),
                )
                np.testing.assert_array_equal(
                    dataset.negative_test_scores,
                    np.full(5, 0.2),
                )
                np.testing.assert_array_equal(
                    dataset.positive_external_scores,
                    np.full(3, 0.3),
                )
                np.testing.assert_array_equal(
                    dataset.negative_external_scores,
                    np.full(5, 0.4),
                )

    def test_v3_grid_aucs_match_independent_density_integrals_and_pairwise_order(self):
        # Integrate the normalized Beta density in each bin, without using
        # the production CDF or ROC routines. Then count pairwise bin wins,
        # assigning half a win to ties as required by the grid AUC.
        expected_aucs = (0.8219053925748699, 0.7298989020330722, 0.6200056706877085)
        historical_aucs = (0.956375336, 0.85039269104, 0.711772688)
        grid = np.linspace(0.0, 1.0, 11)
        actual_aucs = []
        for scenario, expected_auc, historical_auc in zip(
            SIMULATION_SCENARIOS, expected_aucs, historical_aucs, strict=True,
        ):
            with self.subTest(scenario=scenario):
                positive_shape, negative_shape = simulation_beta_parameters(scenario)
                self.assertEqual(sum(positive_shape), 6.0)
                self.assertEqual(negative_shape, positive_shape[::-1])
                self.assertGreater(positive_shape[0], positive_shape[1])
                self.assertGreater(min(positive_shape), 1.0)
                integrated = []
                for a, b in (positive_shape, negative_shape):
                    normalizer = beta_function(a, b)
                    probabilities = np.array([
                        quad(
                            lambda x: x ** (a - 1) * (1 - x) ** (b - 1) / normalizer,
                            lower, upper, epsabs=1e-13, epsrel=1e-13,
                        )[0]
                        for lower, upper in zip(grid[:-1], grid[1:], strict=True)
                    ])
                    self.assertTrue(np.all(probabilities > 0.0))
                    self.assertAlmostEqual(probabilities.sum(), 1.0, places=12)
                    integrated.append(probabilities)
                q_plus, q_minus = integrated
                independent_auc = sum(
                    q_plus[positive] * q_minus[negative]
                    * (1.0 if positive > negative else 0.5 if positive == negative else 0.0)
                    for positive in range(10) for negative in range(10)
                )
                _, actual_plus, actual_minus = beta_bin_probabilities(
                    10, positive_shape, negative_shape,
                )
                np.testing.assert_allclose(actual_plus, q_plus, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(actual_minus, q_minus, rtol=0.0, atol=1e-12)
                actual_auc = prior_expected_grid_auc(actual_plus, actual_minus)
                self.assertAlmostEqual(actual_auc, independent_auc, places=12)
                self.assertAlmostEqual(actual_auc, expected_auc, places=12)
                self.assertLess(actual_auc, historical_auc)
                self.assertTrue(0.5 < actual_auc < 13 / 15)
                actual_aucs.append(actual_auc)
        self.assertGreater(actual_aucs[0], actual_aucs[1])
        self.assertGreater(actual_aucs[1], actual_aucs[2])

    def test_real_generator_returns_matched_finite_score_vectors(self):
        dataset = generate_simulation_dataset(
            "Medium",
            n_positive=np.int64(7),
            n_negative=np.int64(11),
            rng=np.random.default_rng(734),
        )

        self.assertEqual(dataset.positive_test_scores.shape, (7,))
        self.assertEqual(dataset.negative_test_scores.shape, (11,))
        self.assertEqual(dataset.positive_external_scores.shape, (7,))
        self.assertEqual(dataset.negative_external_scores.shape, (11,))

        for scores in (
            dataset.positive_test_scores,
            dataset.negative_test_scores,
            dataset.positive_external_scores,
            dataset.negative_external_scores,
        ):
            self.assertTrue(np.all(np.isfinite(scores)))
            self.assertTrue(np.all((0.0 <= scores) & (scores <= 1.0)))

        gamma = fixed_uniform_grid(K=10)
        for scores, expected_count in (
            (dataset.positive_test_scores, 7),
            (dataset.negative_test_scores, 11),
            (dataset.positive_external_scores, 7),
            (dataset.negative_external_scores, 11),
        ):
            histogram = histogram_from_scores(scores, gamma)
            self.assertEqual(histogram.shape, (10,))
            self.assertEqual(int(histogram.sum()), expected_count)

    def test_same_seed_reproduces_all_four_samples(self):
        first = generate_simulation_dataset(
            "Hard",
            n_positive=8,
            n_negative=6,
            rng=np.random.default_rng(9182),
        )
        second = generate_simulation_dataset(
            "Hard",
            n_positive=8,
            n_negative=6,
            rng=np.random.default_rng(9182),
        )

        for first_scores, second_scores in zip(
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
            np.testing.assert_array_equal(first_scores, second_scores)

    def test_advancing_generator_matches_the_documented_draw_order(self):
        actual_rng = np.random.default_rng(126)
        first = generate_simulation_dataset(
            "Easy",
            n_positive=4,
            n_negative=3,
            rng=actual_rng,
        )
        second = generate_simulation_dataset(
            "Easy",
            n_positive=4,
            n_negative=3,
            rng=actual_rng,
        )

        reference_rng = np.random.default_rng(126)
        expected_samples = []
        for _ in range(2):
            expected_samples.append(
                (
                    reference_rng.beta(3.75, 2.25, size=4),
                    reference_rng.beta(2.25, 3.75, size=3),
                    reference_rng.beta(3.75, 2.25, size=4),
                    reference_rng.beta(2.25, 3.75, size=3),
                )
            )

        for dataset, expected in zip((first, second), expected_samples):
            for actual_scores, expected_scores in zip(
                (
                    dataset.positive_test_scores,
                    dataset.negative_test_scores,
                    dataset.positive_external_scores,
                    dataset.negative_external_scores,
                ),
                expected,
            ):
                np.testing.assert_array_equal(actual_scores, expected_scores)

    def test_runtime_assertions_reject_invalid_inputs(self):
        for scenario in ("easy", "Unknown", "", 1):
            with self.subTest(scenario=scenario):
                rng = RecordingBetaRng()
                with self.assertRaises(AssertionError):
                    generate_simulation_dataset(
                        scenario,
                        n_positive=2,
                        n_negative=2,
                        rng=rng,
                    )
                self.assertEqual(rng.calls, [])

        for invalid_count in (0, -1, 1.5, 2.0, True, np.bool_(True)):
            with self.subTest(n_positive=invalid_count):
                rng = RecordingBetaRng()
                with self.assertRaises(AssertionError):
                    generate_simulation_dataset(
                        "Easy",
                        n_positive=invalid_count,
                        n_negative=2,
                        rng=rng,
                    )
                self.assertEqual(rng.calls, [])
            with self.subTest(n_negative=invalid_count):
                rng = RecordingBetaRng()
                with self.assertRaises(AssertionError):
                    generate_simulation_dataset(
                        "Easy",
                        n_positive=2,
                        n_negative=invalid_count,
                        rng=rng,
                    )
                self.assertEqual(rng.calls, [])

        with self.assertRaises(AssertionError):
            generate_simulation_dataset(
                "Easy",
                n_positive=2,
                n_negative=2,
                rng=None,
            )

    def test_runtime_assertions_check_generated_arrays(self):
        invalid_outputs = (
            np.array([0.2]),
            np.array([[0.2], [0.3]]),
            np.array([0.2, np.nan]),
            np.array([0.2, np.inf]),
            np.array([-0.1, 0.2]),
            np.array([0.2, 1.1]),
        )

        for output in invalid_outputs:
            with self.subTest(output=output):
                with self.assertRaises(AssertionError):
                    generate_simulation_dataset(
                        "Easy",
                        n_positive=2,
                        n_negative=2,
                        rng=FixedOutputRng(output),
                    )

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.simulation_datasets"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "simulation_datasets relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
