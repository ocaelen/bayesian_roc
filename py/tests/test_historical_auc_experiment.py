"""Independent mathematical and data-flow checks for the historical AUC study.

The numerical references use explicit bin-pair counting or exact fractions;
they deliberately do not call the prior calibration utility being checked.
"""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_experiment_main_evaluation_on_synthetic_data as runner  # noqa: E402
from roc_utils import utilities_experiments as utilities  # noqa: E402
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    fixed_uniform_grid,
    symmetric_dirichlet_priors,
)
from roc_utils.semantic_rng import make_semantic_rng, semantic_stream_id  # noqa: E402
from roc_utils.simulation_datasets import simulation_beta_parameters  # noqa: E402


def _pair_auc(positive_bins, negative_bins):
    """Probability of a positive bin winning, with half credit for ties."""
    return sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positive_bins
        for negative in negative_bins
    ) / (len(positive_bins) * len(negative_bins))


def _history(auc=0.72):
    """Construct exact raw AUCs with all positive scores in the middle bin."""
    wins = round(500 * auc)
    assert wins / 500 == auc
    positive, negative = np.zeros(10, dtype=int), np.zeros(10, dtype=int)
    positive[1] = 500
    negative[0], negative[2] = wins, 500 - wins
    return runner.HistoricalSummary(positive, negative)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class HistoricalAUCExperimentTests(unittest.TestCase):
    def setUp(self):
        self.condition = runner.CLASS_COUNT_CONDITIONS[0]
        self.gamma = fixed_uniform_grid(10)
        self.alpha_plus, self.alpha_minus = symmetric_dirichlet_priors(10, 1.0)
        self.histograms = utilities.build_test_external_histograms(
            np.repeat([0.15, 0.25, 0.55, 0.75], 5),
            np.repeat([0.05, 0.25, 0.35, 0.65], 5),
            np.repeat([0.25, 0.35, 0.65, 0.85], 5),
            np.repeat([0.05, 0.15, 0.45, 0.55], 5),
            self.gamma,
        )

    def assertBandsEqual(self, first, second):
        for field in ("fpr_center", "tpr_center", "auc_samples"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        self.assertEqual(first.roc_radii, second.roc_radii)
        self.assertEqual(first.auc_intervals, second.auc_intervals)

    def _capture_predictions(self, *, auc=0.72, histograms=None, replication=1):
        predictions = {}
        original_timer = runner.build_timed_band

        def capture(builder, **arguments):
            band, duration = original_timer(builder, **arguments)
            predictions[arguments["diagnostic_context"]["setting"]] = band
            return band, duration

        with patch.object(runner, "build_timed_band", side_effect=capture):
            results = runner._evaluate_methods(
                self.condition, replication,
                self.histograms if histograms is None else histograms,
                historical_summary=_history(auc), n_samples=40, master_seed=781,
            )
        self.assertEqual(set(predictions), set(runner.SETTINGS))
        return predictions, results

    def test_historical_restriction_has_exact_endpoints_and_no_hidden_adjustment(self):
        references = (
            (0.0, 0.5), (0.49, 0.5), (0.5, 0.5), (0.729, 0.729),
            (0.86, 0.86), (0.861, 0.86), (1.0, 0.86),
            (np.float64(0.73), 0.73),
        )
        for raw, target in references:
            with self.subTest(raw=raw):
                self.assertEqual(runner.restrict_historical_auc(raw), target)
        for raw in (None, True, False, "0.7", [], np.array([0.7]),
                    float("nan"), float("inf"), -float("inf"), -0.001, 1.001):
            with self.subTest(invalid=raw), self.assertRaises((ValueError, TypeError, AssertionError)):
                runner.restrict_historical_auc(raw)

    def test_endpoint_priors_match_independent_exact_fractions(self):
        # At a0=.86 the normalized profile is [1,13,...,109]/550,
        # independently of the selected concentration.
        references = (
            (0.0, np.full(10, 0.1)),
            (0.5, np.full(10, 0.1)),
            (0.86, np.arange(1.0, 110.0, 12.0) / 550.0),
            (1.0, np.arange(1.0, 110.0, 12.0) / 550.0),
        )
        for mass in (1., 5., 20.):
            for raw, expected in references:
                with self.subTest(raw=raw, mass=mass), patch.object(
                    runner, "build_posterior_predictive_band", return_value=object(),
                ) as posterior:
                    generator = np.random.default_rng(175)
                    result = runner.build_linear_predictive_band(
                        d_plus=self.histograms.d_plus_test,
                        d_minus=self.histograms.d_minus_test,
                        historical_grid_auc=raw, prior_mass=mass, S=40,
                        levels=(0.8, 0.95), rng=generator,
                    )
                    self.assertIs(result, posterior.return_value)
                    arguments = posterior.call_args.kwargs
                    np.testing.assert_allclose(arguments["alpha_plus"], mass * expected,
                                               rtol=0, atol=mass * 2e-15)
                    np.testing.assert_allclose(arguments["alpha_minus"], mass * expected[::-1],
                                               rtol=0, atol=mass * 2e-15)
                    self.assertIs(arguments["d_plus"], self.histograms.d_plus_test)
                    self.assertIs(arguments["d_minus"], self.histograms.d_minus_test)
                    self.assertIs(arguments["rng"], generator)
                    self.assertEqual(arguments["S"], 40)
                    self.assertEqual(tuple(arguments["levels"]), (0.8, 0.95))
                    self.assertAlmostEqual(float(arguments["alpha_plus"].sum()), mass)
                    self.assertAlmostEqual(float(arguments["alpha_minus"].sum()), mass)
                    self.assertGreater(float(arguments["alpha_plus"].min()), 0.0)
                    q_plus = arguments["alpha_plus"] / mass
                    q_minus = arguments["alpha_minus"] / mass
                    independent_auc = sum(
                        q_plus[i] * q_minus[j] * (float(i > j) + 0.5 * float(i == j))
                        for i in range(10) for j in range(10)
                    )
                    self.assertAlmostEqual(independent_auc, min(0.86, max(0.5, raw)), places=14)

    def test_historical_half_auc_matches_uniform_prediction_with_identical_rng(self):
        # At a0=.5 the linear profile is uniform. Equal concentrations must
        # then give identical predictions, not only identical ROC centres.
        for mass in (1., 5., 20.):
            for raw in (0.0, 0.5):
                with self.subTest(raw=raw, mass=mass):
                    historical = runner.build_linear_predictive_band(
                        d_plus=self.histograms.d_plus_test,
                        d_minus=self.histograms.d_minus_test,
                        historical_grid_auc=raw, prior_mass=mass, S=40, levels=(0.8, 0.95),
                        rng=np.random.default_rng(817),
                    )
                    uniform = runner.build_uniform_predictive_band(
                        d_plus=self.histograms.d_plus_test,
                        d_minus=self.histograms.d_minus_test,
                        prior_mass=mass, S=40, levels=(0.8, 0.95),
                        rng=np.random.default_rng(817),
                    )
                    self.assertBandsEqual(historical, uniform)

    def test_historical_profile_changes_centre_with_same_unit_prior_weight(self):
        # At fixed m=1 the difference between posterior bin means is exactly
        # (q_historical - q_uniform)/(n_class+1). Sum those differences at
        # each threshold independently of the ROC/prior utilities.
        q_plus = np.arange(1.0, 110.0, 12.0) / 550.0
        for n_positive, n_negative in ((20, 500), (500, 20)):
            with self.subTest(n_positive=n_positive, n_negative=n_negative):
                d_plus = np.zeros(10, dtype=int)
                d_minus = np.zeros(10, dtype=int)
                d_plus[6], d_minus[3] = n_positive, n_negative
                common = dict(d_plus=d_plus, d_minus=d_minus, S=40, levels=(0.8, 0.95))
                historical = runner.build_linear_predictive_band(
                    **common, historical_grid_auc=0.86, prior_mass=1., rng=np.random.default_rng(916),
                )
                uniform = runner.build_posterior_predictive_band(
                    **common, alpha_plus=self.alpha_plus, alpha_minus=self.alpha_minus,
                    rng=np.random.default_rng(916),
                )
                for field, probabilities, count in (
                    ("tpr_center", q_plus, n_positive),
                    ("fpr_center", q_plus[::-1], n_negative),
                ):
                    expected = np.array([
                        sum(float(probabilities[j]) - 0.1 for j in range(k, 10)) / (count + 1)
                        for k in range(10, -1, -1)
                    ])
                    np.testing.assert_allclose(
                        getattr(historical, field) - getattr(uniform, field),
                        expected, rtol=0, atol=1e-15,
                    )

    def test_historical_auc_is_grid_based_with_half_credit_for_same_bin(self):
        # The raw-score AUC is one, but both scores are in the first grid bin.
        self.assertEqual(runner.historical_auc_from_scores(
            np.array([0.09]), np.array([0.01]), self.gamma,
        ), 0.5)
        positive_bins = np.array([0, 0, 2, 5, 5, 9])
        negative_bins = np.array([0, 2, 2, 4, 7])
        expected = _pair_auc(positive_bins, negative_bins)
        result = runner.historical_auc_from_scores(
            (positive_bins + 0.3) / 10, (negative_bins + 0.7) / 10, self.gamma,
        )
        self.assertAlmostEqual(result, expected, places=15)
        for positive, negative, expected in (
            ([0.0], [1.0], 0.0), ([1.0], [0.0], 1.0),
            ([0.0], [0.0], 0.5), ([1.0], [1.0], 0.5),
        ):
            with self.subTest(positive=positive, negative=negative):
                self.assertEqual(runner.historical_auc_from_scores(
                    np.array(positive), np.array(negative), self.gamma,
                ), expected)

    def test_invalid_historical_samples_fail_before_prior_calibration(self):
        for invalid in ([], [[0.2]], [np.nan], [np.inf], [-0.001], [1.001]):
            for bad_class in ("positive", "negative"):
                with self.subTest(invalid=invalid, bad_class=bad_class):
                    positive = np.asarray(invalid if bad_class == "positive" else [0.4])
                    negative = np.asarray(invalid if bad_class == "negative" else [0.3])
                    with self.assertRaises((ValueError, AssertionError)):
                        runner.historical_auc_from_scores(positive, negative, self.gamma)

    def test_historical_generation_uses_exactly_two_dedicated_beta_draws(self):
        condition = runner.CLASS_COUNT_CONDITIONS[6]
        replication, seed = 27, 9231
        expected_rng = make_semantic_rng(seed, 1, 5, condition.index, replication)
        positive_shape, negative_shape = simulation_beta_parameters(condition.score_scenario)
        expected_positive = expected_rng.beta(*positive_shape, size=500)
        expected_negative = expected_rng.beta(*negative_shape, size=500)
        actual_rng = make_semantic_rng(seed, 1, 5, condition.index, replication)
        calls = []

        class RecordingGenerator:
            def beta(self, *args, **kwargs):
                values = actual_rng.beta(*args, **kwargs)
                calls.append((args, kwargs, values.copy()))
                return values

        with patch.object(runner, "make_semantic_rng", return_value=RecordingGenerator()) as make:
            with patch.object(runner, "historical_summary_from_scores", wraps=runner.historical_summary_from_scores) as auc:
                value = runner._generate_historical_summary(
                    condition, replication, gamma=self.gamma, master_seed=seed,
                )
        make.assert_called_once_with(seed, 1, 5, condition.index, replication)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], positive_shape)
        self.assertEqual(calls[1][0], negative_shape)
        np.testing.assert_array_equal(calls[0][2], expected_positive)
        np.testing.assert_array_equal(calls[1][2], expected_negative)
        self.assertEqual(calls[0][2].shape, (500,))
        self.assertEqual(calls[1][2].shape, (500,))
        auc.assert_called_once()
        independent = _pair_auc(
            np.searchsorted(self.gamma[1:-1], expected_positive, side="right"),
            np.searchsorted(self.gamma[1:-1], expected_negative, side="right"),
        )
        self.assertAlmostEqual(value.grid_auc, independent, places=14)

    def test_new_streams_leave_data_and_baseline_streams_unchanged(self):
        condition = runner.CLASS_COUNT_CONDITIONS[8]
        self.assertEqual(runner._stream_keys(condition, 71), ((1, 1, 8, 71), (1, 2, 8, 71)))
        expected = {
            "dip_uniform_m1": 2, "dip_uniform_m5": 7, "dip_uniform_m20": 11,
            "dip_linear_m1": 6, "dip_linear_m5": 8, "dip_linear_m20": 12,
            "dip_histograms_m1": 9, "dip_histograms_m5": 10, "dip_histograms_m20": 13,
        }
        self.assertEqual(set(runner.SETTINGS), set(expected))
        keys = set()
        for setting, role in expected.items():
            key = runner._simulation_stream_key(condition, 71, setting)
            self.assertEqual(key, (1, role, 8, 71))
            self.assertEqual(runner._warmup_stream_key(8, setting), (1, 4, role, 8, 0))
            keys.add(key)
        self.assertEqual(len(keys), 9)
        self.assertNotIn((1, 3, 8, 71), keys)
        self.assertNotIn((1, 5, 8, 71), keys)

    def test_test_and_external_scores_keep_the_original_four_draw_sequence(self):
        condition, replication, seed = runner.CLASS_COUNT_CONDITIONS[8], 19, 819
        generator = make_semantic_rng(seed, 1, 1, condition.index, replication)
        positive_shape, negative_shape = simulation_beta_parameters(condition.score_scenario)
        expected = (
            generator.beta(*positive_shape, size=condition.n_positive),
            generator.beta(*negative_shape, size=condition.n_negative),
            generator.beta(*positive_shape, size=condition.n_positive),
            generator.beta(*negative_shape, size=condition.n_negative),
        )
        with patch.object(runner, "build_test_external_histograms",
                          wraps=runner.build_test_external_histograms) as histograms:
            runner._evaluate_one_replication(
                condition, replication, gamma=self.gamma,
                n_samples=40, master_seed=seed,
            )
        histograms.assert_called_once()
        for actual, reference in zip(histograms.call_args.args[:4], expected, strict=True):
            np.testing.assert_array_equal(actual, reference)

    def test_adjustment_metadata_treats_both_exact_endpoints_as_unadjusted(self):
        _, result = self._capture_predictions()
        references = ((0.0, 0.5, 1), (0.5, 0.5, 0), (0.72, 0.72, 0),
                      (0.86, 0.86, 0), (1.0, 0.86, 1))
        for mass, role, position in ((1, 6, 4), (5, 8, 5), (20, 12, 6)):
            setting = f"dip_linear_m{mass}"
            for raw, target, adjusted in references:
                with self.subTest(raw=raw, mass=mass):
                    rows = runner._raw_rows_for_replication(
                        self.condition, 0, result[setting][0],
                        n_replications=2, n_samples=40, master_seed=781,
                        data_key=(1, 1, 0, 0), simulation_key=(1, role, 0, 0),
                        setting=setting, historical_summary=_history(raw),
                        computation_time_ns=123, execution_position=position,
                    )
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["historical_grid_auc"], raw)
                        self.assertEqual(row["prior_expected_grid_auc"], target)
                        self.assertEqual(row["historical_auc_adjusted"], adjusted)
                        self.assertEqual(row["prior_mass"], float(mass))
                        self.assertEqual(row["historical_n_positive"], 500)
                        self.assertEqual(row["historical_n_negative"], 500)

    def test_empirical_endpoint_roundoff_does_not_create_historical_adjustments(self):
        # Both references have exactly 500 observations per class. Their
        # trapezoidal floating-point sums previously fell just outside the
        # restriction interval despite exact bin-pair AUCs of .5 and .86.
        equal = [50, 53, 50, 46, 54, 48, 44, 64, 43, 48]
        references = (
            (equal, equal, 250000, 0.5),
            ([0, 0, 0, 0, 0, 10, 0, 0, 490, 0],
             [55, 52, 48, 61, 40, 54, 54, 48, 42, 46], 430000, 0.86),
        )
        _, result = self._capture_predictions()
        for positive, negative, twice_wins, expected in references:
            with self.subTest(expected=expected):
                independent_numerator = sum(
                    positive[i] * negative[j] * (2 * int(i > j) + int(i == j))
                    for i in range(10) for j in range(10)
                )
                self.assertEqual(independent_numerator, twice_wins)
                self.assertEqual(twice_wins / 500000, expected)
                auc = runner.historical_auc_from_scores(
                    np.repeat((np.arange(10) + 0.5) / 10, positive),
                    np.repeat((np.arange(10) + 0.5) / 10, negative), self.gamma,
                )
                self.assertEqual(auc, expected)
                rows = runner._raw_rows_for_replication(
                    self.condition, 0, result["dip_linear_m1"][0],
                    n_replications=2, n_samples=40, master_seed=781,
                    data_key=(1, 1, 0, 0), simulation_key=(1, 6, 0, 0),
                    setting="dip_linear_m1", historical_summary=runner.HistoricalSummary(np.array(positive), np.array(negative)),
                    computation_time_ns=123, execution_position=4,
                )
                for row in rows:
                    self.assertEqual(row["historical_grid_auc"], expected)
                    self.assertEqual(row["prior_expected_grid_auc"], expected)
                    self.assertEqual(row["historical_auc_adjusted"], 0)

    def test_near_endpoint_empirical_aucs_retain_real_restriction_events(self):
        # AUC increments can be as small as 1/(2*500*500). A true value
        # outside the interval must still be adjusted; no tolerance is used.
        references = (
            ([499, 1, 0], [499, 0, 1], 249999, 0.5, True),
            ([499, 0, 1], [499, 1, 0], 250001, 0.500002, False),
            ([140, 360, 0], [499, 1, 0], 429500, 0.859, False),
            ([139, 360, 1], [499, 1, 0], 430001, 0.86, True),
        )
        for positive, negative, numerator, target, adjusted in references:
            with self.subTest(numerator=numerator):
                auc = runner.historical_auc_from_scores(
                    np.repeat([0.05, 0.15, 0.25], positive),
                    np.repeat([0.05, 0.15, 0.25], negative), self.gamma,
                )
                self.assertEqual(auc, numerator / 500000)
                self.assertEqual(runner.restrict_historical_auc(auc), target)
                self.assertEqual(target != auc, adjusted)

    def test_historical_information_changes_only_the_informed_prediction(self):
        lower, first_results = self._capture_predictions(auc=0.5)
        upper, second_results = self._capture_predictions(auc=0.86)
        for setting in ("dip_uniform_m1", "dip_uniform_m5", "dip_uniform_m20"):
            self.assertBandsEqual(lower[setting], upper[setting])
            self.assertEqual(first_results[setting][0], second_results[setting][0])
        for setting in ("dip_linear_m1", "dip_linear_m5", "dip_linear_m20",
                        "dip_histograms_m1", "dip_histograms_m5", "dip_histograms_m20"):
            with self.subTest(setting=setting):
                self.assertFalse(np.array_equal(lower[setting].fpr_center, upper[setting].fpr_center))
        for setting in runner.SETTINGS:
            self.assertEqual(first_results[setting][0].external_grid_auc,
                             second_results[setting][0].external_grid_auc)

    def test_reordering_builds_preserves_each_settings_scientific_draws(self):
        first, first_results = self._capture_predictions(replication=1)
        with patch.object(runner, "_execution_order", return_value=tuple(reversed(runner.SETTINGS))):
            second, second_results = self._capture_predictions(replication=1)
        for setting in runner.SETTINGS:
            self.assertBandsEqual(first[setting], second[setting])
            self.assertEqual(first_results[setting][0], second_results[setting][0])
            self.assertEqual(first_results[setting][1], second_results[setting][1])

    def test_different_historical_arrays_with_same_grid_auc_give_same_prediction(self):
        # Different raw values and class counts carry no information after the
        # historical sample has been reduced to its sole permitted AUC summary.
        first_auc = runner.historical_auc_from_scores(
            np.array([0.09]), np.array([0.01]), self.gamma,
        )
        second_auc = runner.historical_auc_from_scores(
            np.full(7, 0.62), np.full(11, 0.68), self.gamma,
        )
        self.assertEqual(first_auc, second_auc)
        first, _ = self._capture_predictions(auc=first_auc)
        second, _ = self._capture_predictions(auc=second_auc)
        for setting in ("dip_linear_m1", "dip_linear_m5", "dip_linear_m20"):
            with self.subTest(setting=setting):
                self.assertBandsEqual(first[setting], second[setting])

    def test_external_data_change_evaluation_but_never_predictions(self):
        first, first_results = self._capture_predictions()
        changed = utilities.build_test_external_histograms(
            np.repeat([0.15, 0.25, 0.55, 0.75], 5),
            np.repeat([0.05, 0.25, 0.35, 0.65], 5),
            np.full(20, 0.01), np.full(20, 0.99), self.gamma,
        )
        second, second_results = self._capture_predictions(histograms=changed)
        for setting in runner.SETTINGS:
            self.assertBandsEqual(first[setting], second[setting])
            self.assertNotEqual(first_results[setting][0].external_grid_auc,
                                second_results[setting][0].external_grid_auc)

    def test_shared_test_histograms_are_read_only_and_all_bands_precede_evaluation(self):
        original_timer = runner.build_timed_band
        original_evaluation = runner.evaluate_replication
        completed = []
        snapshots = [self.histograms.d_plus_test.copy(), self.histograms.d_minus_test.copy()]

        def capture(builder, **arguments):
            self.assertIs(arguments["d_plus"], self.histograms.d_plus_test)
            self.assertIs(arguments["d_minus"], self.histograms.d_minus_test)
            for name in ("d_plus", "d_minus"):
                self.assertFalse(arguments[name].flags.writeable)
                with self.assertRaises(ValueError):
                    arguments[name][0] += 1
            setting = arguments["diagnostic_context"]["setting"]
            self.assertEqual("historical_grid_auc" in arguments, runner.SETTING_FAMILIES[setting] == "linear")
            self.assertNotIn("d_plus_external", arguments)
            self.assertNotIn("d_minus_external", arguments)
            result = original_timer(builder, **arguments)
            completed.append(setting)
            return result

        def evaluate(histograms, band):
            self.assertEqual(set(completed), set(runner.SETTINGS))
            self.assertIs(histograms, self.histograms)
            return original_evaluation(histograms, band)

        with patch.object(runner, "build_timed_band", side_effect=capture):
            with patch.object(runner, "evaluate_replication", side_effect=evaluate):
                runner._evaluate_methods(
                    self.condition, 1, self.histograms,
                    historical_summary=_history(0.72), n_samples=40, master_seed=91,
                )
        np.testing.assert_array_equal(self.histograms.d_plus_test, snapshots[0])
        np.testing.assert_array_equal(self.histograms.d_minus_test, snapshots[1])

    def test_complete_nine_replication_rotation_and_single_warmup_per_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.csv"
            with patch.object(runner, "warm_up_band", wraps=runner.warm_up_band) as warm:
                with patch.object(runner, "_generate_historical_summary", wraps=runner._generate_historical_summary) as historical:
                    runner.run_experiment(output, n_replications=9, n_samples=40,
                                          conditions=(self.condition,), show_progress=False)
            rows = _read_csv(output)
        self.assertEqual(len(rows), 162)
        self.assertEqual(warm.call_count, 9)
        self.assertEqual(historical.call_count, 9)
        for replication in range(9):
            order = runner.SETTINGS[replication:] + runner.SETTINGS[:replication]
            selected = [r for r in rows if int(r["replication"]) == replication]
            self.assertEqual([r["setting"] for r in selected],
                             [setting for setting in runner.SETTINGS for _ in runner.NOMINAL_LEVELS])
            for row in selected:
                self.assertEqual(int(row["execution_position"]), order.index(row["setting"]) + 1)
        for setting in runner.SETTINGS:
            self.assertEqual({int(row["execution_position"]) for row in rows if row["setting"] == setting}, set(range(1, 10)))
        self.assertEqual(runner._execution_order(9), runner.SETTINGS)
        self.assertEqual(runner._execution_order(10), runner.SETTINGS[1:] + runner.SETTINGS[:1])

    def test_warmup_and_logging_do_not_change_scientific_results_or_historical_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.csv", Path(directory) / "second.csv"
            arguments = dict(n_replications=3, n_samples=40, conditions=(self.condition,))
            runner.run_experiment(first, **arguments, show_progress=False)
            with patch.object(runner, "warm_up_band"), redirect_stdout(io.StringIO()):
                runner.run_experiment(second, **arguments, show_progress=True)
            first_rows, second_rows = _read_csv(first), _read_csv(second)
        for first_row, second_row in zip(first_rows, second_rows, strict=True):
            first_row.pop("computation_time_ns")
            second_row.pop("computation_time_ns")
        self.assertEqual(first_rows, second_rows)

    def test_all_prior_construction_is_timed_but_history_and_evaluation_are_not(self):
        active, clock_calls = False, 0
        events = []
        def clock():
            nonlocal active, clock_calls
            active = not active
            clock_calls += 1
            return clock_calls * 100
        def inside(name, original):
            def call(*args, **kwargs):
                self.assertTrue(active, name + " must be timed")
                events.append(name)
                return original(*args, **kwargs)
            return call
        def outside(name, original):
            def call(*args, **kwargs):
                self.assertFalse(active, name + " must not be timed")
                if name == "evaluation":
                    self.assertEqual(clock_calls, 18)
                events.append(name)
                return original(*args, **kwargs)
            return call
        with (
            patch.object(utilities.time, "perf_counter_ns", side_effect=clock),
            patch.object(runner, "historical_summary_from_scores", side_effect=outside("history", runner.historical_summary_from_scores)),
            patch.object(runner, "symmetric_dirichlet_priors", side_effect=inside("uniform", runner.symmetric_dirichlet_priors)),
            patch.object(runner, "auc_calibrated_prior_bin_probabilities", side_effect=inside("linear", runner.auc_calibrated_prior_bin_probabilities)),
            patch.object(runner, "historical_histogram_prior_bin_probabilities", side_effect=inside("histograms", runner.historical_histogram_prior_bin_probabilities)),
            patch.object(runner, "dirichlet_parameters_from_bin_probabilities", side_effect=inside("scaling", runner.dirichlet_parameters_from_bin_probabilities)),
            patch.object(runner, "restrict_historical_auc", side_effect=inside("restriction", runner.restrict_historical_auc)),
            patch.object(runner, "make_semantic_rng", side_effect=outside("rng", runner.make_semantic_rng)),
            patch.object(runner, "evaluate_replication", side_effect=outside("evaluation", runner.evaluate_replication)),
        ):
            results, _, _ = runner._evaluate_one_replication(
                self.condition, 1, gamma=self.gamma, n_samples=40, master_seed=195,
            )
        self.assertFalse(active)
        self.assertEqual(clock_calls, 18)
        for family in ("uniform", "linear", "histograms", "restriction"):
            self.assertEqual(events.count(family), 3)
        self.assertEqual(events.count("scaling"), 6)
        self.assertEqual(events.count("history"), 1)
        for result in results.values():
            self.assertEqual(result[2], 100)

    def test_historical_preparation_failures_keep_original_exception_and_stream_context(self):
        condition, replication, seed = runner.CLASS_COUNT_CONDITIONS[9], 13, 623
        for failed_stage in ("beta_draw", "auc_calculation"):
            with self.subTest(stage=failed_stage):
                error = AssertionError(f"Injected failure during {failed_stage}")

                class FailingGenerator:
                    def beta(self, *args, **kwargs):
                        raise error

                if failed_stage == "beta_draw":
                    failure = patch.object(runner, "make_semantic_rng", return_value=FailingGenerator())
                else:
                    failure = patch.object(runner, "historical_summary_from_scores", side_effect=error)
                with failure, self.assertRaises(AssertionError) as raised:
                    runner._generate_historical_summary(
                        condition, replication, gamma=self.gamma, master_seed=seed,
                    )
                self.assertIs(raised.exception, error)
                diagnostics = "\n".join(error.__notes__)
                for required in (
                    "historical_preparation", "dip_linear_m1", condition.condition_id,
                    "replication", str(replication),
                    semantic_stream_id(seed, 1, 5, condition.index, replication),
                    semantic_stream_id(seed, 1, 1, condition.index, replication),
                ):
                    self.assertIn(required, diagnostics)

    def test_invalid_historical_arrays_cannot_mask_original_preparation_failure(self):
        error = ValueError("Original historical sample validation failure")

        class InvalidGenerator:
            def beta(self, *args, **kwargs):
                return np.full(kwargs["size"], "invalid", dtype=object)

        with (
            patch.object(runner, "make_semantic_rng", return_value=InvalidGenerator()),
            patch.object(runner, "historical_summary_from_scores", side_effect=error),
            self.assertRaises(ValueError) as raised,
        ):
            runner._generate_historical_summary(
                self.condition, 13, gamma=self.gamma, master_seed=623,
            )
        self.assertIs(raised.exception, error)
        diagnostics = "\n".join(error.__notes__)
        self.assertIn("historical_preparation", diagnostics)
        self.assertIn("positive_diagnostics_unavailable", diagnostics)
        self.assertIn("negative_diagnostics_unavailable", diagnostics)
        self.assertIn("(500,)", diagnostics)

    def test_pre_v4_uniform_baseline_remains_bit_for_bit_scientifically_identical(self):
        fixture_path = Path(__file__).parent / "fixtures" / "main_synthetic_v3_baselines.json"
        fixture = json.loads(fixture_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.csv"
            runner.run_experiment(output, n_replications=fixture["B"], n_samples=fixture["S"],
                                  master_seed=fixture["master_seed"], show_progress=False)
            actual = [row for row in _read_csv(output) if row["setting"] == "dip_uniform_m1"]
        expected = [row for row in fixture["rows"] if row["method"] == "DiP-ROC"]
        self.assertEqual(len(actual), 60)
        self.assertEqual(len(expected), 60)
        for historical, current in zip(expected, actual, strict=True):
            self.assertEqual({key: current[key] for key in historical}, historical)

    def test_v5_both_dip_baselines_remain_bit_for_bit_scientifically_identical(self):
        fixture_path = Path(__file__).parent / "fixtures" / "main_synthetic_v5_baselines.json"
        fixture = json.loads(fixture_path.read_text())
        names = {"dip_historical": "dip_linear_m1", "dip_uniform": "dip_uniform_m1"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot.csv"
            runner.run_experiment(output, n_replications=fixture["n_replications"], n_samples=fixture["n_samples"],
                                  master_seed=fixture["master_seed"], show_progress=False)
            actual = {(row["setting"], row["condition_index"], row["replication"], row["level"]): row
                      for row in _read_csv(output)}
        self.assertEqual(len(fixture["rows"]), 180)
        expected_rows = [row for row in fixture["rows"] if row["setting"] in names]
        self.assertEqual(len(expected_rows), 120)
        for expected in expected_rows:
            key = (names[expected["setting"]], expected["condition_index"], expected["replication"], expected["level"])
            with self.subTest(key=key):
                self.assertEqual({field: actual[key][field] for field in expected if field != "setting"},
                                 {field: value for field, value in expected.items() if field != "setting"})

    def test_frozen_v6_six_dip_settings_remain_bit_for_bit_scientifically_identical(self):
        fixture_path = Path(__file__).parent / "fixtures" / "main_synthetic_v6_baselines.json"
        fixture = json.loads(fixture_path.read_text())
        self.assertEqual(fixture["source_protocol"], 6)
        self.assertEqual((fixture["n_replications"], fixture["n_samples"], fixture["master_seed"]),
                         (2, 40, 20250731))
        legacy_settings = {
            "dip_uniform_m1", "dip_uniform_m5", "dip_linear_m1", "dip_linear_m5",
            "dip_histograms_m1", "dip_histograms_m5",
        }
        self.assertEqual(len(fixture["rows"]), 420)
        self.assertEqual({row["setting"] for row in fixture["rows"]}, legacy_settings | {"bootp"})
        self.assertEqual({int(row["condition_index"]) for row in fixture["rows"]}, set(range(15)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v8_pilot.csv"
            runner.run_experiment(
                output, n_replications=2, n_samples=40, master_seed=20250731,
                show_progress=False,
            )
            all_rows = _read_csv(output)
        self.assertEqual(len(all_rows), 540)
        ignored = {"schema_version", "computation_time_ns", "execution_position"}
        def key(row):
            return (row["setting"], row["condition_index"], row["replication"], row["level"])
        actual = {key(row): {field: value for field, value in row.items() if field not in ignored}
                  for row in all_rows if row["setting"] in legacy_settings}
        expected = {key(row): row for row in fixture["rows"] if row["setting"] in legacy_settings}
        self.assertEqual(len(expected), 360)
        self.assertEqual(set(actual), set(expected))
        for identity in expected:
            with self.subTest(identity=identity):
                self.assertEqual(actual[identity], expected[identity])

    def test_uniform_constructor_scales_each_class_without_changing_observed_counts(self):
        for mass in (1., 5., 20.):
            with self.subTest(mass=mass), patch.object(
                runner, "build_posterior_predictive_band", return_value=object(),
            ) as posterior:
                runner.build_uniform_predictive_band(
                    self.histograms.d_plus_test, self.histograms.d_minus_test,
                    mass, 40, (0.8, 0.95), np.random.default_rng(19),
                )
                arguments = posterior.call_args.kwargs
                for field in ("alpha_plus", "alpha_minus"):
                    np.testing.assert_array_equal(arguments[field], np.full(10, mass / 10))
                    self.assertAlmostEqual(float(arguments[field].sum()), mass)
                self.assertIs(arguments["d_plus"], self.histograms.d_plus_test)
                self.assertIs(arguments["d_minus"], self.histograms.d_minus_test)

    def test_histogram_constructor_matches_independent_prior_and_preserves_test_counts(self):
        history = _history(1.0)
        for mass in (1., 5., 20.):
            with self.subTest(mass=mass), patch.object(runner, "build_posterior_predictive_band", return_value=object()) as posterior:
                runner.build_histogram_predictive_band(
                    self.histograms.d_plus_test, self.histograms.d_minus_test,
                    history.d_plus, history.d_minus, mass, 40, runner.NOMINAL_LEVELS,
                    np.random.default_rng(6))
                args = posterior.call_args.kwargs
                for field, counts in (("alpha_plus", history.d_plus), ("alpha_minus", history.d_minus)):
                    expected = np.array([mass * (int(count) + 0.1) / 501 for count in counts])
                    np.testing.assert_allclose(args[field], expected, rtol=3e-16, atol=0)
                    self.assertAlmostEqual(args[field].sum(), mass)
                    self.assertTrue(np.all(args[field] > 0))
                self.assertIs(args["d_plus"], self.histograms.d_plus_test)
                self.assertEqual(args["d_plus"].sum(), 20)

    def test_histogram_constructor_does_not_restrict_auc_or_require_mirrored_counts(self):
        positive = np.zeros(10, dtype=int); positive[9] = 500
        negative = np.zeros(10, dtype=int); negative[0] = 400; negative[1] = 100
        for mass in (1, 5, 20):
            with self.subTest(mass=mass), patch.object(
                runner, "restrict_historical_auc", side_effect=AssertionError("No AUC restriction permitted"),
            ):
                result = runner.build_histogram_predictive_band(
                    self.histograms.d_plus_test, self.histograms.d_minus_test,
                    positive, negative, float(mass), 40, runner.NOMINAL_LEVELS,
                    np.random.default_rng(19),
                )
                metadata = runner._historical_metadata(
                    f"dip_histograms_m{mass}", runner.HistoricalSummary(positive, negative),
                )
            self.assertEqual(result.fpr_center.size, 11)
            self.assertGreater(metadata["prior_expected_grid_auc"], 0.86)
            self.assertEqual(metadata["historical_auc_adjusted"], "")

    def test_equal_auc_different_histograms_affect_only_histogram_predictions(self):
        first_history = _history(.5)
        equal = np.zeros(10, dtype=int); equal[7] = 500
        second_history = runner.HistoricalSummary(equal, equal)
        self.assertEqual(first_history.grid_auc, second_history.grid_auc)
        def collect(history):
            predictions = {}
            timer = runner.build_timed_band
            def capture(builder, **args):
                result = timer(builder, **args)
                predictions[args["diagnostic_context"]["setting"]] = result[0]
                return result
            with patch.object(runner, "build_timed_band", side_effect=capture):
                runner._evaluate_methods(self.condition, 1, self.histograms, historical_summary=history,
                                         n_samples=40, master_seed=781)
            return predictions
        first, second = collect(first_history), collect(second_history)
        for setting in runner.SETTINGS:
            if runner.SETTING_FAMILIES[setting] == "histograms":
                self.assertFalse(np.array_equal(first[setting].tpr_center, second[setting].tpr_center))
            else:
                self.assertBandsEqual(first[setting], second[setting])

    def test_historical_counts_are_owned_read_only_and_keep_their_auc(self):
        d_plus, d_minus = np.full(10, 50), np.full(10, 50)
        history = runner.HistoricalSummary(d_plus, d_minus)
        d_plus[0] += 1
        self.assertEqual(history.d_plus[0], 50)
        self.assertEqual(history.grid_auc, .5)
        for values in (history.d_plus, history.d_minus):
            self.assertFalse(values.flags.writeable)
            with self.assertRaises(ValueError):
                values[0] += 1
        for invalid in (np.zeros(10, dtype=int), np.ones(10) * .5, np.ones(10, dtype=bool),
                        np.full(10, -1), np.ones(9, dtype=int)):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                runner.HistoricalSummary(invalid, np.ones(10, dtype=int))


if __name__ == "__main__":
    unittest.main()
