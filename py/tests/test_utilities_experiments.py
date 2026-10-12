import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import roc_utils.utilities_experiments as experiment_utils  # noqa: E402
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    fixed_uniform_grid,
    symmetric_dirichlet_priors,
)
from roc_utils.utilities_experiments import (  # noqa: E402
    PredictiveBand,
    ReplicationMeasurements,
    TestExternalHistograms,
    build_posterior_predictive_band,
    build_bootstrap_predictive_band,
    build_timed_band,
    build_test_external_histograms,
    evaluate_replication,
    summarize_replication_measurements,
    warm_up_band,
)


class SharedExperimentUtilityTests(unittest.TestCase):
    def test_builds_exact_k2_histograms_and_preserves_class_counts(self):
        gamma = fixed_uniform_grid(2)
        histograms = build_test_external_histograms(
            [0.0, 0.5, 1.0],
            [0.25, 0.75],
            [0.1, 0.4, 0.9],
            [0.0, 1.0],
            gamma,
        )

        np.testing.assert_array_equal(gamma, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(histograms.d_plus_test, [1, 2])
        np.testing.assert_array_equal(histograms.d_minus_test, [1, 1])
        np.testing.assert_array_equal(histograms.d_plus_external, [2, 1])
        np.testing.assert_array_equal(histograms.d_minus_external, [1, 1])
        self.assertEqual(histograms.d_plus_test.sum(), histograms.d_plus_external.sum())
        self.assertEqual(histograms.d_minus_test.sum(), histograms.d_minus_external.sum())
        for counts in (
            histograms.d_plus_test,
            histograms.d_minus_test,
            histograms.d_plus_external,
            histograms.d_minus_external,
        ):
            self.assertFalse(counts.flags.writeable)
            with self.assertRaises(ValueError):
                counts[0] = 0

    def test_one_band_build_produces_exact_external_deviation_and_auc(self):
        gamma = fixed_uniform_grid(2)
        histograms = build_test_external_histograms(
            [0.1, 0.2, 0.6, 0.7],
            [0.1, 0.2, 0.6, 0.7],
            [0.1, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.8],
            gamma,
        )
        band = PredictiveBand(
            fpr_center=np.array([0.0, 0.2, 1.0]),
            tpr_center=np.array([0.0, 0.8, 1.0]),
            roc_radii={0.80: 0.07, 0.95: 0.08},
            auc_intervals={0.80: (0.74, 0.76), 0.95: (0.70, 0.90)},
            auc_samples=np.linspace(0.70, 0.80, 20),
        )

        with patch.object(
            experiment_utils,
            "build_posterior_predictive_band",
            return_value=band,
        ) as mocked_band:
            constructed = experiment_utils.build_posterior_predictive_band(
                d_plus=histograms.d_plus_test,
                d_minus=histograms.d_minus_test,
                alpha_plus=np.array([0.5, 0.5]),
                alpha_minus=np.array([0.5, 0.5]),
                S=20,
                levels=(0.80, 0.95),
                rng=np.random.default_rng(123),
            )
            measurements = evaluate_replication(histograms, constructed)

        self.assertEqual(mocked_band.call_count, 1)
        band_arguments = mocked_band.call_args.kwargs
        np.testing.assert_array_equal(
            band_arguments["d_plus"],
            histograms.d_plus_test,
        )
        np.testing.assert_array_equal(
            band_arguments["d_minus"],
            histograms.d_minus_test,
        )
        self.assertFalse(
            np.array_equal(
                band_arguments["d_plus"],
                histograms.d_plus_external,
            )
        )
        self.assertEqual(band_arguments["levels"], (0.80, 0.95))
        self.assertAlmostEqual(
            measurements.external_max_roc_deviation,
            np.sqrt(0.005),
        )
        self.assertAlmostEqual(measurements.external_grid_auc, 0.75)
        self.assertEqual(measurements.roc_radii, band.roc_radii)
        self.assertEqual(measurements.auc_intervals, band.auc_intervals)

    def test_perfect_external_auc_survives_trapezoid_roundoff(self):
        # Exact AUC=1 by complete separation of the occupied class bins.
        # The old trapezoidal sum returned 1.0000000000000002 for this ROC.
        positive = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 20])
        negative = np.array([0, 0, 6, 5, 6, 1, 2, 0, 0, 0])
        histograms = TestExternalHistograms(
            positive, negative, positive.copy(), negative.copy(),
        )
        band = build_bootstrap_predictive_band(
            positive, negative, 4000, (0.80, 0.95), np.random.default_rng(123),
        )
        measurements = evaluate_replication(histograms, band)
        self.assertEqual(measurements.external_grid_auc, 1.0)
        self.assertEqual(measurements.external_max_roc_deviation, 0.0)
        summaries = summarize_replication_measurements(
            [measurements, measurements], (0.80, 0.95),
        )
        for summary in summaries.values():
            self.assertEqual(summary.auc_coverage, 1.0)
            self.assertEqual(summary.roc_coverage, 1.0)
            self.assertGreaterEqual(summary.auc_interval_width_mean, 0.0)

    def test_aggregates_coverage_and_sharpness_at_each_level(self):
        deviations = (0.10, 0.25, 0.20, 0.50)
        external_aucs = (0.10, 0.80, 0.50, 0.95)
        radii_080 = (0.10, 0.20, 0.30, 0.40)
        radii_095 = (0.20, 0.30, 0.40, 0.50)
        intervals_080 = (
            (0.0, 0.2),
            (0.3, 0.7),
            (0.2, 0.8),
            (0.1, 0.9),
        )
        replications = [
            ReplicationMeasurements(
                external_max_roc_deviation=deviations[index],
                external_grid_auc=external_aucs[index],
                roc_radii={
                    0.80: radii_080[index],
                    0.95: radii_095[index],
                },
                auc_intervals={
                    0.80: intervals_080[index],
                    0.95: (0.0, 1.0),
                },
            )
            for index in range(4)
        ]

        summaries = summarize_replication_measurements(
            replications,
            levels=(0.80, 0.95),
        )

        self.assertEqual(set(summaries), {0.80, 0.95})

        summary_080 = summaries[0.80]
        self.assertEqual(summary_080.n_replications, 4)
        self.assertAlmostEqual(summary_080.roc_coverage, 0.5)
        self.assertAlmostEqual(summary_080.roc_coverage_standard_error, 0.25)
        self.assertAlmostEqual(summary_080.auc_coverage, 0.5)
        self.assertAlmostEqual(summary_080.auc_coverage_standard_error, 0.25)
        self.assertAlmostEqual(summary_080.roc_radius_mean, 0.25)
        self.assertAlmostEqual(
            summary_080.roc_radius_standard_error,
            np.sqrt(1.0 / 240.0),
        )
        self.assertAlmostEqual(summary_080.auc_interval_width_mean, 0.5)
        self.assertAlmostEqual(
            summary_080.auc_interval_width_standard_error,
            np.sqrt(1.0 / 60.0),
        )

        summary_095 = summaries[0.95]
        self.assertEqual(summary_095.n_replications, 4)
        self.assertAlmostEqual(summary_095.roc_coverage, 1.0)
        self.assertAlmostEqual(summary_095.roc_coverage_standard_error, 0.0)
        self.assertAlmostEqual(summary_095.auc_coverage, 1.0)
        self.assertAlmostEqual(summary_095.auc_coverage_standard_error, 0.0)
        self.assertAlmostEqual(summary_095.roc_radius_mean, 0.35)
        self.assertAlmostEqual(
            summary_095.roc_radius_standard_error,
            np.sqrt(1.0 / 240.0),
        )
        self.assertAlmostEqual(summary_095.auc_interval_width_mean, 1.0)
        self.assertAlmostEqual(summary_095.auc_interval_width_standard_error, 0.0)

    def test_rejects_duplicate_or_missing_levels_and_unequal_class_counts(self):
        gamma = fixed_uniform_grid(2)
        with self.assertRaises(AssertionError):
            build_test_external_histograms(
                [0.1, 0.9],
                [0.1, 0.5, 0.9],
                [0.1, 0.5, 0.9],
                [0.1, 0.9],
                gamma,
            )

        fractional_histograms = TestExternalHistograms(
            d_plus_test=np.array([1, 1]),
            d_minus_test=np.array([1, 1]),
            d_plus_external=np.array([0.5, 1.5]),
            d_minus_external=np.array([1, 1]),
        )
        with patch.object(
            experiment_utils,
            "build_posterior_predictive_band",
        ) as mocked_band:
            with self.assertRaises(AssertionError):
                evaluate_replication(
                    fractional_histograms,
                    PredictiveBand(
                        fpr_center=np.array([0.0, 0.5, 1.0]),
                        tpr_center=np.array([0.0, 0.5, 1.0]),
                        roc_radii={0.80: 0.2, 0.95: 0.3},
                        auc_intervals={0.80: (0.3, 0.7), 0.95: (0.2, 0.8)},
                        auc_samples=np.linspace(0.0, 1.0, 20),
                    ),
                )
        mocked_band.assert_not_called()

        complete = ReplicationMeasurements(
            external_max_roc_deviation=0.1,
            external_grid_auc=0.7,
            roc_radii={0.80: 0.2, 0.95: 0.3},
            auc_intervals={0.80: (0.5, 0.8), 0.95: (0.4, 0.9)},
        )
        with self.assertRaises(ValueError):
            summarize_replication_measurements(
                [complete, complete],
                levels=(0.80, 0.80),
            )
        with self.assertRaises(ValueError):
            summarize_replication_measurements(
                [complete],
                levels=(0.80, 0.95),
            )

        missing_level = ReplicationMeasurements(
            external_max_roc_deviation=0.1,
            external_grid_auc=0.7,
            roc_radii={0.80: 0.2},
            auc_intervals={0.80: (0.5, 0.8)},
        )
        with self.assertRaises(AssertionError):
            summarize_replication_measurements(
                [missing_level, missing_level],
                levels=(0.80, 0.95),
            )

    def test_requires_explicit_rng_and_nested_multilevel_summaries(self):
        histograms = build_test_external_histograms(
            [0.1, 0.6],
            [0.1, 0.6],
            [0.2, 0.7],
            [0.2, 0.7],
            fixed_uniform_grid(2),
        )
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(2, 1.0)

        with patch.object(
            experiment_utils,
            "sample_posterior_predictive_rocs",
        ) as mocked_sampler:
            with self.assertRaisesRegex(
                AssertionError,
                "caller-supplied numpy.random.Generator",
            ):
                build_posterior_predictive_band(
                    histograms.d_plus_test,
                    histograms.d_minus_test,
                    alpha_plus,
                    alpha_minus,
                    S=20,
                    levels=(0.80, 0.95),
                    rng=None,
                )
        mocked_sampler.assert_not_called()

        nonnested_band = PredictiveBand(
            fpr_center=np.array([0.0, 0.5, 1.0]),
            tpr_center=np.array([0.0, 0.5, 1.0]),
            roc_radii={0.80: 0.20, 0.95: 0.30},
            auc_intervals={0.80: (0.30, 0.70), 0.95: (0.40, 0.80)},
            auc_samples=np.linspace(0.0, 1.0, 20),
        )
        with self.assertRaisesRegex(
            AssertionError,
            "Higher-level AUC intervals",
        ):
            evaluate_replication(histograms, nonnested_band)

    def test_evaluation_module_does_not_import_experiment_orchestration(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.evaluation_criteria; "
            "assert 'roc_utils.utilities_experiments' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_tiny_complete_evaluation_is_seed_reproducible(self):
        histograms = build_test_external_histograms(
            [0.1, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.8],
            [0.2, 0.5, 0.7, 0.9],
            [0.1, 0.4, 0.6, 0.8],
            fixed_uniform_grid(2),
        )
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(2, 1.0)

        def run_once():
            band = build_posterior_predictive_band(
                histograms.d_plus_test,
                histograms.d_minus_test,
                alpha_plus,
                alpha_minus,
                S=40,
                levels=(0.80, 0.95),
                rng=np.random.default_rng(123),
            )
            return evaluate_replication(histograms, band)

        first = run_once()
        second = run_once()

        self.assertEqual(
            first.external_max_roc_deviation,
            second.external_max_roc_deviation,
        )
        self.assertEqual(first.external_grid_auc, second.external_grid_auc)
        self.assertEqual(first.roc_radii, second.roc_radii)
        self.assertEqual(first.auc_intervals, second.auc_intervals)
        self.assertFalse(hasattr(first, "auc_samples"))

    def test_two_centers_share_the_external_auc_but_have_distinct_deviations(self):
        histograms = build_test_external_histograms(
            [0.1, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.8],
            [0.1, 0.2, 0.3, 0.8], [0.1, 0.2, 0.6, 0.8],
            fixed_uniform_grid(2),
        )
        shared = {
            "roc_radii": {0.80: 1.8, 0.95: 2.0},
            "auc_intervals": {0.80: (0.2, 0.8), 0.95: (0.0, 1.0)},
            "auc_samples": np.array([0.25, 0.75]),
        }
        first_band = PredictiveBand(
            fpr_center=np.array([0.0, 0.25, 1.0]),
            tpr_center=np.array([0.0, 0.75, 1.0]), **shared,
        )
        second_band = PredictiveBand(
            fpr_center=np.array([0.0, 0.5, 1.0]),
            tpr_center=np.array([0.0, 0.5, 1.0]), **shared,
        )
        with (
            patch.object(experiment_utils, "build_posterior_predictive_band") as dip_builder,
            patch.object(experiment_utils, "build_bootstrap_predictive_band") as boot_builder,
            patch.object(experiment_utils, "sample_posterior_predictive_rocs") as dip_sampler,
            patch.object(experiment_utils, "sample_stratified_bootstrap_histograms") as boot_sampler,
        ):
            first = evaluate_replication(histograms, first_band)
            second = evaluate_replication(histograms, second_band)
        for mock in (dip_builder, boot_builder, dip_sampler, boot_sampler):
            mock.assert_not_called()
        self.assertEqual(first.external_grid_auc, 0.375)
        self.assertEqual(first.external_grid_auc, second.external_grid_auc)
        self.assertAlmostEqual(first.external_max_roc_deviation, np.sqrt(5.0) / 4.0)
        self.assertEqual(second.external_max_roc_deviation, 0.25)
        self.assertEqual(first.roc_radii, {0.80: 1.8, 0.95: 2.0})
        self.assertEqual(second.roc_radii, first.roc_radii)

    def test_external_changes_leave_both_methods_predictions_unchanged(self):
        first_histograms = build_test_external_histograms(
            [0.1, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.8],
            [0.1, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.8],
            fixed_uniform_grid(2),
        )
        second_histograms = build_test_external_histograms(
            [0.1, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.8],
            [0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9],
            fixed_uniform_grid(2),
        )
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(2, 1.0)
        for builder, extra in (
            (build_posterior_predictive_band, {"alpha_plus": alpha_plus, "alpha_minus": alpha_minus}),
            (build_bootstrap_predictive_band, {}),
        ):
            with self.subTest(builder=builder.__name__):
                bands = [
                    builder(
                        histograms.d_plus_test, histograms.d_minus_test,
                        S=40, levels=(0.80, 0.95), rng=np.random.default_rng(71),
                        **extra,
                    )
                    for histograms in (first_histograms, second_histograms)
                ]
                first = evaluate_replication(first_histograms, bands[0])
                second = evaluate_replication(second_histograms, bands[1])
                self.assertEqual(first.roc_radii, second.roc_radii)
                self.assertEqual(first.auc_intervals, second.auc_intervals)
                np.testing.assert_array_equal(bands[0].auc_samples, bands[1].auc_samples)
                np.testing.assert_array_equal(bands[0].fpr_center, bands[1].fpr_center)
                np.testing.assert_array_equal(bands[0].tpr_center, bands[1].tpr_center)
                self.assertNotEqual(first.external_grid_auc, second.external_grid_auc)
                self.assertNotEqual(first.external_max_roc_deviation, second.external_max_roc_deviation)

    def test_timing_encloses_only_one_construction_inside_the_thread_limit(self):
        events = []
        band = PredictiveBand(
            np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]),
            {0.80: 0.2, 0.95: 0.3}, {0.80: (0.3, 0.7), 0.95: (0.2, 0.8)},
            np.array([0.4, 0.6]),
        )
        rng = np.random.default_rng(12)
        histogram = np.array([1, 1])

        @contextmanager
        def thread_limit(*, limits):
            self.assertEqual(limits, 1)
            events.append("enter thread limit")
            try:
                yield
            finally:
                events.append("leave thread limit")

        ticks = iter((100, 175))

        def clock():
            tick = next(ticks)
            events.append(("clock", tick))
            return tick

        def builder(**kwargs):
            self.assertEqual(set(kwargs), {"d_plus", "d_minus", "S", "levels", "rng"})
            self.assertIs(kwargs["d_plus"], histogram)
            self.assertIs(kwargs["d_minus"], histogram)
            self.assertIs(kwargs["rng"], rng)
            self.assertEqual(kwargs["S"], 40)
            self.assertEqual(kwargs["levels"], (0.80, 0.95))
            events.append("construct both levels")
            return band

        with (
            patch.object(experiment_utils, "threadpool_limits", side_effect=thread_limit),
            patch.object(experiment_utils.time, "perf_counter_ns", side_effect=clock) as timer,
        ):
            returned, elapsed = build_timed_band(
                builder, d_plus=histogram, d_minus=histogram,
                S=40, levels=(0.80, 0.95), rng=rng,
                diagnostic_context={"method": "BootP-ROC", "replication_zero_based": 7},
            )
        self.assertIs(returned, band)
        self.assertEqual(elapsed, 75)
        self.assertIs(type(elapsed), int)
        self.assertEqual(timer.call_count, 2)
        self.assertEqual(events, [
            "enter thread limit", ("clock", 100), "construct both levels",
            ("clock", 175), "leave thread limit",
        ])

    def test_warm_up_is_unmeasured_and_timing_failure_restores_thread_limit(self):
        events = []

        @contextmanager
        def thread_limit(*, limits):
            self.assertEqual(limits, 1)
            events.append("enter")
            try:
                yield
            finally:
                events.append("leave")

        def builder(**kwargs):
            self.assertEqual(kwargs["levels"], (0.80, 0.95))
            events.append("warm up")

        with (
            patch.object(experiment_utils, "threadpool_limits", side_effect=thread_limit),
            patch.object(experiment_utils.time, "perf_counter_ns") as timer,
        ):
            self.assertIsNone(warm_up_band(builder, levels=(0.80, 0.95)))
        timer.assert_not_called()
        self.assertEqual(events, ["enter", "warm up", "leave"])

        events.clear()

        def fail():
            events.append("failed construction")
            raise RuntimeError("injected construction failure")

        with (
            patch.object(experiment_utils, "threadpool_limits", side_effect=thread_limit),
            patch.object(experiment_utils.time, "perf_counter_ns", return_value=100),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected construction failure"):
                build_timed_band(fail)
        self.assertEqual(events, ["enter", "failed construction", "leave"])

    def test_failure_notes_preserve_exception_rng_inputs_and_timing_scope(self):
        for wrapper, phase, expected_clock_calls in (
            (warm_up_band, "warmup", 0),
            (build_timed_band, "construction", 1),
        ):
            with self.subTest(phase=phase):
                events = []

                class DiagnosticError(AssertionError):
                    def add_note(self, note):
                        events.append("diagnostic")
                        super().add_note(note)

                failure = DiagnosticError("injected invalid AUC")
                rng = np.random.default_rng(321)
                reference_rng = np.random.default_rng(321)
                histogram = np.array([1, 2])
                context = {"method": "BootP-ROC", "replication_zero_based": 7}

                @contextmanager
                def thread_limit(*, limits):
                    self.assertEqual(limits, 1)
                    events.append("enter")
                    try:
                        yield
                    finally:
                        events.append("leave")

                def fail(**kwargs):
                    self.assertNotIn("diagnostic_context", kwargs)
                    self.assertIs(kwargs["rng"], rng)
                    rng.random()
                    events.append("builder")
                    raise failure

                with (
                    patch.object(experiment_utils, "threadpool_limits", side_effect=thread_limit),
                    patch.object(experiment_utils.time, "perf_counter_ns", return_value=100) as timer,
                ):
                    with self.assertRaises(DiagnosticError) as caught:
                        wrapper(
                            fail, d_plus=histogram, d_minus=histogram,
                            S=40, levels=(0.80, 0.95), rng=rng,
                            diagnostic_context=context,
                        )
                self.assertIs(caught.exception, failure)
                self.assertEqual(timer.call_count, expected_clock_calls)
                self.assertEqual(events, ["enter", "builder", "leave", "diagnostic"])
                self.assertEqual(len(failure.__notes__), 1)
                note = failure.__notes__[0]
                for text in (
                    "BootP-ROC", "replication_zero_based", "7", phase,
                    "'d_plus': [1, 2]", "'d_minus': [1, 2]", "'S': 40",
                ):
                    self.assertIn(text, note)
                reference_rng.random()
                self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
                np.testing.assert_array_equal(histogram, [1, 2])
                self.assertEqual(context, {"method": "BootP-ROC", "replication_zero_based": 7})

    def test_timer_accepts_zero_duration_and_rejects_invalid_clock_results(self):
        band = PredictiveBand(
            np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]),
            {0.80: 0.2, 0.95: 0.3}, {0.80: (0.3, 0.7), 0.95: (0.2, 0.8)},
            np.array([0.4, 0.6]),
        )
        with patch.object(experiment_utils.time, "perf_counter_ns", side_effect=[100, 100]):
            returned, elapsed = build_timed_band(lambda: band)
        self.assertIs(returned, band)
        self.assertEqual(elapsed, 0)
        for ticks in ([100, 99], [100, 100.5]):
            with self.subTest(ticks=ticks):
                with patch.object(experiment_utils.time, "perf_counter_ns", side_effect=ticks):
                    with self.assertRaises(AssertionError):
                        build_timed_band(lambda: band)

    def test_warm_up_with_its_own_rng_preserves_the_scientific_draws(self):
        histogram = np.array([1, 3])
        options = {
            "d_plus": histogram,
            "d_minus": histogram[::-1].copy(),
            "S": 40,
            "levels": (0.80, 0.95),
        }
        for builder, prior_options in (
            (build_bootstrap_predictive_band, {}),
            (build_posterior_predictive_band,
             {"alpha_plus": np.array([0.5, 0.5]), "alpha_minus": np.array([0.5, 0.5])}),
        ):
            with self.subTest(builder=builder.__name__):
                scientific_rng = np.random.default_rng(113)
                warm_up_band(builder, **options, **prior_options, rng=np.random.default_rng(112))
                after_warm_up = builder(**options, **prior_options, rng=scientific_rng)
                reference = builder(**options, **prior_options, rng=np.random.default_rng(113))
                np.testing.assert_array_equal(after_warm_up.auc_samples, reference.auc_samples)
                self.assertEqual(after_warm_up.roc_radii, reference.roc_radii)
                self.assertEqual(after_warm_up.auc_intervals, reference.auc_intervals)

    def test_external_evaluation_rejects_invalid_band_centers_and_summaries(self):
        histograms = build_test_external_histograms(
            [0.1, 0.6], [0.1, 0.6], [0.2, 0.7], [0.2, 0.7], fixed_uniform_grid(2),
        )
        valid = {
            "fpr_center": np.array([0.0, 0.5, 1.0]),
            "tpr_center": np.array([0.0, 0.5, 1.0]),
            "roc_radii": {0.80: 0.2, 0.95: 0.3},
            "auc_intervals": {0.80: (0.3, 0.7), 0.95: (0.2, 0.8)},
            "auc_samples": np.array([0.4, 0.6]),
        }
        for changed in (
            {"fpr_center": np.array([0.0, 1.0])},
            {"tpr_center": np.array([0.0, np.nan, 1.0])},
            {"roc_radii": {0.80: 0.2, 0.95: 2.01}},
            {"roc_radii": {0.80: 0.4, 0.95: 0.3}},
            {"auc_intervals": {0.80: (0.3, 0.7)}},
            {"auc_intervals": {0.80: (0.7, 0.3), 0.95: (0.2, 0.8)}},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(AssertionError):
                    evaluate_replication(histograms, PredictiveBand(**{**valid, **changed}))

    def test_external_evaluation_enforces_complete_roc_endpoints_at_numerical_tolerance(self):
        histograms = build_test_external_histograms(
            [0.1, 0.6], [0.1, 0.6], [0.2, 0.7], [0.2, 0.7], fixed_uniform_grid(2),
        )
        # The external ROC is exactly (0,0), (1/2,1/2), (1,1).
        # Check both coordinates at both endpoints, allowing only 1e-12
        # arithmetic excursions rather than NumPy's default relative tolerance.
        for coordinate in ("fpr_center", "tpr_center"):
            for endpoint, accepted, rejected in (
                (0, (0.0, 0.5e-12, 1e-12), (np.nextafter(1e-12, np.inf), 1e-9)),
                (-1, (1.0, 1.0 - 0.5e-12, 1.0 - 1e-12),
                 (np.nextafter(1.0 - 1e-12, 0.0), 0.999999)),
            ):
                for value in accepted + rejected:
                    with self.subTest(coordinate=coordinate, endpoint=endpoint, value=value):
                        arrays = {
                            "fpr_center": np.array([0.0, 0.5, 1.0]),
                            "tpr_center": np.array([0.0, 0.5, 1.0]),
                        }
                        arrays[coordinate][endpoint] = value
                        band = PredictiveBand(
                            **arrays, roc_radii={0.8: 0.2},
                            auc_intervals={0.8: (0.3, 0.7)}, auc_samples=np.array([0.4, 0.6]),
                        )
                        if value in rejected:
                            with self.assertRaisesRegex(AssertionError, "ROC curve must (start|end)"):
                                evaluate_replication(histograms, band)
                        else:
                            measurements = evaluate_replication(histograms, band)
                            self.assertLessEqual(measurements.external_max_roc_deviation, 1e-12)
                            self.assertEqual(measurements.external_grid_auc, 0.5)

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import roc_utils.utilities_experiments"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "utilities_experiments relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
