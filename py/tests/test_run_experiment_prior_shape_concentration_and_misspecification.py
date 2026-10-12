import ast
import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_experiment_sensitivity_to_prior_specification as runner  # noqa: E402
from roc_utils.evaluation_criteria import is_external_auc_covered  # noqa: E402
from roc_utils.semantic_rng import semantic_stream_id  # noqa: E402
from roc_utils.utilities_experiments import (  # noqa: E402
    ReplicationMeasurements,
    TestExternalHistograms,
    summarize_replication_measurements,
)


def _fixed_measurements() -> ReplicationMeasurements:
    return ReplicationMeasurements(
        external_max_roc_deviation=0.2 + 0.5e-12,
        external_grid_auc=0.4 - 0.5e-12,
        roc_radii={0.80: 0.2, 0.95: 0.3},
        auc_intervals={0.80: (0.4, 0.7), 0.95: (0.3, 0.8)},
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sort_key(row: dict[str, str]):
    return (
        int(row["condition_index"]),
        int(row["replication"]),
        float(row["level"]),
    )


def _scientific_rows(rows):
    # Wall-clock measurements and execution positions are run metadata.
    return [{key: value for key, value in row.items()
             if key not in ("computation_time_ns", "execution_position")}
            for row in rows]


class PriorShapeExperimentTests(unittest.TestCase):
    def test_exact_design_profiles_aucs_and_nine_dirichlet_priors(self):
        self.assertEqual(
            runner.EXPERIMENT_NAME,
            "prior_shape_concentration_and_misspecification",
        )
        self.assertEqual(
            runner.RESULTS_DIRECTORY_NAME,
            "sensitivity_to_prior_specification",
        )
        self.assertEqual(
            runner.DEFAULT_OUTPUT_CSV.relative_to(runner.PY_DIR),
            Path(
                "results/current_experiments/"
                "sensitivity_to_prior_specification/raw_replications_v3.csv"
            ),
        )
        self.assertEqual(runner.SCORE_SCENARIO, "Medium")
        self.assertEqual(runner.N_POSITIVE, 50)
        self.assertEqual(runner.N_NEGATIVE, 50)
        self.assertEqual(runner.N_BINS, 10)
        self.assertEqual(
            runner.PRIOR_PROFILES,
            ("oracle", "auc_compatible", "symmetric"),
        )
        self.assertEqual(runner.PRIOR_MASSES, (1, 20, 50))
        self.assertEqual(runner.NOMINAL_LEVELS, (0.80, 0.95))
        self.assertEqual(runner.OFFICIAL_REPLICATIONS, 10_000)
        self.assertEqual(runner.OFFICIAL_SAMPLES, 4000)
        self.assertEqual(
            [condition.condition_id for condition in runner.PRIOR_CONDITIONS],
            [
                "oracle_m_1",
                "oracle_m_20",
                "auc_compatible_m_1",
                "auc_compatible_m_20",
                "symmetric_m_1",
                "symmetric_m_20",
                "oracle_m_50",
                "auc_compatible_m_50",
                "symmetric_m_50",
            ],
        )
        self.assertEqual(10 * 10_000 * 2, 200_000)

        profiles = runner.build_prior_profiles()
        # Independent reference: integrate the beta PDF in each bin, then
        # sum positive/negative bin-pair wins (half credit for a tie).
        normalization = math.gamma(6.0) / (math.gamma(3.5) * math.gamma(2.5))
        oracle_plus = np.array([
            quad(lambda x: normalization * x**2.5 * (1.0 - x)**1.5,
                 k / 10.0, (k + 1) / 10.0, epsabs=1e-14, epsrel=1e-13)[0]
            for k in range(10)
        ])
        oracle_minus = oracle_plus[::-1]
        true_auc = sum(
            oracle_plus[i] * oracle_minus[j] * (1.0 if i > j else 0.5 if i == j else 0.0)
            for i in range(10) for j in range(10)
        )
        calibrated_plus = np.array([
            0.1 + 6.0 * (true_auc - 0.5) * (k - 5.5) / 99.0
            for k in range(1, 11)
        ])
        np.testing.assert_allclose(
            profiles["oracle"].q_plus,
            oracle_plus,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            profiles["oracle"].q_minus,
            oracle_plus[::-1],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            profiles["auc_compatible"].q_plus,
            calibrated_plus,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            profiles["auc_compatible"].q_minus,
            calibrated_plus[::-1],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            profiles["symmetric"].q_plus,
            np.full(10, 0.1),
        )
        np.testing.assert_array_equal(
            profiles["symmetric"].q_minus,
            np.full(10, 0.1),
        )
        self.assertAlmostEqual(true_auc, 0.7298989020330722, places=14)
        self.assertAlmostEqual(
            profiles["oracle"].expected_grid_auc,
            true_auc,
            places=15,
        )
        self.assertAlmostEqual(
            profiles["auc_compatible"].expected_grid_auc,
            true_auc,
            places=15,
        )
        self.assertLess(true_auc, 13 / 15)
        self.assertEqual(profiles["symmetric"].expected_grid_auc, 0.5)
        self.assertFalse(
            np.allclose(
                profiles["oracle"].q_plus,
                profiles["auc_compatible"].q_plus,
            )
        )
        for profile in profiles.values():
            self.assertFalse(profile.q_plus.flags.writeable)
            self.assertFalse(profile.q_minus.flags.writeable)

        priors = runner._build_condition_priors(profiles)
        for condition in runner.PRIOR_CONDITIONS:
            alpha_plus, alpha_minus = priors[condition.index]
            profile = profiles[condition.prior_profile]
            np.testing.assert_allclose(
                alpha_plus,
                condition.prior_mass * profile.q_plus,
                rtol=0.0,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                alpha_minus,
                condition.prior_mass * profile.q_minus,
                rtol=0.0,
                atol=1e-15,
            )
            self.assertAlmostEqual(float(np.sum(alpha_plus)), condition.prior_mass)
            self.assertAlmostEqual(float(np.sum(alpha_minus)), condition.prior_mass)
            self.assertTrue(np.all(alpha_plus > 0.0))
            self.assertTrue(np.all(alpha_minus > 0.0))

        conditions_by_setting = {
            (condition.prior_profile, condition.prior_mass): condition
            for condition in runner.PRIOR_CONDITIONS
        }
        for profile_name in runner.PRIOR_PROFILES:
            alpha_one = priors[conditions_by_setting[(profile_name, 1)].index]
            alpha_twenty = priors[
                conditions_by_setting[(profile_name, 20)].index
            ]
            alpha_fifty = priors[
                conditions_by_setting[(profile_name, 50)].index
            ]
            for class_index in (0, 1):
                np.testing.assert_allclose(
                    alpha_fifty[class_index],
                    50.0 * alpha_one[class_index],
                    rtol=0.0,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    alpha_fifty[class_index],
                    2.5 * alpha_twenty[class_index],
                    rtol=0.0,
                    atol=1e-14,
                )
        self.assertFalse(hasattr(runner, "linear_informative_priors"))

        with self.assertRaises(AssertionError):
            runner.PriorCondition(True, "oracle", 1)
        with self.assertRaises(AssertionError):
            runner.PriorCondition(0, "oracle", True)

    def test_semantic_streams_encode_the_paired_design(self):
        master_seed = 813
        for replication in (0, 1, 9_999):
            data_key = runner._data_stream_key(replication)
            self.assertEqual(data_key, (2, 1, replication))
            posterior_keys = {
                runner._posterior_stream_key(condition, replication)
                for condition in runner.PRIOR_CONDITIONS
            }
            self.assertEqual(len(posterior_keys), 9)
            data_id = semantic_stream_id(master_seed, *data_key)
            posterior_ids = {
                semantic_stream_id(master_seed, *key)
                for key in posterior_keys
            }
            self.assertNotIn(data_id, posterior_ids)
            self.assertTrue(data_id.startswith("PCG64:813:2:1:"))

    def test_bootstrap_stream_is_prior_independent_and_warmup_is_reserved(self):
        for replication in (0, 1, 9999):
            self.assertEqual(runner._bootstrap_stream_key(replication), (2, 3, replication))
            keys = {runner._data_stream_key(replication), runner._bootstrap_stream_key(replication)}
            keys.update(runner._posterior_stream_key(condition, replication)
                        for condition in runner.PRIOR_CONDITIONS)
            self.assertEqual(len(keys), 11)
            warm_keys = {runner._warmup_stream_key(condition.index, "DiP-ROC")
                         for condition in runner.PRIOR_CONDITIONS}
            warm_keys.add(runner._warmup_stream_key(9, "BootP-ROC"))
            self.assertEqual(len(warm_keys), 10)
            self.assertTrue(warm_keys.isdisjoint(keys))
            self.assertEqual(runner._warmup_stream_key(9, "BootP-ROC"), (2, 4, 3, 9, 0))

    def test_construction_failures_identify_prior_or_shared_bootstrap_and_phase(self):
        condition = runner.PRIOR_CONDITIONS[3]
        priors = runner._build_condition_priors(runner.build_prior_profiles())
        for method, name, role in (
            ("DiP-ROC", "build_posterior_predictive_band", 2),
            ("BootP-ROC", "build_bootstrap_predictive_band", 3),
        ):
            for phase, replication in (("warmup", 0), ("construction", 1)):
                with self.subTest(method=method, phase=phase):
                    histograms, _ = runner._generate_replication_histograms(
                        replication, gamma=runner.fixed_uniform_grid(10), master_seed=17,
                    )
                    error = AssertionError("injected builder failure")
                    captured = {}

                    def fail(**kwargs):
                        self.assertNotIn("diagnostic_context", kwargs)
                        captured.update(rng=kwargs["rng"], state=kwargs["rng"].bit_generator.state)
                        raise error

                    with patch.object(runner, name, new=fail):
                        with self.assertRaises(AssertionError) as raised:
                            if phase == "warmup":
                                runner._warm_up_conditions(
                                    histograms, (condition,), priors, n_samples=40, master_seed=17,
                                )
                            elif method == "DiP-ROC":
                                runner._evaluate_prior_condition(
                                    condition, replication, histograms, *priors[condition.index],
                                    n_samples=40, master_seed=17,
                                )
                            else:
                                runner._evaluate_bootstrap_condition(
                                    replication, histograms, n_samples=40, master_seed=17,
                                )
                    self.assertIs(raised.exception, error)
                    details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
                    index = condition.index if method == "DiP-ROC" else 9
                    condition_id = condition.condition_id if method == "DiP-ROC" else "bootstrap"
                    if phase == "warmup":
                        simulation_key = (2, 4, role, index, 0)
                    elif method == "DiP-ROC":
                        simulation_key = (2, 2, 3, replication)
                    else:
                        simulation_key = (2, 3, replication)
                    expected = {
                        "experiment": runner.EXPERIMENT_NAME, "method": method,
                        "condition_id": condition_id, "condition_index": index,
                        "replication_zero_based": replication, "K": 10,
                        "master_seed": 17, "phase": phase, "S": 40,
                        "data_stream_id": semantic_stream_id(17, 2, 1, replication),
                        "simulation_stream_id": semantic_stream_id(17, *simulation_key),
                    }
                    self.assertEqual({key: details[key] for key in expected}, expected)
                    self.assertEqual(sum(details["d_plus"]), 50)
                    self.assertEqual(sum(details["d_minus"]), 50)
                    self.assertEqual(captured["rng"].bit_generator.state, captured["state"])
                    if method == "BootP-ROC":
                        self.assertNotIn("alpha_plus", details)
                        self.assertNotIn("alpha_minus", details)

    def test_builder_failure_after_completed_replication_preserves_published_csv(self):
        condition = runner.PRIOR_CONDITIONS[0]
        original_builder = runner.build_bootstrap_predictive_band
        original_writerows = csv.DictWriter.writerows
        written_rows = []
        calls = 0
        error = AssertionError("failure after completed replication")

        def write_rows(writer, rows):
            written_rows.extend(rows)
            return original_writerows(writer, rows)

        def fail_on_second_replication(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                self.assertEqual(len(written_rows), 4)
                self.assertEqual({row["replication"] for row in written_rows}, {0})
                raise error
            return original_builder(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            output.write_bytes(b"previous published results\n")
            with (
                patch.object(runner, "build_bootstrap_predictive_band", new=fail_on_second_replication),
                patch.object(csv.DictWriter, "writerows", autospec=True, side_effect=write_rows),
            ):
                with self.assertRaises(AssertionError) as raised:
                    runner.run_experiment(output, n_replications=2, n_samples=40,
                                          conditions=(condition,), overwrite=True, show_progress=False)
            self.assertIs(raised.exception, error)
            self.assertEqual(output.read_bytes(), b"previous published results\n")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])
            details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
            self.assertEqual(details["replication_zero_based"], 1)
            self.assertEqual(details["condition_id"], "bootstrap")

    def test_raw_rows_preserve_prior_metadata_and_inclusive_boundaries(self):
        condition = runner.PRIOR_CONDITIONS[0]
        profiles = runner.build_prior_profiles()
        data_key = runner._data_stream_key(0)
        posterior_key = runner._posterior_stream_key(condition, 0)
        rows = runner._raw_rows_for_condition_replication(
            condition,
            0,
            _fixed_measurements(),
            profiles[condition.prior_profile],
            true_grid_auc=profiles["oracle"].expected_grid_auc,
            n_replications=2,
            n_samples=40,
            master_seed=91,
            data_key=data_key,
            simulation_key=posterior_key,
            method="DiP-ROC", computation_time_ns=0, execution_position=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(tuple(rows[0]), runner.RAW_FIELDNAMES)
        self.assertEqual(rows[0]["condition_id"], "oracle_m_1")
        self.assertEqual(rows[0]["prior_mass"], 1)
        self.assertEqual(
            rows[0]["true_grid_auc"],
            profiles["oracle"].expected_grid_auc,
        )
        self.assertEqual(
            rows[0]["prior_expected_grid_auc"],
            profiles["oracle"].expected_grid_auc,
        )
        self.assertEqual(rows[0]["roc_covered"], 1)
        self.assertEqual(rows[0]["auc_covered"], 1)
        self.assertEqual(rows[0]["data_stream_id"], rows[1]["data_stream_id"])
        self.assertEqual(
            rows[0]["simulation_stream_id"],
            rows[1]["simulation_stream_id"],
        )

        with self.assertRaises(AssertionError):
            runner._raw_rows_for_condition_replication(
                condition,
                0,
                _fixed_measurements(),
                profiles[condition.prior_profile],
                true_grid_auc=profiles["oracle"].expected_grid_auc,
                n_replications=2,
                n_samples=40,
                master_seed=91,
                data_key=posterior_key,
                simulation_key=posterior_key,
            method="DiP-ROC", computation_time_ns=0, execution_position=2,
            )

    def test_orchestration_reuses_one_bootstrap_for_nine_priors_and_alternates_blocks(self):
        with self.assertRaises(AssertionError):
            runner._generate_replication_histograms(0, gamma=np.linspace(0, 1, 11) ** 2, master_seed=17)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            with (
                patch.object(runner, "generate_simulation_dataset", wraps=runner.generate_simulation_dataset) as generate,
                patch.object(runner, "build_test_external_histograms", wraps=runner.build_test_external_histograms) as histogram,
                patch.object(runner, "build_timed_band", wraps=runner.build_timed_band) as timed,
                patch.object(runner, "warm_up_band", wraps=runner.warm_up_band) as warm,
                patch.object(runner, "evaluate_replication", return_value=_fixed_measurements()) as evaluate,
            ):
                runner.run_experiment(output, master_seed=17, n_replications=3,
                                      n_samples=40, conditions=runner.PRIOR_CONDITIONS,
                                      show_progress=False)
            rows = _read_rows(output)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(histogram.call_count, 3)
        self.assertEqual(evaluate.call_count, 30)
        self.assertEqual(timed.call_count, 30)
        self.assertEqual(warm.call_count, 10)
        expected_priors = runner._build_condition_priors(runner.build_prior_profiles())
        for replication in range(3):
            calls = timed.call_args_list[10 * replication:10 * (replication + 1)]
            expected = [runner.build_posterior_predictive_band] * 9
            if replication % 2 == 0:
                expected.insert(0, runner.build_bootstrap_predictive_band)
            else:
                expected.append(runner.build_bootstrap_predictive_band)
            self.assertEqual([call.args[0] for call in calls], expected)
            self.assertEqual(len({id(call.kwargs["d_plus"]) for call in calls}), 1)
            self.assertEqual(len({id(call.kwargs["d_minus"]) for call in calls}), 1)
            self.assertEqual(len({id(call.kwargs["rng"]) for call in calls}), 10)
            prior_calls = [call for call in calls if call.args[0] is runner.build_posterior_predictive_band]
            for condition, call in zip(runner.PRIOR_CONDITIONS, prior_calls):
                np.testing.assert_array_equal(call.kwargs["alpha_plus"], expected_priors[condition.index][0])
                np.testing.assert_array_equal(call.kwargs["alpha_minus"], expected_priors[condition.index][1])
            bootstrap = next(call for call in calls if call.args[0] is runner.build_bootstrap_predictive_band)
            self.assertEqual(set(bootstrap.kwargs), {"d_plus", "d_minus", "S", "levels", "rng", "diagnostic_context"})
            for call in calls:
                self.assertEqual(call.kwargs["S"], 40)
                self.assertEqual(call.kwargs["levels"], (0.80, 0.95))
                self.assertFalse(call.kwargs["d_plus"].flags.writeable)
            replication_rows = [row for row in rows if int(row["replication"]) == replication]
            self.assertEqual(len({row["simulation_stream_id"] for row in replication_rows}), 10)
            self.assertEqual(len({row["data_stream_id"] for row in replication_rows}), 1)
            self.assertEqual([row["method"] for row in replication_rows], ["DiP-ROC"] * 18 + ["BootP-ROC"] * 2)
            boot_rows = replication_rows[-2:]
            self.assertEqual({row["execution_position"] for row in boot_rows}, {"1" if replication % 2 == 0 else "10"})
            for row in boot_rows:
                self.assertEqual(row["condition_id"], "bootstrap")
                self.assertEqual(row["condition_index"], "9")
                self.assertEqual(row["prior_profile"], "")
                self.assertEqual(row["prior_mass"], "")
                self.assertEqual(row["prior_expected_grid_auc"], "")
        self.assertEqual(len(rows), 60)

    def test_tiny_true_integration_is_byte_reproducible_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"
            arguments = {
                "master_seed": 2874,
                "n_replications": 2,
                "n_samples": 40,
                "conditions": runner.PRIOR_CONDITIONS,
                "show_progress": False,
            }
            runner.run_experiment(first, **arguments)
            runner.run_experiment(second, **arguments)

            self.assertEqual(_scientific_rows(_read_rows(first)), _scientific_rows(_read_rows(second)))
            rows = _read_rows(first)
            self.assertEqual(len(rows), 40)
            for replication in range(2):
                deviations = {
                    row["external_max_roc_deviation"]
                    for row in rows
                    if int(row["replication"]) == replication
                    and float(row["level"]) == 0.80
                }
                self.assertGreater(len(deviations), 1)
            self.assertEqual(
                runner.validate_raw_results_csv(
                    first,
                    master_seed=arguments["master_seed"],
                    n_replications=arguments["n_replications"],
                    n_samples=arguments["n_samples"],
                    conditions=arguments["conditions"],
                ),
                40,
            )

    def test_condition_order_and_subset_do_not_change_scientific_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            forward_path = Path(temporary_directory) / "forward.csv"
            reverse_path = Path(temporary_directory) / "reverse.csv"
            subset_path = Path(temporary_directory) / "subset.csv"
            common = {
                "master_seed": 886,
                "n_replications": 2,
                "n_samples": 30,
                "show_progress": False,
            }
            runner.run_experiment(
                forward_path,
                conditions=runner.PRIOR_CONDITIONS,
                **common,
            )
            runner.run_experiment(
                reverse_path,
                conditions=tuple(reversed(runner.PRIOR_CONDITIONS)),
                **common,
            )
            subset_condition = runner.PRIOR_CONDITIONS[2]
            runner.run_experiment(
                subset_path,
                conditions=(subset_condition,),
                **common,
            )

            forward_rows = sorted(_read_rows(forward_path), key=_sort_key)
            reverse_rows = sorted(_read_rows(reverse_path), key=_sort_key)
            self.assertEqual(_scientific_rows(forward_rows), _scientific_rows(reverse_rows))
            expected_subset = [
                row
                for row in forward_rows
                if row["condition_id"] in (subset_condition.condition_id, "bootstrap")
            ]
            self.assertEqual(
                _scientific_rows(sorted(_read_rows(subset_path), key=_sort_key)),
                _scientific_rows(expected_subset),
            )

    def test_raw_csv_reconstructs_all_canonical_condition_summaries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "raw.csv"
            runner.run_experiment(
                output,
                master_seed=312,
                n_replications=3,
                n_samples=40,
                conditions=runner.PRIOR_CONDITIONS,
                show_progress=False,
            )
            rows = _read_rows(output)

        for condition in runner.PRIOR_CONDITIONS:
            condition_rows = [
                row
                for row in rows
                if row["condition_id"] == condition.condition_id
            ]
            by_replication: dict[int, dict[float, dict[str, str]]] = {}
            for row in condition_rows:
                by_replication.setdefault(int(row["replication"]), {})[
                    float(row["level"])
                ] = row
            measurements = []
            for replication in range(3):
                level_rows = by_replication[replication]
                reference = level_rows[0.80]
                measurements.append(
                    ReplicationMeasurements(
                        external_max_roc_deviation=float(
                            reference["external_max_roc_deviation"]
                        ),
                        external_grid_auc=float(reference["external_grid_auc"]),
                        roc_radii={
                            level: float(level_rows[level]["roc_radius"])
                            for level in runner.NOMINAL_LEVELS
                        },
                        auc_intervals={
                            level: (
                                float(
                                    level_rows[level]["auc_interval_lower"]
                                ),
                                float(
                                    level_rows[level]["auc_interval_upper"]
                                ),
                            )
                            for level in runner.NOMINAL_LEVELS
                        },
                    )
                )
            summaries = summarize_replication_measurements(
                measurements,
                runner.NOMINAL_LEVELS,
            )
            for level in runner.NOMINAL_LEVELS:
                level_rows = [
                    by_replication[replication][level]
                    for replication in range(3)
                ]
                self.assertEqual(
                    summaries[level].roc_coverage,
                    np.mean(
                        [int(row["roc_covered"]) for row in level_rows]
                    ),
                )
                self.assertEqual(
                    summaries[level].auc_coverage,
                    np.mean(
                        [int(row["auc_covered"]) for row in level_rows]
                    ),
                )

    def test_mocked_official_design_writes_exactly_200000_rows(self):
        histogram = np.full(10, 5, dtype=int)
        fixed_histograms = TestExternalHistograms(
            d_plus_test=histogram.copy(),
            d_minus_test=histogram.copy(),
            d_plus_external=histogram.copy(),
            d_minus_external=histogram.copy(),
        )

        def fake_generation(replication, **unused_arguments):
            return fixed_histograms, runner._data_stream_key(replication)

        def fake_evaluation(
            condition,
            replication,
            histograms,
            alpha_plus,
            alpha_minus,
            **unused_arguments,
        ):
            self.assertIs(histograms, fixed_histograms)
            self.assertEqual(alpha_plus.shape, (10,))
            self.assertEqual(alpha_minus.shape, (10,))
            return (
                _fixed_measurements(),
                runner._posterior_stream_key(condition, replication),
                100,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "official_raw.csv"
            with patch.object(
                runner,
                "_generate_replication_histograms",
                side_effect=fake_generation,
            ) as generation_mock, patch.object(
                runner,
                "_evaluate_prior_condition",
                side_effect=fake_evaluation,
            ) as evaluation_mock, patch.object(
                runner, "_evaluate_bootstrap_condition",
                side_effect=lambda replication, histograms, **kwargs: (
                    _fixed_measurements(), runner._bootstrap_stream_key(replication), 100),
            ) as bootstrap_mock, patch.object(runner, "_warm_up_conditions"):
                runner.run_experiment(
                    output,
                    master_seed=runner.DEFAULT_MASTER_SEED,
                    n_replications=runner.OFFICIAL_REPLICATIONS,
                    n_samples=runner.OFFICIAL_SAMPLES,
                    conditions=runner.PRIOR_CONDITIONS,
                    show_progress=False,
                )

            self.assertEqual(generation_mock.call_count, 10_000)
            self.assertEqual(evaluation_mock.call_count, 90_000)
            self.assertEqual(bootstrap_mock.call_count, 10_000)
            rows = _read_rows(output)
            self.assertEqual(len(rows), 200_000)
            self.assertEqual(
                len(
                    {
                        (
                            row["method"],
                    row["condition_id"],
                            row["replication"],
                            row["level"],
                        )
                        for row in rows
                    }
                ),
                200_000,
            )
            data_ids = {row["data_stream_id"] for row in rows}
            posterior_ids = {row["simulation_stream_id"] for row in rows}
            self.assertEqual(len(data_ids), 10_000)
            self.assertEqual(len(posterior_ids), 100_000)
            self.assertTrue(data_ids.isdisjoint(posterior_ids))

    def test_validator_rejects_prior_pairing_and_csv_corruptions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            valid_path = Path(temporary_directory) / "valid.csv"
            arguments = {
                "master_seed": 44,
                "n_replications": 2,
                "n_samples": 20,
                "conditions": runner.PRIOR_CONDITIONS,
                "show_progress": False,
            }
            runner.run_experiment(valid_path, **arguments)
            rows = _read_rows(valid_path)
            corruptions: dict[str, list[dict[str, str]]] = {}

            wrong_width = [dict(row) for row in rows]
            wrong_width[0]["auc_width"] = "0.12345678901234568"
            corruptions["wrong width"] = wrong_width

            wrong_flag = [dict(row) for row in rows]
            wrong_flag[0]["roc_covered"] = str(
                1 - int(wrong_flag[0]["roc_covered"])
            )
            corruptions["wrong coverage flag"] = wrong_flag

            wrong_true_auc = [dict(row) for row in rows]
            wrong_true_auc[0]["true_grid_auc"] = "0.7"
            corruptions["wrong true AUC"] = wrong_true_auc

            wrong_prior_auc = [dict(row) for row in rows]
            wrong_prior_auc[0]["prior_expected_grid_auc"] = "0.6"
            corruptions["wrong prior AUC"] = wrong_prior_auc

            wrong_stream = [dict(row) for row in rows]
            wrong_stream[0]["simulation_stream_id"] += ":corrupt"
            corruptions["wrong posterior stream"] = wrong_stream

            cross_prior_auc = [dict(row) for row in rows]
            target_rows = [
                row
                for row in cross_prior_auc
                if row["condition_id"] == "oracle_m_1"
                and row["replication"] == "0"
            ]
            self.assertEqual(len(target_rows), 2)
            for row in target_rows:
                row["external_grid_auc"] = "0.12345678901234568"
                row["auc_covered"] = str(
                    int(
                        is_external_auc_covered(
                            float(row["external_grid_auc"]),
                            (
                                float(row["auc_interval_lower"]),
                                float(row["auc_interval_upper"]),
                            ),
                        )
                    )
                )
            corruptions["cross-prior external AUC mismatch"] = cross_prior_auc

            for field, bad_value in (
                ("method", "unknown"), ("computation_time_ns", "-1"),
                ("execution_position", "0"), ("timing_threads", "2"),
            ):
                corrupted = [dict(row) for row in rows]
                corrupted[0][field] = bad_value
                corruptions["method_metadata_" + field] = corrupted
            inconsistent_time = [dict(row) for row in rows]
            inconsistent_time[0]["computation_time_ns"] = str(int(inconsistent_time[0]["computation_time_ns"]) + 1)
            corruptions["cross_level_timing"] = inconsistent_time
            bootstrap_prior = [dict(row) for row in rows]
            next(row for row in bootstrap_prior if row["method"] == "BootP-ROC")["prior_mass"] = "1"
            corruptions["bootstrap_prior_leak"] = bootstrap_prior
            wrong_dip_radius = [dict(row) for row in rows]
            wrong_dip_radius[0]["roc_radius"] = "1.8"
            wrong_dip_radius[0]["roc_covered"] = "1"
            corruptions["dip_radius_bound"] = wrong_dip_radius
            cross_method_auc = [dict(row) for row in rows]
            first_boot = next(row for row in cross_method_auc if row["method"] == "BootP-ROC")
            for row in cross_method_auc:
                if (row["method"] == "BootP-ROC" and row["replication"] == first_boot["replication"]
                        and row["condition_id"] == first_boot["condition_id"]):
                    row["external_grid_auc"] = "0.12345678901234568"
                    row["auc_covered"] = str(int(is_external_auc_covered(
                        float(row["external_grid_auc"]),
                        (float(row["auc_interval_lower"]), float(row["auc_interval_upper"])))))
            corruptions["cross_method_external_auc"] = cross_method_auc

            corruptions["missing level"] = rows[:-1]
            corruptions["missing condition bundle"] = rows[:-2]
            corruptions["duplicate row"] = rows + [rows[0]]

            validation_arguments = {
                "master_seed": arguments["master_seed"],
                "n_replications": arguments["n_replications"],
                "n_samples": arguments["n_samples"],
                "conditions": arguments["conditions"],
            }
            for index, (name, corrupted_rows) in enumerate(corruptions.items()):
                with self.subTest(corruption=name):
                    path = Path(temporary_directory) / f"corrupt_{index}.csv"
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=runner.RAW_FIELDNAMES,
                        )
                        writer.writeheader()
                        writer.writerows(corrupted_rows)
                    with self.assertRaises(AssertionError):
                        runner.validate_raw_results_csv(
                            path,
                            **validation_arguments,
                        )

            wrong_header = Path(temporary_directory) / "wrong_header.csv"
            with wrong_header.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=runner.RAW_FIELDNAMES[:-1],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "v3 synthetic comparison CSV header"):
                runner.validate_raw_results_csv(
                    wrong_header,
                    **validation_arguments,
                )

    def test_invalid_arguments_atomic_failure_and_official_main(self):
        condition = runner.PRIOR_CONDITIONS[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "raw.csv"
            arguments = {
                "master_seed": 9,
                "n_replications": 1,
                "n_samples": 20,
                "conditions": (condition,),
                "show_progress": False,
            }
            runner.run_experiment(output, **arguments)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(output, **arguments)
            with self.assertRaises(TypeError):
                runner.run_experiment(output, overwrite=0, **arguments)
            runner.run_experiment(output, overwrite=True, **arguments)

            preserved_bytes = output.read_bytes()
            with patch.object(
                runner,
                "_generate_replication_histograms",
                side_effect=RuntimeError("injected failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    runner.run_experiment(
                        output,
                        overwrite=True,
                        **arguments,
                    )
            self.assertEqual(output.read_bytes(), preserved_bytes)
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

            with self.assertRaises(ValueError):
                runner.run_experiment(
                    Path(temporary_directory) / "raw.txt",
                    **arguments,
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    Path(temporary_directory) / "bad_b.csv",
                    **{**arguments, "n_replications": 0},
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    Path(temporary_directory) / "bad_s.csv",
                    **{**arguments, "n_samples": 1},
                )
            with self.assertRaises(TypeError):
                runner.run_experiment(
                    Path(temporary_directory) / "bad_seed.csv",
                    **{**arguments, "master_seed": True},
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    Path(temporary_directory) / "duplicate.csv",
                    **{**arguments, "conditions": (condition, condition)},
                )

        expected_path = Path("mocked.csv")
        with patch.object(
            runner,
            "run_experiment",
            return_value=expected_path,
        ) as run_mock:
            self.assertEqual(runner.main([]), expected_path)
        self.assertEqual(run_mock.call_args.kwargs["n_replications"], 10_000)
        self.assertEqual(run_mock.call_args.kwargs["n_samples"], 4000)
        self.assertEqual(
            run_mock.call_args.kwargs["conditions"],
            runner.PRIOR_CONDITIONS,
        )

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        import_command = (
            "import sys; "
            f"sys.path.insert(0, {str(PY_DIR)!r}); "
            "import run_experiment_sensitivity_to_prior_specification"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", import_command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "This experiment relies on runtime assertions",
            completed.stderr,
        )


    def test_v3_csv_rejects_previous_mixed_and_relabelled_score_protocols(self):
        self.assertEqual(runner.SYNTHETIC_DESIGN_VERSION, 3)
        self.assertEqual(runner.RAW_SCHEMA_VERSION, "prior_shape_raw_v3")
        arguments = dict(conditions=runner.PRIOR_CONDITIONS[:1], n_replications=2,
                         n_samples=20, master_seed=31)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            runner.run_experiment(source, show_progress=False, **arguments)
            rows = _read_rows(source)
            beta_fields = ("positive_beta_a", "positive_beta_b", "negative_beta_a", "negative_beta_b")
            for case in ("old v1", "old v2", "mixed versions", "old beta", "mixed beta"):
                with self.subTest(case=case):
                    changed = [dict(row) for row in rows]
                    if case in ("old v1", "old v2", "mixed versions"):
                        selected = changed[-1:] if case == "mixed versions" else changed
                        version = 1 if case == "old v1" else 2
                        for row in selected:
                            row["schema_version"] = f"prior_shape_raw_v{version}"
                        message = "old or mixed synthetic protocol"
                    else:
                        selected = changed[-1:] if case == "mixed beta" else changed
                        for row in selected:
                            row.update(dict(zip(beta_fields, map(str, (5.0, 3.0, 3.0, 5.0)))))
                        message = "Beta metadata"
                    corrupted = Path(directory) / "old_or_mixed.csv"
                    with corrupted.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=runner.RAW_FIELDNAMES)
                        writer.writeheader()
                        writer.writerows(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_raw_results_csv(corrupted, **arguments)

    def test_previous_csv_paths_and_renamed_protocol_files_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_paths = [root / name for name in (
                "raw_replications.csv", "raw_replications_v1.csv", "raw_replications_v2.csv",
            )]
            previous_paths += [root / name / "custom.csv" for name in (
                "latex_tables", "latex_tables_v1", "latex_tables_v2",
            )]
            for path in previous_paths:
                with self.subTest(path=path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("previous results\n")
                    with patch.object(runner, "generate_simulation_dataset") as generate:
                        with self.assertRaisesRegex(ValueError, "protected"):
                            runner.run_experiment(path, overwrite=True, show_progress=False,
                                                  conditions=runner.PRIOR_CONDITIONS[:1],
                                                  n_replications=2, n_samples=20)
                        generate.assert_not_called()
                    self.assertEqual(path.read_text(), "previous results\n")
            # A CLI override cannot bypass the same safeguard.
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.main(["--output", str(previous_paths[0]), "--overwrite"])
            renamed = root / "custom.csv"
            for version in (1, 2):
                old_text = f"schema_version\nprior_shape_raw_v{version}\n"
                renamed.write_text(old_text)
                with self.assertRaisesRegex(ValueError, "protected"):
                    runner.run_experiment(renamed, overwrite=True, show_progress=False,
                                          conditions=runner.PRIOR_CONDITIONS[:1],
                                          n_replications=2, n_samples=20)
                self.assertEqual(renamed.read_text(), old_text)

if __name__ == "__main__":
    unittest.main()
