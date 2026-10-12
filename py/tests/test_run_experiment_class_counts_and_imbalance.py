import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_experiment_main_evaluation_on_synthetic_data as runner  # noqa: E402
from roc_utils.evaluation_criteria import is_external_auc_covered  # noqa: E402
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    equal_tailed_interval,
    fixed_uniform_grid,
    nearest_rank_quantile,
    symmetric_dirichlet_priors,
)
from roc_utils.semantic_rng import (  # noqa: E402
    make_semantic_rng,
    semantic_stream_id,
)
from roc_utils.utilities_experiments import (  # noqa: E402
    ReplicationMeasurements,
    summarize_replication_measurements,
)


def _fixed_measurements() -> ReplicationMeasurements:
    return ReplicationMeasurements(
        external_max_roc_deviation=0.2 + 0.5e-12,
        external_grid_auc=0.4 - 0.5e-12,
        roc_radii={0.80: 0.2, 0.95: 0.3},
        auc_intervals={0.80: (0.4, 0.7), 0.95: (0.3, 0.8)},
    )


def _history(auc=.75):
    positive, negative = np.zeros(10, dtype=int), np.zeros(10, dtype=int)
    positive[1] = 500
    negative[0], negative[2] = round(500 * auc), 500 - round(500 * auc)
    return runner.HistoricalSummary(positive, negative)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scientific_rows(rows):
    # Wall-clock measurements and execution positions are run metadata.
    return [{key: value for key, value in row.items()
             if key not in ("computation_time_ns", "execution_position")}
            for row in rows]


class ClassCountsAndImbalanceExperimentTests(unittest.TestCase):
    def test_exact_manuscript_design_and_symmetric_prior(self):
        self.assertEqual(runner.EXPERIMENT_NAME, "class_counts_and_imbalance")
        self.assertEqual(
            runner.RESULTS_DIRECTORY_NAME,
            "main_evaluation_on_synthetic_data",
        )
        self.assertEqual(
            runner.DEFAULT_OUTPUT_CSV.relative_to(runner.PY_DIR),
            Path(
                "results/current_experiments/"
                "main_evaluation_on_synthetic_data/raw_replications_v8.csv"
            ),
        )
        self.assertEqual(runner.SIMULATION_SCENARIOS, ("Easy", "Medium", "Hard"))
        self.assertEqual(
            runner.CLASS_COUNT_PAIRS,
            (
                (20, 20),
                (100, 100),
                (500, 500),
                (20, 500),
                (500, 20),
            ),
        )
        self.assertEqual(len(runner.CLASS_COUNT_CONDITIONS), 15)
        self.assertEqual(
            [condition.index for condition in runner.CLASS_COUNT_CONDITIONS],
            list(range(15)),
        )
        self.assertEqual(
            len(
                {
                    condition.condition_id
                    for condition in runner.CLASS_COUNT_CONDITIONS
                }
            ),
            15,
        )
        self.assertIn(
            "easy_n_positive_20_n_negative_500",
            {
                condition.condition_id
                for condition in runner.CLASS_COUNT_CONDITIONS
            },
        )
        self.assertIn(
            "easy_n_positive_500_n_negative_20",
            {
                condition.condition_id
                for condition in runner.CLASS_COUNT_CONDITIONS
            },
        )
        self.assertEqual(runner.N_BINS, 10)
        self.assertEqual(runner.SETTING_MASSES["dip_uniform_m1"], 1.0)
        self.assertEqual(runner.MAIN_PROTOCOL_VERSION, 8)
        self.assertEqual(runner.METHODS, ("DiP-ROC",))
        self.assertEqual(runner.HISTORICAL_N_POSITIVE, 500)
        self.assertEqual(runner.HISTORICAL_N_NEGATIVE, 500)
        self.assertEqual(runner.SETTING_MASSES["dip_linear_m1"], 1.)
        self.assertEqual(runner.SETTING_MASSES["dip_linear_m1"], runner.SETTING_MASSES["dip_uniform_m1"])
        self.assertEqual(runner.HISTORICAL_AUC_BOUNDS, (0.5, 0.86))
        self.assertEqual(runner.SETTINGS, (
            "dip_uniform_m1", "dip_uniform_m5", "dip_uniform_m20",
            "dip_linear_m1", "dip_linear_m5", "dip_linear_m20",
            "dip_histograms_m1", "dip_histograms_m5", "dip_histograms_m20",
        ))
        self.assertEqual(tuple(runner.SETTING_MASSES[s] for s in runner.SETTINGS),
                         (1., 5., 20., 1., 5., 20., 1., 5., 20.))
        self.assertEqual(tuple(runner.SETTING_FAMILIES[s] for s in runner.SETTINGS),
                         ("uniform",) * 3 + ("linear",) * 3 + ("histograms",) * 3)
        self.assertEqual(tuple(runner.SETTING_PROFILES[s] for s in runner.SETTINGS),
                         ("symmetric",) * 3 + ("historical_auc_linear",) * 3
                         + ("historical_histograms",) * 3)
        self.assertEqual(set(runner.SETTING_METHODS.values()), {"DiP-ROC"})
        self.assertEqual(runner._INFORMED_SETTINGS, (
            "dip_linear_m1", "dip_linear_m5", "dip_linear_m20",
            "dip_histograms_m1", "dip_histograms_m5", "dip_histograms_m20",
        ))
        self.assertEqual(runner.NOMINAL_LEVELS, (0.80, 0.95))
        self.assertEqual(runner.OFFICIAL_REPLICATIONS, 10_000)
        self.assertEqual(runner.OFFICIAL_SAMPLES, 4000)
        self.assertEqual(
            15
            * runner.OFFICIAL_REPLICATIONS
            * len(runner.NOMINAL_LEVELS) * len(runner.SETTINGS),
            2_700_000,
        )

        gamma = fixed_uniform_grid(runner.N_BINS)
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(
            runner.N_BINS,
            runner.SETTING_MASSES["dip_uniform_m1"],
        )
        np.testing.assert_array_equal(gamma, np.linspace(0.0, 1.0, 11))
        np.testing.assert_array_equal(alpha_plus, np.full(10, 0.1))
        np.testing.assert_array_equal(alpha_minus, np.full(10, 0.1))
        self.assertAlmostEqual(float(np.sum(alpha_plus)), 1.0)
        self.assertAlmostEqual(float(np.sum(alpha_minus)), 1.0)

        ordered_values = np.arange(1.0, 4001.0)
        self.assertEqual(nearest_rank_quantile(ordered_values, 0.80), 3200.0)
        self.assertEqual(nearest_rank_quantile(ordered_values, 0.95), 3800.0)
        self.assertEqual(
            equal_tailed_interval(ordered_values, 0.80),
            (400.0, 3600.0),
        )
        self.assertEqual(
            equal_tailed_interval(ordered_values, 0.95),
            (100.0, 3900.0),
        )

        invalid_integer_fields = (
            {"index": True, "n_positive": 20, "n_negative": 20},
            {"index": 0, "n_positive": True, "n_negative": 20},
            {"index": 0, "n_positive": 20, "n_negative": True},
        )
        for fields in invalid_integer_fields:
            with self.subTest(fields=fields), self.assertRaises(AssertionError):
                runner.ClassCountCondition(
                    score_scenario="Easy",
                    **fields,
                )

    def test_semantic_rng_is_reproducible_distinct_and_order_independent(self):
        key_a = (1, 1, 4, 27)
        key_b = (1, 2, 4, 27)
        first = make_semantic_rng(12345, *key_a).random(12)
        second = make_semantic_rng(12345, *key_a).random(12)
        different_role = make_semantic_rng(12345, *key_b).random(12)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different_role))
        self.assertEqual(
            semantic_stream_id(12345, *key_a),
            "PCG64:12345:1:1:4:27",
        )
        self.assertNotEqual(
            semantic_stream_id(12345, *key_a),
            semantic_stream_id(12346, *key_a),
        )

        forward = {
            key: make_semantic_rng(765, *key).integers(0, 2**31, size=5)
            for key in (key_a, key_b)
        }
        reverse = {
            key: make_semantic_rng(765, *key).integers(0, 2**31, size=5)
            for key in (key_b, key_a)
        }
        for key in forward:
            np.testing.assert_array_equal(forward[key], reverse[key])

        with self.assertRaises(TypeError):
            make_semantic_rng(True, *key_a)
        with self.assertRaises(ValueError):
            make_semantic_rng(-1, *key_a)
        with self.assertRaises(ValueError):
            make_semantic_rng(1)
        with self.assertRaises(ValueError):
            semantic_stream_id(1, 1, -1)

    def test_warmup_keys_and_warmup_do_not_change_scientific_results(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        self.assertEqual(runner._simulation_stream_key(condition, 7, "dip_uniform_m1"), (1, 2, 0, 7))
        self.assertEqual(runner._warmup_stream_key(0, "dip_uniform_m1"), (1, 4, 2, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            warm = Path(directory) / "warm.csv"
            cold = Path(directory) / "cold.csv"
            arguments = dict(n_replications=2, n_samples=20, conditions=(condition,), show_progress=False)
            runner.run_experiment(warm, **arguments)
            with patch.object(runner, "warm_up_band"):
                runner.run_experiment(cold, **arguments)
            self.assertEqual(_scientific_rows(_read_rows(warm)), _scientific_rows(_read_rows(cold)))

    def test_main_experiment_never_constructs_bootstrap_bands_or_uses_reserved_role(self):
        self.assertFalse(hasattr(runner, "build_bootstrap_predictive_band"))
        self.assertNotIn("bootp", runner.SETTINGS)
        self.assertNotIn(3, runner._SETTING_STREAM_ROLES.values())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dip_only.csv"
            with (
                patch("roc_utils.utilities_experiments.build_bootstrap_predictive_band",
                      side_effect=AssertionError("Bootstrap must not run in the main experiment")) as bootstrap,
                patch.object(runner, "make_semantic_rng", wraps=runner.make_semantic_rng) as random_streams,
            ):
                runner.run_experiment(output, n_replications=2, n_samples=20,
                                      conditions=runner.CLASS_COUNT_CONDITIONS[:1], show_progress=False)
            bootstrap.assert_not_called()
            rows = _read_rows(output)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["method"] for row in rows}, {"DiP-ROC"})
        self.assertEqual({row["setting"] for row in rows}, set(runner.SETTINGS))
        for call in random_streams.call_args_list:
            key = call.args[1:]
            self.assertNotEqual(key[1], 3)
            if key[1] == 4:
                self.assertNotEqual(key[2], 3)

    def test_construction_failures_identify_each_setting_and_actual_stream(self):
        condition = runner.CLASS_COUNT_CONDITIONS[3]
        names = {"uniform": "build_uniform_predictive_band", "linear": "build_linear_predictive_band",
                 "histograms": "build_histogram_predictive_band"}
        for setting in runner.SETTINGS:
            name = names[runner.SETTING_FAMILIES[setting]]
            original = getattr(runner, name)
            role = runner._SETTING_STREAM_ROLES[setting]
            for phase, replication in (("warmup", 0), ("construction", 1)):
                with self.subTest(setting=setting, phase=phase):
                    error = AssertionError("injected builder failure")
                    captured = {}
                    def fail(**kwargs):
                        if kwargs["prior_mass"] != runner.SETTING_MASSES[setting]:
                            return original(**kwargs)
                        self.assertNotIn("diagnostic_context", kwargs)
                        captured.update(rng=kwargs["rng"], state=kwargs["rng"].bit_generator.state)
                        raise error
                    with patch.object(runner, name, new=fail), self.assertRaises(AssertionError) as raised:
                        runner._evaluate_one_replication(condition, replication, gamma=fixed_uniform_grid(10),
                                                         n_samples=40, master_seed=17)
                    self.assertIs(raised.exception, error)
                    details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
                    simulation_key = ((1, 4, role, 3, 0) if phase == "warmup" else (1, role, 3, replication))
                    expected = {"experiment": runner.EXPERIMENT_NAME, "method": runner.SETTING_METHODS[setting], "setting": setting,
                                "prior_family": runner.SETTING_FAMILIES[setting], "prior_mass": runner.SETTING_MASSES[setting],
                                "condition_id": condition.condition_id, "condition_index": 3,
                                "replication_zero_based": replication, "K": 10, "master_seed": 17, "phase": phase, "S": 40,
                                "data_stream_id": semantic_stream_id(17, 1, 1, 3, replication),
                                "simulation_stream_id": semantic_stream_id(17, *simulation_key)}
                    self.assertEqual({key: details[key] for key in expected}, expected)
                    self.assertEqual(sum(details["d_plus"]), 20)
                    self.assertEqual(sum(details["d_minus"]), 500)
                    self.assertEqual(captured["rng"].bit_generator.state, captured["state"])
                    if runner.SETTING_FAMILIES[setting] == "histograms":
                        self.assertEqual(sum(details["historical_counts_positive"]), 500)
                        self.assertEqual(sum(details["historical_counts_negative"]), 500)

    def test_v8_high_replication_with_official_budget_preserves_auc_bounds(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(10, 1.0)
        with patch.object(runner, "build_test_external_histograms",
                          wraps=runner.build_test_external_histograms) as histogram:
            results, data_key, historical_auc = runner._evaluate_one_replication(
                condition, 8421, gamma=fixed_uniform_grid(10),
                n_samples=4000, master_seed=20250731,
            )
        self.assertEqual(data_key, (1, 1, 0, 8421))
        observed = runner.build_test_external_histograms(*histogram.call_args.args)
        # The original v2 witness is frozen in the bootstrap unit tests.
        # This exercises the new score laws with the same large stream index.
        self.assertEqual(observed.d_plus_test.shape, (10,))
        self.assertEqual(observed.d_minus_test.shape, (10,))
        self.assertEqual(int(observed.d_plus_test.sum()), 20)
        self.assertEqual(int(observed.d_minus_test.sum()), 20)
        self.assertEqual(results["dip_uniform_m20"][1], (1, 11, 0, 8421))
        for setting in runner.SETTINGS:
            measurements, key, elapsed, position = results[setting]
            rows = runner._raw_rows_for_replication(
                condition, 8421, measurements, n_replications=10000, n_samples=4000,
                master_seed=20250731, data_key=data_key, simulation_key=key,
                setting=setting, historical_summary=historical_auc if setting in runner._INFORMED_SETTINGS else None, computation_time_ns=elapsed, execution_position=position,
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(0.0 <= measurements.external_grid_auc <= 1.0)
            for lower, upper in measurements.auc_intervals.values():
                self.assertTrue(0.0 <= lower <= upper <= 1.0)

    def test_test_external_preparation_failures_preserve_exception_and_stream_context(self):
        condition, replication, seed = runner.CLASS_COUNT_CONDITIONS[3], 27, 819
        for function in ("generate_simulation_dataset", "build_test_external_histograms"):
            with self.subTest(function=function):
                error = ValueError("injected test/external preparation failure")
                with patch.object(runner, function, side_effect=error), self.assertRaises(ValueError) as raised:
                    runner._evaluate_one_replication(
                        condition, replication, gamma=fixed_uniform_grid(10),
                        n_samples=40, master_seed=seed,
                    )
                self.assertIs(raised.exception, error)
                details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
                expected = {
                    "stage": "test_external_preparation", "protocol": 8,
                    "condition_id": condition.condition_id, "condition_index": 3,
                    "replication_zero_based": 27, "master_seed": 819,
                    "data_stream_id": "PCG64:819:1:1:3:27",
                    "n_positive": 20, "n_negative": 500, "K": 10, "S": 40,
                }
                self.assertEqual({key: details[key] for key in expected}, expected)

    def test_csv_serialization_failures_preserve_exception_setting_and_published_output(self):
        condition = runner.CLASS_COUNT_CONDITIONS[3]
        for phase in ("row_construction", "row_writing"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "published.csv"
                previous_bytes = b"previous published results\n"
                output.write_bytes(previous_bytes)
                error = OSError("injected serialization failure")
                original_rows = runner._raw_rows_for_replication
                original_write = csv.DictWriter.writerows

                def fail_rows(*args, **kwargs):
                    if kwargs["setting"] == "dip_histograms_m20" and args[1] == 1:
                        raise error
                    return original_rows(*args, **kwargs)

                def fail_write(writer, rows):
                    if rows[0]["setting"] == "dip_histograms_m20" and rows[0]["replication"] == 1:
                        raise error
                    return original_write(writer, rows)

                failure = (patch.object(runner, "_raw_rows_for_replication", side_effect=fail_rows)
                           if phase == "row_construction" else
                           patch.object(csv.DictWriter, "writerows", autospec=True, side_effect=fail_write))
                with failure, self.assertRaises(OSError) as raised:
                    runner.run_experiment(output, conditions=(condition,), n_replications=2,
                                          n_samples=20, master_seed=819,
                                          overwrite=True, show_progress=False)
                self.assertIs(raised.exception, error)
                self.assertEqual(output.read_bytes(), previous_bytes)
                self.assertFalse(list(Path(directory).glob("*.tmp")))
                details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
                expected = {
                    "stage": "csv_serialization", "setting": "dip_histograms_m20",
                    "condition_id": condition.condition_id, "condition_index": 3,
                    "replication_zero_based": 1, "master_seed": 819,
                    "data_stream_id": "PCG64:819:1:1:3:1",
                    "simulation_stream_id": "PCG64:819:1:13:3:1",
                    "historical_stream_id": "PCG64:819:1:5:3:1",
                    "prior_family": "histograms", "prior_mass": 20., "S": 20,
                    "output_csv": str(output),
                }
                self.assertEqual({key: details[key] for key in expected}, expected)
                self.assertEqual(sum(details["historical_counts_positive"]), 500)
                self.assertEqual(sum(details["historical_counts_negative"]), 500)

    def test_csv_validation_failure_preserves_exception_and_identifies_failing_row(self):
        condition = runner.CLASS_COUNT_CONDITIONS[3]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "valid.csv"
            options = dict(conditions=(condition,), n_replications=2, n_samples=20, master_seed=819)
            runner.run_experiment(output, show_progress=False, **options)
            rows = _read_rows(output)
            failing_index = next(index for index, row in enumerate(rows)
                                 if row["setting"] == "dip_histograms_m20"
                                 and row["replication"] == "1" and row["level"] == "0.95")
            failing_row = rows[failing_index]
            error = ValueError("injected CSV record validation failure")
            original = runner.parse_raw_level_measurements

            def fail_record(row):
                if row == failing_row:
                    raise error
                return original(row)

            with patch.object(runner, "parse_raw_level_measurements", side_effect=fail_record):
                with self.assertRaises(ValueError) as raised:
                    runner.validate_raw_results_csv(output, **options)
            self.assertIs(raised.exception, error)
            details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
            self.assertEqual(details["stage"], "csv_validation")
            self.assertEqual(details["protocol"], 8)
            self.assertEqual(details["csv_path"], str(output))
            self.assertEqual(details["csv_line"], failing_index + 2)
            for field in ("setting", "condition_id", "replication", "level", "master_seed",
                          "data_stream_id", "simulation_stream_id", "historical_stream_id"):
                self.assertEqual(details[field], failing_row[field])

    def test_builder_failure_after_completed_replication_preserves_published_csv(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        original_builder = runner.build_uniform_predictive_band
        original_writerows = csv.DictWriter.writerows
        written_rows = []
        calls = 0
        error = AssertionError("failure after completed replication")

        def write_rows(writer, rows):
            written_rows.extend(rows)
            return original_writerows(writer, rows)

        def fail_on_second_replication(**kwargs):
            nonlocal calls
            if kwargs["prior_mass"] != 1.0:
                return original_builder(**kwargs)
            calls += 1
            if calls == 3:  # Warm-up, replication 0, then replication 1.
                self.assertEqual(len(written_rows), 18)
                self.assertEqual({row["replication"] for row in written_rows}, {0})
                raise error
            return original_builder(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            output.write_bytes(b"previous published results\n")
            with (
                patch.object(runner, "build_uniform_predictive_band", new=fail_on_second_replication),
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
            self.assertEqual(details["method"], "DiP-ROC")
            self.assertEqual(details["setting"], "dip_uniform_m1")

    def test_raw_rows_use_inclusive_boundaries_and_preserve_full_precision(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        data_key, posterior_key = runner._stream_keys(condition, 0)
        rows = runner._raw_rows_for_replication(
            condition,
            0,
            _fixed_measurements(),
            n_replications=2,
            n_samples=40,
            master_seed=91,
            data_key=data_key,
            simulation_key=posterior_key,
            setting="dip_uniform_m1", computation_time_ns=0, execution_position=1,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(tuple(rows[0]), runner.RAW_FIELDNAMES)
        self.assertEqual(rows[0]["roc_covered"], 1)
        self.assertEqual(rows[0]["auc_covered"], 1)
        self.assertEqual(rows[0]["auc_width"], 0.7 - 0.4)
        self.assertEqual(rows[1]["auc_width"], 0.8 - 0.3)
        self.assertEqual(rows[0]["simulation_stream_id"], rows[1]["simulation_stream_id"])

        with self.assertRaises(AssertionError):
            runner._raw_rows_for_replication(
                condition,
                0,
                _fixed_measurements(),
                n_replications=2,
                n_samples=40,
                master_seed=91,
                data_key=posterior_key,
                simulation_key=posterior_key,
            setting="dip_uniform_m1", computation_time_ns=0, execution_position=1,
            )

        outside = ReplicationMeasurements(
            external_max_roc_deviation=0.2 + 2.0e-12,
            external_grid_auc=0.4 - 2.0e-12,
            roc_radii={0.80: 0.2, 0.95: 0.3},
            auc_intervals={0.80: (0.4, 0.7), 0.95: (0.3, 0.8)},
        )
        outside_rows = runner._raw_rows_for_replication(
            condition,
            0,
            outside,
            n_replications=2,
            n_samples=40,
            master_seed=91,
            data_key=data_key,
            simulation_key=posterior_key,
            setting="dip_uniform_m1", computation_time_ns=0, execution_position=1,
        )
        self.assertEqual(outside_rows[0]["roc_covered"], 0)
        self.assertEqual(outside_rows[0]["auc_covered"], 0)
        self.assertEqual(outside_rows[1]["roc_covered"], 1)
        self.assertEqual(outside_rows[1]["auc_covered"], 1)

    def test_runner_evaluates_once_per_setting_and_reuses_histograms_and_levels(self):
        condition = runner.CLASS_COUNT_CONDITIONS[3]
        with self.assertRaises(AssertionError):
            runner._evaluate_one_replication(condition, 0, gamma=np.linspace(0, 1, 11) ** 2, n_samples=40, master_seed=17)
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
                                      n_samples=40, conditions=(condition,), show_progress=False)
            rows = _read_rows(output)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(histogram.call_count, 3)
        self.assertEqual(evaluate.call_count, 27)
        self.assertEqual(timed.call_count, 27)
        self.assertEqual(warm.call_count, 9)
        for replication in range(3):
            calls = timed.call_args_list[9 * replication:9 * (replication + 1)]
            order = runner.SETTINGS[replication:] + runner.SETTINGS[:replication]
            self.assertEqual([call.kwargs["diagnostic_context"]["setting"] for call in calls], list(order))
            for call in calls:
                self.assertIs(call.kwargs["d_plus"], calls[0].kwargs["d_plus"])
                self.assertIs(call.kwargs["d_minus"], calls[0].kwargs["d_minus"])
                self.assertEqual(call.kwargs["S"], 40)
                self.assertEqual(call.kwargs["levels"], (0.80, 0.95))
                self.assertFalse(call.kwargs["d_plus"].flags.writeable)
                self.assertFalse(call.kwargs["d_minus"].flags.writeable)
                setting = call.kwargs["diagnostic_context"]["setting"]
                family = runner.SETTING_FAMILIES[setting]
                self.assertEqual("historical_grid_auc" in call.kwargs, family == "linear")
                self.assertEqual("historical_d_plus" in call.kwargs, family == "histograms")
                self.assertEqual("historical_d_minus" in call.kwargs, family == "histograms")
                if family == "histograms":
                    self.assertFalse(call.kwargs["historical_d_plus"].flags.writeable)
                    self.assertFalse(call.kwargs["historical_d_minus"].flags.writeable)
            self.assertEqual(len({id(call.kwargs["rng"]) for call in calls}), 9)
            evaluations = evaluate.call_args_list[9 * replication:9 * (replication + 1)]
            self.assertTrue(all(call.args[0] is evaluations[0].args[0] for call in evaluations))
        self.assertEqual(len(rows), 54)
        self.assertEqual([row["setting"] for row in rows[:18]],
                         [setting for setting in runner.SETTINGS for _ in range(2)])

    def test_full_design_tiny_integration_is_reproducible_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"
            arguments = {
                "master_seed": 2874,
                "n_replications": 2,
                "n_samples": 40,
                "conditions": runner.CLASS_COUNT_CONDITIONS,
                "show_progress": False,
            }
            runner.run_experiment(first, **arguments)
            runner.run_experiment(second, **arguments)

            self.assertEqual(_scientific_rows(_read_rows(first)), _scientific_rows(_read_rows(second)))
            self.assertEqual(len(_read_rows(first)), 540)
            self.assertEqual(
                runner.validate_raw_results_csv(
                    first,
                    master_seed=arguments["master_seed"],
                    n_replications=arguments["n_replications"],
                    n_samples=arguments["n_samples"],
                    conditions=arguments["conditions"],
                ),
                540,
            )

            rows = _read_rows(first)
            primary_keys = {
                (
                    row["setting"],
                    row["condition_id"],
                    int(row["replication"]),
                    float(row["level"]),
                )
                for row in rows
            }
            self.assertEqual(len(primary_keys), 540)

    def test_progress_does_not_change_csv_or_enter_timed_constructions(self):
        clock_calls = 0
        inside_timing = False

        def clock():
            nonlocal clock_calls, inside_timing
            clock_calls += 1
            inside_timing = not inside_timing
            return clock_calls * 1_234_567

        def check_print(*args, **kwargs):
            self.assertFalse(inside_timing, "Printing entered a timed construction.")
            self.assertIs(kwargs.get("flush"), True)

        arguments = dict(
            n_replications=3, n_samples=40,
            conditions=runner.CLASS_COUNT_CONDITIONS[:2],
        )
        with tempfile.TemporaryDirectory() as directory:
            quiet = Path(directory) / "quiet.csv"
            verbose = Path(directory) / "verbose.csv"
            with patch("roc_utils.utilities_experiments.time.perf_counter_ns", side_effect=clock), \
                    patch("builtins.print", side_effect=check_print) as printed:
                runner.run_experiment(quiet, show_progress=False, **arguments)
                printed.assert_not_called()
                self.assertEqual(clock_calls, 108)
                self.assertFalse(inside_timing)
                clock_calls = 0
                runner.run_experiment(verbose, show_progress=True, **arguments)
                self.assertGreater(printed.call_count, 0)
                self.assertEqual(clock_calls, 108)
                self.assertFalse(inside_timing)
            # With an identical clock, even the timing metadata must match.
            self.assertEqual(quiet.read_bytes(), verbose.read_bytes())

    def test_progress_means_match_csv_and_restart_for_each_condition(self):
        conditions = (runner.CLASS_COUNT_CONDITIONS[0], runner.CLASS_COUNT_CONDITIONS[7])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            with patch.object(runner, "_PROGRESS_INTERVAL", 2), \
                    patch.object(runner, "_generate_historical_summary", side_effect=[_history(value) for value in [0.4, 0.9, 0.7] * 2]), \
                    patch("builtins.print") as printed:
                runner.run_experiment(
                    output, n_replications=3, n_samples=40,
                    conditions=conditions, show_progress=True,
                )
            rows = _read_rows(output)

        lines = [call.args[0] for call in printed.call_args_list]
        self.assertIn("2 conditions; B=3, S=40 per setting, K=10", lines[0])
        self.assertIn("levels=(0.8, 0.95)", lines[0])
        self.assertIn(f"master_seed={runner.DEFAULT_MASTER_SEED}", lines[0])
        self.assertIn("construction threads=1", lines[1])
        self.assertIn("AUC restriction=(0.5, 0.86)", lines[1])
        self.assertIn("each use m=1, m=5 and m=20 per class", lines[1])
        for setting in ("DiP-ROC (uniform, m=20)", "DiP-ROC (linear, m=20)",
                        "DiP-ROC (histograms, m=20)"):
            self.assertTrue(any(setting in line for line in lines))
        self.assertTrue(any("validation of all 108 rows" in line for line in lines))
        self.assertTrue(any("rotates through 9 positions" in line for line in lines))
        self.assertEqual(sum(line.startswith("Progress ") for line in lines), 6)
        for condition in conditions:
            for completed in (1, 2, 3):
                header = (
                    f"Progress {condition.condition_id}: {completed}/3 "
                    f"replications ({100.0 * completed / 3:.1f}%); "
                    f"last replication index={completed - 1} (zero-based)."
                )
                self.assertEqual(lines.count(header), 1)
                position = lines.index(header) + 1
                for setting in runner.SETTINGS:
                    selected = [
                        row for row in rows
                        if row["condition_id"] == condition.condition_id
                        and row["setting"] == setting
                        and int(row["replication"]) < completed
                    ]
                    # One timing per replication, despite two level rows.
                    times = [int(row["computation_time_ns"]) for row in selected
                             if float(row["level"]) == 0.80]
                    expected_time = sum(times) / len(times) / 1e6
                    self.assertEqual(
                        lines[position],
                        f"  {runner.SETTING_LABELS[setting]}: mean construction time={expected_time:.3f} ms "
                        "(both levels together)",
                    )
                    position += 1
                    for level in (0.80, 0.95):
                        level_rows = [row for row in selected if float(row["level"]) == level]
                        def mean(field):
                            return sum(float(row[field]) for row in level_rows) / len(level_rows)
                        expected = (
                            f"    level={level:.0%}: "
                            f"ROC coverage={100.0 * mean('roc_covered'):.2f}%, "
                            f"AUC coverage={100.0 * mean('auc_covered'):.2f}%, "
                            f"mean radius={mean('roc_radius'):.6f}, "
                            f"mean AUC width={mean('auc_width'):.6f}"
                        )
                        self.assertEqual(lines[position], expected)
                        position += 1
                historical = [row for row in rows
                              if row["condition_id"] == condition.condition_id
                              and row["setting"] == "dip_linear_m1"
                              and float(row["level"]) == 0.8
                              and int(row["replication"]) < completed]
                aucs = [float(row["historical_grid_auc"]) for row in historical]
                targets = [float(row["prior_expected_grid_auc"]) for row in historical]
                adjusted = sum(int(row["historical_auc_adjusted"]) for row in historical)
                self.assertEqual(lines[position],
                    f"  Historical AUC: mean={sum(aucs) / completed:.6f}, "
                    f"range=[{min(aucs):.6f}, {max(aucs):.6f}]; "
                    f"target: mean={sum(targets) / completed:.6f}, "
                    f"range=[{min(targets):.6f}, {max(targets):.6f}]; "
                    f"adjusted={adjusted}/{completed} ({100 * adjusted / completed:.2f}%).")
        self.assertTrue(any(line.startswith("Simulation complete. Validating all 108") for line in lines))
        self.assertTrue(any(line.startswith("Saved 108 validated raw rows") for line in lines))
        for call in printed.call_args_list:
            self.assertEqual(call.kwargs, {"flush": True})

    def test_single_replication_prints_one_progress_report(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print") as printed:
            runner.run_experiment(
                Path(directory) / "one.csv", n_replications=1, n_samples=20,
                conditions=runner.CLASS_COUNT_CONDITIONS[:1], show_progress=True,
            )
        headers = [call.args[0] for call in printed.call_args_list
                   if call.args[0].startswith("Progress ")]
        self.assertEqual(len(headers), 1)
        self.assertIn("1/1 replications (100.0%); last replication index=0", headers[0])

    def test_failed_validation_does_not_print_a_success_message(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            output.write_text("previous published results\n", encoding="utf-8")
            with patch.object(runner, "validate_raw_results_csv", side_effect=AssertionError("invalid CSV")), \
                    patch("builtins.print") as printed:
                with self.assertRaisesRegex(AssertionError, "invalid CSV"):
                    runner.run_experiment(
                        output, n_replications=1, n_samples=20, overwrite=True,
                        conditions=runner.CLASS_COUNT_CONDITIONS[:1], show_progress=True,
                    )
            self.assertEqual(output.read_text(encoding="utf-8"), "previous published results\n")
            self.assertEqual(list(Path(directory).iterdir()), [output])
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(lines[-1].startswith("Simulation complete. Validating"))
        self.assertFalse(any(line.startswith("Saved ") for line in lines))

    def test_mocked_official_design_writes_exactly_2700000_complete_rows(self):
        history = _history()
        def fake_evaluation(condition, replication, **unused_arguments):
            data_key = runner._stream_keys(condition, replication)[0]
            order = runner._execution_order(replication)
            return {setting: (_fixed_measurements(), runner._simulation_stream_key(condition, replication, setting),
                              100, order.index(setting) + 1) for setting in runner.SETTINGS}, data_key, history
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "official_raw.csv"
            with patch.object(runner, "_evaluate_one_replication", side_effect=fake_evaluation) as evaluate:
                runner.run_experiment(output, show_progress=False)
            self.assertEqual(evaluate.call_count, 150_000)
            # Validation already checks every stream and pairing. Independently
            # verify cardinality and uniqueness without duplicating large sets.
            seen = np.zeros((15, 10000, 9, 2), dtype=bool)
            count = 0
            with output.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (int(row["condition_index"]), int(row["replication"]),
                           runner.SETTINGS.index(row["setting"]), runner.NOMINAL_LEVELS.index(float(row["level"])))
                    self.assertFalse(seen[key])
                    seen[key] = True
                    count += 1
            self.assertEqual(count, 2_700_000)
            self.assertTrue(np.all(seen))

    def test_condition_order_does_not_change_any_scientific_row(self):
        conditions = runner.CLASS_COUNT_CONDITIONS[:2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            forward_path = Path(temporary_directory) / "forward.csv"
            reverse_path = Path(temporary_directory) / "reverse.csv"
            common = {
                "master_seed": 886,
                "n_replications": 2,
                "n_samples": 30,
                "show_progress": False,
            }
            runner.run_experiment(
                forward_path,
                conditions=conditions,
                **common,
            )
            runner.run_experiment(
                reverse_path,
                conditions=tuple(reversed(conditions)),
                **common,
            )

            sort_key = lambda row: (
                int(row["condition_index"]),
                int(row["replication"]),
                float(row["level"]),
            )
            forward_rows = sorted(_read_rows(forward_path), key=sort_key)
            reverse_rows = sorted(_read_rows(reverse_path), key=sort_key)
            self.assertEqual(_scientific_rows(forward_rows), _scientific_rows(reverse_rows))

    def test_raw_csv_reconstructs_the_canonical_condition_summary(self):
        condition = runner.CLASS_COUNT_CONDITIONS[7]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "raw.csv"
            runner.run_experiment(
                output,
                master_seed=312,
                n_replications=3,
                n_samples=40,
                conditions=(condition,),
                show_progress=False,
            )
            rows = [row for row in _read_rows(output) if row["setting"] == "dip_uniform_m1"]

        rows_by_replication: dict[int, dict[float, dict[str, str]]] = {}
        for row in rows:
            rows_by_replication.setdefault(int(row["replication"]), {})[
                float(row["level"])
            ] = row

        measurements = []
        for replication in range(3):
            level_rows = rows_by_replication[replication]
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
                            float(level_rows[level]["auc_interval_lower"]),
                            float(level_rows[level]["auc_interval_upper"]),
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
                rows_by_replication[replication][level]
                for replication in range(3)
            ]
            self.assertEqual(
                summaries[level].roc_coverage,
                np.mean([int(row["roc_covered"]) for row in level_rows]),
            )
            self.assertEqual(
                summaries[level].auc_coverage,
                np.mean([int(row["auc_covered"]) for row in level_rows]),
            )
            self.assertEqual(
                summaries[level].roc_radius_mean,
                np.mean([float(row["roc_radius"]) for row in level_rows]),
            )
            self.assertEqual(
                summaries[level].auc_interval_width_mean,
                np.mean(
                    [
                        float(row["auc_width"])
                        for row in level_rows
                    ]
                ),
            )

    def test_validator_rejects_all_major_csv_corruptions(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            valid_path = Path(temporary_directory) / "valid.csv"
            arguments = {
                "master_seed": 44,
                "n_replications": 2,
                "n_samples": 20,
                "conditions": (condition,),
                "show_progress": False,
            }
            runner.run_experiment(valid_path, **arguments)
            rows = _read_rows(valid_path)

            boundary_radius = [dict(row) for row in rows]
            for row in boundary_radius:
                row["roc_radius"] = repr(float(np.sqrt(2.0)))
                row["roc_covered"] = "1"
            large_path = Path(temporary_directory) / "boundary_dip_radius.csv"
            with large_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=runner.RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(boundary_radius)
            self.assertEqual(runner.validate_raw_results_csv(
                large_path, **{key: value for key, value in arguments.items() if key != "show_progress"}), 36)
            self.assertEqual({float(row["roc_radius"]) for row in _read_rows(large_path)},
                             {float(np.sqrt(2.0))})

            corruptions: dict[str, list[dict[str, str]]] = {}

            wrong_width = [dict(row) for row in rows]
            wrong_width[0]["auc_width"] = "0.12345678901234568"
            corruptions["wrong width"] = wrong_width

            wrong_roc_flag = [dict(row) for row in rows]
            wrong_roc_flag[0]["roc_covered"] = str(
                1 - int(wrong_roc_flag[0]["roc_covered"])
            )
            corruptions["wrong ROC flag"] = wrong_roc_flag

            wrong_auc_flag = [dict(row) for row in rows]
            wrong_auc_flag[0]["auc_covered"] = str(
                1 - int(wrong_auc_flag[0]["auc_covered"])
            )
            corruptions["wrong AUC flag"] = wrong_auc_flag

            wrong_stream = [dict(row) for row in rows]
            wrong_stream[0]["simulation_stream_id"] += ":corrupt"
            corruptions["wrong stream"] = wrong_stream

            mismatched_external = [dict(row) for row in rows]
            mismatched_external[0]["external_grid_auc"] = "0.12345678901234568"
            mismatched_external[0]["auc_covered"] = str(
                int(
                    is_external_auc_covered(
                        float(mismatched_external[0]["external_grid_auc"]),
                        (
                            float(mismatched_external[0]["auc_interval_lower"]),
                            float(mismatched_external[0]["auc_interval_upper"]),
                        ),
                    )
                )
            )
            corruptions["cross-level external mismatch"] = mismatched_external

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
            missing_prior = [dict(row) for row in rows]
            missing_prior[0]["prior_mass"] = ""
            corruptions["missing_dip_prior"] = missing_prior
            legacy_method = [dict(row) for row in rows]
            legacy_method[0]["method"] = "BootP-ROC"
            corruptions["legacy_bootstrap_method"] = legacy_method
            legacy_setting = [dict(row) for row in rows]
            legacy_setting[0]["setting"] = "bootp"
            corruptions["legacy_bootstrap_setting"] = legacy_setting
            wrong_dip_radius = [dict(row) for row in rows]
            wrong_dip_radius[0]["roc_radius"] = "1.8"
            wrong_dip_radius[0]["roc_covered"] = "1"
            corruptions["dip_radius_bound"] = wrong_dip_radius
            cross_setting_auc = [dict(row) for row in rows]
            first_linear = next(row for row in cross_setting_auc if row["setting"] == "dip_linear_m1")
            for row in cross_setting_auc:
                if (row["setting"] == "dip_linear_m1" and row["replication"] == first_linear["replication"]
                        and row["condition_id"] == first_linear["condition_id"]):
                    row["external_grid_auc"] = "0.12345678901234568"
                    row["auc_covered"] = str(int(is_external_auc_covered(
                        float(row["external_grid_auc"]),
                        (float(row["auc_interval_lower"]), float(row["auc_interval_upper"])))))
            corruptions["cross_setting_external_auc"] = cross_setting_auc

            corruptions["missing level row"] = rows[:-1]
            corruptions["duplicate row"] = rows + [rows[0]]
            corruptions["missing complete setting"] = [row for row in rows
                                                        if row["setting"] != "dip_linear_m1"]
            for field, bad in (
                ("setting", "unknown"), ("setting", "dip_uniform_m1"),
                ("historical_n_positive", "499"), ("historical_n_negative", "501"),
                ("historical_stream_id", "PCG64:wrong"),
                ("prior_expected_grid_auc", "0.5"),
                ("prior_profile", "symmetric"), ("prior_mass", "2.0"),
                ("historical_auc_adjusted", "1"),
            ):
                changed = [dict(row) for row in rows]
                next(row for row in changed if row["setting"] == "dip_linear_m1")[field] = bad
                corruptions["historical_" + field + bad] = changed
            for setting in ("dip_uniform_m1", "dip_uniform_m5", "dip_uniform_m20"):
                for field in ("historical_grid_auc", "historical_stream_id", "historical_auc_adjusted"):
                    changed = [dict(row) for row in rows]
                    next(row for row in changed if row["setting"] == setting)[field] = "0"
                    corruptions["history_leak_" + setting + field] = changed
            changed = [dict(row) for row in rows]
            # Keep each individual row valid, but change the summary between levels.
            linear_row = next(row for row in changed
                              if row["setting"] == "dip_linear_m20" and row["level"] == "0.8")
            self.assertNotEqual(linear_row["historical_grid_auc"], "0.7")
            linear_row["historical_grid_auc"] = "0.7"
            linear_row["prior_expected_grid_auc"] = "0.7"
            linear_row["historical_auc_adjusted"] = "0"
            corruptions["cross_level_historical_summary"] = changed

            for setting, role in (("dip_uniform_m20", 11), ("dip_linear_m20", 12),
                                  ("dip_histograms_m20", 13)):
                corruptions["missing_" + setting] = [row for row in rows if row["setting"] != setting]
                for field, bad in (("prior_mass", "5.0"),
                                   ("simulation_stream_id", f"PCG64:44:1:{role - 1}:0:0"),
                                   ("execution_position", "10")):
                    changed = [dict(row) for row in rows]
                    next(row for row in changed if row["setting"] == setting)[field] = bad
                    corruptions[setting + "_" + field] = changed

            validation_arguments = {
                "master_seed": arguments["master_seed"],
                "n_replications": arguments["n_replications"],
                "n_samples": arguments["n_samples"],
                "conditions": arguments["conditions"],
            }
            for name, corrupted_rows in corruptions.items():
                with self.subTest(corruption=name):
                    corrupt_path = (
                        Path(temporary_directory)
                        / f"corrupt_{len(name)}_{name.count(' ')}.csv"
                    )
                    with corrupt_path.open(
                        "w",
                        encoding="utf-8",
                        newline="",
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=runner.RAW_FIELDNAMES,
                        )
                        writer.writeheader()
                        writer.writerows(corrupted_rows)
                    expected_exception = ValueError if name == "missing_dip_prior" else AssertionError
                    with self.assertRaises(expected_exception):
                        runner.validate_raw_results_csv(
                            corrupt_path,
                            **validation_arguments,
                        )

            wrong_header_path = Path(temporary_directory) / "wrong_header.csv"
            with wrong_header_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=runner.RAW_FIELDNAMES[:-1],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "v8 synthetic comparison CSV header"):
                runner.validate_raw_results_csv(
                    wrong_header_path,
                    **validation_arguments,
                )

    def test_histogram_metadata_validation_and_shared_history_pairing(self):
        arguments = dict(conditions=runner.CLASS_COUNT_CONDITIONS[:1], n_replications=2, n_samples=20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.csv"
            with patch.object(runner, "_generate_historical_summary", return_value=_history(.5)):
                runner.run_experiment(path, show_progress=False, **arguments)
            source = _read_rows(path)
            def write_validate(rows):
                changed = Path(directory) / "changed.csv"
                with changed.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=runner.RAW_FIELDNAMES)
                    writer.writeheader(); writer.writerows(rows)
                return runner.validate_raw_results_csv(changed, **arguments)
            # Row order is immaterial to scientific validation.
            self.assertEqual(write_validate(list(reversed(source))), 36)
            malformed_counts = ("[]", "[1,2]", "[true,0,0,0,0,0,0,0,0,499]",
                                "[-1,501,0,0,0,0,0,0,0,0]", "[0.5,499.5,0,0,0,0,0,0,0,0]",
                                "[0,499,0,0,0,0,0,0,0,0]", "not JSON")
            corruptions = [("historical_counts_positive", value) for value in malformed_counts]
            corruptions += [("historical_auc_adjusted", "0"), ("prior_mass", "500"),
                            ("prior_expected_grid_auc", "0.86"), ("historical_grid_auc", "0.75")]
            for field, bad in corruptions:
                with self.subTest(field=field, value=bad):
                    rows = [dict(row) for row in source]
                    next(row for row in rows if row["setting"] == "dip_histograms_m20")[field] = bad
                    with self.assertRaises((ValueError, AssertionError)):
                        write_validate(rows)
            # Individually valid histogram rows with identical raw and prior AUC
            # still fail when their historical profiles differ across masses.
            rows = [dict(row) for row in source]
            for row in rows:
                if row["setting"] == "dip_histograms_m20":
                    row["historical_counts_positive"] = json.dumps([50] * 10, separators=(",", ":"))
                    row["historical_counts_negative"] = row["historical_counts_positive"]
                    row["prior_expected_grid_auc"] = "0.5"
            with self.assertRaisesRegex(AssertionError, "same observed counts"):
                write_validate(rows)
            # Each linear row is valid, but all three masses must share one AUC.
            rows = [dict(row) for row in source]
            for row in rows:
                if row["setting"] == "dip_linear_m20":
                    row["historical_grid_auc"] = row["prior_expected_grid_auc"] = "0.7"
            with self.assertRaisesRegex(AssertionError, "share one historical sample"):
                write_validate(rows)
            for setting in ("dip_uniform_m1", "dip_uniform_m5", "dip_uniform_m20",
                            "dip_linear_m1", "dip_linear_m5", "dip_linear_m20"):
                rows = [dict(row) for row in source]
                next(row for row in rows if row["setting"] == setting)["historical_counts_positive"] = "[50,50,50,50,50,50,50,50,50,50]"
                with self.subTest(setting=setting), self.assertRaises(AssertionError):
                    write_validate(rows)

    def test_invalid_arguments_and_overwrite_policy_are_explicit(self):
        condition = runner.CLASS_COUNT_CONDITIONS[0]
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
                "_evaluate_one_replication",
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

    def test_competing_csv_publication_cannot_defeat_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "competing.csv"
            competing_bytes = b"another process published this result\n"
            original_publish = runner.publish_run_metadata

            def create_competing_output(*args, **kwargs):
                # A competitor wins after the runner's final existence check.
                self.assertFalse(output.exists())
                output.write_bytes(competing_bytes)
                return original_publish(*args, **kwargs)

            with patch.object(runner, "publish_run_metadata", side_effect=create_competing_output) as publish:
                with self.assertRaises(FileExistsError):
                    runner.run_experiment(
                        output, n_replications=1, n_samples=20,
                        conditions=runner.CLASS_COUNT_CONDITIONS[:1],
                        overwrite=False, show_progress=False,
                    )
            publish.assert_called_once()
            self.assertEqual(output.read_bytes(), competing_bytes)
            self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    def test_official_main_rejects_changes_to_the_historical_protocol(self):
        changed_constants = {
            "HISTORICAL_N_POSITIVE": 499,
            "HISTORICAL_N_NEGATIVE": 501,
            "SETTING_MASSES": {**runner.SETTING_MASSES, "dip_linear_m20": 5.0},
            "SETTING_FAMILIES": {**runner.SETTING_FAMILIES, "dip_histograms_m20": "linear"},
            "SETTING_PROFILES": {**runner.SETTING_PROFILES, "dip_linear_m20": "symmetric"},
            "_SETTING_STREAM_ROLES": {**runner._SETTING_STREAM_ROLES, "dip_uniform_m20": 7},
            "HISTORICAL_AUC_BOUNDS": (0.5, 0.87),
            "SETTINGS": ("dip_uniform_m1", "dip_linear_m1"),
        }
        for name, value in changed_constants.items():
            with self.subTest(constant=name), patch.object(runner, name, value):
                with patch.object(runner, "run_experiment") as run_mock:
                    with self.assertRaises(AssertionError):
                        runner.main([])
                    run_mock.assert_not_called()

    def test_official_main_forwards_the_exact_design_and_cli_arguments(self):
        expected_path = Path("mocked.csv")
        with patch.object(
            runner,
            "run_experiment",
            return_value=expected_path,
        ) as run_mock:
            returned = runner.main(
                [
                    "--output",
                    str(expected_path),
                    "--master-seed",
                    "123",
                    "--overwrite",
                ]
            )

        self.assertEqual(returned, expected_path)
        call = run_mock.call_args
        self.assertEqual(call.args[0], expected_path)
        self.assertEqual(call.kwargs["master_seed"], 123)
        self.assertEqual(call.kwargs["n_replications"], 10_000)
        self.assertEqual(call.kwargs["n_samples"], 4000)
        self.assertEqual(call.kwargs["conditions"], runner.CLASS_COUNT_CONDITIONS)
        self.assertTrue(call.kwargs["overwrite"])

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        for module_name, message in (
            (
                "roc_utils.semantic_rng",
                "semantic_rng relies on runtime assertions",
            ),
            (
                "run_experiment_main_evaluation_on_synthetic_data",
                "This experiment relies on runtime assertions",
            ),
        ):
            with self.subTest(module=module_name):
                import_command = (
                    "import sys; "
                    f"sys.path.insert(0, {str(PY_DIR)!r}); "
                    f"import {module_name}"
                )
                completed = subprocess.run(
                    [sys.executable, "-O", "-c", import_command],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)


    def test_v8_csv_rejects_previous_mixed_and_relabelled_score_protocols(self):
        self.assertEqual(runner.SYNTHETIC_DESIGN_VERSION, 3)
        self.assertEqual(runner.RAW_SCHEMA_VERSION, "class_counts_raw_v8")
        arguments = dict(conditions=runner.CLASS_COUNT_CONDITIONS[:1], n_replications=2,
                         n_samples=20, master_seed=31)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "valid.csv"
            runner.run_experiment(source, show_progress=False, **arguments)
            rows = _read_rows(source)
            beta_fields = ("positive_beta_a", "positive_beta_b", "negative_beta_a", "negative_beta_b")
            for case in ("old v1", "old v2", "old v3", "old v4", "old v5", "old v6", "old v7", "mixed versions", "old beta", "mixed beta"):
                with self.subTest(case=case):
                    changed = [dict(row) for row in rows]
                    if case in ("old v1", "old v2", "old v3", "old v4", "old v5", "old v6", "old v7", "mixed versions"):
                        selected = changed[-1:] if case == "mixed versions" else changed
                        version = int(case[-1]) if case.startswith("old v") else 4
                        for row in selected:
                            row["schema_version"] = f"class_counts_raw_v{version}"
                        message = "old or mixed synthetic protocol"
                    else:
                        selected = changed[-1:] if case == "mixed beta" else changed
                        for row in selected:
                            row.update(dict(zip(beta_fields, map(str, (5.0, 2.0, 2.0, 5.0)))))
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
                "raw_replications.csv", "raw_replications_v1.csv", "raw_replications_v2.csv", "raw_replications_v3.csv", "raw_replications_v4.csv", "raw_replications_v5.csv", "raw_replications_v6.csv", "raw_replications_v7.csv",
            )]
            previous_paths += [root / name / "custom.csv" for name in (
                "latex_tables", "latex_tables_v1", "latex_tables_v2", "latex_tables_v3", "latex_tables_v4", "latex_tables_v5", "latex_tables_v6", "latex_tables_v7",
            )]
            for path in previous_paths:
                with self.subTest(path=path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("previous results\n")
                    with patch.object(runner, "generate_simulation_dataset") as generate:
                        with self.assertRaisesRegex(ValueError, "protected"):
                            runner.run_experiment(path, overwrite=True, show_progress=False,
                                                  conditions=runner.CLASS_COUNT_CONDITIONS[:1],
                                                  n_replications=2, n_samples=20)
                        generate.assert_not_called()
                    self.assertEqual(path.read_text(), "previous results\n")
            file_alias = root / "renamed_v6.csv"
            file_alias.symlink_to(root / "raw_replications_v6.csv")
            directory_alias = root / "renamed_tables"
            directory_alias.symlink_to(root / "latex_tables_v6", target_is_directory=True)
            for alias in (file_alias, directory_alias / "custom.csv"):
                with self.subTest(alias=alias), patch.object(runner, "generate_simulation_dataset") as generate:
                    with self.assertRaisesRegex(ValueError, "protected"):
                        runner.run_experiment(alias, overwrite=True, show_progress=False,
                                              conditions=runner.CLASS_COUNT_CONDITIONS[:1],
                                              n_replications=2, n_samples=20)
                    generate.assert_not_called()
                self.assertEqual(alias.read_text(), "previous results\n")
            # A CLI override cannot bypass the same safeguard.
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.main(["--output", str(previous_paths[0]), "--overwrite"])
            renamed = root / "custom.csv"
            for version in (1, 2, 3, 4, 5, 6, 7):
                old_text = f"schema_version\nclass_counts_raw_v{version}\n"
                renamed.write_text(old_text)
                with self.assertRaisesRegex(ValueError, "protected"):
                    runner.run_experiment(renamed, overwrite=True, show_progress=False,
                                          conditions=runner.CLASS_COUNT_CONDITIONS[:1],
                                          n_replications=2, n_samples=20)
                self.assertEqual(renamed.read_text(), old_text)
            # A valid-looking first row must not hide historical rows later on.
            mixed_text = "schema_version\nclass_counts_raw_v8\nclass_counts_raw_v7\n"
            renamed.write_text(mixed_text)
            with self.assertRaisesRegex(ValueError, "protected"):
                runner.run_experiment(renamed, overwrite=True, show_progress=False,
                                      conditions=runner.CLASS_COUNT_CONDITIONS[:1],
                                      n_replications=2, n_samples=20)
            self.assertEqual(renamed.read_text(), mixed_text)

if __name__ == "__main__":
    unittest.main()
