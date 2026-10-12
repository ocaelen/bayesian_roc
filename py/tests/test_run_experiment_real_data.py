import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_experiment_empirical_evaluation_on_real_world_datasets as runner  # noqa
from roc_utils.evaluation_criteria import (  # noqa: E402
    is_external_auc_covered,
    is_external_roc_covered,
)
from roc_utils.posterior_predictive_roc_sampler import (  # noqa: E402
    symmetric_dirichlet_priors,
)
from roc_utils.real_datasets import (  # noqa: E402
    LoadedRealDataset,
    PreparedRealDataset,
    real_dataset_metadata,
    split_and_encode_real_dataset,
)
from roc_utils.real_model import RealDatasetScores  # noqa: E402
from roc_utils.semantic_rng import semantic_stream_id  # noqa: E402
from roc_utils.utilities_experiments import (  # noqa: E402
    ReplicationMeasurements,
    build_test_external_histograms,
    summarize_replication_measurements,
)


FERTILITY_INDEX = runner.REAL_DATASET_NAMES.index("Fertility")
FERTILITY_CONDITIONS = tuple(
    condition
    for condition in runner.REAL_DATA_CONDITIONS
    if condition.dataset_index == FERTILITY_INDEX
)
K5_CONDITIONS = tuple(
    condition
    for condition in runner.REAL_DATA_CONDITIONS
    if condition.n_bins == 5
)
ALL_REAL_DATA_CACHES_PRESENT = all(
    (
        runner.DEFAULT_REAL_DATA_DIR
        / f"{real_dataset_metadata(name).slug}.arff"
    ).is_file()
    for name in runner.REAL_DATASET_NAMES
)


def _readonly(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values.setflags(write=False)
    return values


def _fixed_measurements(
    *,
    external_deviation: float = 0.2 + 0.5e-12,
    external_auc: float = 0.4 - 0.5e-12,
) -> ReplicationMeasurements:
    return ReplicationMeasurements(
        external_max_roc_deviation=external_deviation,
        external_grid_auc=external_auc,
        roc_radii={0.80: 0.2, 0.95: 0.3},
        auc_intervals={0.80: (0.4, 0.7), 0.95: (0.3, 0.8)},
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames=runner.RAW_FIELDNAMES,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(row: dict[str, str]):
    return (
        int(row["condition_index"]),
        int(row["replication"]),
        float(row["level"]),
    )


def _fertility_loaded(source_path: Path) -> LoadedRealDataset:
    metadata = real_dataset_metadata("Fertility")
    source_path.write_text("synthetic offline fixture", encoding="utf-8")
    rng = np.random.default_rng(8831)
    negative = rng.normal(0.0, 1.0, size=(metadata.n_negative, 2))
    positive = rng.normal(0.55, 1.0, size=(metadata.n_positive, 2))
    X_raw = np.asarray(np.vstack((negative, positive)), dtype=object)
    y = np.asarray(
        [0] * metadata.n_negative + [1] * metadata.n_positive,
        dtype=np.int64,
    )
    source_indices = np.arange(metadata.n_complete_rows, dtype=np.int64)
    X_raw.setflags(write=False)
    y.setflags(write=False)
    source_indices.setflags(write=False)
    return LoadedRealDataset(
        name=metadata.name,
        slug=metadata.slug,
        source="synthetic pinned-metadata fixture",
        source_path=source_path,
        openml_data_id=metadata.openml_data_id,
        openml_version=metadata.openml_version,
        openml_file_id=metadata.openml_file_id,
        md5_checksum=metadata.md5_checksum,
        n_source_rows=metadata.n_source_rows,
        n_complete_rows=metadata.n_complete_rows,
        n_removed_rows=metadata.n_removed_rows,
        feature_names=("signal", "noise"),
        categorical_feature_names=(),
        X_raw=X_raw,
        y=y,
        source_row_indices=source_indices,
    )


def _prepared_fixture(metadata=None) -> PreparedRealDataset:
    if metadata is None:
        metadata = real_dataset_metadata("Fertility")
    train_positive, train_negative, held_positive, held_negative = (
        runner._split_counts(metadata)
    )

    def arrays(negative_count, positive_count):
        y = np.asarray(
            [0] * negative_count + [1] * positive_count,
            dtype=np.int64,
        )
        X = np.linspace(-1.0, 1.0, y.size).reshape(-1, 1)
        return X, y

    X_train, y_train = arrays(train_negative, train_positive)
    X_test, y_test = arrays(held_negative, held_positive)
    X_external, y_external = arrays(held_negative, held_positive)
    train_stop = y_train.size
    test_stop = train_stop + y_test.size
    external_stop = test_stop + y_external.size
    assert external_stop == metadata.n_complete_rows
    return PreparedRealDataset(
        name=metadata.name,
        slug=metadata.slug,
        source="fixture",
        source_path=Path("/unused/source.arff"),
        openml_data_id=metadata.openml_data_id,
        openml_version=metadata.openml_version,
        openml_file_id=metadata.openml_file_id,
        md5_checksum=metadata.md5_checksum,
        n_source_rows=metadata.n_source_rows,
        n_complete_rows=metadata.n_complete_rows,
        n_removed_rows=metadata.n_removed_rows,
        numeric_feature_names=("x",),
        categorical_feature_names=(),
        feature_names=("x",),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_external=X_external,
        y_external=y_external,
        train_row_indices=np.arange(train_stop, dtype=np.int64),
        test_row_indices=np.arange(train_stop, test_stop, dtype=np.int64),
        external_row_indices=np.arange(
            test_stop,
            external_stop,
            dtype=np.int64,
        ),
    )


def _score_fixture(metadata=None, n_iterations: int = 7) -> RealDatasetScores:
    if metadata is None:
        metadata = real_dataset_metadata("Fertility")
    _, _, held_positive, held_negative = runner._split_counts(metadata)
    return RealDatasetScores(
        positive_test_scores=_readonly(np.linspace(0.55, 0.85, held_positive)),
        negative_test_scores=_readonly(np.linspace(0.15, 0.45, held_negative)),
        positive_external_scores=_readonly(
            np.linspace(0.50, 0.80, held_positive)
        ),
        negative_external_scores=_readonly(
            np.linspace(0.20, 0.50, held_negative)
        ),
        n_iterations=n_iterations,
    )


def _scientific_rows(rows):
    # Wall-clock measurements and execution positions are run metadata.
    return [{key: value for key, value in row.items()
             if key not in ("computation_time_ns", "execution_position")}
            for row in rows]


class RealDataExperimentTests(unittest.TestCase):
    def test_main_and_supplementary_grids_preserve_original_condition_streams(self):
        self.assertEqual(runner.MAIN_N_BINS, 10)
        self.assertEqual(runner.SUPPLEMENTARY_N_BINS_VALUES, (5, 20))
        self.assertEqual(runner.N_BINS_VALUES, (5, 10, 20))
        for dataset_index, name in enumerate(runner.REAL_DATASET_NAMES):
            conditions = tuple(c for c in runner.REAL_DATA_CONDITIONS if c.dataset_name == name)
            self.assertEqual([c.index for c in conditions], [3 * dataset_index + i for i in range(3)])
            self.assertEqual([c.n_bins for c in conditions], [5, 10, 20])
            self.assertEqual([runner._posterior_stream_key(c, 7) for c in conditions],
                             [(3, 2, 3 * dataset_index + i, 7) for i in range(3)])

    def test_grid_rotation_rejects_invalid_replications_and_mixed_dataset_bundles(self):
        for replication in (-1, 0.5, True):
            with self.subTest(replication=replication), self.assertRaises(AssertionError):
                runner._execution_order(FERTILITY_CONDITIONS, replication)
        foreign = runner.REAL_DATA_CONDITIONS[0]
        for conditions in ((), FERTILITY_CONDITIONS + (FERTILITY_CONDITIONS[0],),
                           FERTILITY_CONDITIONS + (foreign,)):
            with self.subTest(conditions=conditions), self.assertRaises(AssertionError):
                runner._execution_order(conditions, 0)

    def test_shared_score_arrays_are_revalidated_after_construction(self):
        metadata = real_dataset_metadata("Fertility")
        for name in ("positive_test_scores", "negative_test_scores",
                     "positive_external_scores", "negative_external_scores"):
            for invalid in ("writeable", float("nan"), -0.01, 1.01):
                scores = _score_fixture()
                values = getattr(scores, name)
                values.setflags(write=True)
                if invalid != "writeable":
                    values[0] = invalid
                    values.setflags(write=False)
                with self.subTest(name=name, invalid=invalid), self.assertRaises(AssertionError):
                    runner._assert_scores(scores, metadata)

    def test_exact_design_priors_and_source_class_counts(self):
        self.assertEqual(runner.EXPERIMENT_NAME, "real_data")
        self.assertEqual(
            runner.RESULTS_DIRECTORY_NAME,
            "empirical_evaluation_on_real_world_datasets",
        )
        self.assertEqual(
            runner.DEFAULT_OUTPUT_CSV.relative_to(runner.PY_DIR),
            Path(
                "results/current_experiments/"
                "empirical_evaluation_on_real_world_datasets/"
                "raw_replications_v3.csv"
            ),
        )
        self.assertEqual(len(runner.REAL_DATASET_NAMES), 12)
        self.assertEqual(runner.N_BINS_VALUES, (5, 10, 20))
        self.assertEqual(runner.PRIOR_PROFILE, "symmetric")
        self.assertEqual(runner.PRIOR_MASS, 1.0)
        self.assertEqual(runner.NOMINAL_LEVELS, (0.80, 0.95))
        self.assertEqual(runner.OFFICIAL_REPLICATIONS, 300)
        self.assertEqual(runner.OFFICIAL_SAMPLES, 4000)
        self.assertEqual(len(runner.REAL_DATA_CONDITIONS), 36)
        self.assertEqual(12 * 3 * 300 * 2, 21_600)
        self.assertEqual(runner.REAL_PROTOCOL_VERSION, 3)
        self.assertEqual(runner.RAW_SCHEMA_VERSION, "real_data_raw_v3")
        self.assertEqual(runner.METHODS, ("DiP-ROC",))
        self.assertEqual(
            [condition.n_bins for condition in runner.REAL_DATA_CONDITIONS[:6]],
            [5, 10, 20, 5, 10, 20],
        )
        self.assertEqual(
            runner.REAL_DATA_CONDITIONS[0].condition_id,
            "abalone_k_5",
        )
        self.assertEqual(
            runner.REAL_DATA_CONDITIONS[-1].condition_id,
            "contraceptive_method_choice_k_20",
        )

        for n_bins in runner.N_BINS_VALUES:
            alpha_plus, alpha_minus = symmetric_dirichlet_priors(n_bins, 1.0)
            np.testing.assert_array_equal(alpha_plus, alpha_minus)
            np.testing.assert_allclose(
                alpha_plus,
                np.full(n_bins, 1.0 / n_bins),
                rtol=0.0,
                atol=1e-15,
            )
            self.assertAlmostEqual(float(np.sum(alpha_plus)), 1.0)

        expected_counts = {
            "Abalone": (1249, 1258, 416, 419),
            "Adult": (6726, 20410, 2241, 6802),
            "Alzheimer": (55, 51, 17, 17),
            "Banknote Authentication": (366, 458, 122, 152),
            "Diabetes": (162, 300, 53, 100),
            "Fertility": (8, 54, 2, 17),
            "Haberman": (49, 135, 16, 45),
            "Mammographic": (243, 257, 80, 85),
            "Sonar": (67, 59, 22, 19),
            "Titanic": (427, 894, 142, 298),
            "Transfusion": (108, 342, 35, 114),
            "Contraceptive Method Choice": (508, 379, 168, 125),
        }
        for name, expected in expected_counts.items():
            metadata = real_dataset_metadata(name)
            self.assertEqual(runner._split_counts(metadata), expected)

        with self.assertRaises(AssertionError):
            runner.RealDataCondition(True, 0, "Abalone", 5)
        with self.assertRaises(AssertionError):
            runner.RealDataCondition(0, True, "Abalone", 5)
        with self.assertRaises(AssertionError):
            runner.RealDataCondition(0, 0, "Abalone", True)

    def test_semantic_streams_encode_the_paired_grid_design(self):
        master_seed = 923
        for replication in (0, 1, 299):
            data_key = runner._data_stream_key(FERTILITY_INDEX, replication)
            self.assertEqual(data_key, (3, 1, FERTILITY_INDEX, replication))
            posterior_keys = {
                runner._posterior_stream_key(condition, replication)
                for condition in FERTILITY_CONDITIONS
            }
            self.assertEqual(len(posterior_keys), 3)
            data_id = semantic_stream_id(master_seed, *data_key)
            posterior_ids = {
                semantic_stream_id(master_seed, *key) for key in posterior_keys
            }
            self.assertEqual(len(posterior_ids), 3)
            self.assertNotIn(data_id, posterior_ids)
        self.assertNotEqual(
            runner._data_stream_key(FERTILITY_INDEX, 0),
            runner._data_stream_key(FERTILITY_INDEX + 1, 0),
        )

    def test_dip_and_warmup_streams_preserve_dataset_grid_identity(self):
        for condition in runner.REAL_DATA_CONDITIONS:
            self.assertEqual(runner._simulation_stream_key(condition, 7, "DiP-ROC"),
                             (3, 2, condition.index, 7))
            self.assertEqual(runner._warmup_stream_key(condition.index, "DiP-ROC"),
                             (3, 4, 2, condition.index, 0))
            with self.assertRaises(AssertionError):
                runner._simulation_stream_key(condition, 7, "BootP-ROC")

    def test_raw_rows_preserve_provenance_model_and_boundaries(self):
        metadata = real_dataset_metadata("Fertility")
        condition = FERTILITY_CONDITIONS[0]
        prepared = _prepared_fixture(metadata)
        scores = _score_fixture(metadata)
        data_key = runner._data_stream_key(condition.dataset_index, 0)
        posterior_key = runner._posterior_stream_key(condition, 0)
        rows = runner._raw_rows_for_condition_replication(
            condition,
            0,
            _fixed_measurements(),
            metadata,
            prepared,
            scores,
            n_replications=1,
            n_samples=40,
            master_seed=73,
            data_key=data_key,
            simulation_key=posterior_key,
            method="DiP-ROC", computation_time_ns=0, execution_position=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["openml_data_id"], metadata.openml_data_id)
        self.assertEqual(rows[0]["md5_checksum"], metadata.md5_checksum)
        self.assertEqual(rows[0]["model_penalty"], "l2")
        self.assertEqual(rows[0]["model_C"], 1.0)
        self.assertEqual(rows[0]["model_l1_ratio"], 0.0)
        self.assertEqual(
            rows[0]["numeric_scaling"],
            "training_only_numeric_standardization",
        )
        self.assertEqual(rows[0]["model_n_iter"], scores.n_iterations)
        self.assertEqual(rows[0]["n_positive"], 2)
        self.assertEqual(rows[0]["n_negative"], 17)
        self.assertEqual(rows[0]["auc_width"], 0.7 - 0.4)
        self.assertEqual(rows[0]["roc_covered"], 1)
        self.assertEqual(rows[0]["auc_covered"], 1)
        self.assertEqual(rows[0]["data_stream_id"], rows[1]["data_stream_id"])
        self.assertEqual(
            rows[0]["simulation_stream_id"],
            rows[1]["simulation_stream_id"],
        )

        overlapping_partition = replace(
            prepared,
            external_row_indices=prepared.test_row_indices.copy(),
        )
        with self.assertRaisesRegex(AssertionError, "pairwise disjoint"):
            runner._assert_prepared_replication(
                overlapping_partition,
                metadata,
            )

        with self.assertRaises(AssertionError):
            runner._raw_rows_for_condition_replication(
                condition,
                0,
                _fixed_measurements(),
                metadata,
                prepared,
                scores,
                n_replications=1,
                n_samples=40,
                master_seed=73,
                data_key=posterior_key,
                simulation_key=posterior_key,
            method="DiP-ROC", computation_time_ns=0, execution_position=2,
            )

        with self.assertRaises(AssertionError):
            runner._raw_rows_for_condition_replication(
                condition,
                0,
                _fixed_measurements(),
                real_dataset_metadata("Abalone"),
                prepared,
                scores,
                n_replications=1,
                n_samples=40,
                master_seed=73,
                data_key=data_key,
                simulation_key=posterior_key,
            method="DiP-ROC", computation_time_ns=0, execution_position=2,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.csv"
            provenance = runner._collect_experiment_metadata((condition,), 1, 40, 73)
            text_rows = [{field: str(row[field]) for field in runner.RAW_FIELDNAMES}
                         for row in rows]
            _write_rows(path, text_rows)
            with self.assertRaises(AssertionError):
                runner.validate_raw_results_csv(path, conditions=(condition,),
                    n_replications=1, n_samples=40, master_seed=73, metadata=provenance)
            for row in text_rows:
                row["execution_position"] = "1"
            _write_rows(path, text_rows)
            self.assertEqual(runner.validate_raw_results_csv(path, conditions=(condition,),
                n_replications=1, n_samples=40, master_seed=73, metadata=provenance), 2)

    def test_one_split_model_and_histograms_are_shared_by_methods_and_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = _fertility_loaded(Path(directory) / "fertility.arff")
            output = Path(directory) / "raw.csv"
            metadata = real_dataset_metadata("Fertility")
            with self.assertRaises(AssertionError):
                runner._prepare_dataset_replication(loaded, metadata, FERTILITY_INDEX + 1, 0, master_seed=7)
            condition = FERTILITY_CONDITIONS[0]
            alpha_plus, alpha_minus = symmetric_dirichlet_priors(condition.n_bins, 1.0)
            with self.assertRaises(AssertionError):
                runner._evaluate_grid_condition(condition, 0, _score_fixture(metadata),
                    np.linspace(0, 1, condition.n_bins + 1) ** 2, alpha_plus, alpha_minus,
                    n_samples=40, master_seed=7)
            with (
                patch.object(runner, "load_real_dataset", return_value=loaded) as load,
                patch.object(runner, "split_and_encode_real_dataset", wraps=split_and_encode_real_dataset) as split,
                patch.object(runner, "fit_logistic_regression_scores", side_effect=lambda prepared: _score_fixture()) as score,
                patch.object(runner, "build_test_external_histograms", wraps=build_test_external_histograms) as histogram,
                patch.object(runner, "build_timed_band", wraps=runner.build_timed_band) as timed,
                patch.object(runner, "warm_up_band", wraps=runner.warm_up_band) as warm,
                patch.object(runner, "evaluate_replication", return_value=_fixed_measurements()) as evaluate,
            ):
                runner.run_experiment(output, master_seed=901, n_replications=3,
                                      n_samples=20, conditions=FERTILITY_CONDITIONS, show_progress=False)
            rows = _read_rows(output)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(split.call_count, 3)
        self.assertEqual(score.call_count, 3)
        self.assertEqual(histogram.call_count, 9)
        self.assertEqual(timed.call_count, 9)
        self.assertEqual(evaluate.call_count, 9)
        self.assertEqual(warm.call_count, 3)
        self.assertEqual({row["method"] for row in rows}, {"DiP-ROC"})
        expected_orders = ((5, 10, 20), (10, 20, 5), (20, 5, 10))
        for replication in range(3):
            calls = histogram.call_args_list[3 * replication:3 * (replication + 1)]
            self.assertEqual(tuple(call.args[-1].size - 1 for call in calls), expected_orders[replication])
            for score_position in range(4):
                self.assertEqual(len({id(call.args[score_position]) for call in calls}), 1)
            for position, call in enumerate(timed.call_args_list[3 * replication:3 * (replication + 1)], 1):
                self.assertIs(call.args[0], runner.build_posterior_predictive_band)
                self.assertEqual(call.kwargs["S"], 20)
                self.assertEqual(call.kwargs["levels"], (0.80, 0.95))
                self.assertFalse(call.kwargs["d_plus"].flags.writeable)
            replicated_rows = [row for row in rows if int(row["replication"]) == replication]
            self.assertEqual([int(row["K"]) for row in replicated_rows], [5, 5, 10, 10, 20, 20])
            self.assertEqual([int(row["execution_position"]) for row in replicated_rows],
                             [expected_orders[replication].index(k) + 1 for k in (5, 5, 10, 10, 20, 20)])
        self.assertEqual(len(rows), 18)

    def test_construction_failures_identify_dataset_grid_method_and_actual_stream(self):
        condition = FERTILITY_CONDITIONS[1]
        alpha_plus, alpha_minus = symmetric_dirichlet_priors(10, 1.0)
        for method, name, role in (
            ("DiP-ROC", "build_posterior_predictive_band", 2),
        ):
            for phase, replication in (("warmup", 0), ("construction", 1)):
                with self.subTest(method=method, phase=phase):
                    error = AssertionError("injected builder failure")
                    captured = {}

                    def fail(**kwargs):
                        self.assertNotIn("diagnostic_context", kwargs)
                        captured.update(rng=kwargs["rng"], state=kwargs["rng"].bit_generator.state)
                        raise error

                    with patch.object(runner, name, new=fail):
                        with self.assertRaises(AssertionError) as raised:
                            runner._evaluate_grid_condition(
                                condition, replication, _score_fixture(),
                                runner.fixed_uniform_grid(10), alpha_plus, alpha_minus,
                                n_samples=40, master_seed=17,
                            )
                    self.assertIs(raised.exception, error)
                    details = ast.literal_eval(error.__notes__[-1].split(": ", 1)[1])
                    simulation_key = ((3, 4, role, 16, 0) if phase == "warmup"
                                      else (3, role, 16, replication))
                    expected = {
                        "experiment": runner.EXPERIMENT_NAME, "method": method,
                        "condition_id": "fertility_k_10", "condition_index": 16,
                        "replication_zero_based": replication, "K": 10,
                        "master_seed": 17, "phase": phase, "S": 40,
                        "data_stream_id": semantic_stream_id(17, 3, 1, 5, replication),
                        "simulation_stream_id": semantic_stream_id(17, *simulation_key),
                    }
                    self.assertEqual({key: details[key] for key in expected}, expected)
                    self.assertEqual(sum(details["d_plus"]), 2)
                    self.assertEqual(sum(details["d_minus"]), 17)
                    self.assertEqual(captured["rng"].bit_generator.state, captured["state"])

    def test_builder_failure_after_completed_replication_preserves_published_csv(self):
        condition = FERTILITY_CONDITIONS[0]
        original_builder = runner.build_posterior_predictive_band
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
                self.assertEqual(len(written_rows), 2)
                self.assertEqual({row["replication"] for row in written_rows}, {0})
                raise error
            return original_builder(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.csv"
            output.write_bytes(b"previous published results\n")
            loaded = _fertility_loaded(Path(directory) / "fertility.arff")
            with (
                patch.object(runner, "load_real_dataset", return_value=loaded),
                patch.object(runner, "build_posterior_predictive_band", new=fail_on_second_replication),
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
            self.assertEqual(details["condition_id"], "fertility_k_5")

    def test_tiny_true_integration_is_byte_reproducible_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "fertility.arff"
            loaded = _fertility_loaded(source_path)
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"
            arguments = {
                "data_dir": Path(temporary_directory) / "data",
                "master_seed": 322,
                "n_replications": 2,
                "n_samples": 40,
                "conditions": FERTILITY_CONDITIONS,
                "show_progress": False,
            }
            with patch.object(runner, "load_real_dataset", return_value=loaded):
                runner.run_experiment(first, **arguments)
                runner.run_experiment(second, **arguments)

            self.assertEqual(_scientific_rows(_read_rows(first)), _scientific_rows(_read_rows(second)))
            self.assertEqual(len(_read_rows(first)), 12)
            self.assertEqual(
                runner.validate_raw_results_csv(
                    first,
                    master_seed=arguments["master_seed"],
                    n_replications=arguments["n_replications"],
                    n_samples=arguments["n_samples"],
                    conditions=arguments["conditions"],
                ),
                12,
            )

    @unittest.skipUnless(
        ALL_REAL_DATA_CACHES_PRESENT,
        "all twelve checksum-pinned ARFF files must already be cached",
    )
    def test_cached_all_datasets_true_reduced_integration_is_reproducible(self):
        self.assertEqual(len(K5_CONDITIONS), len(runner.REAL_DATASET_NAMES))
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"
            arguments = {
                "data_dir": runner.DEFAULT_REAL_DATA_DIR,
                "master_seed": runner.DEFAULT_MASTER_SEED,
                "n_replications": 1,
                "n_samples": 20,
                "conditions": K5_CONDITIONS,
                "show_progress": False,
            }
            runner.run_experiment(first, **arguments)
            runner.run_experiment(second, **arguments)

            self.assertEqual(_scientific_rows(_read_rows(first)), _scientific_rows(_read_rows(second)))
            self.assertEqual(
                runner.validate_raw_results_csv(
                    first,
                    master_seed=arguments["master_seed"],
                    n_replications=arguments["n_replications"],
                    n_samples=arguments["n_samples"],
                    conditions=arguments["conditions"],
                ),
                24,
            )
            rows = _read_rows(first)

        self.assertEqual(len(rows), 24)
        self.assertEqual({int(row["K"]) for row in rows}, {5})
        self.assertEqual(
            {row["dataset_name"] for row in rows},
            set(runner.REAL_DATASET_NAMES),
        )
        self.assertEqual(len({row["data_stream_id"] for row in rows}), 12)
        self.assertEqual(
            len({row["simulation_stream_id"] for row in rows}),
            12,
        )

    def test_condition_order_and_subset_do_not_change_scientific_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded = _fertility_loaded(
                Path(temporary_directory) / "fertility.arff"
            )
            forward = Path(temporary_directory) / "forward.csv"
            reverse = Path(temporary_directory) / "reverse.csv"
            subset = Path(temporary_directory) / "subset.csv"
            sparse_subset = Path(temporary_directory) / "sparse.csv"
            common = {
                "data_dir": Path(temporary_directory) / "data",
                "master_seed": 77,
                "n_replications": 3,
                "n_samples": 30,
                "show_progress": False,
            }
            with patch.object(runner, "load_real_dataset", return_value=loaded):
                runner.run_experiment(
                    forward,
                    conditions=FERTILITY_CONDITIONS,
                    **common,
                )
                runner.run_experiment(
                    reverse,
                    conditions=tuple(reversed(FERTILITY_CONDITIONS)),
                    **common,
                )
                selected = (FERTILITY_CONDITIONS[1],)
                runner.run_experiment(subset, conditions=selected, **common)
                runner.run_experiment(sparse_subset, conditions=(FERTILITY_CONDITIONS[2], FERTILITY_CONDITIONS[0]), **common)

            self.assertEqual(_scientific_rows(_read_rows(forward)), _scientific_rows(_read_rows(reverse)))
            expected_subset = [
                row
                for row in _read_rows(forward)
                if row["condition_id"] == selected[0].condition_id
            ]
            self.assertEqual(
                _scientific_rows(sorted(_read_rows(subset), key=_sort_key)),
                _scientific_rows(sorted(expected_subset, key=_sort_key)),
            )

            sparse_rows = _read_rows(sparse_subset)
            expected_sparse = [row for row in _read_rows(forward) if row["K"] in ("5", "20")]
            self.assertEqual(_scientific_rows(sparse_rows), _scientific_rows(expected_sparse))
            for replication in range(3):
                positions = {int(row["K"]): int(row["execution_position"]) for row in sparse_rows
                             if int(row["replication"]) == replication}
                self.assertEqual(positions, {5: 1, 20: 2} if replication % 2 == 0 else {5: 2, 20: 1})

    def test_raw_csv_reconstructs_the_canonical_summary(self):
        condition = FERTILITY_CONDITIONS[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded = _fertility_loaded(
                Path(temporary_directory) / "fertility.arff"
            )
            output = Path(temporary_directory) / "raw.csv"
            with patch.object(runner, "load_real_dataset", return_value=loaded):
                runner.run_experiment(
                    output,
                    master_seed=29,
                    n_replications=3,
                    n_samples=40,
                    conditions=(condition,),
                    show_progress=False,
                )
            rows = [row for row in _read_rows(output) if row["method"] == "DiP-ROC"]

        by_replication: dict[int, dict[float, dict[str, str]]] = {}
        for row in rows:
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
                by_replication[replication][level] for replication in range(3)
            ]
            self.assertEqual(
                summaries[level].roc_coverage,
                np.mean([int(row["roc_covered"]) for row in level_rows]),
            )
            self.assertEqual(
                summaries[level].auc_coverage,
                np.mean([int(row["auc_covered"]) for row in level_rows]),
            )

    def test_mocked_official_design_writes_exactly_21600_rows(self):
        minimal_prepared = _prepared_fixture()
        minimal_scores = _score_fixture()
        load_count = 0
        prepare_count = 0
        evaluation_count = 0

        def fake_load(*unused_arguments, **unused_keywords):
            nonlocal load_count
            load_count += 1
            return object()

        def fake_prepare(
            loaded,
            metadata,
            dataset_index,
            replication,
            *,
            master_seed,
        ):
            nonlocal prepare_count
            prepare_count += 1
            return (
                minimal_prepared,
                minimal_scores,
                runner._data_stream_key(dataset_index, replication),
            )

        def fake_evaluation(condition, replication, scores, gamma, alpha_plus, alpha_minus,
                            *, n_samples, master_seed, execution_position=1):
            nonlocal evaluation_count
            evaluation_count += 1
            return {method: (
                _fixed_measurements(external_deviation=0.1 + condition.n_bins / 1000.0,
                                    external_auc=0.45 + condition.n_bins / 1000.0),
                runner._simulation_stream_key(condition, replication, method),
                100, execution_position)
                for method in runner.METHODS}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "official.csv"
            with (
                patch.object(runner, "load_real_dataset", side_effect=fake_load),
                patch.object(runner, "_assert_loaded_source"),
                patch.object(
                    runner,
                    "_prepare_dataset_replication",
                    side_effect=fake_prepare,
                ),
                patch.object(
                    runner,
                    "_evaluate_grid_condition",
                    side_effect=fake_evaluation,
                ),
                patch.object(runner, "_assert_prepared_replication"),
                patch.object(runner, "_assert_scores"),
            ):
                runner.run_experiment(
                    output,
                    data_dir=Path(temporary_directory) / "data",
                    master_seed=817,
                    n_replications=300,
                    n_samples=4000,
                    conditions=runner.REAL_DATA_CONDITIONS,
                    show_progress=False,
                )
            rows = _read_rows(output)

        self.assertEqual(load_count, 12)
        self.assertEqual(prepare_count, 3_600)
        self.assertEqual(evaluation_count, 10_800)
        self.assertEqual(len(rows), 21_600)
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
            21_600,
        )
        self.assertEqual(len({row["data_stream_id"] for row in rows}), 3_600)
        self.assertEqual(
            len({row["simulation_stream_id"] for row in rows}),
            10_800,
        )
        self.assertTrue(
            {row["data_stream_id"] for row in rows}.isdisjoint(
                {row["simulation_stream_id"] for row in rows}
            )
        )

    def test_validator_rejects_pairing_provenance_and_csv_corruptions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loaded = _fertility_loaded(
                Path(temporary_directory) / "fertility.arff"
            )
            valid_path = Path(temporary_directory) / "valid.csv"
            with patch.object(runner, "load_real_dataset", return_value=loaded):
                runner.run_experiment(
                    valid_path,
                    master_seed=41,
                    n_replications=1,
                    n_samples=30,
                    conditions=FERTILITY_CONDITIONS,
                    show_progress=False,
                )
            valid_rows = _read_rows(valid_path)
            provenance = runner.read_run_metadata(valid_path, metadata_schema=runner.RUN_METADATA_SCHEMA)

            permitted = [row.copy() for row in valid_rows]
            for row in permitted:
                if row["K"] == "10":
                    row["external_max_roc_deviation"] = "0.25"
                    row["roc_covered"] = str(
                        int(
                            is_external_roc_covered(
                                0.25,
                                float(row["roc_radius"]),
                            )
                        )
                    )
                    row["external_grid_auc"] = "0.52"
                    row["auc_covered"] = str(
                        int(
                            is_external_auc_covered(
                                0.52,
                                (
                                    float(row["auc_interval_lower"]),
                                    float(row["auc_interval_upper"]),
                                ),
                            )
                        )
                    )
            permitted_path = Path(temporary_directory) / "permitted.csv"
            _write_rows(permitted_path, permitted)
            self.assertEqual(
                runner.validate_raw_results_csv(
                    permitted_path,
                    conditions=FERTILITY_CONDITIONS,
                    n_replications=1,
                    n_samples=30,
                    master_seed=41, metadata=provenance,
                ),
                6,
            )

            corruptions = {}
            wrong_checksum = [row.copy() for row in valid_rows]
            wrong_checksum[0]["md5_checksum"] = "0" * 32
            corruptions["checksum"] = wrong_checksum
            wrong_count = [row.copy() for row in valid_rows]
            wrong_count[0]["n_positive"] = "3"
            corruptions["count"] = wrong_count
            wrong_model = [row.copy() for row in valid_rows]
            wrong_model[0]["model_C"] = "2.0"
            corruptions["model"] = wrong_model
            wrong_width = [row.copy() for row in valid_rows]
            wrong_width[0]["auc_width"] = "0.123"
            corruptions["width"] = wrong_width
            wrong_flag = [row.copy() for row in valid_rows]
            wrong_flag[0]["auc_covered"] = str(
                1 - int(wrong_flag[0]["auc_covered"])
            )
            corruptions["flag"] = wrong_flag
            wrong_stream = [row.copy() for row in valid_rows]
            wrong_stream[0]["data_stream_id"] = "PCG64:wrong"
            corruptions["stream"] = wrong_stream
            wrong_features = [row.copy() for row in valid_rows]
            for row in wrong_features:
                if row["K"] == "10":
                    row["encoded_feature_count"] = str(
                        int(row["encoded_feature_count"]) + 1
                    )
            corruptions["cross_grid_features"] = wrong_features
            for field, bad_value in (
                ("method", "unknown"), ("computation_time_ns", "-1"),
                ("execution_position", "0"), ("timing_threads", "2"),
            ):
                corrupted = [dict(row) for row in valid_rows]
                corrupted[0][field] = bad_value
                corruptions["method_metadata_" + field] = corrupted
            inconsistent_time = [dict(row) for row in valid_rows]
            inconsistent_time[0]["computation_time_ns"] = str(int(inconsistent_time[0]["computation_time_ns"]) + 1)
            corruptions["cross_level_timing"] = inconsistent_time
            wrong_prior = [dict(row) for row in valid_rows]
            wrong_prior[0]["prior_mass"] = "5"
            corruptions["wrong_prior"] = wrong_prior
            wrong_dip_radius = [dict(row) for row in valid_rows]
            wrong_dip_radius[0]["roc_radius"] = "1.8"
            wrong_dip_radius[0]["roc_covered"] = "1"
            corruptions["dip_radius_bound"] = wrong_dip_radius
            inconsistent_auc = [dict(row) for row in valid_rows]
            inconsistent_auc[0]["external_grid_auc"] = "0.12345678901234568"
            inconsistent_auc[0]["auc_covered"] = str(int(is_external_auc_covered(
                float(inconsistent_auc[0]["external_grid_auc"]),
                (float(inconsistent_auc[0]["auc_interval_lower"]), float(inconsistent_auc[0]["auc_interval_upper"])))))
            corruptions["cross_level_external_auc"] = inconsistent_auc

            missing = [row.copy() for row in valid_rows[:-1]]
            corruptions["missing"] = missing
            duplicate = [row.copy() for row in valid_rows]
            duplicate.append(valid_rows[0].copy())
            corruptions["duplicate"] = duplicate

            for name, rows in corruptions.items():
                with self.subTest(corruption=name):
                    path = Path(temporary_directory) / f"{name}.csv"
                    _write_rows(path, rows)
                    with self.assertRaises(AssertionError):
                        runner.validate_raw_results_csv(
                            path,
                            conditions=FERTILITY_CONDITIONS,
                            n_replications=1,
                            n_samples=30,
                            master_seed=41, metadata=provenance,
                        )

            bad_header = Path(temporary_directory) / "header.csv"
            _write_rows(bad_header, valid_rows, runner.RAW_FIELDNAMES[:-1])
            with self.assertRaises((AssertionError, ValueError)):
                runner.validate_raw_results_csv(
                    bad_header,
                    conditions=FERTILITY_CONDITIONS,
                    n_replications=1,
                    n_samples=30,
                    master_seed=41, metadata=provenance,
                )

    def test_invalid_arguments_atomic_failure_and_official_main(self):
        condition = FERTILITY_CONDITIONS[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    temporary_path / "raw.txt",
                    conditions=(condition,),
                    show_progress=False,
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    temporary_path / "raw.csv",
                    conditions=(condition,),
                    n_replications=0,
                    show_progress=False,
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    temporary_path / "raw.csv",
                    conditions=(condition,),
                    n_samples=1,
                    show_progress=False,
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    temporary_path / "raw.csv",
                    conditions=(),
                    show_progress=False,
                )
            with self.assertRaises(ValueError):
                runner.run_experiment(
                    temporary_path / "raw.csv",
                    conditions=(condition, condition),
                    show_progress=False,
                )
            with self.assertRaises((TypeError, ValueError)):
                runner.run_experiment(
                    temporary_path / "raw.csv",
                    conditions=(condition,),
                    master_seed=True,
                    show_progress=False,
                )

            output = temporary_path / "existing.csv"
            output.write_bytes(b"existing result\n")
            with self.assertRaises(FileExistsError):
                runner.run_experiment(
                    output,
                    conditions=(condition,),
                    show_progress=False,
                )
            with self.assertRaises(TypeError):
                runner.run_experiment(
                    output,
                    conditions=(condition,),
                    overwrite=0,
                    show_progress=False,
                )

            loaded = _fertility_loaded(temporary_path / "fertility.arff")
            with (
                patch.object(runner, "load_real_dataset", return_value=loaded),
                patch.object(
                    runner,
                    "_evaluate_grid_condition",
                    side_effect=RuntimeError("injected failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    runner.run_experiment(
                        output,
                        conditions=(condition,),
                        n_replications=1,
                        n_samples=20,
                        overwrite=True,
                        show_progress=False,
                    )
            self.assertEqual(output.read_bytes(), b"existing result\n")
            self.assertEqual(
                list(temporary_path.glob(f".{output.name}.*.tmp")),
                [],
            )

            with patch.object(runner, "run_experiment", return_value=output) as run:
                returned = runner.main(
                    [
                        "--output",
                        str(output),
                        "--data-dir",
                        str(temporary_path / "data"),
                        "--master-seed",
                        "123",
                        "--overwrite",
                    ]
                )
            self.assertEqual(returned, output)
            call = run.call_args
            self.assertEqual(call.args[0], output)
            self.assertEqual(call.kwargs["n_replications"], 300)
            self.assertEqual(call.kwargs["n_samples"], 4000)
            self.assertEqual(call.kwargs["conditions"], runner.REAL_DATA_CONDITIONS)
            self.assertEqual(call.kwargs["master_seed"], 123)
            self.assertTrue(call.kwargs["overwrite"])

    def test_prior_and_shared_preparation_stay_outside_the_clock(self):
        import roc_utils.utilities_experiments as utilities
        events = []
        clock_calls = 0
        original_prior = runner.symmetric_dirichlet_priors
        original_histograms = runner.build_test_external_histograms
        original_builder = runner.build_posterior_predictive_band

        def prior(*args, **kwargs):
            events.append("prior")
            return original_prior(*args, **kwargs)

        def histogram(*args, **kwargs):
            events.append("histogram")
            self.assertFalse(args[-1].flags.writeable)
            return original_histograms(*args, **kwargs)

        def build(**kwargs):
            events.append("build")
            for name in ("alpha_plus", "alpha_minus", "d_plus", "d_minus"):
                self.assertFalse(kwargs[name].flags.writeable)
            return original_builder(**kwargs)

        def clock():
            nonlocal clock_calls
            events.append("clock start" if clock_calls % 2 == 0 else "clock end")
            clock_calls += 1
            return 25 * clock_calls

        def score(prepared):
            events.append("score")
            return _score_fixture()

        with tempfile.TemporaryDirectory() as directory:
            loaded = _fertility_loaded(Path(directory) / "fertility.arff")
            output = Path(directory) / "raw.csv"
            with (
                patch.object(runner, "load_real_dataset", return_value=loaded),
                patch.object(runner, "fit_logistic_regression_scores", side_effect=score),
                patch.object(runner, "symmetric_dirichlet_priors", side_effect=prior),
                patch.object(runner, "build_test_external_histograms", side_effect=histogram),
                patch.object(runner, "build_posterior_predictive_band", side_effect=build),
                patch.object(utilities.time, "perf_counter_ns", side_effect=clock),
            ):
                runner.run_experiment(output, conditions=FERTILITY_CONDITIONS,
                                      n_replications=2, n_samples=20, show_progress=False)
            rows = _read_rows(output)
        self.assertEqual(events[:3], ["prior"] * 3)
        self.assertEqual(events.count("prior"), 3)
        self.assertEqual(events.count("score"), 2)
        self.assertEqual(events.count("histogram"), 6)
        self.assertEqual(events.count("build"), 9)
        self.assertEqual(clock_calls, 12)
        for index, event in enumerate(events):
            if event == "clock start":
                self.assertEqual(events[index:index + 3], ["clock start", "build", "clock end"])
        self.assertEqual({row["computation_time_ns"] for row in rows}, {"25"})

    @unittest.skipUnless(ALL_REAL_DATA_CACHES_PRESENT, "all twelve pinned caches are required")
    def test_frozen_v2_dip_results_are_scientifically_unchanged(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "real_data_v2_baseline.json").read_text())
        self.assertEqual(fixture["source_protocol"], 2)
        self.assertEqual((fixture["B"], fixture["S"], fixture["master_seed"]), (2, 40, 20250731))
        self.assertEqual(len(fixture["rows"]), 144)
        self.assertEqual({int(row["condition_index"]) for row in fixture["rows"]}, set(range(36)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v3.csv"
            runner.run_experiment(output, n_replications=2, n_samples=40,
                                  master_seed=20250731, show_progress=False)
            rows = _read_rows(output)
        self.assertEqual(len(rows), 144)
        fields = fixture["fields"]
        expected = {_sort_key(row): row for row in fixture["rows"]}
        actual = {_sort_key(row): {field: row[field] for field in fields} for row in rows}
        self.assertEqual(set(actual), set(expected))
        for identity in expected:
            with self.subTest(identity=identity, software=fixture["software"]):
                self.assertEqual(actual[identity], expected[identity])

    def test_optimized_mode_cannot_disable_scientific_assertions(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-O",
                str(
                    PY_DIR
                    / "run_experiment_empirical_evaluation_on_real_world_datasets.py"
                ),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "This experiment relies on runtime assertions",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
