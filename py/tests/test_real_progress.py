"""Progress must describe the recorded experiment without changing its draws."""

import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_experiment_empirical_evaluation_on_real_world_datasets as runner
from test_run_experiment_real_data import FERTILITY_CONDITIONS, _fertility_loaded, _score_fixture


class RealProgressTests(unittest.TestCase):
    def test_progress_matches_written_rows_at_first_periodic_and_final_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = _fertility_loaded(root / "source.arff")
            output = root / "run.csv"
            capture = io.StringIO()
            with patch.object(runner, "load_real_dataset", return_value=loaded), patch.object(
                runner, "fit_logistic_regression_scores", return_value=_score_fixture(),
            ), contextlib.redirect_stdout(capture):
                runner.run_experiment(output, conditions=FERTILITY_CONDITIONS,
                                      n_replications=26, n_samples=40, master_seed=71)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            lines = capture.getvalue().splitlines()
            self.assertIn("prior=symmetric, mass_per_class=1.0", lines[0])
            self.assertIn("expected_rows=156", lines[0])
            self.assertIn("solver=lbfgs", lines[0])
            self.assertIn("main K=10 (52 rows); supplementary K=(5, 20) (104 rows)", lines[1])
            self.assertIn("share each split, fitted classifier and scores", lines[1])
            self.assertIn("complete_cases=100", capture.getvalue())
            checkpoints = [(i, line) for i, line in enumerate(lines)
                           if line.startswith("Fertility: completed")]
            self.assertEqual(len(checkpoints), 3)
            for completed, (i, line) in zip((1, 25, 26), checkpoints):
                self.assertIn(f"{completed}/26", line)
                expected = []
                for condition in FERTILITY_CONDITIONS:
                    for level in runner.NOMINAL_LEVELS:
                        selected = [row for row in rows if row["condition_id"] == condition.condition_id
                                    and float(row["level"]) == level and int(row["replication"]) < completed]
                        self.assertEqual(len(selected), completed)
                        mean = lambda field: sum(float(row[field]) for row in selected) / completed
                        expected.append(
                            f"  {condition.condition_id} level={level:.2f}: "
                            f"ROC coverage={100 * mean('roc_covered'):.2f}%, "
                            f"AUC coverage={100 * mean('auc_covered'):.2f}%, "
                            f"mean radius={mean('roc_radius'):.6f}, "
                            f"mean AUC width={mean('auc_width'):.6f}, "
                            f"mean construction={mean('computation_time_ns') / 1_000_000:.3f} ms"
                        )
                self.assertEqual(lines[i + 1:i + 7], expected)

    def test_disabling_progress_leaves_scientific_rows_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = _fertility_loaded(root / "source.arff")
            results = []
            for progress in (False, True):
                output = root / f"run_{progress}.csv"
                capture = io.StringIO()
                with patch.object(runner, "load_real_dataset", return_value=loaded), patch.object(
                    runner, "fit_logistic_regression_scores", return_value=_score_fixture(),
                ), contextlib.redirect_stdout(capture):
                    runner.run_experiment(output, conditions=FERTILITY_CONDITIONS,
                                          n_replications=2, n_samples=40, master_seed=71,
                                          show_progress=progress)
                self.assertEqual(bool(capture.getvalue()), progress)
                with output.open() as handle:
                    results.append([{key: value for key, value in row.items()
                                     if key != "computation_time_ns"} for row in csv.DictReader(handle)])
            self.assertEqual(results[0], results[1])

    def test_subset_progress_does_not_claim_missing_main_grid_results(self):
        for grids, expected in (((10,), "main K=10 (4 rows); supplementary K=(5, 20) (0 rows)"),
                                ((5, 20), "main K=10 (0 rows); supplementary K=(5, 20) (8 rows)")):
            with self.subTest(grids=grids), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                loaded = _fertility_loaded(root / "source.arff")
                capture = io.StringIO()
                conditions = tuple(c for c in FERTILITY_CONDITIONS if c.n_bins in grids)
                with patch.object(runner, "load_real_dataset", return_value=loaded), patch.object(
                    runner, "fit_logistic_regression_scores", return_value=_score_fixture(),
                ), contextlib.redirect_stdout(capture):
                    runner.run_experiment(root / "run.csv", conditions=conditions,
                                          n_replications=2, n_samples=40)
                self.assertIn(expected, capture.getvalue())


if __name__ == "__main__":
    unittest.main()
