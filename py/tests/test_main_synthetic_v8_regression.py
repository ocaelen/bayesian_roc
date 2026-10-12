"""Frozen scientific outputs from before the DiP-only protocol change."""

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[1]
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import run_experiment_main_evaluation_on_synthetic_data as runner
from roc_utils.posterior_predictive_roc_sampler import fixed_uniform_grid


class MainSyntheticV8RegressionTests(unittest.TestCase):
    def test_all_nine_settings_preserve_frozen_v7_scientific_measurements(self):
        """Check all 15 conditions, including m=20, independently of elapsed time."""
        reference = json.loads((Path(__file__).parent / "fixtures" /
                                "main_synthetic_v7_dip_baselines.json").read_text())
        entries = reference["replications"]
        self.assertEqual(len(entries), 30)
        self.assertEqual({(entry["condition"], entry["replication"]) for entry in entries},
                         {(index, replication) for index in range(15) for replication in (0, 1)})
        for entry in entries:
            with self.subTest(condition=entry["condition"], replication=entry["replication"]):
                condition = runner.CLASS_COUNT_CONDITIONS[entry["condition"]]
                results, data_key, history = runner._evaluate_one_replication(
                    condition, entry["replication"], gamma=fixed_uniform_grid(10),
                    n_samples=reference["n_samples"], master_seed=reference["master_seed"],
                )
                actual = dict(condition=condition.index, replication=entry["replication"],
                    data_key=data_key, history_plus=history.d_plus.tolist(),
                    history_minus=history.d_minus.tolist(), history_auc=history.grid_auc,
                    measurements={setting: asdict(values[0]) for setting, values in results.items()})
                # JSON normalizes tuple/list representations without rounding floats.
                self.assertEqual(json.loads(json.dumps(actual)), entry)


if __name__ == "__main__":
    unittest.main()
