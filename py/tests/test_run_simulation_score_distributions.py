"""Check the three density panels without overwriting manuscript figures."""

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import beta

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

import run_simulation_score_distributions as figures  # noqa: E402
from roc_utils.simulation_datasets import (  # noqa: E402
    SIMULATION_SCENARIOS,
    simulation_beta_parameters,
)


class SimulationScoreDistributionFigureTests(unittest.TestCase):
    def test_figure_registry_and_density_limits_match_current_scenarios(self):
        self.assertEqual(tuple(case[0] for case in figures.SCENARIOS), SIMULATION_SCENARIOS)
        self.assertEqual(len({case[3] for case in figures.SCENARIOS}), 3)
        scores = np.linspace(0.0, 1.0, 1001)
        for scenario, positive, negative, filename in figures.SCENARIOS:
            with self.subTest(scenario=scenario):
                self.assertEqual((positive, negative), simulation_beta_parameters(scenario))
                self.assertEqual(filename, f"simulation_{scenario.lower()}_score_distributions.pdf")
                densities = []
                for a, b in (positive, negative):
                    density = beta.pdf(scores, a, b)
                    self.assertTrue(np.all(np.isfinite(density)))
                    self.assertTrue(np.all(density >= 0.0))
                    self.assertLess(density.max(), 3.0)
                    # For a,b>1 the exact mode also bounds peaks between
                    # the points used by the plotting script.
                    self.assertGreater(min(a, b), 1.0)
                    self.assertLess(beta.pdf((a - 1) / (a + b - 2), a, b), 3.0)
                    densities.append(density)
                np.testing.assert_allclose(densities[0], densities[1][::-1], rtol=0.0, atol=1e-12)

    def test_main_writes_three_matching_pdf_panels_only_in_a_temporary_directory(self):
        captured_axes = []
        original_subplots = figures.plt.subplots

        def capture_subplots(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured_axes.append(axes)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch.object(figures, "__file__", str(directory / "run_simulation_score_distributions.py")),
                patch.object(figures.plt, "subplots", side_effect=capture_subplots),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                figures.main()

            paths = sorted((directory / "figures").glob("*.pdf"))
            self.assertEqual({path.name for path in paths}, {case[3] for case in figures.SCENARIOS})
            page_boxes = []
            for path in paths:
                contents = path.read_bytes()
                self.assertTrue(contents.startswith(b"%PDF-"))
                self.assertIn(b"%%EOF", contents[-1024:])
                self.assertGreater(len(contents), 1000)
                # Matplotlib writes this page dictionary uncompressed;
                # reading its MediaBox needs no additional PDF dependency.
                match = re.search(rb"/MediaBox\s*\[([^\]]+)\]", contents)
                self.assertIsNotNone(match)
                page_box = tuple(float(value) for value in match.group(1).split())
                self.assertEqual(len(page_box), 4)
                self.assertGreater(page_box[2] - page_box[0], 0.0)
                self.assertGreater(page_box[3] - page_box[1], 0.0)
                page_boxes.append(page_box)
            self.assertEqual(len(set(page_boxes)), 1)
            np.testing.assert_allclose(page_boxes[0], (0.0, 0.0, 2.15 * 72, 2.55 * 72))

        self.assertEqual(len(captured_axes), 3)
        for axes, (_, positive, negative, _) in zip(captured_axes, figures.SCENARIOS, strict=True):
            self.assertEqual(axes.get_xlim(), (0.0, 1.0))
            self.assertEqual(axes.get_ylim(), (0.0, 3.0))
            self.assertEqual(len(axes.lines), 2)
            for line, shape in zip(axes.lines, (negative, positive), strict=True):
                np.testing.assert_array_equal(line.get_ydata(), beta.pdf(line.get_xdata(), *shape))


if __name__ == "__main__":
    unittest.main()
