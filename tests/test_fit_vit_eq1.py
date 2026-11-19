import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import fit_vit_eq1 as fit
import fit_vit_eq1_fast as fast


class Eq1FitTests(unittest.TestCase):
    def test_relative_line_matches_scaled_least_squares(self):
        x = np.array([0.1, 0.2, 0.4, 0.75])
        y = np.array([0.3, 0.6, 1.0, 1.2])
        weights = 1 / y**2
        expected = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]) / y[:, None],
                                   np.ones_like(y), rcond=None)[0]
        np.testing.assert_allclose(fit.nonnegative_line(x, y, weights/weights.sum()), expected)

    def test_scale_includes_one_and_maximizes_interval_overlap(self):
        y = np.ones(5)
        self.assertEqual(fast.calibrated_scale(y, y), 1.)
        pred = np.array([2., 2., 2., 1., 1.])
        scale = fast.calibrated_scale(pred, y)
        self.assertEqual(fast.hit_mask(scale*pred, y).sum(), 3)

    def test_two_branches(self):
        f = np.arange(100., 3100., 10.)
        y = 0.07 + np.maximum(200 / f, 0.2)
        curve, _ = fit.fit_curves(f, y)
        np.testing.assert_allclose([curve.intercept, curve.slope, curve.knee], [0.07, 200, 1000])
        self.assertEqual(fit.assess_knee(curve, f, y)[0], "stable_on_training_splits")

    def test_grid_can_recover_missing_frequency_dependence(self):
        f = np.linspace(200., 3200., 121)
        y = 0.31 + np.maximum(150/f, .15)
        constant = fit.Curve(float(y.mean()), 0.)
        curve = fast.search_shape(f, y, constant)
        self.assertGreater(fast.hit_mask(curve.predict(f), y).mean(), .9)
        self.assertLess(fast.hit_mask(constant.predict(f), y).mean(), .9)

    def test_shape_selection_keeps_baseline_on_tie(self):
        f = np.arange(100., 500., 10.)
        y = np.ones_like(f)
        curve = fit.Curve(1., 0.)
        chosen, details = fast.select_shape(f, y, curve)
        self.assertEqual(curve, chosen)
        self.assertFalse(details["shape_changed"])

    def test_single_branch_has_no_identified_knee(self):
        f = np.arange(100., 3000., 10.)
        for y in (np.full_like(f, 0.2), 0.1 + 250 / f):
            curve, _ = fit.fit_curves(f, y)
            self.assertIsNone(curve.knee)
            np.testing.assert_allclose(curve.predict(f), y)

    def test_held_out_labels_do_not_change_curve(self):
        with TemporaryDirectory() as directory:
            p = Path(directory) / "run.csv"
            def run(offset):
                with p.open("w", newline="") as out:
                    writer = csv.writer(out)
                    writer.writerow(["gpu_uuid", "layer_name", "requested_gpu_freq", "latency_ms"])
                    for i, f in enumerate(range(180, 3292, 8)):
                        y = 0.1 + max(150 / f, 0.1) + (offset if i % 5 == 2 else 0)
                        writer.writerow(["GPU-A", "mlp", f, y])
                rows, _, summary = fit.analyze(p)
                fast_rows, _, report = fast.compare(p)
                shape_rows, _, shape_report = fast.compare(p, refine_shape=True)
                self.assertEqual(shape_report["old_hits"], report["new_hits"])
                self.assertEqual(shape_report["old_WAPE_pct"], report["new_WAPE_pct"])
                self.assertEqual(report["held_out"], 78)
                self.assertEqual(report["new_hits"] - report["old_hits"], report["gained"] - report["lost"])
                self.assertEqual(json.loads(json.dumps(summary)), summary)
                self.assertEqual((rows[0]["train_count"], summary["held_out_cells"]), (311, 78))
                return rows[0], fast_rows[0], shape_rows[0]
            (first, first_fast, first_shape), (second, second_fast, second_shape) = run(0), run(1)
            self.assertEqual(first_shape, second_shape)
            self.assertEqual(first_fast, second_fast)
            for key in ("intercept_ms", "C_ms_MHz", "M_ms", "knee_MHz", "knee_support"):
                self.assertEqual(first[key], second[key])
            self.assertGreater(second["eq1_WAPE_pct"], first["eq1_WAPE_pct"])
