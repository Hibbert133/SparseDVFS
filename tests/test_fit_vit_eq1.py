import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import fit_vit_eq1 as fit


class Eq1FitTests(unittest.TestCase):
    def test_two_branches(self):
        f = np.arange(100., 3100., 10.)
        y = 0.07 + np.maximum(200 / f, 0.2)
        curve, _ = fit.fit_curves(f, y)
        np.testing.assert_allclose([curve.intercept, curve.slope, curve.knee], [0.07, 200, 1000])
        self.assertEqual(fit.assess_knee(curve, f, y)[0], "stable_on_training_splits")

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
                self.assertEqual(json.loads(json.dumps(summary)), summary)
                self.assertEqual((rows[0]["train_count"], summary["held_out_cells"]), (311, 78))
                return rows[0]
            first, second = run(0), run(1)
            for key in ("intercept_ms", "C_ms_MHz", "M_ms", "knee_MHz", "knee_support"):
                self.assertEqual(first[key], second[key])
            self.assertGreater(second["eq1_WAPE_pct"], first["eq1_WAPE_pct"])
