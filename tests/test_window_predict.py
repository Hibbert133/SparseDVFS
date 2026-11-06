import unittest
from unittest.mock import patch
from rtx5090 import window_predict as wp


def rows(value):
    return [dict(summary={k:value for k in wp.TARGETS}, identity='id', evidence_sha256=str(i)) for i in range(3)]


class PredictionTests(unittest.TestCase):
    def model(self):
        with patch.object(wp, 'rows_at', return_value=(rows(10), {'train'}, {})) as read:
            model = wp.fit('fixture')
            read.assert_called_once_with('fixture', 'reference')
            return model

    def test_holdout_not_used_for_fit_and_errors_reported(self):
        model = self.model()
        with patch.object(wp, 'rows_at', return_value=(rows(20), {'test'}, {})):
            r = wp.validate('fixture', model)
        self.assertEqual(r['status'], 'not_passed')
        self.assertEqual(r['scores']['mean_request_s']['wape'], .5)
        self.assertEqual(model['predictions']['mean_request_s'], 10)
        self.assertFalse(r['equation_1_validated'])

    def test_overlap_rejected(self):
        model = self.model()
        with patch.object(wp, 'rows_at', return_value=(rows(10), {'train'}, {})):
            with self.assertRaises(ValueError): wp.validate('fixture', model)

    def test_exact_prediction_passes_only_scoped_model(self):
        model = self.model()
        with patch.object(wp, 'rows_at', return_value=(rows(10), {'test'}, {})):
            self.assertEqual(wp.validate('fixture', model)['status'], 'passed')
