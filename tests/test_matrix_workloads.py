import unittest
from rtx5090.matrix_workloads import account


class AccountingTests(unittest.TestCase):
    def test_known_linear_and_no_calibration_claim(self):
        j = dict(m=2, n=3, k=4, phase='prefill', module='example',
                 row=dict(signature_id='fixture', signature=dict(kernel_names=['fixture'])))
        r = account(j, 0)
        self.assertEqual(r['work_flop'], 48)
        self.assertEqual(r['logical_data_bytes'], 52)
        self.assertFalse(r['calibration_ready'])
        self.assertIsNone(r['latency_prediction_s'])
        self.assertIsNone(r['internal_workspace_and_reduction_bytes'])
