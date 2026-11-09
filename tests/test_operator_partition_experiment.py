import unittest
from rtx5090.operator_partition_experiment import experiment
from tests.test_operator_pair_contract import fixture

class ExperimentTests(unittest.TestCase):
    def test_partition_coverage_and_assumptions(self):
        reports=experiment(fixture())
        self.assertEqual(len(reports),16)
        for r in reports:
            self.assertFalse(r['deployable']);self.assertFalse(r['parameters_validated'])
            self.assertEqual(sum(len(b['nodes']) for b in r['blocks']),2)
            self.assertEqual(len(r['blocks']),1)
    def test_zero_is_not_replaced_with_epsilon(self):
        data=fixture()
        for w in data['windows']:
            for row in w['operators'].values():row['median_s']=0
        with self.assertRaises(ValueError):experiment(data)
