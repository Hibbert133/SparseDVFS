import copy
import unittest
import numpy as np
from rtx5090.eq2_candidate_check import candidates, prediction


class CandidateTests(unittest.TestCase):
    def fixture(self):
        x=np.array([[1,1,3],[2,3,1],[3,1,1],[2,2,3],[3,3,2]],float)
        c=np.array([2.,3.,4.])
        return dict(training=[dict(mean_basis=row.tolist(),energy_j=float(row@c),duration_s=1,energy_budget=.005)
                              for row in x],audit=dict(candidate_parameters=dict(zip(('alpha','k1','k2'),c))))

    def test_all_corners_and_no_mutation(self):
        data=self.fixture(); before=copy.deepcopy(data)
        center,corners=candidates(data)
        self.assertEqual(corners.shape,(32,3))
        r=prediction(center,corners,dict(integral_basis=[2,2,2],duration_s=2),18,.005)
        self.assertAlmostEqual(r['predicted_energy_j'],18)
        self.assertAlmostEqual(r['predicted_power_w'],9)
        self.assertTrue(r['observed_within_prediction_range'])
        self.assertEqual(data,before)

    def test_reject_modified_candidate(self):
        data=self.fixture(); data['audit']['candidate_parameters']['alpha']=5
        with self.assertRaisesRegex(ValueError,'frozen'): candidates(data)


if __name__=='__main__': unittest.main()
