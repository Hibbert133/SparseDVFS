import unittest
import numpy as np
from rtx5090.eq2_identifiability import audit_matrix, features


class Eq2AuditTests(unittest.TestCase):
    def test_constant_voltage_frequency_not_identifiable(self):
        a=audit_matrix([[1e9,40,1],[1e9,45,1],[1e9,50,1]],[100,110,120],[.005]*3)
        self.assertIsNone(a['parameters'])
        self.assertIn('rank_deficient',a['blockers'])

    def test_units_do_not_change_condition(self):
        x=np.array([[1,1,3],[2,3,1],[3,1,1]],float)
        y=x @ np.array([2,3,4])
        a=audit_matrix(x,y,[.00001]*3)
        b=audit_matrix(x*np.array([1e9,1,1]),y,[.00001]*3)
        self.assertIsNotNone(a['parameters'])
        self.assertAlmostEqual(a['normalized_condition_number'],b['normalized_condition_number'])
        self.assertAlmostEqual(b['parameters']['alpha']*1e9,2)

    def test_budget_can_prevent_fit(self):
        x=np.array([[1,1,3],[2,3,1],[3,1,1]],float)
        a=audit_matrix(x,x @ np.array([2,3,4]),[.2]*3)
        self.assertIsNone(a['parameters'])
        self.assertIn('coefficient_energy_budget_sensitivity_exceeds_20_percent',a['blockers'])

    def test_time_integrated_features(self):
        frames=[]
        for i in range(12):
            t=i*100000000
            frames.append([dict(uuid='u',timestamp_ns=t,temperature_c=40,clock_mhz=2000,
                field_query_times_ns={k:dict(start=t,end=t) for k in ('temperature_c','clock_mhz')},
                voltage=dict(status=0,reported_core_voltage_v=1,binding={'uuid':'u'},query_start_ns=t,query_end_ns=t))])
        data=dict(frames=frames,host_window_start_ns=50000000,host_window_end_ns=1050000000)
        f=features(data,'u')
        np.testing.assert_allclose(f['mean_basis'],[2e9,40,1])
        np.testing.assert_allclose(f['integral_basis'],[2e9,40,1])
        data['host_window_end_ns']=2000000000
        with self.assertRaisesRegex(ValueError,'bracketed'): features(data,'u')


if __name__=='__main__': unittest.main()
