import unittest
from collections import Counter
from rtx5090.resource_frequencies import SWEEP_RATES,sweep_jobs,merge_training
from rtx5090.resource_sweep_report import report


class SweepTests(unittest.TestCase):
    def test_two_passes_cover_all_conditions(self):
        jobs=sweep_jobs(list(SWEEP_RATES),2,100)
        self.assertEqual(len(jobs),72)
        self.assertEqual(jobs[0]['requested_mhz'],2250)
        self.assertEqual([j['requested_mhz'] for j in jobs[:36:3]],
                         list(reversed([j['requested_mhz'] for j in jobs[36::3]])))
        counts=Counter((j['requested_mhz'],j['case']['family'],j['case']['size']) for j in jobs)
        self.assertEqual(set(counts.values()),{2})
        self.assertEqual(len({j['case']['seed'] for j in jobs}),72)

    def test_invalid_grid_and_repeat_rejected(self):
        for rates,repeats in (([2250],2),(list(SWEEP_RATES),0),(list(SWEEP_RATES),4)):
            with self.assertRaises(ValueError):sweep_jobs(rates,repeats,1)

    def test_failed_sweep_is_reported_not_relabelled_complete(self):
        data=dict(collection_role='sweep',protocol=sweep_jobs(list(SWEEP_RATES),2,1),
            rows=[],status='failed',error='energy budget',restoration={'verified':True})
        r=report(data)
        self.assertEqual(r['collection_status'],'failed')
        self.assertEqual(len(r['missing_rank_resource_settings']),72)
        self.assertFalse(r['parameters_fitted'])

    def test_board_window_power_not_per_call_energy(self):
        row=dict(rank=0,family='copy',shape=[1048576],requested_mhz=2250,frequency_hz=2200e6,
            voltage_v=.9,feature_ranges={'temperature_c':[40,45]},latency_s=2e-5,
            energy_j=6000,host_window_s=60,energy_relative_budget=.01,memory_mhz=13801)
        data=dict(collection_role='sweep',protocol=sweep_jobs(list(SWEEP_RATES),1,1),rows=[row],
                  status='failed',restoration={'verified':True})
        r=report(data)
        self.assertEqual(r['rows'][0]['board_power_mean_w'],100)
        self.assertNotIn('operator_energy_j',r['rows'][0])

if __name__=='__main__':unittest.main()
