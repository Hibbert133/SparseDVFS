import unittest
from rtx5090.module_cost_audit import audit
from rtx5090.eager_core import UNITS
from rtx5090.observe_core import GPU_UUIDS
from rtx5090.partition_dryrun import partition,demo


class ModuleAuditTests(unittest.TestCase):
    def test_energy_missing_does_not_report_latency_missing(self):
        d=demo()
        for rows in d['candidates'].values():
            for row in rows.values(): row['pair_energy_j']=None
        blocker=partition(d)['blockers'][0]
        self.assertEqual(blocker['missing_latency_count'],0)
        self.assertEqual(blocker['missing_energy_count'],8)

    def test_weighted_sum_not_median_ratio(self):
        windows=[]
        for i in range(3):
            for rate in (2250,2527):
                ops={}
                for r in (0,1):
                    for c in ['prefill']+['decode.%d'%j for j in range(63)]:
                        for u in UNITS:
                            t=(100 if rate==2527 else 120) if u=='layer.0' else 1
                            ops['%d:%s/%s'%(r,c,u)]=dict(uuid=GPU_UUIDS[r],median_s=t)
                windows.append(dict(repeat=i,requested_mhz=rate,prompt_hash=str(i)+str(rate),operators=ops))
        data=dict(status='two_setting_modules_timing_complete',identity_fingerprint='test',
                  restoration=dict(verified=True),windows=windows)
        result=audit(data)
        row=result['phase_sums'][0]
        self.assertAlmostEqual(row['ratio'],(120+len(UNITS)-1)/(100+len(UNITS)-1))
        self.assertGreater(row['ratio'],1.1)
        self.assertFalse(result['deployable'])
        del windows[0]['operators']['0:prefill/embed']
        with self.assertRaises(ValueError): audit(data)


if __name__=='__main__': unittest.main()
