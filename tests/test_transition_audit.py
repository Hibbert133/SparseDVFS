import copy
import unittest
from rtx5090.transition_audit import analyze
from rtx5090.observe_core import GPU_UUIDS


def fixture():
    frames=[]
    for i in range(4):
        start=(3+i*30)*1000000
        frames.append([dict(uuid=u,begin_ns=start,end_ns=start+10000000,
            raw=dict(timestamp_ns=9000000000+i*30000000,clock_mhz=2200 if i==0 else 2450,
                     utilization_pct=80,errors={},field_query_times_ns={'clock_mhz':dict(
                         start=9000000000+i*30000000,end=9001000000+i*30000000)})) for u in GPU_UUIDS])
    return dict(loaded_samples=frames,begin_ns=0,api_return_ns=2000000,confirmed_ns=103000000,
                confirmed=True,from_requested_mhz=2250,to_requested_mhz=2527)


class AuditTests(unittest.TestCase):
    def test_exact_nonoverlapping_decomposition(self):
        r,flat=analyze(fixture(),{u:[2440,2490] for u in GPU_UUIDS})
        self.assertEqual(r['decomposition'],dict(api_ms=2,initial_dispatch_ms=1,polling_groups_ms=40,
                                                between_group_gaps_ms=60,total_ms=103))
        self.assertEqual(r['first_pair_loaded_in_band_ms'],43)
        self.assertEqual(r['after_first_pair_until_confirmation_ms'],60)
        self.assertEqual(r['by_uuid'][0]['max_remote_telemetry_ms'],1)
        self.assertEqual(len(flat),8)

    def test_utilization_gate_separated_from_clock(self):
        data=fixture()
        for s in data['loaded_samples'][0]:
            s['raw'].update(clock_mhz=2450,utilization_pct=0)
        r,_=analyze(data,{u:[2440,2490] for u in GPU_UUIDS})
        self.assertEqual(r['first_pair_clock_in_band_ms'],13)
        self.assertEqual(r['first_pair_loaded_in_band_ms'],43)

    def test_reject_bad_confirmation(self):
        data=fixture();data['confirmed_ns']+=1
        with self.assertRaises(ValueError): analyze(data,{u:[2440,2490] for u in GPU_UUIDS})


if __name__=='__main__':unittest.main()
