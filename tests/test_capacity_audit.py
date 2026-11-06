import unittest
from rtx5090.capacity_audit import audit
from rtx5090.observe_core import GPU_UUIDS


class AuditTests(unittest.TestCase):
    def fixture(self):
        snapshot = dict(uuid=GPU_UUIDS[0],sm_mhz=2800,memory_mhz=13801,temperature_c=50)
        calls = [dict(phase=p,module='down_proj',parent='model.layers.0.mlp.down_proj',
            M=512 if p=='prefill' else 1,N=5120,K=8704,kernel_names_verified=False,
            cache_state_matched=False,timing_scope='repeated_calls') for p in ('prefill','decode')]
        obs = dict(identity_fingerprint='id',status='complete',coverage={'passed':True},
            matrix_replay=[dict(rank=0,uuid=GPU_UUIDS[0],rows=calls)])
        coll = dict(identity_fingerprint='id',ranks=[dict(rank=0,uuid=GPU_UUIDS[0],
            status='conditional_primitive_model_collected',rows=[dict(name='copy_train',
            seconds=[1,2,3],before=snapshot,after=snapshot)],
            MLP_transfer_diagnostic={'rows':[dict(name=p+'/down_proj',observed_s=1,
                predicted_s=.8,relative_error=-.2) for p in ('prefill','decode')]},
            primitive_holdout={'scores':dict(wape=.17,p90_relative_error=.19)})])
        return coll,obs

    def test_does_not_certify_or_invent_kernel(self):
        c,o = self.fixture(); r = audit(c,o)
        self.assertFalse(r['equation_1_parameters_validated'])
        self.assertFalse(r['ranks'][0]['primitive_numeric_gate_passed'])
        self.assertIsNone(r['ranks'][0]['linear_calls'][0]['cuda_kernel_names'])
        self.assertAlmostEqual(r['ranks'][0]['phases']['decode']['wape'],.2)
        self.assertEqual(r['ranks'][0]['primitives'][0]['spread_percent'],100)

    def test_identity_change_rejected(self):
        c,o = self.fixture(); o['identity_fingerprint']='other'
        with self.assertRaises(ValueError): audit(c,o)

    def test_snapshot_uuid_rejected(self):
        c,o = self.fixture(); c['ranks'][0]['rows'][0]['before']['uuid']='wrong'
        with self.assertRaises(ValueError): audit(c,o)

    def test_blocked_preserved(self):
        c,o = self.fixture(); c['ranks'][0]['status']='blocked_insufficient_free_memory'
        c['ranks'][0]['before_context']={'free_mib':384}
        r=audit(c,o)['ranks'][0]
        self.assertEqual(r['memory_evidence']['before_context']['free_mib'],384)
        self.assertEqual(r['linear_calls'],[])


if __name__ == '__main__': unittest.main()
