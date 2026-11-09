import unittest
from rtx5090.operator_partition_inventory import inventory
from rtx5090.observe_core import GPU_UUIDS


class InventoryTests(unittest.TestCase):
    def source(self):
        context = dict(phase='prefill', prompt_tokens=512, admitted_computed_tokens=0,
                       resumed=False, preempted=False, speculative_tokens=0,
                       num_computed_tokens=0, num_scheduled_tokens=512)
        op = dict(rank=0, phase='prefill', context=context, parent='model.layers.0.attn',
                  overload='custom.fused_attention.default', opaque=True, inputs={}, outputs={},
                  gpu_activities=[dict(name='fused_kernel', category='kernel')])
        return dict(status='complete', rank=0, identity=dict(gpu_uuid=GPU_UUIDS[0]),
                    operators=[op, dict(op)])

    def test_preserve_fusion_occurrences_and_unknown_costs(self):
        result = inventory(self.source(), 0, ['prefill'])
        self.assertEqual(len(result['nodes']), 2)
        self.assertEqual(len(result['signatures']), 1)
        self.assertNotEqual(result['nodes'][0]['id'], result['nodes'][1]['id'])
        self.assertEqual(result['nodes'][0]['kernels'], ['fused_kernel'])
        self.assertIsNone(result['nodes'][0]['energy_by_frequency'])
        self.assertIsNone(result['dependencies'])
        self.assertFalse(result['deployable'])

    def test_missing_context_and_wrong_uuid_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.source(), 0, ['decode.0'])
        source = self.source()
        source['identity']['gpu_uuid'] = GPU_UUIDS[1]
        with self.assertRaises(ValueError):
            inventory(source, 0, ['prefill'])

    def test_missing_kernel_not_invented(self):
        source = self.source()
        source['operators'][0]['gpu_activities'] = []
        self.assertEqual(inventory(source, 0, ['prefill'])['nodes'][0]['kernels'], [])
