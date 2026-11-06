"""CPU contract checks only; no CUDA kernel match claimed."""
import copy
import unittest
from rtx5090.matrix_probe import candidates, compare_kernels
from rtx5090.observe_core import digest


def fixture(rank=0):
    def tensor(m, n):
        return dict(tensor=True, shape=[m, n], stride=[n, 1], dtype='torch.bfloat16',
                    device=f'cuda:{rank}', layout='torch.strided')
    rows = []
    for phase, m in [('prefill', 512), ('decode', 1)]:
        for module, n, k in [('gate_up_proj', 17408, 5120), ('down_proj', 5120, 8704)]:
            sig = dict(rank=rank, phase=phase, overload='aten.linear.default',
                inputs=dict(args=[tensor(m, k), tensor(n, k)], kwargs={}),
                outputs=tensor(m, n), kernel_names=['gemm', 'reduce'] if module == 'down_proj' else ['gemm'])
            rows.append(dict(signature=sig, signature_id=digest(sig), opaque=False,
                             parents=['language_model.model.layers.0.mlp.'+module]))
    return rows


class ProbeTests(unittest.TestCase):
    def test_both_ranks_four_cases(self):
        for rank in (0, 1):
            self.assertEqual(len(candidates(fixture(rank), rank)), 4)

    def test_wrong_rank_hash_and_missing_case(self):
        with self.assertRaises(ValueError): candidates(fixture(1), 0)
        rows = fixture(); rows[0]['signature_id'] = 'wrong'
        with self.assertRaises(ValueError): candidates(rows, 0)
        with self.assertRaises(ValueError): candidates(fixture()[:-1], 0)

    def test_shape_stride_bias_and_dtype_not_silently_changed(self):
        for key, value in [('stride', [1, 512]), ('shape', [512, 5121]), ('dtype', 'torch.float32')]:
            rows = fixture()
            rows[0]['signature']['inputs']['args'][0][key] = value
            rows[0]['signature_id'] = digest(rows[0]['signature'])
            with self.assertRaises(ValueError): candidates(rows, 0)
        rows = fixture()
        rows[0]['signature']['inputs']['kwargs']['bias'] = None
        rows[0]['signature_id'] = digest(rows[0]['signature'])
        with self.assertRaises(ValueError): candidates(rows, 0)

    def test_multiple_variants_rejected(self):
        rows = fixture(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, 'Multiple'): candidates(rows, 0)

    def test_multikernel_path_requires_reduction(self):
        events = [dict(cat='kernel', ph='X', name='gemm')]
        self.assertFalse(compare_kernels(events, ['gemm', 'reduce'])['kernel_name_set_match'])
        events.append(dict(cat='kernel', ph='X', name='reduce'))
        r = compare_kernels(events, ['gemm', 'reduce'])
        self.assertTrue(r['kernel_name_set_match'])
        self.assertFalse(r['launch_order_and_multiplicity_verified'])
        self.assertFalse(compare_kernels([], [])['kernel_name_set_match'])
        events.append(dict(cat='kernel', ph='X', name='unexpected'))
        self.assertFalse(compare_kernels(events, ['gemm', 'reduce'])['kernel_name_set_match'])
