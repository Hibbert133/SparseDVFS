import copy
import unittest
from rtx5090.linear_latency_model import fit, validate
from rtx5090.observe_core import GPU_UUIDS, digest


def fixture():
    ranks = []
    for rank in (0, 1):
        rows = []
        for phase in ('prefill', 'decode'):
            for module in ('gate_up_proj', 'down_proj'):
                rows.append(dict(phase=phase, module=module, parent='layer.0.'+module,
                    M=512 if phase == 'prefill' else 1, N=4, K=8,
                    input_metadata=[{'dtype': 'bf16'}], output_metadata={'dtype': 'bf16'},
                    collection_protocol='kernel_verified_batch_counts_v1', kernel_names_verified=True,
                    kernel_comparison={'source_sequence': [{'name': 'gemm', 'grid': [1], 'block': [32]}]},
                    batches=[dict(repeats=n, mean_call_s=.001) for n in (32, 128) for _ in range(3)]))
        ranks.append(dict(rank=rank, uuid=GPU_UUIDS[rank], rows=rows))
    return dict(status='complete', kernel_mapping_passed=True, kernel_launch_geometry_passed=True,
                identity_fingerprint='identity', matrix_replay=ranks,
                cache_isolation={'miss_verified': True},
                reply=dict(prompt_hash='train', request_id='train', usage=dict(prompt_tokens=512, completion_tokens=64)))


class FrozenValidationTests(unittest.TestCase):
    def setUp(self):
        self.training = fixture()
        self.model = fit(self.training)[0]
        self.new = copy.deepcopy(self.training)
        self.new['reply'].update(prompt_hash='fresh', request_id='fresh')

    def test_all_batches_no_refit(self):
        before = digest(self.model)
        report = validate(self.model, self.training, self.new)
        self.assertTrue(report['all_cases_within_project_error_limits'])
        self.assertTrue(all(len(r['check_values_s']) == 6 for r in report['rows']))
        self.assertEqual(before, digest(self.model))

    def test_error_not_dropped(self):
        self.new['matrix_replay'][0]['rows'][0]['batches'][0]['mean_call_s'] = .1
        self.assertFalse(validate(self.model, self.training, self.new)['all_cases_within_project_error_limits'])

    def test_reject_invalid_evidence(self):
        for kind in ('prompt', 'identity', 'geometry', 'missing', 'uuid', 'model'):
            with self.subTest(kind=kind):
                new, model = copy.deepcopy(self.new), copy.deepcopy(self.model)
                row = new['matrix_replay'][0]['rows'][0]
                if kind == 'prompt': new['reply']['prompt_hash'] = 'train'
                if kind == 'identity': new['identity_fingerprint'] = 'different'
                if kind == 'geometry': row['kernel_comparison']['source_sequence'][0]['grid'] = [2]
                if kind == 'missing': row['batches'].pop()
                if kind == 'uuid': new['matrix_replay'][0]['uuid'] = 'wrong'
                if kind == 'model': model['operators']['0:prefill/gate_up_proj']['latency_s'] = .002
                with self.assertRaises(ValueError): validate(model, self.training, new)


if __name__ == '__main__': unittest.main()
