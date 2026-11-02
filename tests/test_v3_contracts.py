import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sparsedvfs_vllm.contracts import (PROTOCOL, UNITS, WORKER, PhaseTracker, engine_args,
                                      server_args, validate_engine, validate_plan)
from sparsedvfs_vllm.http_client import completion
from sparsedvfs_vllm.host import replace_paths
from sparsedvfs_vllm.report import controller_audit, coverage, stats
from sparsedvfs_vllm.state import restored


def scheduler(*, rid='r', prompt=None, computed=0, count=1, finished=()):
    sp = NS(prompt_logprobs=None, logprobs=None)
    new = [NS(req_id=rid, prompt_token_ids=list(range(prompt)), num_computed_tokens=computed,
              mm_features=[], sampling_params=sp)] if prompt is not None else []
    cached = NS(req_ids=[] if new else [rid], num_computed_tokens=[] if new else [computed],
                resumed_req_ids=set())
    return NS(scheduled_new_reqs=new, scheduled_cached_reqs=cached,
              num_scheduled_tokens={rid: count} if count else {}, total_num_scheduled_tokens=count,
              finished_req_ids=set(finished), scheduled_spec_decode_tokens={},
              scheduled_encoder_inputs={}, preempted_req_ids=None)


def valid_plan():
    return dict(schema=5, granularity='module', identity={'backend': PROTOCOL}, units=UNITS, switch_us=10, blocks=[
        dict(phase=phase, block_id=i, operators=UNITS, triplet=[1, 2, 3], estimated_ms=1,
             amortized_strict_gt=True) for i, phase in enumerate(('prefill', 'decode'))])


class Contracts(unittest.TestCase):
    def test_scheduler_prefill_decode_and_finished_request(self):
        tracker = PhaseTracker()
        a = tracker.step(scheduler(prompt=512, count=512))
        b = tracker.step(scheduler(computed=512))
        c = tracker.step(scheduler(computed=513))
        self.assertEqual([a['phase'], b['phase'], c['phase']], ['prefill', 'decode', 'decode'])
        self.assertEqual(c['num_computed_tokens'], 513)
        self.assertEqual(tracker.step(scheduler(rid='next', prompt=1, count=1, finished=['r']))['phase'], 'prefill')

    def test_one_token_prompt_is_prefill(self):
        self.assertEqual(PhaseTracker().step(scheduler(prompt=1))['phase'], 'prefill')

    def test_reject_chunked_prefill(self):
        with self.assertRaisesRegex(RuntimeError, 'partial prefill'):
            PhaseTracker().step(scheduler(prompt=512, count=128))

    def test_reject_mixed_batch(self):
        s = scheduler(prompt=1)
        s.num_scheduled_tokens['other'] = 1
        s.total_num_scheduled_tokens = 2
        with self.assertRaisesRegex(RuntimeError, 'Mixed/batched'):
            PhaseTracker().step(s)

    def test_reject_resume_and_speculation(self):
        for field in ('resume', 'spec'):
            s = scheduler(prompt=1)
            if field == 'resume':
                s.scheduled_cached_reqs.resumed_req_ids = {'r'}
            else:
                s.scheduled_spec_decode_tokens = {'r': [1]}
            with self.assertRaises(RuntimeError):
                PhaseTracker().step(s)

    def test_reject_logprobs_and_unknown_cached_request(self):
        s = scheduler(prompt=1)
        s.scheduled_new_reqs[0].sampling_params.prompt_logprobs = 1
        with self.assertRaisesRegex(RuntimeError, 'logprobs'):
            PhaseTracker().step(s)
        with self.assertRaisesRegex(RuntimeError, 'Unknown cached'):
            PhaseTracker().step(scheduler(computed=3))

    def test_reject_nonsequential_decode(self):
        tracker = PhaseTracker()
        tracker.step(scheduler(prompt=3, count=3))
        with self.assertRaisesRegex(RuntimeError, 'sequential decode'):
            tracker.step(scheduler(computed=9))

    def test_v3_plan_rejects_old_backend_and_missing_units(self):
        plan = valid_plan()
        validate_plan(plan)
        old = copy.deepcopy(plan)
        old['identity']['backend'] = 'transformers'
        with self.assertRaisesRegex(ValueError, 'v2 Transformers'):
            validate_plan(old)
        plan['blocks'][0]['operators'] = UNITS[:-1]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_plan(plan)

    def test_v3_plan_identity_and_duplicate_block_ids(self):
        plan = valid_plan()
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_plan(plan, {'backend': PROTOCOL, 'changed': True})
        plan['blocks'][1]['block_id'] = 0
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_plan(plan)

    def test_engine_args_preserve_cache_and_disable_confounds(self):
        args = engine_args()
        self.assertEqual(args['worker_cls'], WORKER)
        self.assertFalse(args['enable_prefix_caching'])
        self.assertTrue(args['enforce_eager'])
        self.assertNotIn('kv_cache_memory_bytes', args)
        self.assertNotIn('mamba_cache_mode', args)
        native = server_args(custom=False)
        self.assertNotIn('--worker-cls', native)
        self.assertIn('--no-enable-chunked-prefill', native)
        production = server_args(custom=False, production=True)
        self.assertNotIn('--enforce-eager', production)
        self.assertNotIn('--worker-cls', production)

    def test_config_gate_graph_multiworker_and_async(self):
        config = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
                    model_config=NS(enforce_eager=True, dtype='torch.bfloat16', quantization=None,
                                    hf_text_config=NS(num_hidden_layers=64)),
                    scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False, async_scheduling=False),
                    cache_config=NS(enable_prefix_caching=False), speculative_config=None)
        validate_engine(config, '0.19.0+thor')
        for field, key, value in [('model_config', 'enforce_eager', False),
                                   ('parallel_config', 'data_parallel_size', 2),
                                   ('scheduler_config', 'async_scheduling', True)]:
            changed = copy.deepcopy(config)
            setattr(getattr(changed, field), key, value)
            with self.assertRaises(RuntimeError):
                validate_engine(changed, '0.19.0')

    def test_only_path_prefixes_are_remapped(self):
        result = replace_paths({'p': ['/sys/cpu/policy0/scaling_min_freq', '/sys/cpu2/a', 3]},
                               [('/sys/cpu', '/host-dvfs/cpu')])
        self.assertEqual(result['p'], ['/host-dvfs/cpu/policy0/scaling_min_freq', '/sys/cpu2/a', 3])

    def test_restoration_compares_controls_not_unlocked_current_rate(self):
        before = dict(domains=[1], bpmp=dict(lock=0, halt=0, lock_path='l', halt_path='h', rate_path='r', rate=100))
        after = copy.deepcopy(before)
        after['bpmp']['rate'] = 200
        self.assertTrue(restored(before, after))
        before['bpmp']['lock'] = after['bpmp']['lock'] = 1
        self.assertFalse(restored(before, after))

    def test_streamed_chunks_are_counted_by_token_ids(self):
        data = dict(input_token_ids=[4, 5, 6], input_tokens=3, output_tokens=4)
        events = [dict(id='c', choices=[dict(index=0, prompt_token_ids=[4, 5, 6], token_ids=[1, 2], text='ab')]),
                  dict(id='c', choices=[dict(index=0, token_ids=[3, 4], text='cd')]),
                  dict(id='c', choices=[], usage=dict(prompt_tokens=3, completion_tokens=4))]
        stream = io.BytesIO((''.join('data: ' + json.dumps(x) + '\n\n' for x in events) + 'data: [DONE]\n\n').encode())
        with patch('urllib.request.urlopen', return_value=stream):
            result = completion('http://localhost', data)
        self.assertEqual(len(result['chunks']), 2)
        self.assertEqual(len(result['output_token_ids']), 4)

    def test_usage_mismatch_rejected(self):
        stream = io.BytesIO(b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}\n\ndata: [DONE]\n')
        with patch('urllib.request.urlopen', return_value=stream), self.assertRaisesRegex(RuntimeError, 'mismatched'):
            completion('http://localhost', dict(input_token_ids=[1, 2, 3], input_tokens=3, output_tokens=2))

    def test_report_detects_stale_write_and_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'fuse_trace.csv').write_text('ticket,kind\n1,gpu_stream_boundary\n2,gpu_stream_boundary\n1,gpu_event_lookahead\n')
            self.assertEqual(len(controller_audit(path)['stale_write_violations']), 1)
        self.assertFalse(coverage({'rows': [], 'steps': []}, 128)['valid'])
        self.assertIsNone(stats([]))


if __name__ == '__main__':
    unittest.main()
