import asyncio
import copy
import unittest
from types import SimpleNamespace

from rtx5090.linear_frequency_probe import summarize, compare, transaction
from rtx5090.observe_core import GPU_UUIDS


def observation():
    return dict(status='complete', identity_fingerprint='same',
                cache_isolation=dict(miss_verified=True), coverage=dict(passed=True),
                reply=dict(usage=dict(prompt_tokens=512, completion_tokens=64)),
                matrix_replay=[dict(rank=r, uuid=GPU_UUIDS[r], rows=[
                    dict(phase=p, module=m, parent='layer.0.'+m, M=512 if p=='prefill' else 1,
                         N=100, K=100, batches=[dict(repeats=32, mean_call_s=v)
                                              for v in (.001, .002, .003)])
                    for p in ('prefill', 'decode') for m in ('gate_up_proj', 'down_proj')])
                    for r in (0, 1)])


class ProbeTests(unittest.TestCase):
    def test_full_dual_rank_summary_and_comparison(self):
        operators = summarize(observation(), 'same')
        self.assertEqual(len(operators), 8)
        windows = [dict(requested_mhz=f, operators=copy.deepcopy(operators))
                   for _ in range(3) for f in (2250, 2527)]
        self.assertTrue(all(r['latency_ratio_2250_over_2527']==1 for r in compare(windows)))
        windows[2]['operators']['0:prefill/down_proj']['shape'] = [1,2,3]
        with self.assertRaises(ValueError): compare(windows)

    def test_reject_invalid_observation(self):
        for mutate in (
            lambda o:o.update(identity_fingerprint='other'),
            lambda o:o['matrix_replay'].pop(),
            lambda o:o['reply']['usage'].update(completion_tokens=1),
            lambda o:o['matrix_replay'][0]['rows'][0]['batches'].pop(),
            lambda o:o['matrix_replay'][0].update(uuid=GPU_UUIDS[1]),
        ):
            o = observation(); mutate(o)
            with self.assertRaises(ValueError): summarize(o, 'same')

    def test_failure_and_interrupt_restore(self):
        for error in (RuntimeError('single card write failed'), KeyboardInterrupt('stop')):
            calls=[]
            controller=SimpleNamespace(restore_result=dict(required=True))
            async def work():
                raise error
            async def restore():
                calls.append('reset-and-verify')
            with self.assertRaises(type(error)):
                asyncio.run(transaction(controller, work, restore))
            self.assertEqual(calls, ['reset-and-verify'])

    def test_failed_restore_never_becomes_success(self):
        controller=SimpleNamespace(restore_result=dict(required=True))
        async def work(): return 1
        async def restore(): raise RuntimeError('restore failed')
        with self.assertRaisesRegex(RuntimeError, 'restore failed'):
            asyncio.run(transaction(controller, work, restore))


if __name__ == '__main__': unittest.main()
