"""CPU-only integration of the real worker adapter, spans and Schedule.

The vLLM base class, Qwen modules, CUDA events and controller are test doubles.
No real CUDA execution or native FUSE execution is claimed by this suite.
"""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch
import test_v3_worker as fixtures
from test_v3_contracts import scheduler
from sparsedvfs.qwen_runner import Schedule
from sparsedvfs_vllm.contracts import UNITS


def make_plan():
    blocks = []
    # The last block spans norm/logits in execute_model and sample_tokens.
    cuts = (0, 9, 33, 65, 68)
    vectors = ([400000, 200000000, 1000000000],
               [800000, 500000000, 2000000000],
               [1000000, 1000000000, 1000000000],
               [400000, 200000000, 1000000000])
    for phase in ('prefill', 'decode'):
        for i in range(4):
            blocks.append(dict(phase=phase, block_id=len(blocks),
                               operators=UNITS[cuts[i]:cuts[i+1]], triplet=vectors[i],
                               estimated_ms=3, amortized_strict_gt=i != 2))
    return dict(synthetic=True, deployable=False, switch_us=100, blocks=blocks)


class RecordingController:
    """Record requests, without pretending to emulate GPU concurrency."""
    boost_khz = 1000000

    def __init__(self):
        self.events, self.open_ticket = [], None
        self.counter = 0

    def cpu(self, frequency):
        self.events.append(dict(kind='cpu', frequency=frequency))

    def cuda_begin(self, stream, block, upcoming, lead, lookahead):
        if self.open_ticket is not None:
            raise RuntimeError('Overlapping submission scopes')
        self.counter += 1
        self.open_ticket = self.counter
        self.events.append(dict(kind='begin', ticket=self.counter, stream=stream,
                                phase=block['phase'], block_id=block['block_id'],
                                units=list(block['operators']), triplet=block['triplet'],
                                next_block=upcoming['block_id'] if upcoming else None,
                                lookahead_eligible=lookahead, lead_us=lead))
        return self.counter

    def cuda_end(self, ticket):
        if ticket != self.open_ticket:
            raise RuntimeError('Wrong submission ticket')
        self.events.append(dict(kind='end', ticket=ticket))
        self.open_ticket = None

    def cuda_drain(self):
        if self.open_ticket is not None:
            raise RuntimeError('Undrained submission')
        self.events.append(dict(kind='drain'))


class VirtualChainTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkerTests('test_execute_and_sample_are_one_complete_step')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.worker = self.fixture.worker
        self.plan = make_plan()
        self.controller = RecordingController()
        # Intentionally use the fixture's instrumented configuration to bypass
        # production identity/profile gates. This plan is synthetic and is NOT
        # accepted here as a validated/deployable runtime plan.
        self.worker._sd_fuse = self.controller
        self.worker._sd_schedule = Schedule(self.plan, self.controller, 'lookahead')

    def step(self, output):
        self.assertIsNone(self.worker.execute_model(output))
        # Final block stays open across the original worker method boundary.
        self.assertIsNotNone(self.worker._sd_schedule.ticket)
        self.assertEqual(self.worker._sd_trace.order, UNITS[:-1])
        self.assertEqual(self.worker.sample_tokens(None), 'sampled')
        self.assertIsNone(self.worker._sd_schedule.ticket)

    def test_prefill_three_decodes_and_cross_method_blocks(self):
        self.step(scheduler(prompt=3, count=3))
        for computed in (3, 4, 5):
            self.step(scheduler(computed=computed))
        self.worker._sd_schedule.finalize()
        data = self.worker.sdvfs_export()
        self.assertEqual([s['phase'] for s in data['steps']], ['prefill'] + ['decode']*3)
        self.assertEqual(len(data['rows']), 68*4)
        self.assertTrue(all(s['order'] == UNITS for s in data['steps']))
        starts = [e for e in self.controller.events if e['kind'] == 'begin']
        ends = [e for e in self.controller.events if e['kind'] == 'end']
        self.assertEqual([e['block_id'] for e in starts], [0,1,2,3] + [4,5,6,7]*3)
        self.assertEqual([e['ticket'] for e in starts], [e['ticket'] for e in ends])
        for step in range(4):
            current = starts[step*4:step*4+4]
            self.assertEqual([u for e in current for u in e['units']], UNITS)
            self.assertEqual([e['next_block'] for e in current],
                             [current[1]['block_id'], current[2]['block_id'], current[3]['block_id'], None])
            self.assertEqual([e['lookahead_eligible'] for e in current], [True,True,False,True])
        trace_dir = os.environ.get('VIRTUAL_TRACE_DIR')
        if trace_dir:
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path/'virtual_chain.json').write_text(json.dumps(dict(
                synthetic=True, deployable=False, real_vllm=False, real_cuda=False,
                native_fuse=False, profile_identity_gates_exercised=False,
                timing_valid=False, plan=self.plan, events=self.controller.events,
                phases=[s['phase'] for s in data['steps']],
                unit_visits=len(data['rows']), block_begins=len(starts), block_ends=len(ends)
            ), indent=2), encoding='utf-8')

    def test_block_interiors_do_not_trigger_extra_boundaries(self):
        self.step(scheduler(prompt=3, count=3))
        self.assertEqual(len([e for e in self.controller.events if e['kind']=='begin']), 4)
        self.assertEqual(len(self.worker._sd_trace.rows), 68)

    def test_serial_disables_lookahead(self):
        self.worker._sd_schedule = Schedule(self.plan, self.controller, 'serial')
        self.step(scheduler(prompt=3, count=3))
        self.assertFalse(any(e.get('lookahead_eligible') for e in self.controller.events))

    def test_missing_plan_unit_is_rejected_before_submission(self):
        plan = make_plan()
        plan['blocks'][0]['operators'] = plan['blocks'][0]['operators'][1:]
        schedule = Schedule(plan, self.controller, 'serial')
        with self.assertRaises(KeyError):
            schedule.before('prefill', 'embed')
        self.assertEqual(self.controller.events, [])

    def test_wrong_unit_order_is_rejected(self):
        trace = self.worker._sd_trace
        trace.start(dict(phase='prefill', step=0, request_id='r'), 'synthetic')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected execution order'):
            trace.begin('layer.0')
        self.assertEqual(self.controller.events, [])

    def test_unclosed_block_cannot_finalize_or_start_next_step(self):
        schedule = self.worker._sd_schedule
        schedule.before('prefill', 'embed')
        with self.assertRaisesRegex(RuntimeError, 'Unclosed'):
            schedule.finalize()
        with self.assertRaisesRegex(RuntimeError, 'not closed'):
            schedule.start_forward('decode', 1)


if __name__ == '__main__':
    unittest.main()
