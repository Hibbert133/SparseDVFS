"""Offline fixtures test reporting and grid gates, never produce device results."""
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_core import fake_hardware
from test_v3_contracts import valid_plan
from sparsedvfs.hardware import write_json, load_json, snapshot
from sparsedvfs_vllm.contracts import REVISION, PROTOCOL, UNITS
from sparsedvfs_vllm.host import hardware, plan_command
from sparsedvfs_vllm.report import report


class Workflow(unittest.TestCase):
    def test_missing_emc_table_allows_only_current_quick_and_never_writes_clocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hw = fake_hardware(root / 'nodes')
            before = snapshot(hw)
            hw['emc']['frequencies'] = []
            results = root / 'results'
            results.mkdir()
            args = NS(results=results, emc_evidence=None, power_source=None)
            with patch('sparsedvfs_vllm.host.probe', return_value=copy.deepcopy(hw)):
                quick = hardware(args, grid='quick')
            self.assertEqual(quick['emc']['frequencies'], [1000000000])
            with patch('sparsedvfs_vllm.host.probe', return_value=copy.deepcopy(hw)), \
                    self.assertRaisesRegex(RuntimeError, 'verified discrete EMC'):
                hardware(args, grid='full')
            after = snapshot(hw)
            self.assertEqual(before['domains'], after['domains'])
            self.assertEqual(before['bpmp'], after['bpmp'])

    def test_v2_profile_cannot_create_v3_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / 'scenario_512_128'
            case.mkdir()
            write_json(case / 'profile.json', {'identity': {'backend': 'transformers'}})
            with self.assertRaisesRegex(ValueError, 'v2/Transformers'):
                plan_command(NS(results=root, input_tokens=512, output_tokens=128))
            self.assertFalse((case / 'plan.json').exists())

    def test_complete_report_keeps_negative_savings_and_suppresses_invalid_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = valid_plan()
            plan.update(selector='measured', emc_grid='quick')
            write_json(root / 'plan.json', plan)
            rows = []
            for mode in ('native', 'instrumented', 'fixed', 'serial', 'lookahead'):
                directory = root / mode
                (directory / 'worker/controller').mkdir(parents=True)
                rows.append(dict(mode=mode, repeat=0, run_dir=str(directory), image_id='fixture-image',
                    model_revision=REVISION, workload=dict(input_tokens=1, output_tokens=2),
                    client=dict(ttft_ms=1, tpot_ms=1, request_ms=2, output_sha256='fixture',
                                usage=dict(completion_tokens=2)),
                    measurement=dict(power=dict(valid=True, energy_j=3 if mode == 'lookahead' else 2,
                                                scope='fixture'),
                                     frequency_readbacks=[dict(cpu_khz=[1], gpu_hz=2, emc_hz=3)])))
                if mode == 'native':
                    continue
                steps = [dict(request_id='r', phase=phase, order=UNITS) for phase in ('prefill', 'decode')]
                events = [dict(request_id='r', name=name, phase=phase, stream=42, latency_ms=0.1)
                          for phase in ('prefill', 'decode') for name in UNITS]
                write_json(directory / 'worker/worker_trace.json', dict(identity=plan['identity'], steps=steps, rows=events))
                control = directory / 'worker/controller'
                write_json(control / 'restore_status.json', dict(restored=True))
                write_json(control / 'requested_schedule.json', dict(dvfs_applied=True, events=[
                    dict(kind='cuda_block', ticket=i, block_id=i, lookahead=mode=='lookahead' and i==1) for i in (1, 2)]))
                ahead='1,gpu_event_lookahead\n' if mode=='lookahead' else ''
                (control / 'fuse_trace.csv').write_text('ticket,kind\n1,cpu_submission_boost\n1,gpu_stream_boundary\n'+ahead+'2,cpu_submission_boost\n2,gpu_stream_boundary\n')
                (control / 'cuda_timing.csv').write_text('kind\ngpu_end_observed\ngpu_end_observed\n')
            write_json(root / 'bench_latest.json', dict(comparable=True, results=rows, run_dir=str(root)))
            with patch('sys.stdout', new=io.StringIO()):
                report(root)
            result = load_json(root / 'report.json')
            self.assertTrue(result['mechanism_acceptance'])
            self.assertEqual(result['groups']['lookahead']['energy_saving_vs_eager_native_pct'], -50.0)
            fuse_file=root/'lookahead/worker/controller/fuse_trace.csv'
            fuse_file.write_text(fuse_file.read_text().replace('1,gpu_event_lookahead\n',''))
            with patch('sys.stdout',new=io.StringIO()):report(root)
            self.assertFalse(load_json(root/'report.json')['mechanism_acceptance'])
            fuse_file.write_text(fuse_file.read_text().replace('2,cpu_submission_boost','1,gpu_event_lookahead\n2,cpu_submission_boost'))
            write_json(root / 'lookahead/worker/controller/restore_status.json', dict(restored=False))
            with patch('sys.stdout', new=io.StringIO()):
                report(root)
            result = load_json(root / 'report.json')
            self.assertFalse(result['comparable'])
            self.assertIsNone(result['groups']['lookahead']['energy_saving_vs_eager_native_pct'])


if __name__ == '__main__':
    unittest.main()
