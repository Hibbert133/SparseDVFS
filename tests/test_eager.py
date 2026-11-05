"""CPU-only software checks, never target CUDA/DVFS acceptance."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock

from rtx5090 import eager_core as core, eager_model as model, eager
from rtx5090.eager_ipc import Rendezvous, PairSwitch, LookaheadSwitch
from rtx5090.observe_core import GPU_UUIDS
from rtx5090.service_telemetry import DualClocks
from sparsedvfs.paper_core import greedy_partition, OrderedDAG
from sparsedvfs.paper_core import timeline_seconds, thermal_power_w


def message(rank, seq=0, key=None):
    return dict(rank=rank, uuid=GPU_UUIDS[rank], fingerprint='session', seq=seq,
                gpu_complete=True, key=key or ['request', 'prefill', 0])


def dataset(split='train'):
    windows = []
    for repeat in range(3):
        for frequency in ('auto', '2250', '2527'):
            windows.append(dict(frequency=frequency, status='valid', quality_errors=[],
                cache_status='verified_miss', requests=20, energy_relative_uncertainty=.01,
                prompt_hashes=[f'{split}-{repeat}-{i}' for i in range(20)],
                energy_by_uuid_j={u: 10 for u in GPU_UUIDS}, energy_j=20,
                window_s=1, ttft_s=.1, tpot_s=.01))
    return dict(schema=core.SCHEMA, status='collected', split=split, synthetic=False,
        identity_fingerprint='session', restoration=dict(verified=True),
        protocol=dict(input=512, output=64, concurrency=1, windows=3, requests=20), windows=windows)


def equations(split='train'):
    data = dict(schema=core.SCHEMA, split=split, synthetic=True, identity_fingerprint='session',
        scope='same_path_repeated_unit_board_windows', restoration=dict(verified=True),
        units=dict(latency='s', power='W', voltage='V', temperature='C', frequency='MHz', work='FLOP', data='byte'),
        workloads={}, capacities={}, records=[])
    for rank in (0, 1):
        data['capacities'][str(rank)] = dict(same_path_provenance='CPU_FIXTURE_ONLY',
            peak_flops={'2250': 2e12, '2527': 2.5e12}, bandwidth_Bps=1e12)
        for unit in ('embed', 'layer.0'):
            key = f'{rank}:prefill/{unit}'
            data['workloads'][key] = dict(work_flop=1e9, data_byte=1e6, s_comp=0, s_mem=0,
                signature=key, workload_provenance='CPU_FIXTURE_ONLY', data_semantics='explicit_test_bytes')
            for i in range(8):
                f = core.RATES[i%2]
                v, t = .8+(i%3)*.05, 40+i*2
                seconds = timeline_seconds(1e9, 1e6, 0, 0, data['capacities'][str(rank)]['peak_flops'][str(f)], 1e12, .00001)
                power = thermal_power_w(1e-8, v, f*1e6, .1, 2, t)
                data['records'].append(dict(rank=rank, uuid=GPU_UUIDS[rank], unit=unit, context='prefill',
                    mhz=f, actual_mhz=f, sample_id=f'{split}-{rank}-{unit}-{i}', prompt_hash=f'{split}-{i}',
                    signature=key, same_path_verified=True, path_evidence_sha256='test',
                    power_attribution='repeated_unit_board_measurement', power_evidence_sha256='test',
                    latency_s=seconds, power_w=power, voltage_v=v, temperature_c=t, window_s=20,
                    sensor_interval_s=.1, energy_uncertainty=.01))
    return data


class WeightTests(unittest.TestCase):
    def setup_files(self, root):
        (root/'config.json').write_text(json.dumps(dict(model_type='qwen3_5', text_config=dict(num_hidden_layers=64))))
        (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'w': 'a.safetensors'})))
        header = json.dumps({'w': dict(dtype='BF16', shape=[2], data_offsets=[0, 4])}).encode()
        (root/'a.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'\0'*4)

    def test_complete_and_truncated(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.setup_files(root)
            self.assertEqual(core.weights(root)['tensors'], 1)
            f = root/'a.safetensors'; f.write_bytes(f.read_bytes()[:-1])
            with self.assertRaises(ValueError): core.weights(root)

    def test_missing_shard(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.setup_files(root); (root/'a.safetensors').unlink()
            with self.assertRaises(ValueError): core.weights(root)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.setup_files(root)
            (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'w': '../outside'})))
            with self.assertRaises(ValueError): core.weights(root)

    def test_quantized_model_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path); self.setup_files(root)
            (root/'config.json').write_text(json.dumps(dict(model_type='qwen3_5',num_hidden_layers=64,quantization_config={'quant_method':'awq'})))
            with self.assertRaises(ValueError): core.weights(root)


class CoordinationTests(unittest.TestCase):
    def test_lookahead_cancel_joins_before_restore(self):
        with tempfile.TemporaryDirectory() as path:
            switch = Mock()
            switch.plan = dict(switch_s=.01, contexts={'prefill': [
                dict(mhz=2250,estimated_s=1),dict(mhz=2527,estimated_s=1)]})
            switch.timeout = .1
            queue = LookaheadSwitch(switch, Path(path)/'queue.jsonl')
            queue(['r','prefill',0])
            queue.close()
            self.assertEqual(switch.call_count, 1)
            with self.assertRaises(RuntimeError): queue(['r','prefill',1])

    def test_lookahead_next_boundary_waits_on_queued_write(self):
        with tempfile.TemporaryDirectory() as path:
            switch = Mock(return_value=dict(readback_verified=True))
            switch.plan = dict(switch_s=.01, contexts={'prefill': [
                dict(mhz=2250,estimated_s=.02),dict(mhz=2527,estimated_s=.02)]})
            switch.timeout = .1
            queue = LookaheadSwitch(switch, Path(path)/'queue.jsonl')
            try:
                queue(['r','prefill',0]); queue(['r','prefill',1])
                self.assertEqual(switch.call_count, 2)
            finally: queue.close()

    def test_both_ranks_required_and_multiple_generations(self):
        action = Mock(return_value={'released': True})
        gate = Rendezvous(action, 'session', timeout=.5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            def rank_run(rank):
                return [gate.exchange(message(rank, i, ['req', 'prefill', i])) for i in range(10)]
            a, b = pool.submit(rank_run, 0), pool.submit(rank_run, 1)
            self.assertEqual(a.result(), b.result())
        self.assertEqual(action.call_count, 10)

    def test_timeout_poison_persists(self):
        action = Mock()
        gate = Rendezvous(action, 'session', timeout=.01)
        with self.assertRaises(TimeoutError): gate.exchange(message(0))
        with self.assertRaises(RuntimeError): gate.exchange(message(1))
        action.assert_not_called()

    def test_mismatched_boundary_releases_neither(self):
        action = Mock()
        gate = Rendezvous(action, 'session', timeout=.5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(gate.exchange, message(0))
            b = pool.submit(gate.exchange, message(1, key=['other', 'prefill', 0]))
            for f in (a, b):
                with self.assertRaises((ValueError, RuntimeError)): f.result()
        action.assert_not_called()

    def test_action_failure_propagates_both_ranks(self):
        gate = Rendezvous(Mock(side_effect=RuntimeError('set failed')), 'session', timeout=.5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(gate.exchange, message(i)) for i in (0, 1)]
            for f in futures:
                with self.assertRaises(RuntimeError): f.result()

    def test_wrong_uuid_and_missing_gpu_completion(self):
        for changes in ({'uuid': GPU_UUIDS[1]}, {'gpu_complete': False}, {'seq': 1}, {'fingerprint': 'old'}):
            with self.subTest(changes=changes):
                gate = Rendezvous(Mock(), 'session', .01)
                with self.assertRaises(ValueError): gate.exchange(dict(message(0), **changes))

    def test_pair_failure_restores_both_touched_devices(self):
        devices = [Mock(uuid=u) for u in GPU_UUIDS]
        devices[1].set.side_effect = RuntimeError('second card failed')
        controller = DualClocks(devices, apply=True)
        plan = dict(contexts={'prefill': [dict(mhz=2250)]})
        with tempfile.TemporaryDirectory() as path:
            switch = PairSwitch(controller, plan, Path(path)/'log', timeout=.1)
            with self.assertRaises(RuntimeError): switch(['r', 'prefill', 0])
        for d in devices: d.reset.assert_called_once()
        self.assertFalse(controller.restore_result['verified'])

    def test_set_return_does_not_release_unsettled_pair(self):
        devices = [Mock(uuid=u, read=Mock(return_value=2100)) for u in GPU_UUIDS]
        controller = DualClocks(devices, apply=True)
        with tempfile.TemporaryDirectory() as path:
            switch = PairSwitch(controller, dict(contexts={'prefill':[dict(mhz=2250)]}), Path(path)/'log', timeout=.02, interval=.001)
            with self.assertRaises(TimeoutError): switch(['r', 'prefill', 0])
        for d in devices: d.reset.assert_called_once()

    def test_apply_is_required(self):
        with self.assertRaises(ValueError): DualClocks([Mock(), Mock()]).set(2250)


class DataTests(unittest.TestCase):
    def test_freeze_and_independent_validation(self):
        frozen = core.freeze_window_model(dataset())
        result = core.validate_window_model(dataset('validation'), frozen)
        self.assertTrue(result['passed']); self.assertFalse(result['block_selection_authorized'])

    def test_train_validation_leakage(self):
        data = dataset(); frozen = core.freeze_window_model(data); data['split'] = 'validation'
        with self.assertRaises(ValueError): core.validate_window_model(data, frozen)

    def test_failed_missing_unpaired_or_unrestored_rejected(self):
        for mode in ('failed', 'missing', 'unpaired', 'restore', 'energy'):
            with self.subTest(mode=mode):
                data = dataset()
                if mode == 'failed': data['windows'][0]['status'] = 'invalid'
                if mode == 'missing': data['windows'].pop()
                if mode == 'unpaired': data['windows'][0]['prompt_hashes'][0] = 'different'
                if mode == 'restore': data['restoration']['verified'] = False
                if mode == 'energy': data['windows'][0]['energy_j'] = 19
                with self.assertRaises(ValueError): core.freeze_window_model(data)

    def test_context_tracks_exact_cache_length(self):
        row = dict(prompt_tokens=512, admitted_computed_tokens=0, resumed=False, preempted=False,
            speculative_tokens=0, phase='decode', num_scheduled_tokens=1, num_computed_tokens=530)
        self.assertEqual(core.context_key(row), 'decode.18')
        with self.assertRaises(ValueError): core.context_key(dict(row, admitted_computed_tokens=10))

    def test_partial_prefill_rejected(self):
        row = dict(prompt_tokens=512, admitted_computed_tokens=0, resumed=False, preempted=False,
            speculative_tokens=0, phase='prefill', num_scheduled_tokens=256, num_computed_tokens=0)
        with self.assertRaises(ValueError): core.context_key(row)


class EquationTests(unittest.TestCase):
    def test_fit_shared_thermal_and_independent_validation(self):
        fitted = model.fit(equations(), allow_synthetic=True)
        for t in fitted['thermal'].values():
            self.assertAlmostEqual(t['k1'], .1, places=7)
            self.assertAlmostEqual(t['k2'], 2, places=7)
        result = model.validate(equations('validation'), fitted, allow_synthetic=True)
        self.assertTrue(result['passed']); self.assertTrue(result['synthetic'])

    def test_synthetic_never_deployable(self):
        with self.assertRaises(ValueError): model.fit(equations())

    def test_window_power_cannot_become_unit_label(self):
        data = equations(); data['records'][0]['power_attribution'] = 'request_average'
        with self.assertRaises(ValueError): model.fit(data, allow_synthetic=True)

    def test_raw_zeros_cannot_be_effective_sparsity(self):
        data = equations(); next(iter(data['workloads'].values()))['s_comp'] = .5
        with self.assertRaises(ValueError): model.fit(data, allow_synthetic=True)

    def test_signature_change_rejected(self):
        data = equations(); data['records'][0]['signature'] = 'different-cache-state'
        with self.assertRaises(ValueError): model.fit(data, allow_synthetic=True)

    def test_voltage_temperature_unidentifiable(self):
        data = equations()
        for r in data['records']: r.update(voltage_v=1, temperature_c=50)
        with self.assertRaises(ValueError): model.fit(data, allow_synthetic=True)

    def test_algorithm1_tail_and_gpu_pair_projection(self):
        blocks, decisions = greedy_partition(OrderedDAG.chain(['a', 'b', 'c']),
            lambda u: (2250,2250) if u == 'a' else (2527,2527),
            lambda units, f: .2*len(units), lambda a,b: a == b,
            .1, 5, frequency_factory=model.PairFrequency.of)
        self.assertEqual(blocks[0]['nodes'], ['a','b','c'])
        self.assertEqual(blocks[0]['triplet'], [2527,2527])
        self.assertTrue(decisions[0]['short'])
        short, _ = greedy_partition(OrderedDAG.chain(['a']), lambda u: (2250,2250),
            lambda u,f: .01, lambda a,b: True, .1, 5, frequency_factory=model.PairFrequency.of)
        self.assertFalse(short[0]['amortized_strict_gt'])

    def test_no_cpu_emc_sentinels(self):
        with self.assertRaises(ValueError): model.PairFrequency.of((1,2250,1))


class CLITests(unittest.TestCase):
    def test_no_native_profiler_route_means_no_stop_request(self):
        import asyncio
        from unittest.mock import patch
        from rtx5090.service_model import stop_profile
        with patch('rtx5090.service_client.http') as http:
            asyncio.run(stop_profile(dict(profiler_routes={})))
            asyncio.run(stop_profile(dict(profiler_routes={'/start_profile':False,'/stop_profile':False})))
            http.assert_not_called()

    def test_minimal_commands_parse(self):
        for args in (['preflight','--run','x'], ['observe','--identity','i','--out','o','--seed','1'],
                     ['paper-fit','--data','d','--out','o']):
            self.assertIsNotNone(eager.parser().parse_args(args))

    def test_partial_arm_is_disarmed(self):
        from unittest.mock import patch
        def call(path, command, timeout):
            if command.get('mode') == 'units' and str(path).endswith('rank-1.sock'):
                raise RuntimeError('rank1 failed')
            return dict(armed=command.get('mode'))
        with patch('rtx5090.eager.rpc', side_effect=call) as mocked:
            with self.assertRaises(RuntimeError):
                eager.rank_commands(dict(runtime='x',session='s'),dict(op='arm',mode='units'))
            self.assertEqual(sum(c.args[1].get('mode') == 'off' for c in mocked.call_args_list), 2)


if __name__ == '__main__': unittest.main()
