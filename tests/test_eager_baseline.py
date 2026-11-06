"""Batch orchestration tests with synthetic telemetry; no hardware certification."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rtx5090 import eager_baseline as batch
from rtx5090.observe_core import GPU_UUIDS


class Client:
    source_sha256, tokenizer_fingerprint = 'mock-source', 'mock-tokenizer'
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def prepare(self, length, seed, request_id):
        return dict(length=length, seed=seed, request_id=request_id, prompt_hash=str(seed), tokens=[seed])


def row(valid=True):
    return dict(status='valid' if valid else 'invalid', requests=20,
        cache_status='verified_miss' if valid else 'cache_hit',
        replies=[dict(latency_s=4)]*20, window_s=80, ttft_s=.2, tpot_s=.06,
        energy_j=100, energy_by_uuid_j={u: 50 for u in GPU_UUIDS},
        energy_relative_uncertainty=.01, quality_errors=[] if valid else ['cache hit'],
        clock_ranges={u: dict(memory_mode=13801) for u in GPU_UUIDS},
        energy=dict(by_uuid={u: dict(source='counter') for u in GPU_UUIDS}))


class BatchTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_has_seven_disjoint_seed_ranges(self):
        plan = batch.window_plan(100, 20)
        seeds = [s+i for _, s in plan for i in range(20)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(len(plan), 7)
        with self.assertRaises(ValueError): batch.window_plan(1, 19)

    async def execute(self, valid=True):
        devices = [Mock(uuid=u) for u in GPU_UUIDS]
        calls = []
        async def measured(client, identity, devs, requests, out, name, **kwargs):
            calls.append((requests, kwargs))
            return copy.deepcopy(row(valid))
        async def isolate(identity, request):
            return dict(method='unique_request_cache_salt', miss_verified=False)
        with tempfile.TemporaryDirectory() as folder:
            args = NS(identity='mock', telemetry='mock', seed=100, requests=20)
            identity = dict(base_url='mock', models={'data':[{'id':'model'}]})
            result = dict(status='failed', windows=[])
            with patch.object(batch, 'read', return_value=[]), \
                 patch.object(batch.telemetry, 'open_devices', return_value=devices), \
                 patch.object(batch.client_api, 'BenchClient', return_value=Client()), \
                 patch.object(batch.service, 'stop_profile', new=AsyncMock()), \
                 patch.object(batch.service, 'warmup', new=AsyncMock()), \
                 patch.object(batch.service, 'measured_window', side_effect=measured), \
                 patch.object(batch.service, 'emit'), \
                 patch.object(batch.eager, 'checked', return_value=identity), \
                 patch.object(batch.eager, 'isolate_observation_cache', side_effect=isolate):
                if valid:
                    await batch.collect(args, identity, Path(folder), result)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Invalid window'):
                        await batch.collect(args, identity, Path(folder), result)
            for device in devices:
                device.set.assert_not_called()
                device.reset.assert_not_called()
                device.close.assert_called_once()
            return calls, result

    async def test_all_windows_freeze_sources_and_salt_every_request(self):
        calls, result = await self.execute()
        self.assertEqual(result['status'], 'current_clock_reference_complete')
        self.assertEqual(len(calls), 7)
        salts = [r['cache_salt'] for requests, _ in calls for r in requests]
        self.assertEqual(len(salts), len(set(salts)))
        self.assertTrue(all(len(s) == 64 for s in salts))
        self.assertIsNone(calls[0][1]['sources'])
        for _, kwargs in calls[1:]:
            self.assertEqual(kwargs['sources'], {u:'counter' for u in GPU_UUIDS})
            self.assertIsNone(kwargs['target'])

    async def test_bad_pilot_stops_no_retry_and_keeps_summary(self):
        calls, result = await self.execute(False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['windows'][0]['status'], 'invalid')
