import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from rtx5090 import host_load_check as check


class HostTests(unittest.IsolatedAsyncioTestCase):
    def test_host_file_python38_syntax(self):
        ast.parse(Path(check.__file__).read_text(), feature_version=(3, 8))

    def test_gap_and_clock_validation(self):
        frames = [[dict(uuid='u', timestamp_ns=t, synthetic=False,
                        timestamp_clock={'name':'perf_counter'})]
                  for t in (0, 100000000, 200000000, 600000000)]
        self.assertEqual(check.summarize(frames, ['u'])[0]['gaps_over_350ms'], 1)
        frames[-1][0]['timestamp_ns'] = 200000000
        with self.assertRaises(ValueError): check.summarize(frames, ['u'])

    async def test_ready_precedes_requests_and_finish_follows(self):
        from rtx5090 import eager, service_client, service_model
        order = []
        async def ready():
            order.append('ready')
            return b'{"status":"host_ready","pid":42}\n'
        async def request(value):
            self.assertIn('ready', order)
            order.append('request')
            return {'usage': {'prompt_tokens':512, 'completion_tokens':64}}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.prepare = Mock(side_effect=lambda length, seed, rid: dict(prompt_hash=str(seed)))
        client.run.side_effect = request
        process = NS(returncode=None, stdin=NS(write=lambda b: order.append('finish'),
            drain=AsyncMock(), close=Mock()), stdout=NS(readline=ready, read=AsyncMock(return_value=b'')),
            wait=AsyncMock(return_value=0))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            check.save(out/'windows-summary.json', dict(status='host_sampling_complete',
                by_uuid=[dict(gaps_over_350ms=0)]))
            args = NS(identity='id', windows_python='python')
            result = dict(replies=[])
            with patch.object(eager, 'checked', return_value=dict(fingerprint='id', base_url='mock', models={'data':[{'id':'m'}]})), \
                 patch.object(service_client, 'BenchClient', return_value=client), \
                 patch.object(service_model, 'warmup', new=AsyncMock()), \
                 patch.object(eager, 'isolate_observation_cache', new=AsyncMock(return_value={'method':'unique_request_cache_salt'})), \
                 patch.object(service_model, 'emit'), \
                 patch('subprocess.check_output', return_value='D:\\repo'), \
                 patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)):
                await check.workload(args, out, result)
            self.assertEqual(order, ['ready']+['request']*20+['finish'])
            self.assertEqual(result['status'], 'loaded_host_sampling_passed')
