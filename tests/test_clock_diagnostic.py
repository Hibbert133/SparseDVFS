"""CPU-only failure evidence and single-frequency diagnostic contract."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rtx5090 import service_model as cli
from rtx5090.observe_core import GPU_UUIDS
from rtx5090.service_telemetry import require_clock_gate


class DiagnosticTests(unittest.TestCase):
    def test_failed_settle_keeps_readbacks(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'settle.json'
            client = NS(prepare=Mock(return_value={}), run=AsyncMock())
            with patch.object(cli, 'time', NS(monotonic=Mock(side_effect=[0, 0, 6]))), \
                 patch.object(cli.asyncio, 'sleep', new=AsyncMock()), \
                 patch.object(cli.telemetry, 'sample_device', return_value={'raw': {'clock_mhz': 2100}}):
                with self.assertRaisesRegex(RuntimeError, 'did not settle'):
                    asyncio.run(cli.settle(client, [object(), object()], 2250, 1, 'test', artifact=out))
            report = json.loads(out.read_text())
            self.assertFalse(report['settled'])
            self.assertEqual(len(report['frames'][0]), 2)

    def test_single_candidate_cannot_certify_full_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = [dict(uuid=u, backend='bridge') for u in GPU_UUIDS]
            (root/'telemetry.json').write_text(json.dumps(config))
            args = NS(apply=True, ack_default_auto=True, diagnose_mhz=2250,
                telemetry=str(root/'telemetry.json'), out=str(root/'diag'))
            devices = [Mock(uuid=u) for u in GPU_UUIDS]
            frames = [[dict(uuid=u, raw=dict(clock_mhz=2250, memory_clock_mhz=13801,
                temperature_c=40, utilization_pct=80)) for u in GPU_UUIDS] for _ in range(3)]
            async def restore(controller, *a):
                controller.reset()
                controller.restore_result['verified'] = True
            context = AsyncMock()
            with patch.object(cli.telemetry, 'open_devices', return_value=devices), \
                 patch.object(cli.client_api, 'BenchClient', return_value=context), \
                 patch.object(cli, 'stop_profile', new=AsyncMock()), \
                 patch.object(cli, 'load_frames', new=AsyncMock(return_value=frames)), \
                 patch.object(cli, 'settle', new=AsyncMock(return_value=frames)), \
                 patch.object(cli, 'restore_checked', new=AsyncMock(side_effect=restore)):
                result = asyncio.run(cli.clock_check(args, identity=dict(schema='optimized-service-v1',
                    fingerprint='id', base_url='test', models={'data':[{'id':'model'}]})))
            self.assertEqual(result['status'], 'diagnostic_passed')
            self.assertTrue(result['restoration']['verified'])
            for d in devices:
                d.set.assert_called_once_with(2250)
                d.reset.assert_called_once()
            with self.assertRaises(ValueError): require_clock_gate(result, 'id', result['telemetry_config_sha256'])
            self.assertTrue((root/'diag/clock-check.json').exists())
