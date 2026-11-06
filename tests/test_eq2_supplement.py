import unittest
from unittest.mock import patch
from rtx5090.eq2_supplement import changed_readbacks, collect
from rtx5090.observe_core import GPU_UUIDS
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from contextlib import ExitStack
from pathlib import Path
import tempfile


class ReadbackTests(unittest.TestCase):
    def test_lower_and_distinct_actual_conditions(self):
        baseline=[[dict(uuid=u,raw=dict(clock_mhz=2850,utilization_pct=80)) for u in GPU_UUIDS]]*4
        def frames(f):
            return dict(frames=[[dict(uuid=u,clock_mhz=f,utilization_pct=80) for u in GPU_UUIDS]]*4)
        first=changed_readbacks(frames(2430),baseline,2527)
        self.assertFalse(first[GPU_UUIDS[0]]['exact_lock_verified'])
        changed_readbacks(frames(2150),baseline,2250,first)
        with self.assertRaises(ValueError): changed_readbacks(frames(2850),baseline,2527)
        with self.assertRaises(ValueError): changed_readbacks(frames(2400),baseline,2527,first)


class RestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_workload_failure_restores_before_close(self):
        events=[]
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
        controller=Mock(restore_result=dict(required=False,verified=False))
        def set_clock(target):
            controller.restore_result['required']=True
            events.append('set')
        controller.set.side_effect=set_clock
        controller.reset.side_effect=lambda: events.append('reset')
        async def restore(*args):
            events.append('verify'); controller.restore_result['verified']=True
        device=Mock(); device.close.side_effect=lambda: events.append('close')
        args=SimpleNamespace(telemetry='unused',identity='unused',seed=1,windows_python='unused')
        result=dict(windows=[],status='failed')
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            prefix='rtx5090.eq2_supplement.'
            stack.enter_context(patch(prefix+'read',return_value=[dict(backend='bridge')]*2))
            stack.enter_context(patch(prefix+'telemetry.open_devices',return_value=[device]))
            stack.enter_context(patch(prefix+'telemetry.DualClocks',return_value=controller))
            stack.enter_context(patch(prefix+'service_client.BenchClient',return_value=Client()))
            stack.enter_context(patch(prefix+'service.model_id',return_value='model'))
            stack.enter_context(patch(prefix+'service.load_frames',new=AsyncMock(return_value=[])))
            stack.enter_context(patch(prefix+'service.restore_checked',side_effect=restore))
            stack.enter_context(patch(prefix+'workload',new=AsyncMock(side_effect=RuntimeError('sampling failure'))))
            stack.enter_context(patch(prefix+'service.emit'))
            with self.assertRaisesRegex(RuntimeError,'sampling failure'):
                await collect(args,dict(base_url='unused'),Path(folder),result)
        self.assertEqual(events,['set','reset','verify','close'])
        self.assertTrue(result['restoration']['verified'])


if __name__=='__main__': unittest.main()
