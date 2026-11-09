import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock,AsyncMock
from contextlib import ExitStack
from rtx5090.transition_probe import in_band,probe
from rtx5090.observe_core import GPU_UUIDS


class BandTests(unittest.TestCase):
    def test_both_loaded_and_in_band(self):
        bands={u:[2400,2500] for u in GPU_UUIDS}
        rows=[dict(uuid=u,end_ns=1,raw=dict(clock_mhz=2450,utilization_pct=50,errors={})) for u in GPU_UUIDS]
        self.assertTrue(in_band(rows,bands))
        self.assertFalse(in_band(rows[:1],bands))
        rows[1]['raw']['utilization_pct']=0
        self.assertFalse(in_band(rows,bands))
        rows[1]['raw']['utilization_pct']=50; rows[1]['raw']['clock_mhz']=2200
        self.assertFalse(in_band(rows,bands))


class RestoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_failure_restores(self):
        events=[]
        devices=[]
        for u in GPU_UUIDS:
            d=Mock(uuid=u)
            d.set.side_effect=lambda value: events.append('set')
            d.reset.side_effect=lambda: events.append('reset')
            d.close.side_effect=lambda: events.append('close')
            devices.append(d)
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self,*a): pass
        async def restored(controller,*a):
            events.append('verify'); controller.restore_result['verified']=True
        with tempfile.TemporaryDirectory() as temp,ExitStack() as s:
            prefix='rtx5090.transition_probe.'
            for name,value in [('read',[dict(backend='bridge')]*2),('telemetry.open_devices',devices),
                               ('service_client.BenchClient',Client()),('service.model_id','model'),('service.prepared',[])]:
                s.enter_context(patch(prefix+name,return_value=value))
            s.enter_context(patch(prefix+'service.load_frames',new=AsyncMock(return_value=[])))
            s.enter_context(patch(prefix+'service.restore_checked',side_effect=restored))
            s.enter_context(patch(prefix+'telemetry.sample_device',side_effect=RuntimeError('query failed')))
            result=dict(replies=[],transitions=[])
            args=SimpleNamespace(telemetry='unused',seed=1)
            with self.assertRaises(RuntimeError):
                await probe(args,dict(base_url='unused'),Path(temp),result,{'2250':{u:[2100,2250] for u in GPU_UUIDS}})
            self.assertTrue(result['restoration']['verified'])
            self.assertEqual(events,['set','set','reset','reset','verify','close','close'])


if __name__=='__main__': unittest.main()
