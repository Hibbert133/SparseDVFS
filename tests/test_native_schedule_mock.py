"""Actual C++ core + real Schedule; CPU-only serial host debug transport."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_core import fake_hardware
from test_virtual_chain import make_plan
from sparsedvfs.fuse import FUSE
from sparsedvfs.hardware import snapshot, read
from sparsedvfs.qwen_runner import Schedule
from sparsedvfs_vllm.contracts import UNITS

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless((ROOT/'build/libfuse_dvfs.so').exists(), 'Actual Linux C++ core required')
class NativeScheduleTests(unittest.TestCase):
    def test_four_steps_native_readback_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            hw=fake_hardware(directory)
            hw['mock_root']=directory
            before=snapshot(hw)
            fake=NS(cuda=NS(current_stream=lambda:NS(synchronize=lambda:None)))
            with FUSE(hw,Path(directory)/'logs',mock=True) as controller:
                schedule=Schedule(make_plan(),controller,'serial',transport='host')
                visits=0
                with patch.dict('sys.modules',{'torch':fake}):
                    for step,phase in enumerate(['prefill','decode','decode','decode']):
                        schedule.start_forward(phase,step)
                        for unit in UNITS:
                            block=schedule.mapping[(phase,unit)]
                            schedule.before(phase,unit)
                            # The native writer's requested constraints are read
                            # back from ordinary files, not physical clock telemetry.
                            self.assertEqual(int(read(Path(hw['gpu']['path'])/'min_freq')),block['triplet'][1])
                            self.assertEqual(int(read(Path(hw['emc']['path'])/'rate')),block['triplet'][2])
                            schedule.after(phase,unit)
                            if unit==block['operators'][-1]:
                                for domain in hw['cpu']:
                                    self.assertEqual(int(read(Path(domain['path'])/'scaling_min_freq')),block['triplet'][0])
                            visits+=1
                    schedule.finalize()
                self.assertEqual(visits,272)
                self.assertEqual(len([e for e in controller.events if e['kind']=='boundary']),16)
            after=snapshot(hw)
            self.assertEqual(before['domains'],after['domains'])
            self.assertEqual(before['bpmp'],after['bpmp'])
