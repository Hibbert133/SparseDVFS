import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from rtx5090.worker_primitives import cases,collect
from rtx5090.service_core import read
from rtx5090.eager import parser

class PrimitiveTests(unittest.TestCase):
    def test_split_shapes_disjoint_and_bounded(self):
        rows=cases();self.assertEqual(len(rows),12)
        self.assertEqual(len({r['id'] for r in rows}),12)
        self.assertFalse({r['n'] for r in rows if r['role']=='train'} & {r['n'] for r in rows if r['role']=='holdout'})
        self.assertLess(max(r['n']**2*2*4 for r in rows),4*2**20)
    def test_no_memory_preserves_failure(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,mem_get_info=lambda:(0,0)))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'32 MiB'):
                collect(fake,0,{'gpu_uuid':'test'},root)
            report=read(Path(root)/'rank-0-independent-primitives.json')
            self.assertEqual(report['status'],'failed');self.assertFalse(report['clock_writes'])
            self.assertEqual(report['rows'],[])
    def test_public_cli(self):
        args=parser().parse_args(['observe','--identity','i','--out','o','--seed','123',
                                  '--independent-primitives','--apply'])
        self.assertTrue(args.independent_primitives);self.assertTrue(args.apply)
