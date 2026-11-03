"""Import/path fixtures only; not a Triton kernel or vLLM compatibility claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from rtx5090.native import ROOT, configure_triton
from rtx5090.experiment import launch


class TritonOverrideTests(unittest.TestCase):
    def child(self, root, code):
        env=dict(os.environ)
        env['PYTHONPATH']=str(ROOT)
        env.pop('SPARSEDVFS_TRITON_PATH',None)
        return subprocess.run([sys.executable,'-B','-c',code,str(root)],env=env,capture_output=True,text=True)

    def fixture(self, root, version='3.6.0'):
        (root/'triton').mkdir()
        (root/'triton/__init__.py').write_text('__version__ = '+repr(version)+'\n')

    def test_actual_import_version_not_distribution_metadata_and_child_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            code='''
import json,sys,subprocess
from unittest.mock import patch
from rtx5090.native import configure_triton,configure_process,environment_record
configure_triton(sys.argv[1])
configure_process('model','GPU-T','build')
with patch('rtx5090.native.importlib.metadata.version',return_value='3.7.1'):
    parent=environment_record()['packages']['triton']
child=json.loads(subprocess.check_output([sys.executable,'-B','-c',
    'import json,os; from rtx5090.native import triton_record; print(json.dumps(triton_record(os.environ["SPARSEDVFS_TRITON_PATH"])))'],text=True))
print(json.dumps(dict(parent=parent,child=child)))
'''
            result=self.child(root,code)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(result.stdout)
            self.assertEqual(value['parent']['version'],'3.6.0')
            self.assertEqual(value['parent']['distribution_version'],'3.7.1')
            self.assertEqual(value['parent']['path'],value['child']['path'])
            self.assertEqual(value['parent']['package_sha256'],value['child']['package_sha256'])

    def test_wrong_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(Path(tmp),'3.7.1')
            result=self.child(tmp,'import sys; from rtx5090.native import configure_triton; configure_triton(sys.argv[1])')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Expected explicit Triton 3.6.0',result.stderr)

    def test_missing_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'triton/__init__.py'):configure_triton(tmp)

    def test_package_hash_changes_with_native_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            payload=root/'triton/libtriton.so';payload.write_bytes(b'fixture one')
            code='import json,sys; from rtx5090.native import configure_triton; print(json.dumps(configure_triton(sys.argv[1])))'
            first=json.loads(self.child(root,code).stdout)
            payload.write_bytes(b'fixture two')
            second=json.loads(self.child(root,code).stdout)
            self.assertNotEqual(first['package_sha256'],second['package_sha256'])

    def test_launcher_explicitly_forwards_override_to_window_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=NS(model=tmp,gpu_uuid='GPU-T',library_dir=tmp,dataset=tmp+'/data.json',baseline=tmp+'/base.json',
                    min_seconds=30,calibration=None,power_evidence=None,apply=False,ack_default_auto=False)
            def run(cmd,**kwargs):
                self.assertEqual(cmd[cmd.index('--triton-path')+1],'/chosen/triton360')
                (Path(tmp)/'result.json').write_text('{}')
            with patch.dict(os.environ,{'SPARSEDVFS_TRITON_PATH':'/chosen/triton360'}),\
                 patch('rtx5090.experiment.subprocess.run',side_effect=run),patch('builtins.print'):
                launch(args,tmp,mode='auto')
