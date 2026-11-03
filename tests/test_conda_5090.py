import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import hashlib
from rtx5090.native import configure_process, assert_cuda_uuid, check_vllm_sources
from rtx5090.contracts import engine_args, validate_engine, verify_model


class CondaTests(unittest.TestCase):
    def test_source_fingerprint_rejects_patched_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'runner.py').write_text('audited')
            expected={'runner.py':hashlib.sha256(b'audited').hexdigest()}
            with patch('rtx5090.native.importlib.metadata.version',return_value='0.32.0+cu129'),\
                 patch('rtx5090.native.importlib.util.find_spec',return_value=NS(origin=str(root/'__init__.py'))),\
                 patch('rtx5090.native.V032_SOURCES',expected):
                self.assertEqual(check_vllm_sources(),expected)
                (root/'runner.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'source differs'):check_vllm_sources()

    def test_missing_explicit_library_fails_without_fallback(self):
        from rtx5090.controller import Controller
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'SPARSEDVFS_LIBRARY_DIR':tmp}):
            with self.assertRaises(OSError):
                Controller(dict(backend='mock',uuid='GPU-T',rates_mhz=[900,1500]),Path(tmp)/'trace')

    def test_failed_restore_cannot_publish_calibration(self):
        from rtx5090.run import calibrate
        class Failed:
            rates=[900,1500]
            def __init__(self,*a):self.backend=NS(synthetic=True,read=lambda:self.rate)
            def __enter__(self):return self
            def __exit__(self,*a):raise RuntimeError('reset failed')
            def set(self,vector):self.rate=vector[1]//1000000
        with tempfile.TemporaryDirectory() as tmp,patch('rtx5090.controller.Controller',Failed):
            target=Path(tmp)/'calibration.json'
            with self.assertRaisesRegex(RuntimeError,'reset failed'):
                calibrate(dict(backend='mock'),target)
            self.assertFalse(target.exists())

    def test_paths_and_process_env(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_process('/some/model', 'GPU-example', '/some/libs')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-example')
            self.assertEqual(os.environ['VLLM_USE_V2_MODEL_RUNNER'], '0')
            self.assertIn('SPARSEDVFS_LIBRARY_DIR', os.environ)
        args = engine_args('/some/model')
        self.assertEqual(args['model'], args['tokenizer'])
        self.assertEqual(args['tensor_parallel_size'], 1)

    def test_temporary_triton_and_multi_gpu_rejected(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/tmp/triton-360-test', 'SPARSEDVFS_TRITON_PATH': ''}):
            with self.assertRaisesRegex(RuntimeError, 'Temporary Triton'):
                configure_process('model', 'GPU-example', 'build')
        with self.assertRaises(ValueError):
            configure_process('model', 'GPU-a,GPU-b', 'build')

    def test_uuid_binding(self):
        t = NS(cuda=NS(device_count=lambda: 1, get_device_properties=lambda n: NS(uuid='example')))
        self.assertEqual(assert_cuda_uuid(t, 'GPU-example'), 'GPU-example')
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            assert_cuda_uuid(t, 'GPU-other')
        t.cuda.device_count = lambda: 2
        with self.assertRaises(RuntimeError): assert_cuda_uuid(t, 'GPU-example')

    def test_v032_contract_keeps_real_checks(self):
        cfg = NS(parallel_config=NS(), model_config=NS(enforce_eager=True,
                 quantization='compressed-tensors', dtype='torch.bfloat16', hf_text_config=NS(num_hidden_layers=64)),
                 scheduler_config=NS(max_num_seqs=1, enable_chunked_prefill=False, async_scheduling=False),
                 cache_config=NS(enable_prefix_caching=False), speculative_config=None)
        validate_engine(cfg, '0.32.0+cu129', False)
        with self.assertRaisesRegex(RuntimeError, 'V1'): validate_engine(cfg, '0.32.0', True)
        cfg.model_config.quantization = None
        with self.assertRaisesRegex(RuntimeError, 'INT4'): validate_engine(cfg, '0.32.0')

    def test_unlocked_weights_and_unknown_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): verify_model(tmp)
        with patch('rtx5090.native.importlib.metadata.version', return_value='0.33.0'):
            with self.assertRaisesRegex(RuntimeError, 'audited'): check_vllm_sources()
