import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from rtx5090.native import configure_process, assert_cuda_uuid, check_vllm_sources
from rtx5090.contracts import engine_args, validate_engine, verify_model


class CondaTests(unittest.TestCase):
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
        with patch.dict(os.environ, {'PYTHONPATH': '/tmp/triton-360-test'}):
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
