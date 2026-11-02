"""CPU-only integration of the paper core and retained target contracts."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from test_v3_contracts import valid_plan
from sparsedvfs.hardware import write_json
from sparsedvfs.graph_partitioner import build_plan
from sparsedvfs_vllm.contracts import PROTOCOL, validate_plan
from sparsedvfs_vllm.host import plan_command, software
import paper_demo


class Integration(unittest.TestCase):
    def test_schema4_calibration_and_shared_algorithm_reach_the_plan_builder(self):
        data = paper_demo.fixture()
        plan = build_plan(data['profile'], 100, factor=5, tolerance=.05, max_slowdown=.05,
                          selector='paper', calibration=data['calibration'], allow_synthetic=True, memory_policy='paper')
        self.assertEqual(plan['calibration_schema'], 4)
        self.assertEqual(plan['equation_model'], 'schema4_argument_dependencies')
        self.assertEqual(len(plan['blocks']), 4)
        self.assertFalse(plan['blocks'][-1]['amortized_strict_gt'])
        self.assertIn('partition_decisions', plan)
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            validate_plan(plan)

    def test_target_plan_requires_explicit_tradeoff_and_similarity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / 'scenario_512_128'
            write_json(case / 'profile.json', {'schema': 5, 'granularity': 'module', 'identity': {'backend': PROTOCOL}})
            args = NS(results=root, input_tokens=512, output_tokens=128,
                      selector=None, max_slowdown=None, similarity_tolerance=None, granularity='module')
            with self.assertRaisesRegex(ValueError, 'explicitly'):
                plan_command(args)
            self.assertFalse((case / 'plan.json').exists())

    def test_target_paper_path_rejects_legacy_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / 'scenario_512_128'
            write_json(case / 'profile.json', {'schema': 5, 'granularity': 'module', 'identity': {'backend': PROTOCOL}})
            write_json(root / 'old.json', {'schema': 3, 'operators': {}})
            args = NS(results=root, input_tokens=512, output_tokens=128,
                      selector='paper', max_slowdown=.05, similarity_tolerance=.05, calibration=str(root / 'old.json'), granularity='module')
            with self.assertRaisesRegex(ValueError, 'schema-4'):
                plan_command(args)
            self.assertFalse((case / 'plan.json').exists())

    def test_old_worker_protocol_and_calibration_are_not_deployable(self):
        plan = valid_plan()
        plan['identity']['backend'] = 'v3_vllm019_scheduler_single_request_main_stream'
        with self.assertRaisesRegex(ValueError, 'real v5'):
            validate_plan(plan)
        plan = valid_plan()
        plan.update(selector='paper', calibration_schema=3, calibration={'schema': 3})
        with self.assertRaisesRegex(ValueError, 'schema-4'):
            validate_plan(plan)

    def test_v3_prepared_image_metadata_is_rejected_before_docker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'prepare.json', {'adapter_version': 3})
            with self.assertRaisesRegex(RuntimeError, 'not a v5'):
                software(NS(results=root))


if __name__ == '__main__':
    unittest.main()
