import copy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from rtx5090.eager_matrix_replay import case_spec, validate_rows, MatrixReplay
from rtx5090.eager import parser


def meta(shape):
    return dict(tensor=True, shape=shape, stride=[shape[1], 1], dtype='torch.bfloat16',
                device='cuda:0', layout='torch.strided')


class ReplayTests(unittest.TestCase):
    def test_shapes(self):
        for phase, m in [('prefill', 512), ('decode', 1)]:
            for module, n, k in [('gate_up_proj', 17408, 5120), ('down_proj', 5120, 8704)]:
                row = case_spec('model.layers.0.mlp.'+module, phase, 'aten.linear.default',
                                [meta([m,k]), meta([n,k])], meta([m,n]), 0)
                self.assertEqual(row['work_flop'], 2*m*n*k)

    def test_reject_layout_and_bias(self):
        x, w, y = meta([1, 5120]), meta([17408, 5120]), meta([1, 17408])
        with self.assertRaises(ValueError):
            case_spec('x.mlp.gate_up_proj', 'decode', 'aten.linear.default', [x,w,None], y, 0)
        w['stride'] = [1, 17408]
        with self.assertRaises(ValueError):
            case_spec('x.mlp.gate_up_proj', 'decode', 'aten.linear.default', [x,w], y, 0)

    def test_non_target(self):
        self.assertIsNone(case_spec('attention', 'decode', 'aten.linear.default', [], {}, 0))
        self.assertIsNone(case_spec('x.mlp.down_proj', 'decode', 'aten.mm.default', [], {}, 0))

    def test_cli_excludes_profiler(self):
        args = ['observe', '--identity', 'id', '--out', 'out', '--seed', '1', '--matrix-replay']
        self.assertTrue(parser().parse_args(args).matrix_replay)

    def test_reuses_input_returns_original_and_measures_once(self):
        class Tensor:
            dtype, device, layout = 'torch.bfloat16', 'cuda:0', 'torch.strided'
            def __init__(self, shape): self.shape = shape
            def stride(self): return (self.shape[1], 1)
        x, w = Tensor((1,5120)), Tensor((17408,5120))
        original = Tensor((1,17408))
        class Linear:
            count = 0
            def __str__(self): return 'aten.linear.default'
            def __call__(self, a, b):
                self.count += 1
                assert a is x and b is w
                return original if self.count == 1 else Tensor((1,17408))
        stream = Mock()
        event = lambda **kw: NS(record=Mock(), synchronize=Mock(), elapsed_time=lambda other: 32.)
        torch = NS(Tensor=Tensor, cuda=NS(current_stream=lambda:stream,
            is_current_stream_capturing=lambda:False, Event=event))
        replay = MatrixReplay(NS(named_modules=lambda:[]), 0, lambda:{'phase':'decode'})
        replay.parents = ['model.layers.0.mlp.gate_up_proj']
        func = Linear()
        with patch.dict(sys.modules, {'torch':torch}):
            result = replay.dispatch(func, (x,w), {})
            self.assertIs(result, original)
            self.assertEqual(func.count, 97)
            replay.dispatch(func, (x,w), {})
        self.assertEqual(func.count, 98)
        self.assertEqual(len(replay.rows), 1)
        self.assertEqual(len(replay.rows[0]['batches']), 3)
        self.assertFalse(replay.rows[0]['calibration_ready'])
        self.assertEqual(replay.rows[0]['batches'][0]['mean_call_s'], .001)

    def test_incomplete_and_bad_timings(self):
        rows = [dict(phase=p, module=m, batches=[dict(repeats=32, elapsed_s=.1)]*3)
                for p in ('prefill','decode') for m in ('gate_up_proj','down_proj')]
        validate_rows(rows)
        with self.assertRaises(ValueError): validate_rows(rows[:-1])
        for bad in (float('nan'), -1, 0):
            altered = copy.deepcopy(rows); altered[0]['batches'][0]['elapsed_s'] = bad
            with self.assertRaises(ValueError): validate_rows(altered)


if __name__ == '__main__': unittest.main()
