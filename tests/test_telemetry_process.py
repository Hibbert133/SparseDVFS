"""Actual process-lifecycle CPU checks; fixtures do not certify hardware."""
import os
import time
import unittest
from unittest.mock import Mock, patch
from rtx5090 import telemetry_process as tp


def process_fixture(connection, config, interval):
    if config == 'fail':
        connection.send(dict(kind='finished', pid=os.getpid(), frames=[], errors=['fixture failure']))
    else:
        frames = [dict(at=time.monotonic_ns())]
        connection.send(dict(kind='ready', pid=os.getpid()))
        connection.recv()
        frames.append(dict(at=time.monotonic_ns()))
        connection.send(dict(kind='finished', pid=os.getpid(), frames=frames, errors=[]))
    connection.close()


class ProcessTests(unittest.TestCase):
    def test_spawn_post_window_frame_and_cleanup(self):
        with patch.object(tp, '_sampling_process', process_fixture):
            sampler = tp.ProcessSampler([])
            sampler.start()
            end = time.monotonic_ns()
            sampler.finish()
        self.assertNotEqual(sampler.child_pid, os.getpid())
        self.assertLessEqual(sampler.frames[0]['at'], end)
        self.assertGreaterEqual(sampler.frames[-1]['at'], end)
        self.assertFalse(sampler.thread.is_alive())
        self.assertEqual(sampler.errors, [])
        self.assertEqual(sampler.finish(), sampler.frames)

    def test_child_failure_before_ready_is_not_fallback(self):
        with patch.object(tp, '_sampling_process', process_fixture):
            sampler = tp.ProcessSampler('fail')
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                sampler.start()
        self.assertFalse(sampler.thread.is_alive())
        self.assertEqual(sampler.frames, [])
        self.assertIn('fixture failure', sampler.errors)

    def test_worker_closes_devices_without_clock_writes(self):
        devices = [Mock(), Mock()]
        sampler = Mock(frames=[['fixture']], errors=[], thread=object())
        connection = Mock()
        connection.recv.return_value = 'finish'
        with patch.object(tp, 'open_devices', return_value=devices), \
             patch.object(tp, 'Sampler', return_value=sampler):
            tp._sampling_process(connection, [], .1)
        sampler.start.assert_called_once()
        sampler.finish.assert_called_once()
        for device in devices:
            device.close.assert_called_once()
            device.set.assert_not_called()
            device.reset.assert_not_called()
        sent = [call.args[0] for call in connection.send.call_args_list]
        self.assertEqual([m['kind'] for m in sent], ['ready', 'finished'])
        self.assertEqual(sent[-1]['errors'], [])

    def test_token_redaction(self):
        with patch.dict(os.environ, SPARSEDVFS_BRIDGE_TOKEN='fixture-secret'):
            self.assertNotIn('fixture-secret', tp._error(ValueError('fixture-secret')))
