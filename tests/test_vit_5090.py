"""CPU control fixtures only. These do not establish 5090/CUDA/clock/energy acceptance."""
import ast
from contextlib import contextmanager
import csv
import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import SparseDVFS_5090 as m

UUID = m.DEFAULT_UUID
ROOT = Path(m.__file__).parent


def device(out, apply=True):
    result = m.NvidiaSMI.__new__(m.NvidiaSMI)
    result.uuid, result.out = UUID, Path(out)
    result.smi, result.powershell = "nvidia-smi.exe", "powershell.exe"
    result.allow_apply, result.dirty, result.last_target = apply, False, None
    return result


class ControlTests(unittest.TestCase):
    def tearDown(self):
        logging.shutdown()

    def test_parser_requires_explicit_apply_and_auto_ack(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()):
            for args in ([], ["--apply"], ["--check-only", "--apply"],
                         ["--check-only", "--uuid", "0"]):
                with self.assertRaises(SystemExit):
                    m.parse_args(args)
        self.assertTrue(m.parse_args(["--check-only"]).check_only)
        self.assertTrue(m.parse_args(["--apply", "--ack-default-auto"]).apply)

    def test_no_permission_to_write_means_no_command_even_for_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp, apply=False)
            d.command = Mock()
            with self.assertRaisesRegex(RuntimeError, "--apply"):
                d.set(2250)
            d.dirty = True
            with self.assertRaisesRegex(RuntimeError, "--apply"):
                d.reset()
            d.command.assert_not_called()

    def test_failed_clock_write_is_dirty_and_still_gets_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.run = Mock(side_effect=[RuntimeError("NVML not supported"), "reset"])
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                d.set(2250)
            self.assertTrue(d.dirty)
            self.assertTrue(d.reset())
            self.assertFalse(d.dirty)
            self.assertEqual(d.run.call_args_list[1].args, ("-rgc",))

    def test_candidate_union_does_not_use_only_idle_memory_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.read = Mock(return_value=dict(uuid=UUID, name="RTX 5090"))
            d.run = Mock(return_value="405, 210\n405, 885\n13801, 2250\n13801, 2527\n13801, 2250\n")
            self.assertEqual(d.probe()["rates_mhz"], [210, 885, 2250, 2527])
            d.run.return_value = "N/A, N/A\n"
            with self.assertRaises(RuntimeError):
                d.probe()

    def test_wrong_uuid_and_bad_power_are_not_fallback_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            for value in (f"GPU-WRONG, RTX 5090, 600.1, 2250, 13801, 150\n",
                          f"{UUID}, RTX 5090, 600.1, 2250, 13801, N/A\n",
                          f"{UUID}, RTX 5090, 600.1, 2250, 13801, nan\n"):
                d.run = Mock(return_value=value)
                with self.assertRaises(RuntimeError):
                    d.read()

    def test_command_retains_real_error_and_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            result = NS(returncode=3, stdout="Not Supported", stderr="Access denied")
            with patch.object(m.subprocess, "run", return_value=result) as run:
                with self.assertRaisesRegex(RuntimeError, "Access denied"):
                    d.run("-lgc", "2250,2250")
            self.assertEqual(run.call_args.args[0][:3], ["nvidia-smi.exe", "-i", UUID])
            record = json.loads((Path(tmp) / "commands.jsonl").read_text())
            self.assertEqual(record["returncode"], 3)
            self.assertIn("Not Supported", record["stdout"])

    def test_cuda_identity_rejects_index_or_property_mismatch(self):
        fake = NS(cuda=NS(is_available=lambda: True, device_count=lambda: 1,
                          get_device_properties=lambda _: NS(uuid="00000000-0000-0000-0000-000000000000")))
        with patch.object(m, "torch", fake):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                m.cuda_identity(UUID)

    def test_check_only_no_weights_no_write_no_reset_and_records_pending_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp, apply=False)
            d.probe = Mock(return_value=dict(uuid=UUID, name="RTX 5090", power_watt=100,
                                             rates_mhz=[2250], physical_restore_verified=False))
            d.ps = Mock(return_value="False\n")
            d.set, d.reset = Mock(), Mock()
            fake_torch = NS(hub=NS(get_dir=lambda: tmp))
            fake_vision = NS(models=NS(ViT_B_16_Weights=NS(
                IMAGENET1K_V1=NS(url="https://download.pytorch.org/models/vit_b_16-c867db91.pth")),
                vit_b_16=Mock(side_effect=AssertionError("must not load weights"))))
            with patch.object(m, "load_dependencies", return_value={}), \
                 patch.object(m, "cuda_identity", return_value=dict(uuid=UUID)), \
                 patch.object(m, "torch", fake_torch), patch.object(m, "torchvision", fake_vision):
                report = m.preflight(d)
            self.assertFalse(report["set_reset_verified"])
            self.assertFalse(report["real_cuda_workload_executed"])
            self.assertFalse(report["windows_admin"])
            fake_vision.models.vit_b_16.assert_not_called()
            d.set.assert_not_called()
            d.reset.assert_not_called()

    def test_main_check_only_never_creates_sampler_or_calls_restore_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp, apply=False)
            d.run = Mock()
            out = Path(tmp) / "new"
            report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=False, weights_cached=False)
            with patch.dict(os.environ), patch.object(m, "NvidiaSMI", return_value=d), \
                 patch.object(m, "preflight", return_value=report), \
                 patch.object(m, "PowerSampler") as sampler:
                self.assertEqual(m.main(["--check-only", "--out", str(out)]), 0)
            sampler.assert_not_called()
            d.run.assert_not_called()
            status = json.loads((out / "status.json").read_text())
            self.assertFalse(status["clock_write_attempted"])
            self.assertFalse(json.loads((out / "restore_status.json").read_text())["required"])

    def test_reset_return_is_not_physical_restore_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.dirty, d.last_target = True, 2527
            d.run = Mock(return_value="")
            d.read = Mock(return_value=dict(gpu_freq=210))
            with patch.object(m.time, "sleep"):
                result = m.restore(d, None, None, None, [])
            self.assertTrue(result["reset_returned"])
            self.assertFalse(result["physical_restore_verified"])

    def test_reset_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.dirty, d.last_target = True, 2250
            d.run = Mock(side_effect=RuntimeError("reset denied"))
            result = m.restore(d, None, None, None, [])
            self.assertFalse(result["reset_returned"])
            self.assertIn("reset denied", result["errors"][0])
            self.assertTrue(d.dirty)


class SamplerTests(unittest.TestCase):
    def make_sampler(self, tmp, row_uuid=UUID, power="100"):
        d = device(tmp)
        s = m.PowerSampler(d)
        def powershell(code):
            if "$p=Start-Process" in code:
                s.raw.write_text(
                    f"2026/11/11 10:00:00.001, {row_uuid}, 2250, 13801, {power}, 50, 90\n",
                    encoding="utf-8")
                return "4321\n"
            self.assertIn("CommandLine.Contains", code)
            self.assertIn(str(s.raw), code)
            self.assertIn("Stop-Process -Id $_.ProcessId", code)
            self.assertNotIn("Stop-Process -Name", code)
            return '{"stopped_pids":[4321],"verified":true}'
        d.ps = Mock(side_effect=powershell)
        return d, s

    def test_one_loop_process_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="vit space ") as tmp:
            d, s = self.make_sampler(tmp)
            with patch.object(m, "windows_path", side_effect=str):
                try:
                    s.start()
                    self.assertEqual(s.frames[-1]["power_watt"], 100)
                    launch = d.ps.call_args_list[0].args[0]
                    self.assertIn("-WindowStyle Hidden", launch)
                    self.assertIn("--loop-ms=20", launch)
                    self.assertIn("-PassThru", launch)
                finally:
                    s.close()
            self.assertEqual(d.ps.call_count, 2)
            self.assertTrue(json.loads((Path(tmp) / "power_sampler_cleanup.json").read_text())["verified"])
            self.assertFalse(s.thread.is_alive())

    def test_invalid_telemetry_stops_without_fake_power(self):
        for value, uuid in (("N/A", UUID), ("0", UUID), ("nan", UUID), ("100", "GPU-WRONG")):
            with self.subTest(value=value, uuid=uuid), tempfile.TemporaryDirectory() as tmp:
                _, s = self.make_sampler(tmp, uuid, value)
                with patch.object(m, "windows_path", side_effect=str):
                    try:
                        with self.assertRaises(RuntimeError):
                            s.start()
                    finally:
                        s.close()
                self.assertTrue(s.error)

    def test_no_samples_in_window_is_error(self):
        fake = NS(check=lambda: None, window=lambda *_: [])
        with patch.object(m, "SAMPLER", fake):
            monitor = m.PowerMonitor()
            monitor.start()
            with self.assertRaisesRegex(RuntimeError, "No valid board-power"):
                monitor.stop()

    def test_cleanup_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, s = self.make_sampler(tmp)
            with patch.object(m, "windows_path", side_effect=str):
                s.start()
                d.ps.side_effect = RuntimeError("Cannot stop owned PID")
                with self.assertRaisesRegex(RuntimeError, "owned PID"):
                    s.close()


class AuthorWorkflowTests(unittest.TestCase):
    def tearDown(self):
        logging.shutdown()

    def test_five_warmups_and_five_synchronized_measurements_with_original_energy(self):
        import numpy as np
        import pandas as pd
        events = []
        module = Mock(side_effect=lambda _: events.append("forward") or object())
        model = NS(named_modules=lambda: [("mlp", module)])
        class Progress:
            def update(self, *_): pass
            def close(self): pass
        class Monitor:
            begin_ns, end_ns = 1, 2
            samples = [dict(sample_index=1)]
            def __init__(self, sample_interval): self.interval = sample_interval
            def start(self): events.append("power_start")
            def stop(self): events.append("power_stop"); return 100
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.set = Mock()
            d.read = Mock(return_value=dict(uuid=UUID, gpu_freq=2250, memory_freq=13801))
            cuda = NS(cuda=NS(synchronize=lambda: events.append("sync")))
            times = iter(i * .001 for i in range(10))
            with patch.multiple(m, GPU=d, SAMPLER=NS(check=lambda: None), torch=cuda,
                                np=np, pd=pd, tqdm=lambda **_: Progress()), \
                 patch.object(m, "get_layer_input_tensor", return_value=object()), \
                 patch.object(m, "calculate_sparsity", return_value=.25), \
                 patch.object(m, "PowerMonitor", Monitor), patch.object(m.time, "sleep"), \
                 patch.object(m.time, "perf_counter", side_effect=lambda: next(times)):
                result = m.profile_model(model, "ViT-Base", "transformer", ["mlp"], [2250], object())
            self.assertEqual(module.call_count, 10)
            self.assertEqual(events[:7], ["sync"] + ["forward"] * 5 + ["sync"])
            self.assertEqual(events.count("sync"), 7)
            self.assertEqual(events[7], "power_start")
            self.assertAlmostEqual(result.iloc[0]["latency_ms"], 1)
            self.assertAlmostEqual(result.iloc[0]["energy_joule"], .1)
            with (Path(tmp) / "profiling_results_ViT-Base.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["cpu_freq"], "")
            self.assertEqual(rows[0]["gpu_uuid"], UUID)

    def test_main_exception_or_interrupt_restores_and_cleans_sampler(self):
        for problem in (RuntimeError("injected failure"), KeyboardInterrupt(), SystemExit(143)):
            with self.subTest(problem=type(problem).__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                d = device(out)
                d.run = Mock(return_value="")
                d.read = Mock(return_value=dict(gpu_freq=2800))
                tensor = NS(cuda=lambda: "input")
                model = NS(cuda=lambda: NS(eval=lambda: object()))
                fake_torch = NS(randn=lambda *_: tensor)
                fake_vision = NS(models=NS(vit_b_16=lambda **_: model,
                    ViT_B_16_Weights=NS(IMAGENET1K_V1="weights")))
                sampler = NS(start=lambda: None, close=Mock(), error=None)
                frames = [dict(gpu_freq=2800, utilization_pct=100)] * 3
                def fail(*_):
                    d.dirty, d.last_target = True, 2250
                    raise problem
                report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=True, weights_cached=True)
                with patch.dict(os.environ), \
                     patch.multiple(m, torch=fake_torch, torchvision=fake_vision), \
                     patch.object(m, "NvidiaSMI", return_value=d), \
                     patch.object(m, "PowerSampler", return_value=sampler), \
                     patch.object(m, "preflight", return_value=report), \
                     patch.object(m, "get_all_layers", return_value=["mlp"]), \
                     patch.object(m, "loaded_readbacks", return_value=frames), \
                     patch.object(m, "profile_model", side_effect=fail), \
                     patch.object(m.time, "sleep"):
                    code = m.main(["--apply", "--ack-default-auto", "--out", str(out)])
                self.assertNotEqual(code, 0)
                d.run.assert_called_once_with("-rgc")
                sampler.close.assert_called_once()
                restoration = json.loads((out / "restore_status.json").read_text())
                self.assertTrue(restoration["reset_returned"])
                self.assertTrue(restoration["physical_restore_verified"])
                self.assertEqual(json.loads((out / "status.json").read_text())["status"], "failed")


class SourceAndFailureTests(unittest.TestCase):
    def test_preserved_functions_match_pinned_author_source(self):
        # SHA-256 of verbatim function source at c524c51; no network during tests.
        expected = {
            "calculate_sparsity": "f2a713b2d6256013e0fd0d2359d9d6f1d4534da3689e7a82635f6b60d04ac026",
            "get_layer_input_tensor": "48b2fa53a9da54a923438ac70ccff8d3fb049c1fff02ff1cdda14366838f8e10",
            "get_all_layers": "66cf1446aec5539244d22849912aa1a0642432b3e57e917b72668b4dbcf38765",
            "setup_plot_style": "2c926b102d78022f3c75c73dc8012241f6c78b2a5c3351f85a649b30604f1160",
            "plot_sparsity_distribution": "a893fa9ea1b1e658b3d131b44edb9aad1eb6f1c8d61543f7965f2ebbc4eb0c06",
            "plot_sparsity_performance": "07c22413d3b362ef830b90067a94a8d58a48a4ad9c714c00e60d2be14091638f",
        }
        source = Path(m.__file__).read_text(encoding="utf-8")
        actual = {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                  for node in ast.parse(source).body
                  if isinstance(node, ast.FunctionDef) and node.name in expected}
        self.assertEqual(actual, expected)

    def test_existing_out_is_not_overwritten_and_starts_no_tool(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
             patch.object(m, "NvidiaSMI") as tool:
            with self.assertRaises(FileExistsError):
                m.main(["--check-only", "--out", tmp])
            tool.assert_not_called()

    def test_hung_reader_does_not_skip_windows_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            s = m.PowerSampler(d)
            s.marker = str(s.raw)
            s.cleanup["required"] = True
            s.thread = NS(join=Mock(), is_alive=lambda: True)
            d.ps = Mock(return_value='{"verified":true,"stopped_pids":[1]}')
            with self.assertRaisesRegex(RuntimeError, "thread did not stop"):
                s.close()
            d.ps.assert_called_once()
            self.assertTrue(s.cleanup["verified"])
            self.assertIn("reader_error", s.cleanup)

    def test_normal_restores_before_plots_and_restore_error_prevents_plots(self):
        import pandas as pd
        for reset_fails in (False, True):
            with self.subTest(reset_fails=reset_fails), tempfile.TemporaryDirectory() as tmp:
                out, events = Path(tmp) / "run", []
                d = device(out)
                def run(*_):
                    events.append("reset")
                    if reset_fails:
                        raise RuntimeError("injected reset failure")
                    return ""
                d.run = Mock(side_effect=run)
                d.read = Mock(return_value=dict(gpu_freq=2800))
                fake_torch = NS(randn=lambda *_: NS(cuda=lambda: "input"))
                fake_vision = NS(models=NS(
                    vit_b_16=lambda **_: NS(cuda=lambda: NS(eval=lambda: object())),
                    ViT_B_16_Weights=NS(IMAGENET1K_V1="weights")))
                sampler = NS(start=lambda: None, close=lambda: events.append("close"), error=None)
                frames = [dict(gpu_freq=2800, utilization_pct=100)] * 3
                def profile(*_):
                    d.dirty, d.last_target = True, 2250
                    events.append("profile")
                    return pd.DataFrame([dict(latency_ms=1)])
                report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=True, weights_cached=True)
                try:
                    with patch.dict(os.environ), \
                         patch.multiple(m, torch=fake_torch, torchvision=fake_vision), \
                         patch.object(m, "NvidiaSMI", return_value=d), \
                         patch.object(m, "PowerSampler", return_value=sampler), \
                         patch.object(m, "preflight", return_value=report), \
                         patch.object(m, "get_all_layers", return_value=["mlp"]), \
                         patch.object(m, "loaded_readbacks", return_value=frames), \
                         patch.object(m, "profile_model", side_effect=profile), \
                         patch.object(m.time, "sleep"), \
                         patch.object(m, "plot_sparsity_distribution", side_effect=lambda *_: events.append("plot")), \
                         patch.object(m, "plot_sparsity_performance"), patch.object(m, "plot_gpu_performance"):
                        code = m.main(["--apply", "--ack-default-auto", "--out", str(out)])
                    self.assertEqual(events[:3], ["profile", "reset", "close"])
                    self.assertEqual("plot" in events, not reset_fails)
                    self.assertEqual(code, 1 if reset_fails else 0)
                finally:
                    logging.shutdown()


if __name__ == "__main__":
    unittest.main()
