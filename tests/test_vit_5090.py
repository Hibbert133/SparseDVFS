"""CPU NVML fixtures and real sampling threads; no target GPU acceptance."""
import ast
import csv
import hashlib
import io
import json
import logging
import os
import subprocess
import threading
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


def fake_telemetry(**overrides):
    fields = dict(open=Mock(), close=Mock(), stop_monitors=Mock(return_value=[]), monitors=set())
    fields.update(overrides)
    return NS(**fields)


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

    def test_timeout_keeps_partial_output_without_encoded_command_in_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            command = ["powershell.exe", "-EncodedCommand", "LONG_ENCODED_COMMAND"]
            problem = subprocess.TimeoutExpired(command, 20, output=b"partial PID", stderr=b"real error")
            with patch.object(m.subprocess, "run", side_effect=problem):
                with self.assertRaises(subprocess.TimeoutExpired):
                    d.command(command)
            record = json.loads((Path(tmp) / "commands.jsonl").read_text())
            self.assertEqual(record["stdout"], "partial PID")
            self.assertEqual(record["stderr"], "real error")
            self.assertEqual(record["timeout_seconds"], 20)
            self.assertNotIn("LONG_ENCODED_COMMAND", m.error_message(problem))

    def test_sampler_start_failure_prevents_model_and_frequency_writes(self):
        for problem in (RuntimeError("no samples"), KeyboardInterrupt()):
            with self.subTest(problem=type(problem).__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                d = device(out)
                d.run = Mock()
                sampler = fake_telemetry(open=Mock(side_effect=problem))
                report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=True, weights_cached=True)
                with patch.dict(os.environ), patch.object(m, "NvidiaSMI", return_value=d), \
                     patch.object(m, "preflight", return_value=report), \
                     patch.object(m, "check_power_windows", return_value={"windows_passed": 20}), \
                     patch.object(m, "NvmlReader", return_value=sampler), \
                     patch.object(m, "load_vit") as load:
                    code = m.main(["--apply", "--ack-default-auto", "--out", str(out)])
                self.assertNotEqual(code, 0)
                sampler.close.assert_called_once()
                load.assert_not_called()
                d.run.assert_not_called()
                self.assertIn("Traceback", (out / "errors.log").read_text())
                self.assertFalse(json.loads((out / "status.json").read_text())["clock_write_attempted"])

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

    def test_main_check_only_checks_sampler_without_model_or_clock_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp, apply=False)
            d.run = Mock()
            out = Path(tmp) / "new"
            report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=False, weights_cached=False)
            sampler = fake_telemetry()
            with patch.dict(os.environ), patch.object(m, "NvidiaSMI", return_value=d), \
                 patch.object(m, "preflight", return_value=report), \
                     patch.object(m, "check_power_windows", return_value={"windows_passed": 20}), \
                 patch.object(m, "NvmlReader", return_value=sampler) as factory, \
                 patch.object(m, "load_vit") as model:
                self.assertEqual(m.main(["--check-only", "--out", str(out)]), 0)
            factory.assert_called_once_with(UUID, out.resolve())
            sampler.open.assert_called_once()
            sampler.close.assert_called_once()
            model.assert_not_called()
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
            def __init__(self, sample_interval, context): self.interval = sample_interval
            def start(self): events.append("power_start")
            def stop(self): events.append("power_stop"); return 100
        with tempfile.TemporaryDirectory() as tmp:
            d = device(tmp)
            d.set = Mock(side_effect=lambda _: events.append("set"))
            # A pre-warmup idle clock must not filter the requested candidate.
            def read():
                events.append("read")
                return dict(uuid=UUID, gpu_freq=210, memory_freq=13801)
            d.read = Mock(side_effect=read)
            cuda = NS(cuda=NS(synchronize=lambda: events.append("sync")))
            times = iter(i * .001 for i in range(10))
            with patch.multiple(m, GPU=d, torch=cuda,
                                np=np, pd=pd, tqdm=lambda **_: Progress()), \
                 patch.object(m, "get_layer_input_tensor", return_value=object()), \
                 patch.object(m, "calculate_sparsity", return_value=.25), \
                 patch.object(m, "PowerMonitor", Monitor), patch.object(m.time, "sleep"), \
                 patch.object(m.time, "perf_counter", side_effect=lambda: next(times)), \
                 self.assertLogs(m.logger, level="INFO") as logs:
                result = m.profile_model(model, "ViT-Base", "transformer", ["mlp"], [2250], object())
            self.assertEqual(module.call_count, 10)
            self.assertEqual(events[:9], ["set", "read", "sync"] + ["forward"] * 5 + ["sync"])
            self.assertEqual(events.count("sync"), 7)
            self.assertEqual(events[9], "power_start")
            d.read.assert_called_once()
            self.assertEqual(result.iloc[0]["gpu_freq"], 210)
            self.assertEqual(result.iloc[0]["requested_gpu_freq"], 2250)
            self.assertAlmostEqual(result.iloc[0]["latency_ms"], 1)
            self.assertAlmostEqual(result.iloc[0]["energy_joule"], .1)
            with (Path(tmp) / "profiling_results_ViT-Base.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["cpu_freq"], "")
            self.assertEqual(rows[0]["gpu_uuid"], UUID)
            messages = "\n".join(logs.output)
            expected = ["Initial state stabilization complete", "Preparing input tensor",
                        "[1/1] Profiling:", "Configuration set.", "Measurement successful:", "Sparsity:"]
            positions = [messages.index(message) for message in expected]
            self.assertEqual(positions, sorted(positions))

    def test_compute_failures_continue_but_control_and_telemetry_failures_stop(self):
        import numpy as np
        import pandas as pd
        for problem in ("warmup", "measurement", "control", "telemetry", "interrupt", "measurement_interrupt"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                d = device(tmp)
                d.set = Mock(side_effect=RuntimeError("denied") if problem == "control" else None)
                d.read = Mock(return_value=dict(gpu_freq=210, memory_freq=13801))
                calls = 0
                def forward(_):
                    nonlocal calls
                    calls += 1
                    if calls == (6 if problem in ("measurement", "measurement_interrupt") else 1):
                        if problem in ("interrupt", "measurement_interrupt"):
                            raise KeyboardInterrupt()
                        if problem in ("warmup", "measurement"):
                            raise RuntimeError("layer failed")
                    return object()
                model = NS(named_modules=lambda: [("mlp", forward)])
                monitor = NS(start=lambda: None, begin_ns=1, end_ns=2,
                             samples=[dict(sample_index=1)],
                             stop=Mock(side_effect=RuntimeError("sampling failed")
                                       if problem == "telemetry" else None, return_value=100))
                with patch.multiple(m, GPU=d, np=np, pd=pd,
                                    torch=NS(cuda=NS(synchronize=lambda: None)),
                                    tqdm=lambda **_: NS(update=lambda *_: None, close=lambda: None)), \
                     patch.object(m, "get_layer_input_tensor", return_value=object()), \
                     patch.object(m, "calculate_sparsity", return_value=.25), \
                     patch.object(m, "PowerMonitor", return_value=monitor), \
                     patch.object(m.time, "sleep"):
                    if problem in ("warmup", "measurement"):
                        result = m.profile_model(model, "ViT-Base", "transformer",
                                                 ["mlp"], [210, 2250], object())
                        self.assertEqual(len(result), 1)
                        self.assertEqual(result.iloc[0]["requested_gpu_freq"], 2250)
                        self.assertEqual(d.set.call_count, 2)
                    else:
                        with self.assertRaises(KeyboardInterrupt if problem in ("interrupt", "measurement_interrupt") else RuntimeError):
                            m.profile_model(model, "ViT-Base", "transformer",
                                            ["mlp"], [210, 2250], object())
                        self.assertEqual(d.set.call_count, 1)
                self.assertEqual(monitor.stop.call_count,
                                 2 if problem == "measurement" else
                                 1 if problem in ("warmup", "telemetry", "measurement_interrupt") else 0)
                failures = [json.loads(line) for line in
                            (Path(tmp) / "failures.jsonl").read_text().splitlines()]
                self.assertEqual(len(failures), 1)

    def test_author_vit_fallback_only_after_primary_failure(self):
        model = NS(cuda=lambda: NS(eval=lambda: "vit"))
        for primary_fails, fallback_fails in ((False, False), (True, False), (True, True)):
            with self.subTest(primary=primary_fails, fallback=fallback_fails):
                primary = Mock(return_value=model,
                               side_effect=RuntimeError("primary failed") if primary_fails else None)
                hub = Mock(return_value=model,
                           side_effect=RuntimeError("hub failed") if fallback_fails else None)
                with patch.multiple(m, torch=NS(hub=NS(load=hub)),
                                    torchvision=NS(models=NS(vit_b_16=primary,
                                        ViT_B_16_Weights=NS(IMAGENET1K_V1="weights")))):
                    if fallback_fails:
                        with self.assertRaisesRegex(RuntimeError, "hub failed"):
                            m.load_vit()
                    else:
                        self.assertEqual(m.load_vit(), "vit")
                primary.assert_called_once_with(weights="weights")
                if primary_fails:
                    hub.assert_called_once_with('pytorch/vision:v0.10.0', 'vit_b_16', pretrained=True)
                else:
                    hub.assert_not_called()

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
                sampler = fake_telemetry()
                frames = [dict(gpu_freq=2800, workload_context="after_synchronized_forward")] * 3
                def fail(*_):
                    d.dirty, d.last_target = True, 2250
                    raise problem
                report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=True, weights_cached=True)
                with patch.dict(os.environ), \
                     patch.multiple(m, torch=fake_torch, torchvision=fake_vision), \
                     patch.object(m, "NvidiaSMI", return_value=d), \
                     patch.object(m, "NvmlReader", return_value=sampler), \
                     patch.object(m, "preflight", return_value=report), \
                     patch.object(m, "check_power_windows", return_value={"windows_passed": 20}), \
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
                sampler = fake_telemetry(close=lambda: events.append("close"))
                frames = [dict(gpu_freq=2800, workload_context="after_synchronized_forward")] * 3
                def profile(*_):
                    d.dirty, d.last_target = True, 2250
                    events.append("profile")
                    return pd.DataFrame([dict(latency_ms=1)])
                report = dict(gpu=dict(rates_mhz=[2250]), windows_admin=True, weights_cached=True)
                try:
                    with patch.dict(os.environ), \
                         patch.multiple(m, torch=fake_torch, torchvision=fake_vision), \
                         patch.object(m, "NvidiaSMI", return_value=d), \
                         patch.object(m, "NvmlReader", return_value=sampler), \
                         patch.object(m, "preflight", return_value=report), \
                     patch.object(m, "check_power_windows", return_value={"windows_passed": 20}), \
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


def nvml_fixture(tmp, power_mw=67125, returned_uuid=UUID):
    def handle(uuid, output):
        output._obj.value = 1234
        return 0
    def uuid(handle, output, length):
        output.value = returned_uuid.encode("ascii")
        return 0
    def power(handle, output):
        output._obj.value = power_mw
        return 0
    def clock(handle, kind, output):
        output._obj.value = 2527
        return 0
    lib = NS(nvmlInit_v2=Mock(return_value=0), nvmlShutdown=Mock(return_value=0),
             nvmlErrorString=Mock(return_value=b"Not Supported"),
             nvmlDeviceGetHandleByUUID=Mock(side_effect=handle), nvmlDeviceGetUUID=Mock(side_effect=uuid),
             nvmlDeviceGetPowerUsage=Mock(side_effect=power), nvmlDeviceGetClockInfo=Mock(side_effect=clock))
    reader = m.NvmlReader(UUID, tmp)
    return reader, lib


class NvmlTests(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARNING, force=True)

    def test_uuid_units_timestamps_and_single_initialization_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, lib = nvml_fixture(tmp)
            with patch.object(m.ctypes, "CDLL", return_value=lib) as load:
                reader.open()
                for _ in range(3):
                    frame = reader.read_power()
                    self.assertEqual(frame["power_watt"], 67.125)
                    self.assertLessEqual(frame["query_begin_monotonic_ns"], frame["query_end_monotonic_ns"])
                self.assertEqual(reader.read_clock()["gpu_freq"], 2527)
                reader.close()
                reader.close()
            load.assert_called_once_with("/usr/lib/wsl/lib/libnvidia-ml.so.1")
            lib.nvmlInit_v2.assert_called_once()
            lib.nvmlShutdown.assert_called_once()
            self.assertEqual(lib.nvmlDeviceGetHandleByUUID.call_args.args[0], UUID.encode("ascii"))

    def test_mismatch_and_query_errors_never_become_fake_power(self):
        for problem in ("uuid", "zero", "unsupported"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                reader, lib = nvml_fixture(tmp, power_mw=0 if problem == "zero" else 67125,
                                           returned_uuid="GPU-WRONG" if problem == "uuid" else UUID)
                if problem == "unsupported":
                    lib.nvmlDeviceGetPowerUsage.side_effect = None
                    lib.nvmlDeviceGetPowerUsage.return_value = 3
                with patch.object(m.ctypes, "CDLL", return_value=lib):
                    try:
                        with self.assertRaises(RuntimeError):
                            reader.open()
                            reader.read_power()
                    finally:
                        reader.close()
                lib.nvmlShutdown.assert_called_once()

    def test_missing_library_has_no_command_or_backend_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = m.NvmlReader(UUID, tmp)
            with patch.object(m.ctypes, "CDLL", side_effect=OSError("missing WSL NVML")), \
                 patch.object(m.subprocess, "run") as command:
                with self.assertRaisesRegex(OSError, "missing WSL"):
                    reader.open()
                reader.close()
            command.assert_not_called()

    def test_shutdown_error_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, lib = nvml_fixture(tmp)
            with patch.object(m.ctypes, "CDLL", return_value=lib):
                reader.open()
            lib.nvmlShutdown.return_value = 3
            with self.assertRaisesRegex(RuntimeError, "nvmlShutdown"):
                reader.close()
            self.assertFalse(json.loads((Path(tmp) / "power_sampler_cleanup.json").read_text())["verified"])


class ThreadSamplingTests(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARNING, force=True)
        import numpy as np
        self.enterContext(patch.object(m, "np", np))

    def reader(self, tmp):
        reader, lib = nvml_fixture(tmp)
        with patch.object(m.ctypes, "CDLL", return_value=lib):
            reader.open()
        return reader, lib

    def test_twenty_real_short_windows_no_subprocess_or_file_tailing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, lib = self.reader(tmp)
            with patch.object(m.subprocess, "Popen", side_effect=AssertionError("No sampling subprocess")):
                report = m.check_power_windows(reader)
            self.assertEqual(report["windows_passed"], 20)
            self.assertEqual(len(report["window_sample_counts"]), 20)
            self.assertTrue(all(n > 0 for n in report["window_sample_counts"]))
            self.assertEqual(reader.monitors, set())
            windows = [json.loads(s) for s in (Path(tmp) / "measurement_windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(windows), 20)
            self.assertTrue(all(w["thread_joined"] and w["error"] is None for w in windows))
            reader.close()

    def test_delayed_file_save_does_not_shift_or_feed_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, _ = self.reader(tmp)
            monitor = m.PowerMonitor(telemetry=reader)
            monitor.start()
            time.sleep(.1)
            self.assertFalse((Path(tmp) / "power_samples.csv").exists())
            self.assertGreater(len(monitor.samples), 0)
            original = monitor.save
            def slow_save():
                self.assertFalse(monitor.is_alive())
                count = len(monitor.samples)
                time.sleep(.15)
                self.assertEqual(len(monitor.samples), count)
                original()
            with patch.object(monitor, "save", side_effect=slow_save):
                self.assertAlmostEqual(monitor.stop(), 67.125)
            self.assertLess(monitor.end_ns - monitor.begin_ns, 200_000_000)
            self.assertTrue((Path(tmp) / "power_samples.csv").exists())
            reader.close()

    def test_stop_keeps_query_already_in_flight_with_honest_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, lib = self.reader(tmp)
            entered, release = threading.Event(), threading.Event()
            original = lib.nvmlDeviceGetPowerUsage.side_effect
            def slow_query(*args):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test release missing")
                return original(*args)
            lib.nvmlDeviceGetPowerUsage.side_effect = slow_query
            monitor = m.PowerMonitor(telemetry=reader)
            monitor.start()
            self.assertTrue(entered.wait(1))
            def unblock():
                monitor._stop_event.wait(1)
                release.set()
            helper = threading.Thread(target=unblock)
            helper.start()
            try:
                self.assertAlmostEqual(monitor.stop(), 67.125)
            finally:
                release.set()
                helper.join(1)
            self.assertEqual(len(monitor.samples), 1)
            self.assertLessEqual(monitor.samples[0]["query_begin_monotonic_ns"], monitor.end_ns)
            self.assertGreaterEqual(monitor.samples[0]["query_end_monotonic_ns"], monitor.end_ns)
            reader.close()

    def test_query_error_and_zero_sample_windows_are_saved(self):
        for query_fails in (True, False):
            with self.subTest(query_fails=query_fails), tempfile.TemporaryDirectory() as tmp:
                reader, lib = self.reader(tmp)
                monitor = m.PowerMonitor(telemetry=reader, context=dict(layer_name="mlp"))
                if query_fails:
                    lib.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("native query failed")
                    monitor.start()
                else:
                    # Model an empty window without inventing a power sample.
                    monitor.begin_ns = time.monotonic_ns()
                with self.assertRaisesRegex(RuntimeError, "native query failed|No valid board-power"):
                    monitor.stop()
                window = json.loads((Path(tmp) / "measurement_windows.jsonl").read_text())
                self.assertEqual(window["sample_count"], 0)
                self.assertTrue(window["error"])
                self.assertIsNotNone(window["begin_monotonic_ns"])
                self.assertTrue(window["thread_joined"])
                reader.close()

    def test_hung_thread_skips_shutdown_but_does_not_skip_gpu_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader, lib = self.reader(tmp)
            monitor = m.PowerMonitor(telemetry=reader)
            monitor.begin_ns = time.monotonic_ns()
            monitor._thread = NS(ident=1, join=Mock(), is_alive=lambda: True)
            reader.monitors.add(monitor)
            d = device(tmp)
            d.dirty, d.last_target = True, 180
            d.run = Mock(return_value="")
            d.read = Mock(return_value=dict(gpu_freq=2527))
            with patch.object(m.time, "sleep"):
                result = m.restore(d, object(), object(), reader, [])
            d.run.assert_called_once_with("-rgc")
            self.assertEqual(result["readbacks_skipped"], "power_query_still_running")
            with self.assertRaisesRegex(RuntimeError, "Shutdown skipped"):
                reader.close()
            monitor._thread.join.assert_called_with(timeout=2)
            lib.nvmlShutdown.assert_not_called()
            # Release the fixture after asserting the non-concurrent shutdown contract.
            monitor._thread.is_alive = lambda: False
            reader.monitors.clear()
            reader.close()

    def test_loaded_readbacks_follow_synchronized_forward_without_utilization_filter(self):
        events = []
        def clock():
            events.append("clock")
            return dict(gpu_freq=2527)
        fake = NS(read_clock=clock)
        with patch.object(m, "torch", NS(no_grad=lambda: __import__('contextlib').nullcontext(),
                                         cuda=NS(synchronize=lambda: events.append("sync")))), \
             patch.object(m.time, "monotonic", side_effect=[0, 0, .5, 1]):
            frames = m.loaded_readbacks(lambda _: events.append("forward"), object(), fake)
        self.assertEqual(events, ["forward", "sync", "clock"] * 2)
        self.assertTrue(all(f["workload_context"] == "after_synchronized_forward" for f in frames))


if __name__ == "__main__":
    unittest.main()
