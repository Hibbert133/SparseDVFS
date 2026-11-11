#!/usr/bin/env python3
"""Author ViT profiler with RTX 5090 hardware control and exit recovery.

Upstream: https://github.com/ziyangzhang96/SparseDVFS/blob/
c524c51aac552fd16bec5743b1fb95309b46347b/SparseDVFS.py
Git blob: 50691aa6c3b11b71e4925eb1361481fae7d779ff

WSL Bash, in the existing Conda environment:
    python SparseDVFS_5090.py --uuid GPU-... --check-only
    python SparseDVFS_5090.py --uuid GPU-... --apply --ack-default-auto

Launch WSL from a Windows administrator terminal for real clock writes.
There is no bridge, socket, Windows Python, pynvml, or project-package import.
The Windows driver tools nvidia-smi.exe and powershell.exe must be accessible.
Python starts the sampler directly; PowerShell checks privilege and Windows PID ownership.

Preserved: torchvision vit_b_16 IMAGENET1K_V1, default precision, random
(1,3,224,224), author layer selection/input hooks/gradient scopes, five warmups,
five synchronized perf_counter measurements, and mean(power)*mean(latency).
CPU/memory remain system-managed. Power is ONE GPU BOARD, not system power.
20 ms is a query request, not certified sensor resolution. Energy is the author's
power-times-latency estimate, not a measured isolated-layer energy counter.
See VIT_5090_zh.md for the precise adaptation and recovery procedure.
"""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import base64
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

SOURCE_COMMIT = "c524c51aac552fd16bec5743b1fb95309b46347b"
DEFAULT_UUID = "GPU-22d1cd22-d198-45c4-2b54-974484026ab5"
RESTORE_TOLERANCE_MHZ = 15
logger = logging.getLogger("SparseDVFS_5090")
# Imported only by main, after selecting the UUID and parsing --help.
torch = torchvision = np = pd = plt = sns = tqdm = None
GPU = SAMPLER = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def error_message(exc):
    """Readable terminal error; full command/traceback stays in the output files."""
    if isinstance(exc, subprocess.TimeoutExpired):
        executable = Path(str(exc.cmd[0])).name
        return f"{executable} timed out after {exc.timeout} seconds (see commands.jsonl)"
    return str(exc) or type(exc).__name__


def log_failure(out, stage, exc):
    with (Path(out) / "errors.log").open("a", encoding="utf-8") as stream:
        stream.write(f"\nStage: {stage}\n" + traceback.format_exc())
    logger.error("%s failed: %s. See %s", stage, error_message(exc), Path(out) / "errors.log")


def number(value, field, optional=False):
    value = value.strip()
    if optional and value in ("N/A", "[N/A]", "Not Supported", "[Not Supported]", ""):
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"Invalid {field}: {value!r}")
    return result


def csv_rows(text):
    return list(csv.reader(text.lstrip("\ufeff").splitlines(), skipinitialspace=True))


def find_windows_tool(name):
    candidate = shutil.which(name)
    if not candidate:
        system32 = (Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32"
                    if os.name == "nt" else Path("/mnt/c/Windows/System32"))
        candidate = str(system32 / ("WindowsPowerShell/v1.0/powershell.exe"
                                   if name == "powershell.exe" else name))
    path = Path(candidate)
    if not path.is_file() or path.suffix.lower() != ".exe":
        raise RuntimeError(f"Cannot find Windows {name}; expose the existing Windows "
                           "System32 directory in WSL PATH. No Linux-tool fallback.")
    return str(path.resolve())


def windows_path(path):
    if os.name == "nt":
        return str(Path(path).resolve())
    result = subprocess.run(["wslpath", "-w", str(Path(path).resolve())],
                            capture_output=True, text=True, timeout=10, check=True)
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"wslpath returned no Windows path for {path}")
    return value


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


class NvidiaSMI:
    """One explicit UUID; all writes require the two CLI acknowledgements."""
    def __init__(self, uuid, out, apply=False, ack_default_auto=False):
        self.uuid, self.out = uuid, Path(out)
        self.allow_apply = apply and ack_default_auto
        self.smi = find_windows_tool("nvidia-smi.exe")
        self.powershell = find_windows_tool("powershell.exe")
        self.dirty = False
        self.last_target = None

    def command(self, arguments):
        started = time.monotonic_ns()
        row = dict(argv=list(arguments), started_monotonic_ns=started)
        try:
            result = subprocess.run(arguments, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=20)
            row.update(returncode=result.returncode, stdout=result.stdout,
                       stderr=result.stderr)
            if result.returncode:
                raise RuntimeError(f"Command failed ({result.returncode}): "
                                   f"{result.stderr.strip()} {result.stdout.strip()}")
            return result.stdout
        except BaseException as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.TimeoutExpired):
                row["timeout_seconds"] = exc.timeout
                for field in ("stdout", "stderr"):
                    value = getattr(exc, field, None)
                    row[field] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            raise
        finally:
            row["ended_monotonic_ns"] = time.monotonic_ns()
            append_jsonl(self.out / "commands.jsonl", row)

    def run(self, *arguments):
        return self.command([self.smi, "-i", self.uuid, *arguments])

    def ps(self, code):
        code = "$ErrorActionPreference='Stop'; " + code
        encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
        return self.command([self.powershell, "-NoLogo", "-NoProfile",
                             "-NonInteractive", "-EncodedCommand", encoded])

    def read(self):
        fields = "uuid,name,driver_version,clocks.current.graphics,clocks.current.memory,power.draw"
        rows = csv_rows(self.run("--query-gpu=" + fields, "--format=csv,noheader,nounits"))
        if len(rows) != 1 or len(rows[0]) != 6 or rows[0][0].strip() != self.uuid:
            raise RuntimeError(f"GPU UUID/query mismatch: {rows!r}")
        uuid, name, driver, core, memory, power = [v.strip() for v in rows[0]]
        return dict(uuid=uuid, name=name, driver=driver,
                    gpu_freq=number(core, "GPU MHz"),
                    memory_freq=number(memory, "memory MHz"),
                    power_watt=number(power, "board power W"),
                    received_monotonic_ns=time.monotonic_ns())

    def probe(self):
        identity = self.read()
        rows = csv_rows(self.run("--query-supported-clocks=memory,graphics",
                                 "--format=csv,noheader,nounits"))
        by_memory = {}
        for row in rows:
            if len(row) != 2:
                raise RuntimeError(f"Invalid supported-clock row: {row!r}")
            memory, core = [number(v, "supported MHz") for v in row]
            if core != int(core) or core <= 0 or memory <= 0:
                raise RuntimeError(f"Invalid supported-clock row: {row!r}")
            by_memory.setdefault(str(memory), []).append(int(core))
        rates = sorted({rate for values in by_memory.values() for rate in values})
        if not rates:
            raise RuntimeError("Driver reported no clock candidates; no invented grid.")
        return dict(**identity, rates_mhz=rates, by_memory_mhz=by_memory,
                    cpu_policy="system-managed", memory_policy="system-managed",
                    clock_grid_basis="union_of_reported_candidates_not_verified_under_load",
                    set_reset_verified=False, physical_restore_verified=False)

    def set(self, mhz):
        if not self.allow_apply:
            raise RuntimeError("Clock writes require --apply --ack-default-auto")
        if type(mhz) is not int or mhz <= 0:
            raise ValueError("GPU frequency must be a positive integer MHz")
        # A command can have side effects even when it returns an error.
        self.dirty, self.last_target = True, mhz
        self.run("-lgc", f"{mhz},{mhz}")
        time.sleep(0.5)  # Author set_frequency stabilization.

    def reset(self):
        if not self.dirty:
            return False
        if not self.allow_apply:
            raise RuntimeError("Reset requires --apply --ack-default-auto")
        self.run("-rgc")
        self.dirty = False
        return True


class PowerSampler:
    """One Windows loop process for the run; bounded RAM, full raw CSV on disk."""
    startup_timeout = 10
    fields = ("timestamp", "uuid", "clocks.current.graphics", "clocks.current.memory",
              "power.draw", "temperature.gpu", "utilization.gpu")

    def __init__(self, device):
        self.device = device
        self.raw = device.out / "power_samples.csv"
        self.stderr = device.out / "power_sampler.stderr.log"
        self.stdout = device.out / "power_sampler.stdout.log"
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.frames = deque(maxlen=10000)
        self.error, self.thread, self.marker = None, None, None
        self.pid, self.count = None, 0
        self.process = None  # On WSL this PID is a relay, not the Windows PID.
        self.stdout_handle = self.stderr_handle = None
        self.cleanup = dict(required=False, verified=False)

    def owned_query(self):
        # Match the unique output path, never all nvidia-smi processes.
        predicate = ("$_.CommandLine -and $_.CommandLine.Contains("
                     + ps_literal(self.marker) + ")")
        return ('@(Get-CimInstance Win32_Process -Filter "Name = \'nvidia-smi.exe\'"'
                + " | Where-Object { " + predicate + " })")

    def start(self):
        self.marker = windows_path(self.raw)
        arguments = ["-i", self.device.uuid,
                     "--query-gpu=" + ",".join(self.fields),
                     "--format=csv,noheader,nounits", "--loop-ms=20",
                     "--filename=" + self.marker]
        argv = [self.device.smi, *arguments]
        logger.info("Starting board-power sampler (20 ms query interval)...")
        self.stdout_handle = self.stdout.open("wb")
        self.stderr_handle = self.stderr.open("wb")
        self.cleanup["required"] = True
        launch = dict(argv=argv, started_monotonic_ns=time.monotonic_ns(), mode="Popen")
        try:
            # Files, not PIPEs: startup never waits for a long-lived process/EOF.
            options = dict(stdin=subprocess.DEVNULL, stdout=self.stdout_handle,
                           stderr=self.stderr_handle)
            if os.name == "nt":  # Development-side Windows process tests.
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            self.process = subprocess.Popen(argv, **options)
            launch["relay_pid"] = self.process.pid
        except OSError as exc:
            self.cleanup["required"] = False  # Popen did not create a process.
            launch["error"] = str(exc)
            raise
        finally:
            launch["ended_monotonic_ns"] = time.monotonic_ns()
            append_jsonl(self.device.out / "commands.jsonl", launch)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(self.startup_timeout):
            self.check_process()
            raise RuntimeError(f"Sampler startup: fewer than 3 valid records within "
                               f"{self.startup_timeout} s: " + self.diagnostics())
        self.check()
        code = ("$owned=" + self.owned_query()
                + "; if($owned.Count -ne 1){ throw 'Expected exactly one owned Windows sampler' }; "
                + "$owned[0].ProcessId")
        try:
            output = self.device.ps(code).strip()
            if not output.isdigit():
                raise RuntimeError(f"Invalid Windows PID: {output!r}")
            self.pid = int(output)
            self.check()
        except Exception as exc:
            raise RuntimeError("Sampler PID verification: " + error_message(exc)
                               + "; " + self.diagnostics()) from exc
        write_json(self.device.out / "power_sampler.json",
                   dict(windows_pid=self.pid, relay_pid=self.process.pid,
                        raw_csv=str(self.raw), interval_request_ms=20, startup_valid_records=self.count,
                        timestamp_basis="Windows wall timestamp plus WSL receive monotonic time"))
        logger.info("Power sampler ready: Windows PID=%s, valid records=%s", self.pid, self.count)
        return self

    def diagnostics(self):
        parts = []
        for path in (self.stdout, self.stderr):
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2000))
                    parts.append(path.name + ": " + stream.read().decode("utf-8", errors="replace"))
        return " ".join(parts)

    def check_process(self):
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"Power sampler exited (returncode={code}): " + self.diagnostics())

    def _read(self):
        try:
            deadline = time.monotonic() + self.startup_timeout
            while not self.raw.exists():
                self.check_process()
                if self.stop_event.wait(0.02):
                    return
                if time.monotonic() > deadline:
                    raise RuntimeError("Sampler did not create raw CSV: " + self.diagnostics())
            with self.raw.open(encoding="utf-8-sig", errors="strict") as stream:
                pending, last = "", time.monotonic()
                while not self.stop_event.is_set():
                    self.check_process()
                    part = stream.readline()
                    if not part:
                        if time.monotonic() - last > 5:
                            raise RuntimeError("Sampler has no new data for 5 s: " + self.diagnostics())
                        self.stop_event.wait(0.005)
                        continue
                    pending += part
                    if not pending.endswith("\n"):
                        continue
                    row = csv_rows(pending)
                    pending = ""
                    if len(row) != 1 or len(row[0]) != len(self.fields):
                        raise RuntimeError(f"Invalid power CSV row: {row!r}")
                    stamp, uuid, core, memory, power, temp, util = [v.strip() for v in row[0]]
                    if uuid != self.device.uuid:
                        raise RuntimeError("Wrong UUID in power sampler: " + uuid)
                    frame = dict(windows_timestamp=stamp, uuid=uuid,
                                 gpu_freq=number(core, "GPU MHz"),
                                 memory_freq=number(memory, "memory MHz"),
                                 power_watt=number(power, "board power W"),
                                 temperature_c=number(temp, "temperature C", optional=True),
                                 utilization_pct=number(util, "utilization", optional=True),
                                 received_monotonic_ns=time.monotonic_ns())
                    if frame["power_watt"] <= 0:
                        raise RuntimeError("Non-positive board power; no fallback value.")
                    with self.lock:
                        self.count += 1
                        frame["sample_index"] = self.count
                        self.frames.append(frame)
                    append_jsonl(self.device.out / "power_samples.jsonl", frame)
                    last = time.monotonic()
                    if self.count >= 3:
                        self.ready.set()
        except BaseException as exc:
            self.error = str(exc)
            self.ready.set()

    def check(self):
        if self.error:
            raise RuntimeError("Power sampler failed: " + self.error)
        self.check_process()
        with self.lock:
            if not self.frames or time.monotonic_ns() - self.frames[-1]["received_monotonic_ns"] > 5e9:
                raise RuntimeError("Power samples are missing or stale.")

    def window(self, begin_ns, end_ns):
        self.check()
        with self.lock:
            if self.frames[0]["received_monotonic_ns"] > begin_ns and self.count > len(self.frames):
                raise RuntimeError("Measurement exceeds retained sample buffer.")
            return [dict(f) for f in self.frames
                    if begin_ns <= f["received_monotonic_ns"] <= end_ns]

    def stop_windows(self):
        if self.cleanup["required"] and self.marker:
            query = self.owned_query()
            identity_check = (f'$known=Get-CimInstance Win32_Process -Filter "ProcessId = {self.pid}"; '
                              "if($known -and $known.Name -eq 'nvidia-smi.exe' -and "
                              "-not $known.CommandLine){ throw 'Cannot verify sampler ownership' }; "
                              if self.pid is not None else "")
            code = (identity_check + "$owned=" + query
                    + "; $ids=@($owned | ForEach-Object { $_.ProcessId }); "
                    + "$owned | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -PassThru } "
                    + "| ForEach-Object { if(-not $_.WaitForExit(5000)){ throw 'Sampler did not exit' } }; "
                    + "$left=" + query
                    + "; if($left.Count){ throw 'Owned Windows sampler still running' }; "
                    + "@{stopped_pids=$ids; verified=$true} | ConvertTo-Json -Compress")
            self.cleanup.update(json.loads(self.device.ps(code)))
            if self.cleanup.get("verified") is not True:
                raise RuntimeError("Windows sampler cleanup was not verified")

    def close(self):
        self.stop_event.set()
        errors = []
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.cleanup["reader_error"] = "Power reader thread did not stop."
                errors.append(self.cleanup["reader_error"])
        try:
            self.stop_windows()
        except Exception as exc:
            self.cleanup["verified"] = False
            errors.append(error_message(exc))
        finally:
            try:
                if self.process is not None:
                    try:
                        self.cleanup["relay_returncode"] = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.cleanup["relay_returncode"] = self.process.wait(timeout=2)
                        self.cleanup["relay_forced"] = True
                        # Recheck Windows after the relay is gone, including startup races.
                        self.stop_windows()
                    self.cleanup["relay_reaped"] = True
                elif self.cleanup["required"] and not self.cleanup.get("stopped_pids"):
                    self.cleanup["verified"] = False
                    errors.append("Sampler launch and cleanup could not be verified.")
            except Exception as exc:
                self.cleanup["verified"] = False
                errors.append(error_message(exc))
            finally:
                for handle in (self.stdout_handle, self.stderr_handle):
                    if handle is not None:
                        handle.close()
        if errors:
            self.cleanup["errors"] = errors
        write_json(self.device.out / "power_sampler_cleanup.json", self.cleanup)
        if errors:
            raise RuntimeError("; ".join(errors))


class PowerMonitor:
    """Author start/stop interface, backed by the one continuous 20 ms query."""
    def __init__(self, sample_interval=0.02):
        if sample_interval != 0.02:
            raise ValueError("Preserve the author's 20 ms query interval")
        self.begin_ns = None
        self.samples = []

    def start(self):
        SAMPLER.check()
        self.begin_ns = time.monotonic_ns()

    def stop(self):
        self.end_ns = time.monotonic_ns()
        self.samples = SAMPLER.window(self.begin_ns, self.end_ns)
        if not self.samples:
            raise RuntimeError("No valid board-power sample during author measurement window")
        return np.mean([f["power_watt"] for f in self.samples])


def calculate_sparsity(tensor):
    if not isinstance(tensor, torch.Tensor):
        return 0.0
    num_non_zeros = torch.count_nonzero(tensor).item()
    total_elements = tensor.numel()
    return 1.0 - (num_non_zeros / total_elements) if total_elements > 0 else 0.0


def get_layer_input_tensor(model, layer_name, input_tensor):
    """Get input tensor for specified layer"""
    intermediate_tensor = None

    def hook_fn(module, input):
        nonlocal intermediate_tensor
        if isinstance(input, tuple):
            input = input[0]
        intermediate_tensor = input.clone().detach()

    target_module = dict(model.named_modules()).get(layer_name)
    if target_module is None:
        logger.error(f"Layer '{layer_name}' not found")
        return None
    handle = target_module.register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_tensor)
    except Exception as e:
        logger.error(f"Model execution failed: {str(e)}")
        handle.remove()
        return None
    handle.remove()
    return intermediate_tensor


def get_all_layers(model, model_type="cnn"):
    """Get all executable layers in the model with different strategies for CNNs and Transformers"""
    all_layers = []
    if model_type == "cnn":
        # CNN-specific layer extraction
        for name, module in model.named_modules():
            if name == "":
                continue
            if any(x in name.lower() for x in ["bn", "pool", "downsample"]):
                continue
            if isinstance(module, torch.nn.Sequential):
                continue
            if "fc" in name or "avgpool" in name:
                continue
            all_layers.append(name)
    elif model_type == "transformer":
        # Transformer-specific layer extraction
        for name, module in model.named_modules():
            if name == "":
                continue
            if "mlp" in name or "attn" in name or "norm" in name or "head" in name:
                all_layers.append(name)
    # Limit to 5 layers for testing
    return all_layers[:]

def profile_model(model, model_name, model_type, target_layers, frequency_combinations, input_tensor):
    """Author layer workflow; Windows core-clock interface and explicit evidence."""
    results = []
    total_combinations = len(target_layers) * len(frequency_combinations)
    time.sleep(1)  # Author initial state stabilization.
    logger.info("Initial state stabilization complete")
    pbar = tqdm(total=total_combinations, desc=f"Profiling {model_name}")
    current_combination = 0
    try:
        for layer_name in target_layers:
            logger.info("\nPreparing input tensor for layer '%s'...", layer_name)
            try:
                layer_input = get_layer_input_tensor(model, layer_name, input_tensor)
            except Exception as exc:
                logger.error("Error getting input tensor: %s", exc)
                append_jsonl(GPU.out / "failures.jsonl",
                             dict(layer_name=layer_name, status="input_capture_failed",
                                  error=str(exc), error_type=type(exc).__name__))
                pbar.update(len(frequency_combinations))
                continue
            if layer_input is None:
                logger.warning("Cannot get input tensor for layer '%s', skipping", layer_name)
                append_jsonl(GPU.out / "failures.jsonl",
                             dict(layer_name=layer_name, status="input_capture_failed"))
                pbar.update(len(frequency_combinations))
                continue
            target_module = dict(model.named_modules()).get(layer_name)
            if target_module is None:
                logger.warning("Layer '%s' not found, skipping", layer_name)
                append_jsonl(GPU.out / "failures.jsonl",
                             dict(layer_name=layer_name, status="layer_not_found"))
                pbar.update(len(frequency_combinations))
                continue
            for gpu_freq in frequency_combinations:
                current_combination += 1
                logger.info("\n[%s/%s] Profiling: Layer='%s', GPU=%.0fMHz (CPU/memory system-managed)",
                            current_combination, total_combinations, layer_name, gpu_freq)
                context = dict(layer_name=layer_name, requested_gpu_freq=gpu_freq)
                stage = "control"
                try:
                    SAMPLER.check()
                    GPU.set(gpu_freq)
                    time.sleep(0.5)  # Author additional stabilization.
                    actual = GPU.read()  # Author reads before synchronization/warmup.
                    append_jsonl(GPU.out / "clock_readbacks.jsonl", dict(**context, **actual))
                    logger.info("  > Configuration set. GPU Freq: %.2fMHz, Memory: %.2fMHz (target: %sMHz)",
                                actual["gpu_freq"], actual["memory_freq"], gpu_freq)
                    stage = "compute"
                    torch.cuda.synchronize()
                    # Keep the author's gradient scopes, warmup count and timing.
                    warmup_runs = 5
                    for _ in range(warmup_runs):
                        _ = target_module(layer_input)
                    torch.cuda.synchronize()
                    min_profile_runs = 5
                    latencies = []
                    stage = "telemetry"
                    power_monitor = PowerMonitor(sample_interval=0.02)
                    power_monitor.start()
                    time.sleep(0.1)  # Author monitor startup delay.
                    stage = "compute"
                    for _ in range(min_profile_runs):
                        iter_start = time.perf_counter()
                        output_tensor = target_module(layer_input)
                        torch.cuda.synchronize()
                        iter_end = time.perf_counter()
                        latencies.append(iter_end - iter_start)
                    stage = "telemetry"
                    avg_power_w = power_monitor.stop()
                    stage = "compute"
                    latency_s = np.mean(latencies)
                    energy_j = avg_power_w * latency_s
                    logger.info("  > Measurement successful: Avg latency: %.2fms, Avg power: %.2fW, Energy: %.4fJ",
                                latency_s * 1000, avg_power_w, energy_j)
                    sparsity = calculate_sparsity(output_tensor)
                    logger.info("  > Sparsity: %.2f%%", sparsity * 100)
                    stage = "output"
                    samples = power_monitor.samples
                    row = dict(model=model_name, model_type=model_type, layer_name=layer_name,
                               cpu_freq=None, gpu_freq=actual["gpu_freq"],
                               latency_ms=float(latency_s * 1000),
                               avg_power_watt=float(avg_power_w), energy_joule=float(energy_j),
                               sparsity=float(sparsity), gpu_uuid=GPU.uuid,
                               requested_gpu_freq=gpu_freq, memory_freq=actual["memory_freq"],
                               power_sample_count=len(samples),
                               power_window_begin_ns=power_monitor.begin_ns,
                               power_window_end_ns=power_monitor.end_ns,
                               power_scope="single_GPU_board",
                               energy_method="author_mean_power_times_mean_latency_estimate",
                               cpu_policy="system-managed", memory_policy="system-managed")
                    results.append(row)
                    # Preserve completed cells even on Ctrl+C or a later control failure.
                    path = GPU.out / "profiling_results_ViT-Base.csv"
                    with path.open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
                    append_jsonl(GPU.out / "measurement_windows.jsonl",
                                 dict(**context, latencies_s=latencies,
                                      sample_indices=[f["sample_index"] for f in samples],
                                      sample_selection="WSL_receive_time_not_cross_OS_clock_sync"))
                except BaseException as exc:
                    append_jsonl(GPU.out / "failures.jsonl",
                                 dict(**context, status="failed", error=str(exc),
                                      error_type=type(exc).__name__, stage=stage))
                    if stage == "compute" and isinstance(exc, Exception):
                        logger.error("  > Profiling failed: %s", exc)
                        continue  # Author skips failed layer/frequency computations.
                    raise  # Control/telemetry/output failures and interrupts stop.
                finally:
                    pbar.update(1)
    finally:
        pbar.close()
    return pd.DataFrame(results)


def setup_plot_style():
    """Set up global plot style with Arial font and black text"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def plot_sparsity_distribution(df, model_name):
    """Plot sparsity distribution across layers as a bar chart"""
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # Get unique layers and their sparsity (in order of appearance)
    layers = []
    sparsities = []
    # Use the first occurrence of each layer for ordering
    for layer_name in df['layer_name'].unique():
        if layer_name not in layers:
            layer_data = df[df['layer_name'] == layer_name]
            avg_sparsity = layer_data['sparsity'].mean()
            layers.append(layer_name)
            sparsities.append(avg_sparsity)
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(layers)), sparsities, color='#0972BA', alpha=0.7)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Sparsity', fontsize=12)
    plt.title(f'Sparsity Distribution Across Layers - {model_name}', fontsize=14)
    plt.xticks(range(len(layers)), layers, rotation=45, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save and show
    filename = f"sparsity_distribution_{model_name}.png"
    plt.savefig(filename)
    logger.info(f"Sparsity distribution plot saved to: {filename}")
    plt.close()


def plot_sparsity_performance(df, model_name):
    """Plot sparsity vs. latency and efficiency for each layer (sorted by index)"""
    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return
    setup_plot_style()
    # Prepare data: average values for each layer (in order of appearance)
    layer_data = []
    for layer_name in df['layer_name'].unique():
        layer_df = df[df['layer_name'] == layer_name]
        avg_sparsity = layer_df['sparsity'].mean()
        avg_latency = layer_df['latency_ms'].mean()
        avg_efficiency = 1 / layer_df['energy_joule'].mean()
        layer_data.append({
            'layer': layer_name,
            'sparsity': avg_sparsity,
            'latency': avg_latency,
            'efficiency': avg_efficiency
        })
    # Sort layers by their index (order of appearance) rather than sparsity
    # The layers are already in order of appearance from the unique() call
    # Create the plot
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    # Plot latency (bar chart)
    layers = [d['layer'] for d in layer_data]
    sparsities = [d['sparsity'] for d in layer_data]
    latencies = [d['latency'] for d in layer_data]
    bars = ax1.bar(range(len(layers)), latencies, color='#0972BA', alpha=0.7)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    # Add sparsity as data labels on the x-axis
    x_labels = [f"{layer}(sparsity:{sparsity:.1%})" for layer, sparsity in zip(layers, sparsities)]
    plt.xticks(range(len(layers)), x_labels, rotation=45, fontsize=8)
    # Create secondary axis for efficiency
    ax2 = ax1.twinx()
    efficiencies = [d['efficiency'] for d in layer_data]
    line = ax2.plot(range(len(layers)), efficiencies, 'o-', color='black', linewidth=2, markersize=4,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor='black')
    ax2.set_ylabel('Efficiency (inferences per joule)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    plt.title(f'Layer Sparsity vs. Performance - {model_name}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save and show
    filename = f"sparsity_performance_{model_name}.png"
    plt.savefig(filename)
    logger.info(f"Sparsity-performance plot saved to: {filename}")
    plt.close()

def plot_gpu_performance(df, model_name):
    """Author GPU bar/power view, with the unavailable CPU axis removed."""
    setup_plot_style()
    for layer_name in df["layer_name"].unique():
        layer_df = df[df["layer_name"] == layer_name]
        grouped = layer_df.groupby("gpu_freq", sort=True)
        gpu_freqs, latencies, powers = [], [], []
        for gpu, group in grouped:
            gpu_freqs.append(gpu)
            latencies.append(group["latency_ms"].mean())
            powers.append(group["avg_power_watt"].mean())
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(range(len(gpu_freqs)), latencies, color="#0972BA", alpha=0.7)
        ax1.set_title(f"Layer: {layer_name} (CPU/memory system-managed)", fontsize=14)
        ax1.set_xlabel("GPU Frequency (MHz)", fontsize=12)
        ax1.set_ylabel("Latency (ms)", fontsize=12)
        ax1.set_xticks(range(len(gpu_freqs)))
        ax1.set_xticklabels([f"{gpu:.2f}" for gpu in gpu_freqs], rotation=45, fontsize=10)
        ax2 = ax1.twinx()
        ax2.plot(range(len(gpu_freqs)), powers, "o-", color="black", linewidth=2,
                 markersize=4, markerfacecolor="white", markeredgewidth=2)
        ax2.set_ylabel("GPU board power (W)", fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        from matplotlib.lines import Line2D
        ax1.legend(handles=[Line2D([0], [0], color="#0972BA", lw=4, label="Latency (ms)"),
                            Line2D([0], [0], color="black", marker="o", lw=2,
                                   label="GPU board power (W)")], loc="upper right")
        fig.tight_layout()
        fig.savefig(f"gpu_performance_{model_name}_{layer_name.replace('.', '_')}.png")
        plt.close(fig)
        # Keep the author heatmaps as one CPU-uncontrolled column, not CPU=0 MHz.
        for metric, label, prefix in (("latency_ms", "Latency (ms)", "latency"),
                                      ("avg_power_watt", "GPU board power (W)", "power")):
            values = layer_df.groupby("gpu_freq")[metric].mean().sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(2.5, max(2.5, 0.25 * len(values) + 1.5)))
            sns.heatmap(values.to_frame("CPU system-managed"), ax=ax, cmap="viridis",
                        annot=False, linewidths=0.5, square=True,
                        cbar_kws={"label": label})
            ax.set_ylabel("GPU Frequency (MHz)")
            ax.set_title(f"{prefix.title()} - {model_name}: {layer_name}", fontsize=10)
            fig.tight_layout()
            fig.savefig(f"{prefix}_heatmap_{model_name}_{layer_name.replace('.', '_')}.png")
            plt.close(fig)


def load_dependencies():
    global torch, torchvision, np, pd, plt, sns, tqdm
    versions = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "matplotlib", "seaborn", "tqdm"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Cannot import {name} in {sys.executable}: {exc}. "
                               "Use the existing compatible environment; no automatic install.") from exc
        versions[name] = dict(version=str(getattr(module, "__version__", "unknown")),
                              path=str(getattr(module, "__file__", "unknown")))
        if name == "matplotlib":
            module.use("Agg")  # seaborn itself can import pyplot.
        globals()[{"numpy": "np", "pandas": "pd", "seaborn": "sns"}.get(name, name)] = module
    # Unlike the author's import order, select the headless backend before pyplot.
    plt = importlib.import_module("matplotlib.pyplot")
    tqdm = importlib.import_module("tqdm").tqdm
    return versions


def cuda_identity(uuid):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one visible CUDA GPU selected by UUID.")
    prop = torch.cuda.get_device_properties(0)
    value = getattr(prop, "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    actual = str(value)
    if not actual.startswith("GPU-"):
        actual = "GPU-" + actual
    if actual.lower() != uuid.lower():
        raise RuntimeError(f"CUDA UUID {actual!r} does not match selected {uuid!r}")
    return dict(uuid=actual, name=prop.name, cuda_runtime=str(torch.version.cuda),
                capability=list(torch.cuda.get_device_capability(0)),
                compiled_architectures=torch.cuda.get_arch_list(),
                default_dtype=str(torch.get_default_dtype()),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))


def preflight(device):
    versions = load_dependencies()
    cuda = cuda_identity(device.uuid)
    probe = device.probe()
    if "5090" not in probe["name"]:
        raise RuntimeError(f"Expected RTX 5090, got {probe['name']!r}")
    if probe["power_watt"] <= 0:
        raise RuntimeError("No positive board-power reading; no fallback.")
    # This query does not write GPU or operating-system settings.
    admin = device.ps("([Security.Principal.WindowsPrincipal]"
                      "[Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("
                      "[Security.Principal.WindowsBuiltInRole]::Administrator)").strip()
    if admin.lower() not in ("true", "false"):
        raise RuntimeError(f"Cannot determine Windows privilege: {admin!r}")
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    from urllib.parse import urlparse
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    return dict(source_commit=SOURCE_COMMIT, python=sys.executable,
                dependencies=versions, cuda=cuda, gpu=probe,
                windows_admin=(admin.lower() == "true"),
                weights="ViT_B_16_Weights.IMAGENET1K_V1", weights_url=weights.url,
                weights_cache=str(cache), weights_cached=cache.is_file(),
                check_only_downloads_weights=False,
                real_cuda_workload_executed=False, set_reset_verified=False,
                energy_scope="one_GPU_board_author_power_times_latency_estimate",
                sample_interval_request_ms=20,
                actual_sensor_update_resolution_verified=False)


def loaded_readbacks(model, input_tensor, sampler):
    """Restoration evidence only, outside all author profiling measurements."""
    begin = time.monotonic_ns()
    deadline = time.monotonic() + 1
    with torch.no_grad():
        while time.monotonic() < deadline:
            model(input_tensor)
            torch.cuda.synchronize()
    end = time.monotonic_ns()
    return sampler.window(begin, end)


def restore(device, model, input_tensor, sampler, baseline):
    result = dict(required=device.dirty, reset_returned=False,
                  physical_restore_verified=False, errors=[])
    if not device.dirty:
        return result
    try:
        result["reset_returned"] = device.reset()
        time.sleep(1)  # Author restoration wait.
        result["post_reset_snapshot"] = device.read()
        logger.info("Restored GPU frequency readback: %.2fMHz (automatic mode requested)",
                    result["post_reset_snapshot"]["gpu_freq"])
        if sampler is not None and model is not None:
            after = loaded_readbacks(model, input_tensor, sampler)
            before = [f["gpu_freq"] for f in baseline if (f["utilization_pct"] or 0) > 0]
            loaded = [f["gpu_freq"] for f in after if (f["utilization_pct"] or 0) > 0]
            result["baseline_loaded_clocks_mhz"] = before
            result["restored_loaded_clocks_mhz"] = loaded
            # Idle downclock or a successful -rgc alone is not unlock evidence.
            result["physical_restore_verified"] = bool(
                len(before) >= 3 and len(loaded) >= 3
                and sum(v > device.last_target + RESTORE_TOLERANCE_MHZ for v in loaded) >= 3
                and any(min(before) - RESTORE_TOLERANCE_MHZ <= v
                        <= max(before) + RESTORE_TOLERANCE_MHZ for v in loaded))
            result["verification_basis"] = (
                "loaded_readbacks_above_previous_lock_ceiling_and_within_baseline_range;"
                "not_an_independent_PLL_measurement")
    except BaseException as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uuid", default=DEFAULT_UUID, help="Full physical GPU UUID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Read-only, no weights download")
    mode.add_argument("--apply", action="store_true", help="Explicitly allow GPU clock writes")
    parser.add_argument("--ack-default-auto", action="store_true",
                        help="Confirm this GPU starts in automatic clock mode; restore with -rgc")
    parser.add_argument("--out", help="New output directory; existing directories are refused")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.uuid):
        parser.error("--uuid must be a full GPU UUID, not an index")
    args.uuid = "GPU-" + args.uuid[4:].lower()
    if args.apply and not args.ack_default_auto:
        parser.error("--apply also requires --ack-default-auto")
    return args


def _termination(signum, frame):
    raise SystemExit(128 + signum)


def load_vit():
    """Author primary pretrained model and original Torch Hub fallback."""
    try:
        model = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).cuda().eval()
        logger.info("ViT-Base (Transformer) model loaded successfully")
        return model
    except Exception as exc:
        logger.error("ViT loading failed: %s", exc)
        logger.info("Attempting to load ViT from PyTorch Hub...")
        try:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16',
                                   pretrained=True).cuda().eval()
            logger.info("ViT-Base (Transformer) model loaded successfully from PyTorch Hub")
            return model
        except Exception as fallback_exc:
            logger.error("ViT loading from PyTorch Hub failed: %s", fallback_exc)
            raise


def main(argv=None):
    global GPU, SAMPLER
    args = parse_args(argv)
    GPU = SAMPLER = None
    # Set before importing torch or torchvision. Do not change Triton/proxy/global settings.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    name = datetime.now(timezone.utc).strftime("vit5090-%Y%m%dT%H%M%S-%fZ") + f"-{os.getpid()}"
    out = Path(args.out).resolve() if args.out else Path("runs", name).resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(out / "dvfs_profiler.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    status = dict(status="started", source_commit=SOURCE_COMMIT, gpu_uuid=args.uuid,
                  out=str(out), mode="check-only" if args.check_only else "profile",
                  real_cuda_workload_executed=False, clock_write_attempted=False)
    model = input_tensor = data = None
    baseline = []
    code = 0
    stage = "preflight"
    previous_handlers = {sig: signal.signal(sig, _termination)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        GPU = NvidiaSMI(args.uuid, out, args.apply, args.ack_default_auto)
        report = preflight(GPU)
        write_json(out / "preflight.json", report)
        if args.apply and not report["windows_admin"]:
            raise RuntimeError("Windows administrator rights required: reopen Windows Terminal "
                               "as administrator, enter WSL and reuse this environment. "
                               "WSL sudo does not elevate Windows.")
        stage = "power sampler startup"
        SAMPLER = PowerSampler(GPU)
        SAMPLER.start()
        report["sampler_startup_passed"] = True
        report["sampler_startup_valid_records"] = SAMPLER.count
        write_json(out / "preflight.json", report)
        if args.check_only:
            status["status"] = "preflight_passed_read_only"
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Starting Sparsity and DVFS Analysis")
            logger.info("=" * 60)
            stage = "ViT loading"
            input_tensor = torch.randn(1, 3, 224, 224).cuda()
            model = load_vit()
            target_layers = get_all_layers(model, "transformer")
            if not target_layers:
                raise RuntimeError("Author selection produced no ViT target layers")
            write_json(out / "target_layers.json", target_layers)
            rates = report["gpu"]["rates_mhz"]
            logger.info("Device supported frequency ranges (in MHz):")
            logger.info("  GPU: %.2fMHz - %.2fMHz", min(rates), max(rates))
            logger.info("  CPU/memory: system-managed")
            logger.info("Will test %s frequency combinations (actual reported frequencies)", len(rates))
            stage = "restoration baseline"
            baseline = loaded_readbacks(model, input_tensor, SAMPLER)
            status["real_cuda_workload_executed"] = True
            write_json(out / "baseline_loaded_readbacks.json", baseline)
            logger.info("\nStarting profiling for ViT-Base (transformer)...")
            stage = "ViT profiling"
            data = profile_model(model, "ViT-Base", "transformer", target_layers,
                                 report["gpu"]["rates_mhz"], input_tensor)
            if data.empty:
                raise RuntimeError("No valid measurement cells; inspect failures.jsonl")
            status.update(status="profiled", valid_cells=len(data))
            logger.info("Profiling data saved to: %s", out / "profiling_results_ViT-Base.csv")
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else (
            int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1)
        status.update(status="failed", stage=stage, error=error_message(exc), error_type=type(exc).__name__)
        log_failure(out, stage, exc)
    finally:
        # Finish bounded cleanup even if Ctrl+C is pressed a second time.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        restoration = dict(required=False, reset_returned=False, physical_restore_verified=False)
        try:
            if GPU is not None:
                status["clock_write_attempted"] = GPU.last_target is not None
                if GPU.dirty:
                    logger.info("\nProfiling complete/stopped, restoring default automatic GPU frequency...")
                restoration = restore(GPU, model, input_tensor, SAMPLER, baseline)
            write_json(out / "restore_status.json", restoration)
            if restoration.get("errors"):
                code = 1
                status.update(status="restoration_failed", restoration_errors=restoration["errors"])
                logger.error("RESTORATION FAILED: %s. Stop further clock experiments.",
                             restoration["errors"])
            elif restoration["required"] and not restoration["physical_restore_verified"]:
                code = code or 2
                status["restoration_unverified"] = True
                logger.warning("-rgc returned; physical restoration remains UNVERIFIED. "
                               "Read restore_status.json before further clock experiments.")
        finally:
            try:
                if SAMPLER is not None:
                    SAMPLER.close()
                    if SAMPLER.error:
                        code = 1
                        status.update(status="power_sampler_failed", sampler_error=SAMPLER.error)
            except BaseException as exc:
                code = 1
                status.update(status="sampler_cleanup_failed", sampler_cleanup_error=error_message(exc))
                log_failure(out, "sampler cleanup", exc)
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
    # Restore the GPU and stop the sampler BEFORE the potentially lengthy plotting.
    if data is not None and not data.empty and status["status"] == "profiled":
        previous_cwd = Path.cwd()
        try:
            os.chdir(out)
            plot_sparsity_distribution(data, "ViT-Base")
            plot_sparsity_performance(data, "ViT-Base")
            plot_gpu_performance(data, "ViT-Base")
            status["status"] = ("completed_restore_unverified"
                                if status.get("restoration_unverified") else "completed")
        except BaseException as exc:
            code = 1
            status.update(status="plot_failed", plot_error=str(exc))
            log_failure(out, "plotting", exc)
        finally:
            os.chdir(previous_cwd)
    if code == 0 and args.check_only:
        logger.info("PRECHECK PASS: UUID=%s, candidates=%s, Windows admin=%s, weights cached=%s. "
                    "Sampler startup/cleanup passed; clock writes/restoration have NOT been tested.",
                    args.uuid, len(report["gpu"]["rates_mhz"]),
                    report["windows_admin"], report["weights_cached"])
    elif status["status"] in ("completed", "completed_restore_unverified"):
        logger.info("\nAll profiling and visualization completed. Check output files for detailed results.")
    status["exit_code"] = code
    write_json(out / "status.json", status)
    logger.info("Output: %s | status=%s | exit=%s", out, status["status"], code)
    logging.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
