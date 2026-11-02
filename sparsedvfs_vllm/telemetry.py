"""Frequency readback is evidence, not a claim of physical PLL completion."""
import threading
import time
from sparsedvfs.hardware import frequencies, PowerMonitor, temperatures


class Window:
    def __init__(self, hardware, interval=0.02):
        self.hardware = hardware
        self.power = PowerMonitor(hardware.get('power'), interval)
        self.readbacks, self.stop_event = [], threading.Event()

    def start(self):
        self.power.start()
        self.temperature_before = temperatures(self.hardware)
        def sample():
            self.readbacks.append(dict(host_ns=time.monotonic_ns(), **frequencies(self.hardware)))
        sample()
        def loop():
            while not self.stop_event.wait(0.02):
                sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        self.start_ns = time.monotonic_ns()
        return self

    def stop(self):
        self.end_ns = time.monotonic_ns()
        self.stop_event.set()
        self.thread.join()
        self.power.stop()
        return dict(start_ns=self.start_ns, end_ns=self.end_ns,
                    wall_ms=(self.end_ns-self.start_ns)/1e6,
                    power=self.power.summarize(self.start_ns, self.end_ns),
                    power_samples=self.power.samples, frequency_readbacks=self.readbacks,
                    temperature_before=self.temperature_before,
                    temperature_after=temperatures(self.hardware))
