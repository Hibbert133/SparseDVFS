"""Read-only Windows NVAPI voltage; no public stable-ABI guarantee.

Layout and interface ID follow LibreHardwareMonitor Interop/NvApi.cs:
NvGpuClientVoltRailsStatus, 0x4c bytes, version 1, microvolts at 0x28/0x2c.
Reports driver values, not independently certified rail measurements.
"""
import csv
import ctypes as C
import io
import math
import os
import re
import subprocess
import time


class Voltage(C.Structure):
    _fields_ = [('words', C.c_uint32 * 19)]


def bind_by_bus(nvapi_buses, smi_csv, uuids):
    by_bus = {}
    for row in csv.reader(io.StringIO(smi_csv)):
        if len(row) != 2:
            raise ValueError('Expected UUID, PCI bus ID from Windows nvidia-smi')
        uuid, pci = (x.strip() for x in row)
        match = re.fullmatch(r'[0-9a-fA-F]{4,8}:([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]', pci)
        if not match:
            raise ValueError('Invalid PCI ID: '+pci)
        bus = int(match[1], 16)
        if bus in by_bus:
            raise ValueError('Ambiguous PCI bus; cannot bind NVAPI to UUID')
        by_bus[bus] = dict(uuid=uuid, pci_bus_id=pci)
    if len(set(nvapi_buses)) != len(nvapi_buses):
        raise ValueError('Duplicate NVAPI bus IDs')
    result = {}
    for index, bus in enumerate(nvapi_buses):
        if bus not in by_bus:
            raise ValueError('NVAPI bus absent from NVML inventory')
        entry = by_bus[bus]
        if entry['uuid'] in uuids:
            if entry['uuid'] in result:
                raise ValueError('Duplicate UUID')
            result[entry['uuid']] = dict(entry, nvapi_index=index, pci_bus=bus)
    if set(result) != set(uuids):
        raise ValueError('Missing requested GPU voltage binding')
    return result


class VoltageReader:
    def __init__(self, uuids):
        if os.name != 'nt' or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError('NVAPI voltage requires 64-bit Windows Python')
        system = os.path.join(os.environ['SystemRoot'], 'System32')
        self.dll = C.CDLL(os.path.join(system, 'nvapi64.dll'))
        query = self.dll.nvapi_QueryInterface
        query.argtypes, query.restype = [C.c_uint32], C.c_void_p
        def api(identifier, *args):
            address = query(identifier)
            if not address:
                raise RuntimeError('Missing NVAPI interface 0x%08X' % identifier)
            return C.CFUNCTYPE(C.c_int, *args)(address)
        self.check(api(0x0150E828)(), 'Initialize')
        enum = api(0xE5AC921F, C.POINTER(C.c_void_p), C.POINTER(C.c_uint32))
        get_bus = api(0x1BE0B8E5, C.c_void_p, C.POINTER(C.c_uint32))
        self.get_voltage = api(0x465F9BCF, C.c_void_p, C.POINTER(Voltage))
        self.handles = (C.c_void_p * 64)()
        count = C.c_uint32()
        self.check(enum(self.handles, C.byref(count)), 'EnumPhysicalGPUs')
        if not 1 <= count.value <= 64:
            raise ValueError('Invalid GPU count')
        buses = []
        for index in range(count.value):
            bus = C.c_uint32()
            self.check(get_bus(self.handles[index], C.byref(bus)), 'GetBusId')
            buses.append(bus.value)
        inventory = subprocess.check_output([os.path.join(system, 'nvidia-smi.exe'),
            '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'], text=True, timeout=15)
        self.bindings = bind_by_bus(buses, inventory, uuids)

    @staticmethod
    def check(status, operation):
        if status != 0:
            raise RuntimeError('%s: NVAPI status=%d' % (operation, status))

    def read(self, uuid):
        binding = self.bindings[uuid]
        value = Voltage()
        value.words[0] = C.sizeof(value) | (1 << 16)
        begin = time.perf_counter_ns()
        status = self.get_voltage(self.handles[binding['nvapi_index']], C.byref(value))
        end = time.perf_counter_ns()
        uv = (value.words[10] | (value.words[11] << 32)) if status == 0 else None
        return dict(status=status, reported_core_voltage_uv=uv,
            reported_core_voltage_v=uv/1e6 if uv else None,
            source='NVAPI_0x465F9BCF_v1', binding=binding,
            query_start_ns=begin, query_end_ns=end,
            semantics='driver_reported_core_voltage_not_independent_sensor_certificate')


def summarize_voltage(samples):
    values, intervals, stamps = [], [], []
    for sample in samples:
        v = sample.get('voltage', {})
        value = v.get('reported_core_voltage_v')
        if (v.get('status') != 0 or isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Missing/failed/zero NVAPI voltage sample')
        if v.get('binding', {}).get('uuid') != sample['uuid']:
            raise ValueError('Voltage UUID binding mismatch')
        begin, end = v['query_start_ns'], v['query_end_ns']
        if not sample['timestamp_ns'] <= begin <= end:
            raise ValueError('Invalid voltage query interval')
        stamps.append((begin+end)/2)
        values.append(value)
        intervals.append((end-begin)/1e6)
    if len(values) < 4:
        raise ValueError('Insufficient voltage samples')
    gaps = [(b-a)/1e6 for a,b in zip(stamps, stamps[1:])]
    if any(g <= 0 or g > 350 for g in gaps):
        raise ValueError('Voltage query gap exceeds budget or is nonmonotonic')
    return dict(samples=len(values), range_v=[min(values), max(values)],
        distinct_values=len(set(values)), max_query_ms=max(intervals), max_gap_ms=max(gaps),
        sensor_update_period_verified=False, equation_2_parameters_validated=False)
