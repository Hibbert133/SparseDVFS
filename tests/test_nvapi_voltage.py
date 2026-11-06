import ast
import copy
import ctypes as C
from pathlib import Path
import unittest
from unittest.mock import Mock
from rtx5090 import nvapi_voltage as v
from rtx5090.host_load_check import summarize


def samples():
    return [dict(uuid='a', timestamp_ns=i*100000000, synthetic=False,
        timestamp_clock={'name': 'perf_counter'}, voltage=dict(status=0,
        reported_core_voltage_v=.9+i*.001, binding={'uuid': 'a'},
        query_start_ns=i*100000000+100, query_end_ns=i*100000000+200)) for i in range(4)]


class VoltageTests(unittest.TestCase):
    def test_python38_and_layout(self):
        ast.parse(Path(v.__file__).read_text(), feature_version=(3, 8))
        self.assertEqual(C.sizeof(v.Voltage), 0x4c)

    def test_order_independent_binding(self):
        m = v.bind_by_bus([131, 2], 'a, 00000000:02:00.0\nb, 00000000:83:00.0\n', ['a', 'b'])
        self.assertEqual(m['a']['nvapi_index'], 1)
        self.assertEqual(m['b']['nvapi_index'], 0)

    def test_missing_ambiguous_and_malformed_binding(self):
        for buses, text in [([2], 'a, 0000:02:00.0'),
                            ([2, 2], 'a, 0000:02:00.0\nb, 0000:83:00.0'),
                            ([2, 131], 'a, 0000:02:00.0\nb, 0001:02:00.0'),
                            ([2, 131], 'a, invalid\nb, 0000:83:00.0')]:
            with self.assertRaises(ValueError): v.bind_by_bus(buses, text, ['a', 'b'])

    def test_voltage_read_and_error_preserved(self):
        reader = v.VoltageReader.__new__(v.VoltageReader)
        reader.bindings = {'a': dict(uuid='a', nvapi_index=0)}
        reader.handles = [123]
        def query(handle, pointer):
            self.assertEqual(handle, 123)
            value = C.cast(pointer, C.POINTER(v.Voltage)).contents
            self.assertEqual(value.words[0], 0x1004c)
            value.words[10] = 935000
            return 0
        reader.get_voltage = query
        self.assertEqual(reader.read('a')['reported_core_voltage_v'], .935)
        reader.get_voltage = Mock(return_value=-3)
        r = reader.read('a')
        self.assertEqual(r['status'], -3)
        self.assertIsNone(r['reported_core_voltage_v'])

    def test_summary_and_no_update_period_claim(self):
        r = summarize([[s] for s in samples()], ['a'], True)[0]['voltage']
        self.assertEqual(r['range_v'], [.9, .903])
        self.assertFalse(r['sensor_update_period_verified'])

    def test_invalid_samples(self):
        for field, value in [('status', -1), ('reported_core_voltage_v', 0),
                             ('reported_core_voltage_v', float('nan')),
                             ('reported_core_voltage_v', True), ('binding', {'uuid':'wrong'}),
                             ('query_end_ns', -1)]:
            s = samples(); s[0]['voltage'][field] = value
            with self.assertRaises(ValueError): v.summarize_voltage(s)

    def test_voltage_gap_even_if_nvml_gaps_pass(self):
        s = samples()
        s[-1]['voltage']['query_start_ns'] += 400000000
        s[-1]['voltage']['query_end_ns'] += 400000000
        with self.assertRaises(ValueError): summarize([[r] for r in s], ['a'], True)


if __name__ == '__main__': unittest.main()
