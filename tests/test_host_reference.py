import copy
import unittest
from rtx5090.host_reference import local_energy, freeze_power, validate_power
from rtx5090.observe_core import GPU_UUIDS


def samples():
    frames = []
    for i in range(1001):
        t = i*100000000
        frame = []
        for u in GPU_UUIDS:
            values = dict(energy_mj=1000000+i*10000, temperature_c=45,
                          clock_mhz=2800, memory_clock_mhz=13801)
            frame.append(dict(uuid=u, synthetic=False, timestamp_ns=t,
                timestamp_clock=dict(name='perf_counter', resolution_s=1e-7),
                field_query_times_ns={key:dict(start=t, end=t+1000) for key in values}, **values))
        frames.append(frame)
    return dict(frames=frames, errors=[], clock_domain='Windows_perf_counter_ns',
                host_window_start_ns=50000000, host_window_end_ns=99950000000)


class HostEnergyTests(unittest.TestCase):
    def test_frozen_power_energy_predictions(self):
        def windows(prefix, scale=1):
            return [dict(name=prefix+'-'+str(i), status='valid_host_envelope', requests=20,
                         host_window_s=100, energy_by_uuid_j={u:10000*scale for u in GPU_UUIDS})
                    for i in (1,2,3)]
        model = freeze_power(windows('reference'), 'id')
        self.assertEqual(model['predictions']['pair/mean_power_w'], 200)
        self.assertEqual(model['predictions']['pair/energy_j'], 20000)
        before = copy.deepcopy(model)
        self.assertTrue(validate_power(model, windows('holdout'))['passed'])
        self.assertFalse(validate_power(model, windows('holdout', 2))['passed'])
        self.assertEqual(model, before)
        with self.assertRaises(ValueError): freeze_power(windows('holdout'), 'id')
        with self.assertRaises(ValueError): validate_power(model, windows('holdout')[:2])

    def test_common_host_window_counter_difference(self):
        result = local_energy(samples())
        self.assertAlmostEqual(result['energy_j'], 19980)
        self.assertAlmostEqual(result['host_window_s'], 99.9)
        self.assertEqual(result['quality_errors'], [])
        self.assertIn('not_exact_WSL', result['scope'])

    def test_old_diagnostic_not_relabelled(self):
        data = samples(); data.pop('host_window_end_ns')
        with self.assertRaisesRegex(ValueError, 'boundary markers'):
            local_energy(data)

    def test_gap_still_rejected(self):
        data = samples(); del data['frames'][400:405]
        self.assertTrue(any('sample gap' in e for e in local_energy(data)['quality_errors']))

    def test_boundary_must_be_bracketed(self):
        data = samples(); data['host_window_end_ns'] = 101000000000
        with self.assertRaisesRegex(ValueError, 'not bracketed'): local_energy(data)

    def test_counter_reset_and_memory_change(self):
        data = samples(); data['frames'][500][0]['energy_mj'] = 0
        with self.assertRaisesRegex(ValueError, 'reset/wrap'): local_energy(data)
        data = samples(); data['frames'][500][0]['memory_clock_mhz'] = 14001
        self.assertTrue(local_energy(data)['quality_errors'])

    def test_uuid_mismatch(self):
        data = samples(); data['frames'][0][0]['uuid'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'both UUIDs'): local_energy(data)
