"""Phase/token-boundary repartitioning and optional calibrated thermal feedback."""
import copy
import math
import time
from .graph_partitioner import build_plan
from .hardware import integer


class RuntimePlanner:
    def __init__(self, profile, initial_plan, hardware, calibration=None, memory_policy='calibrated', allow_synthetic=False):
        if profile['identity'] != initial_plan['identity']:
            raise ValueError('runtime profile identity differs from the deployment plan')
        if profile.get('synthetic') and not allow_synthetic:
            raise ValueError('synthetic runtime profile is not deployable')
        self.profile, self.initial = profile, initial_plan
        self.calibration = copy.deepcopy(calibration)
        self.memory_policy, self.allow_synthetic = memory_policy, allow_synthetic
        self.sensor = None
        if calibration is not None:
            name = calibration.get('temperature_sensor')
            matches = [s for s in hardware.get('thermal', []) if s['name'] == name]
            if len(matches) != 1:
                raise ValueError('thermal calibration must name exactly one actual hardware temperature_sensor')
            limits = calibration.get('temperature_range_c', [])
            if (len(limits) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in limits)
                    or limits[0] >= limits[1]):
                raise ValueError('runtime thermal feedback requires a calibrated temperature_range_c')
            self.sensor = matches[0]
            self.limits = limits
            # Fail on unavailable/out-of-range readings before frequency control starts.
            self.temperature()
        # Validate full coefficient/OPP coverage and policy feasibility before a
        # controller is opened. Do not defer a malformed calibration to inference.
        checked_calibration = copy.deepcopy(self.calibration)
        if checked_calibration is not None:
            checked_calibration['temperature_c'] = self.temperature()
        build_plan(self.profile, self.initial['switch_us'], self.initial['aggregation_factor_N'],
                   self.initial['similarity']['tolerance'], self.initial['max_operator_slowdown'],
                   'paper' if checked_calibration is not None else 'measured', checked_calibration,
                   self.allow_synthetic, self.memory_policy)

    def temperature(self):
        if self.sensor is None:
            return None
        value = integer(self.sensor['path'])
        if value is None:
            raise ValueError('temperature sensor is unreadable; no synthetic temperature fallback')
        temp = value / 1000
        if not self.limits[0] <= temp <= self.limits[1]:
            raise ValueError(f'{temp} C is outside the calibrated temperature range {self.limits}')
        return temp

    def replan(self, phase, step):
        start = time.monotonic_ns()
        calibration = copy.deepcopy(self.calibration)
        temp = self.temperature()
        if calibration is not None:
            calibration['temperature_c'] = temp
        plan = build_plan(self.profile, self.initial['switch_us'], self.initial['aggregation_factor_N'],
                          self.initial['similarity']['tolerance'], self.initial['max_operator_slowdown'],
                          'paper' if calibration is not None else 'measured', calibration,
                          self.allow_synthetic, self.memory_policy)
        blocks = [b for b in plan['blocks'] if b['phase'] == phase]
        return plan, {'phase': phase, 'forward_index': step, 'temperature_c': temp,
                      'policy': 'equations_1_2_live_temperature' if calibration is not None else 'fixed_profile_repartition',
                      'partition_ms': (time.monotonic_ns() - start) / 1e6,
                      'blocks': [{'block_id': b['block_id'], 'triplet': b['triplet'],
                                  'operators': b['operators'], 'estimated_ms': b['estimated_ms']} for b in blocks],
                      'effective_sparsity_source': 'calibrated coefficients; no online zero-skipping inference'}
