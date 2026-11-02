"""Dependency-free mathematical core of arXiv:2603.21908v1.

Eq. (1)/(2) and Algorithm 1 are implemented separately from unspecified
selection/similarity/graph timing choices, which callers must supply.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Real


def finite(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number, not {value!r}')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not isfinite(result) or (minimum is not None and result < minimum) or (positive and result <= 0):
        raise ValueError(f'invalid {name}: {value!r}')
    return result


def timeline_seconds(work_flop, data_byte, s_comp, s_mem, peak_flops, bandwidth_Bps, overhead_s):
    """Section 4.1.1, Eq. (1); seconds, FLOP, byte, FLOP/s and byte/s."""
    w, d = finite(work_flop, 'W_comp', 0), finite(data_byte, 'D_mem', 0)
    sc, sm = finite(s_comp, 'S_comp', 0), finite(s_mem, 'S_mem', 0)
    if sc > 1 or sm > 1:
        raise ValueError('sparsity coefficients must be within [0, 1]')
    p, b = finite(peak_flops, 'P_peak', positive=True), finite(bandwidth_Bps, 'B_mem', positive=True)
    overhead = finite(overhead_s, 'T_overhead', 0)
    value = max(w * (1 - sc) / p, d * (1 - sm) / b) + overhead
    return finite(value, 'predicted T_exe', 0)


def thermal_power_w(alpha, voltage_v, frequency_hz, k1, k2, temperature_c):
    """Section 4.1.2, Eq. (2); alpha at the recorded S_comp, one scalar V/f."""
    a = finite(alpha, 'alpha(S_comp)', 0)
    v, f = finite(voltage_v, 'V', positive=True), finite(frequency_hz, 'f', positive=True)
    slope, intercept, temp = finite(k1, 'k1'), finite(k2, 'k2'), finite(temperature_c, 'T')
    return finite(a * v * v * f + (slope * temp + intercept) * v, 'predicted P_total', positive=True)


@dataclass(frozen=True, order=True)
class Frequency:
    cpu_khz: int
    gpu_hz: int
    emc_hz: int

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in self.values):
            raise ValueError('Frequency requires positive integer CPU-kHz/GPC-Hz/EMC-Hz values')

    @property
    def values(self):
        return (self.cpu_khz, self.gpu_hz, self.emc_hz)

    @classmethod
    def of(cls, value):
        return value if isinstance(value, cls) else cls(*value)

    def conservative_merge(self, other):
        return Frequency(*(max(a, b) for a, b in zip(self.values, other.values)))


@dataclass(frozen=True)
class OrderedDAG:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def validate(self):
        if not self.nodes or len(set(self.nodes)) != len(self.nodes) or any(not isinstance(n, str) or not n for n in self.nodes):
            raise ValueError('Algorithm 1 requires nonempty, unique named vertices')
        if len(set(self.edges)) != len(self.edges):
            raise ValueError('Duplicate graph edges')
        positions = {name: i for i, name in enumerate(self.nodes)}
        for edge in self.edges:
            if len(edge) != 2 or any(name not in positions for name in edge):
                raise ValueError('Graph edge references an unknown vertex')
            if positions[edge[0]] >= positions[edge[1]]:
                raise ValueError('Graph must be acyclic and supplied in a valid topological order')
        return self

    @classmethod
    def chain(cls, names):
        names = tuple(names)
        return cls(names, tuple(zip(names, names[1:]))).validate()


def relative_frequency_similarity(tolerance):
    """A caller-chosen interpretation of the paper's undefined approximate sign."""
    tolerance = finite(tolerance, 'frequency similarity tolerance', 0)
    if tolerance > 1:
        raise ValueError('frequency similarity tolerance must be <= 1')
    def compare(a, b):
        return all(abs(x-y) <= tolerance * max(x, y) for x, y in zip(a.values, b.values))
    return compare


def greedy_partition(graph, predict_frequency, calc_time_s, frequencies_similar, switch_s, factor):
    """Literal Algorithm 1 OR/max/final-append semantics, with explicit callbacks.

    calc_time_s must predict the whole current block at the given vector. A
    serialized sum is one possible adapter, not an assumption made here.
    """
    graph.validate()
    switch_s = finite(switch_s, 'T_switch', positive=True)
    factor = finite(factor, 'N', positive=True)
    threshold = finite(factor * switch_s, 'N*T_switch', positive=True)
    members = [graph.nodes[0]]
    frequency = Frequency.of(predict_frequency(members[0]))
    blocks, decisions = [], []

    def duration(nodes, target):
        return finite(calc_time_s(tuple(nodes), target), 'block duration', positive=True)

    def append_current():
        elapsed = duration(members, frequency)
        blocks.append(dict(nodes=list(members), triplet=list(frequency.values), estimated_s=elapsed,
                           amortized_strict_gt=elapsed > threshold))

    for name in graph.nodes[1:]:
        upcoming = Frequency.of(predict_frequency(name))
        estimate = duration(members, frequency)
        short, near = estimate < threshold, bool(frequencies_similar(frequency, upcoming))
        decisions.append(dict(current=list(members), next=name, current_triplet=list(frequency.values),
                              next_triplet=list(upcoming.values), current_duration_s=estimate,
                              threshold_s=threshold, short=short, similar=near, merged=short or near))
        if short or near:
            members.append(name)
            frequency = frequency.conservative_merge(upcoming)
        else:
            append_current()
            members, frequency = [name], upcoming
    append_current()  # Algorithm 1 line 16: even a short/equality tail is kept.
    return blocks, decisions


def minimum_model_energy(candidates, deadline_s):
    """Proposed explicit selector; the paper does not define predict's optimizer.

    Candidates are (Frequency, seconds, watts). Energy scores are model outputs,
    not sensor measurements. Tie-breaking by latency/vector is a declared choice.
    """
    deadline = finite(deadline_s, 'explicit latency budget', positive=True)
    feasible = []
    for frequency, seconds, watts in candidates:
        seconds, watts = finite(seconds, 'candidate latency', positive=True), finite(watts, 'candidate power', positive=True)
        frequency = Frequency.of(frequency)
        if seconds <= deadline:
            feasible.append((seconds * watts, seconds, frequency))
    if not feasible:
        raise ValueError('No candidate meets the explicit latency budget')
    energy, latency, frequency = min(feasible)
    return dict(triplet=list(frequency.values), latency_s=latency, model_energy_j=energy,
                policy='minimum_model_energy_under_user_budget', measured=False)
