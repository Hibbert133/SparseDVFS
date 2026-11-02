"""Algorithm 1 translated literally, with explicit frequency-vector similarity."""
import math
from .offline_modeler import select_operators
from .paper_core import Frequency, OrderedDAG, greedy_partition, relative_frequency_similarity


def similar(a, b, tolerance):
    # The paper leaves approximately-equal undefined. This relative L-infinity
    # rule is a demo choice, exposed in the plan metadata.
    return relative_frequency_similarity(tolerance)(Frequency.of(a), Frequency.of(b))


def partition(operators, duration, switch_us, factor=5, tolerance=0.05, *, include_decisions=False):
    if not math.isfinite(switch_us) or switch_us <= 0:
        raise ValueError('supply positive measured switch latency, in microseconds')
    if not math.isfinite(factor) or factor <= 0 or not 0 <= tolerance <= 1:
        raise ValueError('invalid N or similarity tolerance')
    blocks, decisions, phases = [], [], []
    for op in operators:
        if not phases or phases[-1][0] != op['phase']:
            if any(phase == op['phase'] for phase, _ in phases):
                raise ValueError('Each phase must appear in one contiguous topological sequence')
            phases.append((op['phase'], []))
        phases[-1][1].append(op)
    for phase, rows in phases:
        graph = OrderedDAG.chain([op['name'] for op in rows])
        frequencies = {op['name']: Frequency.of(op['triplet']) for op in rows}
        def seconds(names, vector):
            return duration(list(names), phase, list(vector.values)) / 1000
        raw, trace = greedy_partition(graph, frequencies.__getitem__, seconds,
                                     relative_frequency_similarity(tolerance), switch_us/1e6, factor)
        blocks.extend(dict(phase=phase, operators=b['nodes'], triplet=b['triplet'],
                           estimated_ms=b['estimated_s']*1000, amortized_strict_gt=b['amortized_strict_gt']) for b in raw)
        decisions.extend(dict(row, phase=phase) for row in trace)
    for i, block in enumerate(blocks):
        block['block_id'] = i
    # Algorithm 1 appends the terminal block unconditionally, even if short.
    # Retain it and expose the flag, instead of silently claiming Eq. (3) holds.
    return (blocks, decisions) if include_decisions else blocks


def build_plan(profile, switch_us, factor=5, tolerance=0.05, max_slowdown=0.05,
               selector='measured', calibration=None, allow_synthetic=False, memory_policy='calibrated'):
    if profile.get('synthetic') and not allow_synthetic:
        raise ValueError('synthetic fixture requires --allow-synthetic; never deploy it')
    if not profile.get('dvfs_applied') and not profile.get('synthetic'):
        raise ValueError('profile was collected without real DVFS; refusing a deployment plan')
    ops, duration = select_operators(profile, max_slowdown, selector, calibration, memory_policy)
    blocks, decisions = partition(ops, duration, switch_us, factor, tolerance, include_decisions=True)
    return {'schema': 1, 'origin': 'clean_room_reference_port', 'synthetic': bool(profile.get('synthetic')),
            'identity': profile['identity'], 'units': profile['units'], 'reference': profile['reference'],
            'selector': selector, 'memory_policy': memory_policy, 'switch_us': switch_us, 'aggregation_factor_N': factor,
            'similarity': {'rule': 'relative_L_infinity', 'tolerance': tolerance},
            'max_operator_slowdown': max_slowdown, 'blocks': blocks, 'operator_choices': ops,
            'partition_decisions': decisions,
            'paper_parameter_provenance': {'N': 'tunable; 5 is a Figure 9 example',
                'similarity_tolerance': 'caller/reference choice; not specified by the paper',
                'max_operator_slowdown': 'caller/reference choice; not specified by the paper'},
            'calibration_schema': calibration.get('schema') if calibration else None,
            'graph_scope': 'explicit serialized coarse chain; not a recovered kernel DAG',
            'equation_model': ('schema4_argument_dependencies' if calibration and calibration.get('schema') == 4
                               else 'legacy_calibration_reference' if calibration else 'empirical_table'),
            'degraded_no_emc': bool(profile.get('degraded_no_emc')),
            'power_attribution': ('whole_inference_average_proxy' if selector == 'measured' else 'calibrated_scalar_equation_2'),
            'runtime_granularity': 'Qwen_text_modules_in_execution_order',
            'limitations': ['not the author C++ source', 'explicit stream integration, not transparent CUDA/TensorRT interception',
                            'decoder modules are coarser than the paper operator DAG',
                            'GPU start observation anchors a prediction; driver/physical transition overlap must be measured',
                            'module slowdown limit is not an end-to-end SLO guarantee']}
