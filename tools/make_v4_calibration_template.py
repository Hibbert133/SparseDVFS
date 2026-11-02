"""Generate unfilled schema-4 target calibration; no numerical values are guessed."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparsedvfs.hardware import load_json, write_json
from sparsedvfs_vllm.contracts import PROTOCOL, UNITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    profile = load_json(args.profile)
    if profile.get('synthetic') or profile['identity'].get('backend') != PROTOCOL:
        raise ValueError('A real v5 vLLM hardware profile is required; synthetic/older inputs are not deployable')
    opps = {','.join(map(str, candidate['triplet'])): dict(
        peak_flops=None, bandwidth_Bps=None, voltage_v=None, effective_frequency_hz=None)
        for candidate in profile['candidates']}
    operators = {phase + '/' + name: dict(work_flop=None, data_byte=None, overhead_s=None,
        workload_provenance=None, s_comp=0.0, s_mem=0.0,
        effective_sparsity_verified=False, effective_sparsity_evidence=None,
        activity_factor=None, activity_s_comp=0.0)
        for phase, name in [(r['phase'],r['name']) for r in profile['candidates'][0]['operators']]}
    write_json(args.output, dict(schema=4, synthetic=False, provenance=None,
        scalar_vf_scope=None, temperature_c=None, temperature_sensor=None, temperature_range_c=[None, None],
        device_thermal=dict(k1=None, k2=None), opps=opps, operators=operators,
        v5_validation=dict(deployable=False,status='unfilled template; use calibrate fit/validate'),
        note='Nulls intentionally fail validation. Measure real Thor coefficients. P_peak must agree for equal CPU/GPU; bandwidth for equal EMC. Shared k1/k2, one alpha at recorded S_comp. No paper-reported Thor values exist.'))


if __name__ == '__main__':
    main()
