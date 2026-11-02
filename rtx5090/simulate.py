"""Illustrative event-cost model, NOT a CPU/GPU performance emulator."""
import json

def scenario(name, low_ms, low_w, switch_ms):
    # Four equal blocks, high/low/high/low. Baseline: all high, 10 ms and 200 W each.
    duration=20+2*low_ms; compute_energy=(20*200+2*low_ms*low_w)/1000
    baseline=dict(time_ms=40,energy_j=8)
    serial=dict(time_ms=duration+3*switch_ms,energy_j=compute_energy+3*switch_ms*200/1000)
    # Optimistic overlap: next request can hide up to min(10, low_ms) ms per transition.
    exposed=max(0,switch_ms-min(10,low_ms))
    lookahead=dict(time_ms=duration+3*exposed,energy_j=compute_energy+3*exposed*200/1000)
    return dict(name=name,synthetic=True,assumptions=dict(low_ms=low_ms,low_w=low_w,switch_ms=switch_ms,
        overlap_model='optimistic, ignores early high-clock residency'),fixed=baseline,serial=serial,lookahead=lookahead)

def results():
    return dict(real_hardware_measured=False,caveat='Assumed values demonstrate sensitivity; no RTX 5090 benefit claim',
        scenarios=[scenario('benefit',10,100,1),scenario('no_benefit',10,200,0),scenario('negative',20,180,20)])

if __name__=='__main__': print(json.dumps(results(),indent=2))
