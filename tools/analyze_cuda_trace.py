#!/usr/bin/env python3
"""Report event-observed overlap; never infer PLL settling from a host write."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def analyze(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r['kind'] for r in rows)
    ends = [r for r in rows if r['kind'] == 'gpu_end_observed']
    return {'schema': 1, 'source': str(path), 'events': len(rows), 'counts': dict(counts),
            'blocks_with_completed_event_timing': len(ends),
            'requests_returned_while_block_end_still_pending': counts['lookahead_written_end_pending'],
            'requests_without_confirmed_end_pending_after_write': counts['lookahead_written_end_complete_postcheck'],
            'missed_before_end_observation': counts['lookahead_missed_end_ready'],
            'stale_requests_skipped': counts['lookahead_skipped_stale'],
            'unchanged_target_noops_not_writes': counts['lookahead_noop_same_target'],
            'event_observation_is_a_host_poll_upper_bound': True,
            'physical_clock_transition_hidden': 'NOT_ESTABLISHED_BY_THIS_TRACE',
            'interpretation': 'An end event still pending after a write supports command/computation overlap only. Frequency readback/physical transition and end-to-end latency require separate measurement.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace')
    p.add_argument('--output')
    args = p.parse_args()
    result = analyze(args.trace)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + '\n', encoding='utf-8')
    print(text)
