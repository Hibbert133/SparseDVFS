"""Measure streamed completion token IDs; a network chunk is not a token."""
import json
import time
import urllib.request
from .contracts import digest_json


def completion(base_url, workload, timeout=1200):
    body = dict(model='qwen35', prompt=workload['input_token_ids'], max_tokens=workload['output_tokens'],
                min_tokens=workload['output_tokens'], temperature=0.0, seed=0, ignore_eos=True,
                add_special_tokens=False, stream=True, stream_options={'include_usage': True},
                return_token_ids=True)
    request = urllib.request.Request(base_url + '/v1/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic_ns()
    chunks, ids, usage, prompt_ids, response_id = [], [], None, None, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event['error'])
            response_id = event.get('id', response_id)
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice['index'] != 0:
                    raise RuntimeError('Only one completion is supported')
                if choice.get('prompt_token_ids') is not None:
                    prompt_ids = choice['prompt_token_ids']
                delta = choice.get('token_ids')
                if delta is None and choice.get('text'):
                    raise RuntimeError('Server did not return token IDs; cannot count tokens accurately')
                if delta:
                    now = time.monotonic_ns()
                    chunks.append(dict(received_ns=now, token_count=len(delta)))
                    ids.extend(delta)
    ended = time.monotonic_ns()
    if (not chunks or not usage or prompt_ids != workload['input_token_ids'] or
            usage['prompt_tokens'] != workload['input_tokens'] or
            usage['completion_tokens'] != len(ids) or len(ids) != workload['output_tokens']):
        raise RuntimeError('Missing/mismatched prompt, output token IDs or usage')
    # A chunk may contain several tokens. Their individual arrival times are not
    # recoverable; only the first-token and last-token observation bounds are known.
    return dict(response_id=response_id, output_token_ids=ids, output_sha256=digest_json(ids),
                usage=usage, chunks=chunks, ttft_ms=(chunks[0]['received_ns']-started)/1e6,
                tpot_ms=(chunks[-1]['received_ns']-chunks[0]['received_ns'])/1e6/(len(ids)-1),
                request_ms=(ended-started)/1e6,
                token_timing='client_receive_bounds; coalesced chunks share an observation time')


def main():
    import argparse
    from sparsedvfs.hardware import load_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    args = p.parse_args()
    print(json.dumps(completion(args.url, load_json(args.workload)), indent=2))


if __name__ == '__main__':
    main()
