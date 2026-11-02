"""Explicit recovery after a crashed controller; never imported for implicit writes."""
import argparse
import fcntl
from pathlib import Path
from sparsedvfs.hardware import load_json, restore, integer, write_json


def validate(saved):
    rows = [(d['min_path'], d['min']) for d in saved['domains']] + [(d['max_path'], d['max']) for d in saved['domains']]
    if saved.get('bpmp'):
        b = saved['bpmp']
        rows += [(b[k + '_path'], b[k]) for k in ('rate', 'lock', 'halt')]
    for path, value in rows:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to('/host-dvfs') or not isinstance(value, int) or value < 0:
            raise ValueError('Recovery accepts only saved nonnegative values below /host-dvfs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    saved = load_json(args.snapshot)
    validate(saved)
    if not args.apply:
        print('Validated snapshot. No writes; repeat with --apply after the controller exits.')
        return
    with open('/run/lock/sparsedvfs-thor.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A controller still owns the global lock; stop it before recovery') from exc
        if saved.get('bpmp'):
            Path(saved['bpmp']['halt_path']).write_text('1')
        restore(saved)
        rows = [(d['min_path'], d['min']) for d in saved['domains']] + [(d['max_path'], d['max']) for d in saved['domains']]
        if saved.get('bpmp'):
            b = saved['bpmp']
            rows += [(b[k + '_path'], b[k]) for k in ('lock', 'halt')]
            if b['lock']:
                rows.append((b['rate_path'], b['rate']))
        status = dict(restored=all(integer(p) == value for p, value in rows),
                      readbacks=[dict(path=p, expected=v, actual=integer(p)) for p, v in rows])
        write_json(Path(args.snapshot).with_name('manual_restore_status.json'), status)
        if not status['restored']:
            raise RuntimeError('Recovery readback mismatch')
        print('Recovery readback matched the saved snapshot.')


if __name__ == '__main__':
    main()
