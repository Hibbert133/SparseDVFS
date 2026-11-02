"""Target-container idle shutdown handshake; no network/admin endpoint."""
import argparse
import os
import signal
import time
from pathlib import Path
from sparsedvfs.hardware import load_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-dir', required=True)
    p.add_argument('--action', choices=['prepare', 'close'], default='close')
    args = p.parse_args()
    root = Path(args.worker_dir)
    nonce = str(time.time_ns())
    write_json(root / 'close_request.json', {'nonce': nonce, 'action': args.action})
    pid = load_json(root / 'worker_identity.json')['pid']
    os.kill(pid, signal.SIGUSR2)
    deadline = time.monotonic() + 85
    while time.monotonic() < deadline:
        if (root / 'close_ack.json').exists():
            ack = load_json(root / 'close_ack.json')
            if ack['nonce'] == nonce:
                if not ack['closed']:
                    raise RuntimeError(ack)
                print('Worker ' + args.action + ' acknowledged.')
                return
        time.sleep(0.1)
    raise TimeoutError('Worker close acknowledgement missing; inspect snapshot/restore state')


if __name__ == '__main__':
    main()
