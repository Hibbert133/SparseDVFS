#!/usr/bin/env python3
"""Verify package checksums; verify upstream blob only if distributed."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
failures = []
for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    target = (root / name).resolve()
    if root.resolve() not in target.parents:
        raise ValueError('checksum path escapes package')
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        failures.append(name)
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
if manifest['vendor'].get('distributed', True):
    raw = (root / manifest['vendor']['file']).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != manifest['vendor']['git_blob_sha1']:
        failures.append('upstream Git blob')
if failures:
    raise SystemExit('FAILED: ' + ', '.join(failures))
print('PASS: delivered checksums; upstream reference pinned in SOURCE_MANIFEST.json')
