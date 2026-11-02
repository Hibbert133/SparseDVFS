"""Compare restored controls, allowing an unlocked EMC governor to move rate."""
def restored(before, after):
    if before['domains'] != after['domains']:
        return False
    a, b = before.get('bpmp'), after.get('bpmp')
    if a is None or b is None:
        return a == b
    if any(a[k] != b[k] for k in ('lock', 'halt', 'lock_path', 'halt_path', 'rate_path')):
        return False
    return a['rate'] == b['rate'] if a['lock'] else True
