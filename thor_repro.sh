#!/usr/bin/env bash
set -euo pipefail
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -B -m sparsedvfs_vllm.host "$@"
