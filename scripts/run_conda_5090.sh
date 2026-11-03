#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Use the activated Conda interpreter, or an explicitly selected interpreter.
exec "${SPARSEDVFS_PYTHON:-python}" -B -m rtx5090.conda "$@"
