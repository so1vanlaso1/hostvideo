#!/usr/bin/env bash
# One-command setup after cloning on Vast.
set -euo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/setup_vast.sh" "$@"
