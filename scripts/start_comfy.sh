#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
PYTHON="$H3_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Run setup_vast.sh first.' >&2; exit 1; }
mkdir -p "$H3_ROOT/locks" "$H3_ROOT/logs"
exec 8>"$H3_ROOT/locks/server.lock"
flock -n 8 || { echo 'The managed ComfyUI server is already running.' >&2; exit 1; }
exec "$PYTHON" -m h3_pipeline --root "$H3_ROOT" serve --port "$H3_PORT" --profile "$H3_PROFILE"
