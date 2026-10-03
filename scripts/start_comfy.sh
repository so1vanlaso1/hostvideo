#!/usr/bin/env bash
set -euo pipefail
H3_ROOT="${H3_ROOT:-/data/minimax-h3}"
export H3_ROOT
PORT="${H3_PORT:-8188}"
PYTHON="$H3_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Run setup_vast.sh first.' >&2; exit 1; }
mkdir -p "$H3_ROOT/locks" "$H3_ROOT/logs"
exec 8>"$H3_ROOT/locks/server.lock"
flock -n 8 || { echo 'The managed ComfyUI server is already running.' >&2; exit 1; }
exec "$PYTHON" -m h3_pipeline --root "$H3_ROOT" serve --port "$PORT" --profile "${H3_PROFILE:-primary}"
