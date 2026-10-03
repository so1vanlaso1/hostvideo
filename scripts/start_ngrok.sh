#!/usr/bin/env bash
set -euo pipefail
H3_ROOT="${H3_ROOT:-/data/minimax-h3}"
PORT="${H3_PORT:-8188}"
NGROK_BIN="${NGROK_BIN:-ngrok}"
CONFIG="$H3_ROOT/private/ngrok.yml"
POLICY="$H3_ROOT/private/ngrok-policy.json"
[[ -f "$CONFIG" && -f "$POLICY" ]] || { echo 'Run configure_ngrok.py first.' >&2; exit 1; }
command -v "$NGROK_BIN" >/dev/null || { echo 'Install ngrok v3 from https://ngrok.com/download/linux, or set NGROK_BIN.' >&2; exit 1; }
curl --fail --silent --output /dev/null "http://127.0.0.1:$PORT/system_stats" || { echo 'Start ComfyUI before ngrok.' >&2; exit 1; }
"$NGROK_BIN" config check --config "$CONFIG"
exec "$NGROK_BIN" http "http://127.0.0.1:$PORT" --config "$CONFIG" --traffic-policy-file "$POLICY"
