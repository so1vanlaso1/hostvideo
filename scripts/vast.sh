#!/usr/bin/env bash
# Manage a detached ComfyUI session; it survives an SSH disconnect.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
ACTION="${1:-status}"
WAIT_SECONDS="${H3_START_TIMEOUT:-300}"
[[ "$WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'H3_START_TIMEOUT must be a positive number of seconds.' >&2; exit 2; }
case "$ACTION" in
  start|stop|restart|status|logs) ;;
  *) echo 'Usage: bash scripts/vast.sh {start|stop|restart|status|logs}' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected one action.' >&2; exit 2; }
command -v tmux >/dev/null || { echo 'Run bash setup.sh first (tmux is missing).' >&2; exit 1; }
command -v flock >/dev/null || { echo 'Run bash setup.sh first (flock is missing).' >&2; exit 1; }
mkdir -p "$H3_ROOT/logs" "$H3_ROOT/locks"
SOCKET="h3-$(printf '%s' "$H3_ROOT" | cksum | cut -d ' ' -f 1)"
LOG="$H3_ROOT/logs/comfy.log"
URL="http://127.0.0.1:$H3_PORT"
running() { tmux -L "$SOCKET" has-session -t '=comfy' 2>/dev/null; }
ready() { curl --fail --silent --max-time 2 --output /dev/null "$URL/system_stats"; }

if [[ "$ACTION" == logs ]]; then
  touch "$LOG"
  exec tail -n 60 -f "$LOG"
fi
if [[ "$ACTION" == status ]]; then
  if running; then
    if ready; then echo "ComfyUI ready: $URL"; else echo "ComfyUI starting or unhealthy; see $LOG"; exit 1; fi
  else
    echo "Managed ComfyUI is stopped. Log: $LOG"
    exit 1
  fi
  exit 0
fi

exec 7>"$H3_ROOT/locks/service.lock"
flock -n 7 || { echo 'Another service command is running.' >&2; exit 1; }
if [[ "$ACTION" == stop || "$ACTION" == restart ]]; then
  if running; then
    tmux -L "$SOCKET" kill-session -t '=comfy'
    # Allow the foreground server to release its lock after termination.
    for ((attempt=0; attempt<30; attempt++)); do
      if flock -n "$H3_ROOT/locks/server.lock" true; then break; fi
      sleep 1
    done
    flock -n "$H3_ROOT/locks/server.lock" true || { echo 'Server lock is still held after stopping tmux; inspect the process before restarting.' >&2; exit 1; }
    echo 'ComfyUI stopped.'
  else
    echo 'Managed ComfyUI is already stopped.'
  fi
  [[ "$ACTION" == restart ]] || exit 0
fi
if running; then
  if ready; then echo "ComfyUI already ready: $URL"; exit 0; fi
  echo 'ComfyUI session already exists; checking startup.'
else
  [[ -x "$H3_ROOT/venv/bin/python" ]] || { echo 'Run bash setup.sh first.' >&2; exit 1; }
  flock -n "$H3_ROOT/locks/server.lock" true || { echo 'ComfyUI or setup already holds the server lock.' >&2; exit 1; }
  if "$H3_ROOT/venv/bin/python" - "$H3_PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(2)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
  then
    echo "Port $H3_PORT is already in use; stop the existing server or select another H3_PORT." >&2
    exit 1
  fi
  LAUNCHER="$H3_ROOT/locks/start-comfy.sh"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'export H3_ROOT=%q H3_PROFILE=%q H3_PORT=%q\n' "$H3_ROOT" "$H3_PROFILE" "$H3_PORT"
    printf 'exec bash %q >> %q 2>&1\n' "$SOURCE_DIR/scripts/start_comfy.sh" "$LOG"
  } > "$LAUNCHER"
  # Close the service lock in the tmux child so future commands can acquire it.
  tmux -L "$SOCKET" new-session -d -s comfy -c "$SOURCE_DIR" bash "$LAUNCHER" 7>&-
fi
echo "Waiting for ComfyUI startup; log: $LOG"
DEADLINE=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < DEADLINE )); do
  if ! running; then
    echo "ComfyUI exited during startup. Last log lines:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  if ready; then
    echo "ComfyUI ready: $URL"
    echo 'For browser access, forward this port with the SSH command described in README.md.'
    exit 0
  fi
  sleep 2
done
echo "ComfyUI is still starting or unhealthy after ${WAIT_SECONDS}s; session left running. See $LOG or run bash scripts/vast.sh status." >&2
exit 1
