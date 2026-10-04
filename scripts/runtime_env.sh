#!/usr/bin/env bash
# Shared by the Vast helpers. Explicit environment values override saved settings.
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$SOURCE_DIR/.h3-vast.env" ]]; then
  _h3_root="${H3_ROOT-}"
  _h3_profile="${H3_PROFILE-}"
  _h3_port="${H3_PORT-}"
  source "$SOURCE_DIR/.h3-vast.env"
  H3_ROOT="${_h3_root:-${H3_ROOT-}}"
  H3_PROFILE="${_h3_profile:-${H3_PROFILE-}}"
  H3_PORT="${_h3_port:-${H3_PORT-}}"
  unset _h3_root _h3_profile _h3_port
fi
if [[ -z "${H3_ROOT:-}" ]]; then
  if command -v mountpoint >/dev/null && mountpoint -q /data; then
    H3_ROOT=/data/minimax-h3
  elif [[ -d /workspace ]]; then
    H3_ROOT=/workspace/minimax-h3
  else
    H3_ROOT="$SOURCE_DIR/.runtime"
  fi
fi
if [[ -d "$H3_ROOT" ]]; then
  H3_ROOT="$(cd -- "$H3_ROOT" && pwd -P)"
fi
H3_PROFILE="${H3_PROFILE:-int8-encoder}"
H3_PORT="${H3_PORT:-8188}"
[[ "$H3_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$H3_PORT >= 1 && 10#$H3_PORT <= 65535 )) || { echo 'H3_PORT must be between 1 and 65535.' >&2; return 2; }
H3_PORT=$((10#$H3_PORT))
export H3_ROOT H3_PROFILE H3_PORT
