#!/usr/bin/env bash
# Shared by the Vast helpers. Explicit environment values override saved settings.
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$SOURCE_DIR/.h3-vast.env" ]]; then
  _h3_root="${H3_ROOT-}"
  _h3_profile="${H3_PROFILE-}"
  _h3_port="${H3_PORT-}"
  _h3_memory_mode="${H3_MEMORY_MODE-}"
  _h3_reserve_vram="${H3_RESERVE_VRAM-}"
  source "$SOURCE_DIR/.h3-vast.env"
  H3_ROOT="${_h3_root:-${H3_ROOT-}}"
  H3_PROFILE="${_h3_profile:-${H3_PROFILE-}}"
  H3_PORT="${_h3_port:-${H3_PORT-}}"
  H3_MEMORY_MODE="${_h3_memory_mode:-${H3_MEMORY_MODE-}}"
  H3_RESERVE_VRAM="${_h3_reserve_vram:-${H3_RESERVE_VRAM-}}"
  unset _h3_root _h3_profile _h3_port _h3_memory_mode _h3_reserve_vram
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
H3_MEMORY_MODE="${H3_MEMORY_MODE:-auto}"
case "$H3_MEMORY_MODE" in
  auto|static|dynamic) ;;
  *) echo 'H3_MEMORY_MODE must be auto, static or dynamic.' >&2; return 2 ;;
esac
if [[ -n "${H3_RESERVE_VRAM:-}" ]]; then
  [[ "$H3_RESERVE_VRAM" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'H3_RESERVE_VRAM must be a nonnegative number of GiB.' >&2; return 2; }
  export H3_RESERVE_VRAM
fi
[[ "$H3_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$H3_PORT >= 1 && 10#$H3_PORT <= 65535 )) || { echo 'H3_PORT must be between 1 and 65535.' >&2; return 2; }
H3_PORT=$((10#$H3_PORT))
export H3_ROOT H3_PROFILE H3_PORT H3_MEMORY_MODE
